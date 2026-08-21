"""Strictly read-only Playwright adapter for visible Facebook group posts."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NoReturn, cast
from urllib.parse import parse_qs, urljoin, urlsplit

from playwright.async_api import BrowserContext, Error, Locator, Page, Playwright, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from lead_agent.config import Settings
from lead_agent.facebook_state import (
    FacebookPageState,
    FacebookSafetyStop,
    assess_facebook_page,
)
from lead_agent.groups import FacebookGroup
from lead_agent.models import (
    FacebookPost,
    canonicalize_facebook_url,
    is_facebook_comment_ui_text,
    is_post_text_expansion,
    normalize_post_text,
)
from lead_agent.scanner import (
    FacebookReadDiagnostics,
    FacebookReadResult,
    TransientFacebookReadError,
)

FACEBOOK_HOME = "https://www.facebook.com/"
POST_PATH_PATTERN = re.compile(r"/(?:posts|permalink)/([^/?#]+)", re.IGNORECASE)
SAFE_FILENAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")
STORY_MESSAGE_SELECTOR = (
    '[data-ad-rendering-role="story_message"], '
    '[data-ad-preview="message"], '
    '[data-ad-comet-preview="message"]'
)
LOADING_STATE_SELECTOR = (
    '[role="progressbar"], [aria-busy="true"], [data-visualcompletion="loading-state"]'
)
SEE_MORE_SUFFIX_PATTERN = re.compile(r"(?:\s*(?:…|\.\.\.)?\s*see more)+\s*$", re.IGNORECASE)
REPEATED_MESSAGE_PREFIX_LENGTH = 80
BROWSER_NETWORK_ERROR_MARKERS = (
    "err_internet_disconnected",
    "err_network_changed",
    "err_name_not_resolved",
    "err_connection_timed_out",
    "err_connection_reset",
    "no internet",
    "you are offline",
)


class FacebookBrowserError(RuntimeError):
    """Raised when the dedicated Playwright browser cannot be started safely."""


def browser_launch_arguments(settings: Settings) -> list[str]:
    """Keep a headed automation window out of the user's active work area by default."""
    return ["--start-minimized"] if settings.browser_start_minimized else []


@dataclass(slots=True)
class _FeedTelemetry:
    started_at: float
    iterations: int = 0
    scrolls: int = 0
    story_nodes_seen: int = 0
    visible_articles_seen: int = 0
    detached_nodes: int = 0
    progress_events: int = 0
    top_level_story_nodes_seen: int = 0
    collapsed_unexpanded_observations: int = 0
    comment_observations: int = 0
    nested_article_observations: int = 0
    short_text_observations: int = 0
    feed_movement_events: int = 0
    feed_height_growth_events: int = 0
    loading_observations: int = 0
    duplicate_identities: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class FacebookPostCandidate:
    """Sanitized semantic fields collected from one possible top-level story."""

    full_text: str
    semantic_messages: tuple[str, ...] = ()
    automatic_texts: tuple[str, ...] = ()
    hrefs: tuple[str, ...] = ()
    article_label: str | None = None
    author_name: str | None = None
    is_nested_article: bool = False
    is_collapsed_message: bool = False


def is_facebook_comment_label(label: str | None) -> bool:
    """Identify Facebook's semantic labels for comment and reply articles."""
    normalized = normalize_post_text(label or "").casefold()
    return normalized.startswith(("comment by ", "reply by "))


def clean_facebook_message_text(value: str) -> str:
    """Remove Facebook expansion controls and duplicated collapsed-message prefixes."""
    normalized = normalize_post_text(value)
    normalized = SEE_MORE_SUFFIX_PATTERN.sub("", normalized).strip()
    if len(normalized) >= REPEATED_MESSAGE_PREFIX_LENGTH * 2:
        prefix = normalized[:REPEATED_MESSAGE_PREFIX_LENGTH]
        repeated_at = normalized.find(prefix, REPEATED_MESSAGE_PREFIX_LENGTH)
        if repeated_at >= REPEATED_MESSAGE_PREFIX_LENGTH:
            normalized = normalized[:repeated_at].rstrip(" …")
    return normalize_post_text(normalized)


def message_text_requires_expansion(value: str) -> bool:
    """Identify a collapsed Facebook message whose visible text ends at “See more”."""
    return bool(SEE_MORE_SUFFIX_PATTERN.search(normalize_post_text(value)))


def is_browser_network_error_text(value: str) -> bool:
    """Recognize Chromium's content-free network error pages as transient outages."""
    normalized = " ".join(value.casefold().split())
    return any(marker in normalized for marker in BROWSER_NETWORK_ERROR_MARKERS)


def extract_post_id(url: str) -> str | None:
    """Extract a stable visible Facebook post identifier when the URL exposes one."""
    parts = urlsplit(url)
    match = POST_PATH_PATTERN.search(parts.path)
    if match:
        return match.group(1)
    query = parse_qs(parts.query)
    for key in ("story_fbid", "fbid", "multi_permalinks"):
        values = query.get(key)
        if values and values[0].strip():
            return values[0].strip()
    return None


def facebook_group_key(url: str) -> str | None:
    """Return the configured group path identifier from a Facebook URL."""
    segments = [segment for segment in urlsplit(url).path.split("/") if segment]
    if len(segments) < 2 or segments[0].casefold() != "groups":
        return None
    return segments[1].casefold()


def select_facebook_permalink(hrefs: Sequence[str], group_url: str) -> str | None:
    """Choose the strongest visible Facebook URL that identifies an individual post."""
    candidates: list[tuple[int, int, str]] = []
    for href in hrefs:
        absolute = urljoin(group_url, href)
        parts = urlsplit(absolute)
        hostname = (parts.hostname or "").casefold()
        if hostname != "facebook.com" and not hostname.endswith(".facebook.com"):
            continue
        if extract_post_id(absolute) is None:
            continue
        query = parse_qs(parts.query)
        if POST_PATH_PATTERN.search(parts.path):
            rank = 0
        elif "story_fbid" in query or "multi_permalinks" in query:
            rank = 1
        else:
            continue
        candidates.append((rank, len(candidates), canonicalize_facebook_url(absolute)))
    return min(candidates)[2] if candidates else None


def select_message_text(
    full_text: str,
    semantic_messages: Sequence[str],
    automatic_texts: Sequence[str],
    *,
    min_length: int,
) -> str | None:
    """Prefer Facebook's semantic message nodes, with conservative visible fallbacks."""

    def candidates(values: Sequence[str]) -> list[str]:
        valid: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = clean_facebook_message_text(value)
            if len(normalized) >= min_length and normalized not in seen:
                seen.add(normalized)
                valid.append(normalized)
        return valid

    semantic = candidates(semantic_messages)
    if semantic:
        selected = max(semantic, key=len)
        expansions = [
            value
            for value in candidates(automatic_texts)
            if is_post_text_expansion(selected, value)
        ]
        return max((selected, *expansions), key=len)
    automatic = candidates(automatic_texts)
    if automatic:
        return max(automatic, key=len)
    fallback = candidates([full_text])
    if fallback:
        return fallback[0]
    return None


def build_facebook_post(
    candidate: FacebookPostCandidate,
    group: FacebookGroup,
    *,
    min_length: int,
) -> FacebookPost | None:
    """Convert sanitized DOM semantics into a group-scoped post or reject them safely."""
    post, _ = _build_facebook_post_with_reason(candidate, group, min_length=min_length)
    return post


def _build_facebook_post_with_reason(
    candidate: FacebookPostCandidate,
    group: FacebookGroup,
    *,
    min_length: int,
) -> tuple[FacebookPost | None, str | None]:
    """Build a post while preserving a content-free reason for conservative rejections."""
    if candidate.is_nested_article or is_facebook_comment_label(candidate.article_label):
        return None, "nested_article" if candidate.is_nested_article else "comment"
    post_text = select_message_text(
        candidate.full_text,
        candidate.semantic_messages,
        candidate.automatic_texts,
        min_length=min_length,
    )
    if post_text is None:
        return None, "short_text"
    if is_facebook_comment_ui_text(post_text):
        return None, "comment"
    if candidate.is_collapsed_message and not any(
        is_post_text_expansion(clean_facebook_message_text(value), post_text)
        for value in candidate.semantic_messages
    ):
        return None, "collapsed_unexpanded"
    post_url = select_facebook_permalink(candidate.hrefs, group.url)
    if post_url is not None and facebook_group_key(post_url) != facebook_group_key(group.url):
        return None, "outside_group"
    return (
        FacebookPost(
            external_post_id=extract_post_id(post_url) if post_url else None,
            post_url=post_url,
            group_id=group.id,
            group_name=group.name,
            author_name=candidate.author_name,
            post_text=post_text,
        ),
        None,
    )


def merge_facebook_post(
    collected: dict[str, FacebookPost],
    incoming: FacebookPost,
) -> None:
    """Merge a later permalink-bearing rendering into one earlier content-only story."""
    existing = collected.get(incoming.identity_key)
    if existing is None:
        compatible = {
            id(post): post
            for post in collected.values()
            if post.group_id == incoming.group_id
            and post.text_hash == incoming.text_hash
            and (post.post_url is None or incoming.post_url is None)
        }
        if len(compatible) == 1:
            existing = next(iter(compatible.values()))
    if existing is None:
        collected[incoming.identity_key] = incoming
        return
    if existing.post_url is None and incoming.post_url is not None:
        existing.post_url = incoming.post_url
        existing.external_post_id = incoming.external_post_id
    if existing.author_name is None and incoming.author_name is not None:
        existing.author_name = incoming.author_name
    if is_post_text_expansion(existing.post_text, incoming.post_text):
        existing.post_text = incoming.post_text
        existing.text_hash = incoming.text_hash


def cleanup_old_screenshots(
    directory: Path,
    *,
    retention_days: int,
    now: datetime | None = None,
) -> int:
    """Delete only expired PNG diagnostics from the configured screenshot directory."""
    if not directory.exists():
        return 0
    cutoff = (now or datetime.now(UTC)) - timedelta(days=retention_days)
    deleted = 0
    for screenshot in directory.glob("*.png"):
        modified = datetime.fromtimestamp(screenshot.stat().st_mtime, tz=UTC)
        if modified < cutoff:
            screenshot.unlink()
            deleted += 1
    return deleted


class FacebookReadOnlyBrowser:  # pragma: no cover - requires an interactive Facebook session
    """Persistent, headed browser that exposes no Facebook write operations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._active_feed_telemetry: _FeedTelemetry | None = None

    async def __aenter__(self) -> FacebookReadOnlyBrowser:
        self.settings.require_read_only_mode()
        self.settings.facebook_profile_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.settings.facebook_profile_path.chmod(0o700)
        self.settings.screenshot_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.settings.screenshot_dir.chmod(0o700)
        cleanup_old_screenshots(
            self.settings.screenshot_dir,
            retention_days=self.settings.screenshot_retention_days,
        )
        try:
            self._playwright = await async_playwright().start()
            if self.settings.browser_channel is None:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    str(self.settings.facebook_profile_path),
                    headless=self.settings.browser_headless,
                    accept_downloads=False,
                    locale="en-US",
                    args=browser_launch_arguments(self.settings),
                )
            else:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    str(self.settings.facebook_profile_path),
                    headless=self.settings.browser_headless,
                    accept_downloads=False,
                    locale="en-US",
                    channel=self.settings.browser_channel,
                    args=browser_launch_arguments(self.settings),
                )
        except Exception as error:
            if self._playwright is not None:
                await self._playwright.stop()
            self._playwright = None
            raise FacebookBrowserError(
                "Playwright could not start the dedicated Facebook browser; "
                "verify Chromium is installed and the profile is not already open"
            ) from error
        self._context.set_default_timeout(self.settings.facebook_post_load_timeout_seconds * 1000)
        self._context.set_default_navigation_timeout(
            self.settings.facebook_navigation_timeout_seconds * 1000
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def manual_login(self) -> None:
        """Open Facebook for human-only login and verify the resulting page."""
        page = await self._page(fresh=True)
        try:
            await page.goto(FACEBOOK_HOME, wait_until="domcontentloaded")
        except Error:
            await self._stop(
                page,
                "manual-login",
                FacebookPageState.UNEXPECTED,
                "The Facebook login page could not be opened safely",
            )
        print(
            "Complete Facebook login manually in the browser. "
            "Handle any MFA yourself; this program will not enter credentials."
        )
        await asyncio.to_thread(input, "After Facebook finishes loading, press Enter here... ")
        try:
            await page.wait_for_load_state("domcontentloaded")
        except Error:
            await self._stop(
                page,
                "manual-login",
                FacebookPageState.UNEXPECTED,
                "The Facebook login did not finish loading safely",
            )
        await self._require_normal_page(page, group_id="manual-login")

    async def read_group(
        self,
        group: FacebookGroup,
        *,
        max_posts: int,
        attempt: int = 0,
        known_identities: frozenset[str] = frozenset(),
    ) -> FacebookReadResult:
        """Wait for and read visible posts from one allowlisted group."""
        page = await self._page(fresh=True)
        try:
            await page.goto(group.url, wait_until="domcontentloaded")
            await self._require_normal_page(page, group_id=group.id)
            if facebook_group_key(page.url) != facebook_group_key(group.url):
                await self._stop(
                    page,
                    group.id,
                    FacebookPageState.UNEXPECTED,
                    "Facebook navigated outside the configured group",
                )
        except FacebookSafetyStop:
            raise
        except PlaywrightTimeoutError as error:
            try:
                await self._require_normal_page(page, group_id=group.id)
            except FacebookSafetyStop:
                raise
            except Error:
                # A replaced/closed body during timeout assessment remains a bounded transient
                # failure. It never weakens explicit login, CAPTCHA, checkpoint, or domain stops.
                pass
            raise TransientFacebookReadError(
                stage="navigation",
                kind="timeout",
                screenshot_path=await self._capture_failure(
                    page, group.id, "transient-navigation-timeout"
                ),
            ) from error
        except Error as error:
            raise TransientFacebookReadError(
                stage="navigation",
                kind="playwright_error",
                screenshot_path=await self._capture_failure(
                    page, group.id, "transient-navigation-error"
                ),
            ) from error

        try:
            target_posts = max(1, max_posts - len(known_identities))
            result = await self._wait_for_readable_posts(
                page,
                group,
                max_posts=target_posts,
                attempt=attempt,
                known_identities=known_identities,
            )
            if (
                target_posts > 0
                and len(result.posts) / target_posts
                < self.settings.operations_minimum_group_post_yield_rate
            ):
                screenshot = await self._capture_failure(page, group.id, "severe-partial")
                return replace(result, severe_screenshot_path=screenshot)
            return result
        except FacebookSafetyStop:
            raise
        except PlaywrightTimeoutError as error:
            raise TransientFacebookReadError(
                stage="feed",
                kind="timeout",
                screenshot_path=await self._capture_failure(
                    page, group.id, "transient-feed-timeout"
                ),
            ) from error
        except Error as error:
            raise TransientFacebookReadError(
                stage="feed",
                kind="playwright_error",
                screenshot_path=await self._capture_failure(page, group.id, "transient-feed-error"),
            ) from error

    async def _wait_for_readable_posts(
        self,
        page: Page,
        group: FacebookGroup,
        *,
        max_posts: int,
        attempt: int = 0,
        known_identities: frozenset[str] = frozenset(),
    ) -> FacebookReadResult:
        """Retry through Facebook placeholders and transient feed re-renders."""
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        retry_hydration = attempt > 0
        scan_wait_seconds = (
            self.settings.facebook_retry_scan_max_wait_seconds
            if retry_hydration
            else self.settings.facebook_scan_max_wait_seconds
        )
        idle_seconds = (
            self.settings.facebook_retry_scan_idle_seconds
            if retry_hydration
            else self.settings.facebook_scan_idle_seconds
        )
        settle_seconds = (
            self.settings.facebook_retry_scroll_settle_seconds
            if retry_hydration
            else self.settings.facebook_scroll_settle_seconds
        )
        deadline = started_at + scan_wait_seconds
        initial_grace_deadline = started_at + 2
        permalink_grace_deadline: float | None = None
        last_feed_activity_at = started_at
        previous_feed_observation: tuple[int, int, int, int, int, int, int] | None = None
        visible_article_seen = False
        collected: dict[str, FacebookPost] = {}
        scrolls = 0
        telemetry = _FeedTelemetry(started_at=started_at)
        stop_reason = "timeout"

        self._active_feed_telemetry = telemetry
        try:
            while loop.time() < deadline:
                telemetry.iterations += 1
                await self._require_normal_page(page, group_id=group.id)
                observation = await self._observe_feed(page)
                if observation is not None:
                    if previous_feed_observation is not None:
                        (
                            previous_top,
                            previous_height,
                            previous_loading,
                            previous_container_top,
                            previous_container_height,
                            _previous_container_viewport,
                            previous_story_count,
                        ) = previous_feed_observation
                        (
                            current_top,
                            current_height,
                            current_loading,
                            current_container_top,
                            current_container_height,
                            _current_container_viewport,
                            current_story_count,
                        ) = observation
                        moved = (
                            current_top != previous_top
                            or current_container_top != previous_container_top
                        )
                        grew = (
                            current_height > previous_height
                            or current_container_height > previous_container_height
                        )
                        hydrated = current_loading < previous_loading
                        story_nodes_changed = current_story_count != previous_story_count
                        if moved or grew or hydrated or story_nodes_changed:
                            last_feed_activity_at = loop.time()
                        if moved:
                            telemetry.feed_movement_events += 1
                        if grew:
                            telemetry.feed_height_growth_events += 1
                    if observation[2] > 0:
                        telemetry.loading_observations += 1
                    previous_feed_observation = observation
                before_count = len(collected)
                try:
                    story_node_count = await page.locator(STORY_MESSAGE_SELECTOR).count()
                    telemetry.story_nodes_seen = max(
                        telemetry.story_nodes_seen,
                        story_node_count,
                    )
                    story_posts = await self._extract_story_posts(page, group, max_posts=max_posts)
                except Error:
                    telemetry.detached_nodes += 1
                    story_posts = []
                for story_post in story_posts:
                    if story_post.identity_key in known_identities:
                        telemetry.duplicate_identities.add(story_post.identity_key)
                        continue
                    self._merge_feed_post(collected, story_post, telemetry)

                if not story_posts or any(post.post_url is None for post in collected.values()):
                    try:
                        articles = await self._post_articles(page)
                        count = min(await articles.count(), max(max_posts * 5, 50))
                    except Error:
                        count = 0

                    visible_articles = 0
                    for index in range(count):
                        try:
                            article = articles.nth(index)
                            if not await article.is_visible(timeout=1000):
                                continue
                            visible_articles += 1
                            visible_article_seen = True
                            article_label = await article.get_attribute("aria-label", timeout=1000)
                            if is_facebook_comment_label(article_label):
                                telemetry.comment_observations += 1
                                continue
                            if await self._is_nested_article(article):
                                telemetry.nested_article_observations += 1
                                continue
                            legacy_post = await self._extract_article(article, group)
                        except Error:
                            # Facebook commonly replaces placeholder nodes while the feed hydrates.
                            telemetry.detached_nodes += 1
                            continue
                        if legacy_post is not None:
                            if legacy_post.identity_key in known_identities:
                                telemetry.duplicate_identities.add(legacy_post.identity_key)
                                continue
                            self._merge_feed_post(collected, legacy_post, telemetry)
                    telemetry.visible_articles_seen = max(
                        telemetry.visible_articles_seen,
                        visible_articles,
                    )

                if len(collected) > before_count:
                    telemetry.progress_events += 1

                if len(collected) >= max_posts:
                    selected = list(collected.values())[:max_posts]
                    if all(post.post_url is not None for post in selected):
                        return self._read_result(loop.time(), selected, telemetry, "target_met")
                    if permalink_grace_deadline is None:
                        permalink_grace_deadline = min(deadline, loop.time() + 2)
                    if loop.time() >= permalink_grace_deadline:
                        return self._read_result(
                            loop.time(),
                            selected,
                            telemetry,
                            "target_missing_permalinks",
                        )
                    await page.wait_for_timeout(250)
                    continue

                if not collected and loop.time() < initial_grace_deadline:
                    await page.wait_for_timeout(250)
                    continue
                if scrolls >= self.settings.facebook_max_scrolls:
                    stop_reason = "scroll_limit"
                    break
                if scrolls >= 3 and loop.time() - last_feed_activity_at >= idle_seconds:
                    stop_reason = "idle"
                    break

                try:
                    await self._scroll_for_more(page)
                except Error:
                    # The feed can replace its final story between the count and scroll calls.
                    # Keep already collected posts and let the next loop inspect the new DOM.
                    await page.wait_for_timeout(100)
                scrolls += 1
                telemetry.scrolls = scrolls
                remaining_seconds = max(deadline - loop.time(), 0)
                if remaining_seconds > 0:
                    settle_milliseconds = min(
                        int(settle_seconds * 1000),
                        max(int(remaining_seconds * 1000), 1),
                    )
                    await page.wait_for_timeout(settle_milliseconds)

            if collected or telemetry.top_level_story_nodes_seen:
                return self._read_result(
                    loop.time(),
                    list(collected.values())[:max_posts],
                    telemetry,
                    stop_reason,
                )

            await self._require_normal_page(page, group_id=group.id)
            if visible_article_seen:
                return self._read_result(
                    loop.time(),
                    [],
                    telemetry,
                    "no_readable_text",
                )
            reason = "No visible Facebook posts appeared before the safety timeout"
            await self._stop(page, group.id, FacebookPageState.UNEXPECTED, reason)
        finally:
            self._active_feed_telemetry = None

    @staticmethod
    def _merge_feed_post(
        collected: dict[str, FacebookPost],
        post: FacebookPost,
        telemetry: _FeedTelemetry,
    ) -> None:
        before_count = len(collected)
        merge_facebook_post(collected, post)
        if len(collected) == before_count:
            telemetry.duplicate_identities.add(post.identity_key)

    @staticmethod
    def _read_result(
        completed_at: float,
        posts: list[FacebookPost],
        telemetry: _FeedTelemetry,
        stop_reason: str,
    ) -> FacebookReadResult:
        permalinked = sum(post.post_url is not None for post in posts)
        return FacebookReadResult(
            posts=tuple(posts),
            diagnostics=FacebookReadDiagnostics(
                elapsed_ms=max(0, round((completed_at - telemetry.started_at) * 1000)),
                iterations=telemetry.iterations,
                scrolls=telemetry.scrolls,
                story_nodes_seen=telemetry.story_nodes_seen,
                visible_articles_seen=telemetry.visible_articles_seen,
                readable_posts=len(posts),
                permalinked_posts=permalinked,
                missing_permalinks=max(0, len(posts) - permalinked),
                detached_nodes=telemetry.detached_nodes,
                duplicate_identities=len(telemetry.duplicate_identities),
                progress_events=telemetry.progress_events,
                top_level_story_nodes_seen=telemetry.top_level_story_nodes_seen,
                collapsed_unexpanded_observations=(telemetry.collapsed_unexpanded_observations),
                comment_observations=telemetry.comment_observations,
                nested_article_observations=telemetry.nested_article_observations,
                short_text_observations=telemetry.short_text_observations,
                feed_movement_events=telemetry.feed_movement_events,
                feed_height_growth_events=telemetry.feed_height_growth_events,
                loading_observations=telemetry.loading_observations,
                stop_reason=stop_reason,
            ),
        )

    async def _observe_feed(
        self,
        page: Page,
    ) -> tuple[int, int, int, int, int, int, int] | None:
        """Measure window and inner-feed progress without reading post content."""
        try:
            raw = await page.evaluate(
                """
                selectors => {
                    const root = document.scrollingElement || document.documentElement;
                    const messages = Array.from(
                        document.querySelectorAll(selectors.storyMessage)
                    );
                    const topLevel = messages.filter(node => {
                        const article = node.closest('article, [role=article]');
                        const label = (article?.getAttribute('aria-label') || '').toLowerCase();
                        return Boolean(node.getClientRects().length)
                            && !label.startsWith('comment by ')
                            && !label.startsWith('reply by ')
                            && !article?.parentElement?.closest('article, [role=article]');
                    });
                    const anchor = topLevel[topLevel.length - 1];
                    const ancestors = [];
                    for (let node = anchor?.parentElement; node; node = node.parentElement) {
                        ancestors.push(node);
                    }
                    const candidates = [
                        ...ancestors,
                        anchor?.closest('[role="feed"]'),
                        ...Array.from(
                            document.querySelectorAll('[role="feed"], main, [role="main"]')
                        ),
                        root,
                    ].filter((node, index, values) => node && values.indexOf(node) === index);
                    const isScrollable = node => {
                        if (!node || node.scrollHeight <= node.clientHeight + 20) return false;
                        if (node === root) return true;
                        const overflowY = window.getComputedStyle(node).overflowY;
                        return ['auto', 'scroll', 'overlay'].includes(overflowY);
                    };
                    const scroller = candidates.find(isScrollable) || root;
                    const loading = document.querySelectorAll(selectors.loadingState).length;
                    const topLevelStories = topLevel.length;
                    return [
                        Math.round(window.scrollY || window.pageYOffset || 0),
                        Math.round(root?.scrollHeight || 0),
                        Math.min(loading, 100),
                        Math.round(scroller?.scrollTop || 0),
                        Math.round(scroller?.scrollHeight || 0),
                        Math.round(scroller?.clientHeight || 0),
                        topLevelStories,
                    ];
                }
                """,
                {
                    "storyMessage": STORY_MESSAGE_SELECTOR,
                    "loadingState": LOADING_STATE_SELECTOR,
                },
            )
        except (Error, TypeError):
            return None
        if isinstance(raw, list) and len(raw) == 7 and all(isinstance(value, int) for value in raw):
            return cast(tuple[int, int, int, int, int, int, int], tuple(raw))
        return None

    async def _scroll_for_more(self, page: Page) -> None:
        """Scroll the nearest scrollable feed container after selecting a top-level story."""
        moved = await page.evaluate(
            """
            storyMessageSelector => {
                const messages = Array.from(
                    document.querySelectorAll(storyMessageSelector)
                );
                const topLevel = messages.filter(node => {
                    const article = node.closest('article, [role=article]');
                    const label = (article?.getAttribute('aria-label') || '').toLowerCase();
                    return Boolean(node.getClientRects().length)
                        && !label.startsWith('comment by ')
                        && !label.startsWith('reply by ')
                        && !article?.parentElement?.closest('article, [role=article]');
                });
                const anchor = topLevel[topLevel.length - 1];
                const anchorRect = anchor?.getBoundingClientRect();
                if (anchor && anchorRect && anchorRect.top >= window.innerHeight) {
                    anchor.scrollIntoView({block: 'end', inline: 'nearest'});
                }
                const root = document.scrollingElement || document.documentElement;
                const ancestors = [];
                for (let node = anchor?.parentElement; node; node = node.parentElement) {
                    ancestors.push(node);
                }
                const candidates = [
                    ...ancestors,
                    anchor?.closest('[role="feed"]'),
                    ...Array.from(
                        document.querySelectorAll('[role="feed"], main, [role="main"]')
                    ),
                    root,
                ].filter((node, index, values) => node && values.indexOf(node) === index);
                const isScrollable = node => {
                    if (!node || node.scrollHeight <= node.clientHeight + 20) return false;
                    if (node === root) return true;
                    const overflowY = window.getComputedStyle(node).overflowY;
                    return ['auto', 'scroll', 'overlay'].includes(overflowY);
                };
                const scroller = candidates.find(isScrollable) || root;
                const distance = Math.max(Math.floor(window.innerHeight * 0.85), 600);
                const beforeWindow = Math.round(window.scrollY || window.pageYOffset || 0);
                const beforeScroller = Math.round(scroller.scrollTop || 0);
                if (scroller === root) {
                    root.scrollTop = Math.min(
                        root.scrollTop + distance,
                        root.scrollHeight - root.clientHeight
                    );
                    if (root.scrollTop === beforeScroller) window.scrollBy(0, distance);
                } else {
                    scroller.scrollTop = Math.min(
                        scroller.scrollTop + distance,
                        scroller.scrollHeight - scroller.clientHeight
                    );
                }
                return Math.round(window.scrollY || window.pageYOffset || 0) !== beforeWindow
                    || Math.round(scroller.scrollTop || 0) !== beforeScroller;
            }
            """,
            STORY_MESSAGE_SELECTOR,
        )
        if moved is False:
            viewport = page.viewport_size
            if viewport is None:
                raw_viewport = await page.evaluate("() => [window.innerWidth, window.innerHeight]")
                viewport = (
                    {"width": raw_viewport[0], "height": raw_viewport[1]}
                    if isinstance(raw_viewport, list)
                    and len(raw_viewport) == 2
                    and all(isinstance(value, int) for value in raw_viewport)
                    else None
                )
            if viewport is not None:
                await page.mouse.move(
                    round(viewport["width"] * 0.4),
                    round(viewport["height"] * 0.75),
                )
                await page.mouse.wheel(
                    0,
                    max(round(viewport["height"] * 0.85), 600),
                )

    async def _extract_story_posts(
        self,
        page: Page,
        group: FacebookGroup,
        *,
        max_posts: int,
    ) -> list[FacebookPost]:
        """Extract current Facebook story-message nodes and their nearest post permalinks."""
        messages = page.locator(STORY_MESSAGE_SELECTOR)
        count = min(await messages.count(), max(max_posts * 3, 20))
        collected: dict[str, FacebookPost] = {}

        try:
            snapshots = cast(
                list[dict[str, object]],
                await messages.evaluate_all(
                    r"""
                    (nodes, limit) => nodes.slice(0, limit).flatMap(node => {
                        const visible = Boolean(
                            node.getClientRects().length || node.offsetParent !== null
                        );
                        if (!visible) return [];
                        const article = node.closest('article, [role=article]');
                        const nested = Boolean(
                            article?.parentElement?.closest('article, [role=article]')
                        );
                        let owner = node;
                        let hrefs = [];
                        while (owner && owner.getAttribute('role') !== 'feed') {
                            hrefs = Array.from(owner.querySelectorAll('a[href]'))
                                .map(link => link.href)
                                .filter(href => (
                                    /\/groups\/[^/]+\/(posts|permalink)\//i.test(href)
                                    || /[?&](story_fbid|multi_permalinks)=/i.test(href)
                                ));
                            if (hrefs.length) break;
                            owner = owner.parentElement;
                        }
                        if (!hrefs.length && article) {
                            hrefs = Array.from(article.querySelectorAll('a[href]'))
                                .map(link => link.href)
                                .filter(href => (
                                    /\/groups\/[^/]+\/(posts|permalink)\//i.test(href)
                                    || /[?&](story_fbid|multi_permalinks)=/i.test(href)
                                ));
                        }
                        return [{
                            text: node.innerText || '',
                            content: node.textContent || '',
                            collapsed: /(?:…|\.\.\.)?\s*see more\s*$/i.test(
                                node.innerText || ''
                            ),
                            label: article?.getAttribute('aria-label') || null,
                            nested,
                            hrefs,
                        }];
                    })
                    """,
                    count,
                ),
            )
        except Error:
            snapshots = []

        telemetry = self._active_feed_telemetry
        if telemetry is not None:
            top_level_nodes = 0
            for item in snapshots:
                label_value = item.get("label")
                label = label_value if isinstance(label_value, str) else None
                nested_value = item.get("nested")
                nested = nested_value if isinstance(nested_value, bool) else False
                if not is_facebook_comment_label(label) and not nested:
                    top_level_nodes += 1
            telemetry.top_level_story_nodes_seen = max(
                telemetry.top_level_story_nodes_seen,
                top_level_nodes,
            )
        for snapshot in snapshots:
            label_value = snapshot.get("label")
            comment_label = label_value if isinstance(label_value, str) else None
            if is_facebook_comment_label(comment_label):
                self._record_rejection("comment")
                continue
            nested_value = snapshot.get("nested")
            is_nested_article = nested_value if isinstance(nested_value, bool) else False
            if is_nested_article:
                self._record_rejection("nested_article")
                continue
            text_value = snapshot.get("text")
            content_value = snapshot.get("content")
            collapsed_value = snapshot.get("collapsed")
            post_text = clean_facebook_message_text(
                text_value if isinstance(text_value, str) else ""
            )
            content_text = clean_facebook_message_text(
                content_value if isinstance(content_value, str) else ""
            )
            href_values = snapshot.get("hrefs")
            snapshot_hrefs = (
                tuple(value for value in href_values if isinstance(value, str))
                if isinstance(href_values, list)
                else ()
            )
            post, reason = _build_facebook_post_with_reason(
                FacebookPostCandidate(
                    full_text=post_text,
                    semantic_messages=(post_text,),
                    automatic_texts=(content_text,),
                    hrefs=snapshot_hrefs,
                    article_label=comment_label,
                    is_nested_article=is_nested_article,
                    is_collapsed_message=(
                        collapsed_value if isinstance(collapsed_value, bool) else False
                    ),
                ),
                group,
                min_length=self.settings.min_post_text_length,
            )
            if post is None:
                self._record_rejection(reason)
                continue
            merge_facebook_post(collected, post)

        if snapshots:
            return list(collected.values())[:max_posts]

        for index in range(count):
            try:
                message = messages.nth(index)
                if not await message.is_visible(timeout=1000):
                    continue
                comment_label = cast(
                    str | None,
                    await message.evaluate(
                        "node => node.closest('article, [role=article]')"
                        "?.getAttribute('aria-label')"
                    ),
                )
                if is_facebook_comment_label(comment_label):
                    self._record_rejection("comment")
                    continue
                is_nested_article = bool(
                    await message.evaluate(
                        "node => { const article = node.closest('article, [role=article]'); "
                        "return Boolean("
                        "article?.parentElement?.closest('article, [role=article]')"
                        "); }"
                    )
                )
                if is_nested_article:
                    self._record_rejection("nested_article")
                    continue
                telemetry = self._active_feed_telemetry
                if telemetry is not None:
                    telemetry.top_level_story_nodes_seen = max(
                        telemetry.top_level_story_nodes_seen,
                        1,
                    )
                raw_post_text = await message.inner_text(timeout=1000)
                post_text = clean_facebook_message_text(raw_post_text)
                content_text = clean_facebook_message_text(
                    cast(str, await message.evaluate("node => node.textContent || ''"))
                )
                hrefs = await self._nearest_post_hrefs(message)
                post, reason = _build_facebook_post_with_reason(
                    FacebookPostCandidate(
                        full_text=post_text,
                        semantic_messages=(post_text,),
                        automatic_texts=(content_text,),
                        hrefs=tuple(hrefs),
                        article_label=comment_label,
                        is_nested_article=is_nested_article,
                        is_collapsed_message=message_text_requires_expansion(raw_post_text),
                    ),
                    group,
                    min_length=self.settings.min_post_text_length,
                )
            except Error:
                # Preserve other messages when one story node is detached during hydration.
                continue
            if post is None:
                self._record_rejection(reason)
                continue
            merge_facebook_post(collected, post)
        return list(collected.values())[:max_posts]

    def _record_rejection(self, reason: str | None) -> None:
        """Record a bounded, content-free reason that visible feed text was skipped."""
        telemetry = self._active_feed_telemetry
        if telemetry is None:
            return
        if reason == "collapsed_unexpanded":
            telemetry.collapsed_unexpanded_observations += 1
        elif reason == "comment":
            telemetry.comment_observations += 1
        elif reason == "nested_article":
            telemetry.nested_article_observations += 1
        elif reason == "short_text":
            telemetry.short_text_observations += 1

    async def _nearest_post_hrefs(self, message: Locator) -> list[str]:
        """Find permalink candidates on the smallest ancestor that owns the story message."""
        values = await message.evaluate(
            r"""
            node => {
                let owner = node;
                while (owner && owner.getAttribute('role') !== 'feed') {
                    const hrefs = Array.from(owner.querySelectorAll('a[href]'))
                        .map(link => link.href)
                        .filter(href => (
                            /\/groups\/[^/]+\/(posts|permalink)\//i.test(href)
                            || /[?&](story_fbid|multi_permalinks)=/i.test(href)
                        ));
                    if (hrefs.length) return hrefs;
                    owner = owner.parentElement;
                }
                const article = node.closest('article, [role=article]');
                if (article) {
                    return Array.from(article.querySelectorAll('a[href]'))
                        .map(link => link.href)
                        .filter(href => (
                            /\/groups\/[^/]+\/(posts|permalink)\//i.test(href)
                            || /[?&](story_fbid|multi_permalinks)=/i.test(href)
                        ));
                }
                return [];
            }
            """
        )
        return cast(list[str], values)

    async def _post_articles(self, page: Page) -> Locator:
        """Prefer the visible feed so sidebar cards do not become candidate posts."""
        feeds = page.get_by_role("feed")
        for index in range(min(await feeds.count(), 5)):
            feed = feeds.nth(index)
            if await feed.is_visible(timeout=1000):
                return feed.get_by_role("article")
        return page.get_by_role("article")

    async def _is_nested_article(self, article: Locator) -> bool:
        """Exclude comment articles nested inside a top-level post article."""
        return bool(
            await article.evaluate(
                "node => Boolean(node.parentElement?.closest('article, [role=article]'))"
            )
        )

    async def _owned_article_texts(self, article: Locator, selector: str) -> list[str]:
        """Read matching text owned by this post while excluding nested comment articles."""
        values = await article.evaluate(
            """
            (root, selector) => Array.from(root.querySelectorAll(selector))
                .filter(node => node.closest('article, [role=article]') === root)
                .map(node => node.innerText || '')
            """,
            selector,
        )
        return cast(list[str], values)

    async def _article_text_without_comments(self, article: Locator) -> str:
        """Build a final text fallback after removing nested comment articles."""
        value = await article.evaluate(
            """
            root => {
                const clone = root.cloneNode(true);
                clone.querySelectorAll('article, [role=article]').forEach(node => node.remove());
                return clone.textContent || '';
            }
            """
        )
        return cast(str, value)

    async def _page(self, *, fresh: bool = False) -> Page:
        if self._context is None:
            raise RuntimeError("Facebook browser context is not open")
        if fresh:
            existing_pages = list(self._context.pages)
            page = await self._context.new_page()
            for existing in existing_pages:
                with suppress(Error):
                    await existing.close()
            return page
        if self._context.pages:
            return self._context.pages[0]
        return await self._context.new_page()

    async def _extract_article(
        self,
        article: Locator,
        group: FacebookGroup,
    ) -> FacebookPost | None:
        article_label = await article.get_attribute("aria-label", timeout=1000)
        full_text = await self._article_text_without_comments(article)
        semantic_messages = await self._owned_article_texts(
            article,
            STORY_MESSAGE_SELECTOR,
        )
        automatic_texts = await self._owned_article_texts(article, '[dir="auto"]')
        links = article.get_by_role("link")
        hrefs: list[str] = []
        author_name: str | None = None
        for index in range(min(await links.count(), 50)):
            link = links.nth(index)
            href = await link.get_attribute("href", timeout=1000)
            if href:
                hrefs.append(href)
            if author_name is None:
                candidate = normalize_post_text(await link.inner_text(timeout=1000))
                if candidate and candidate != group.name and 1 < len(candidate) <= 100:
                    author_name = candidate

        return build_facebook_post(
            FacebookPostCandidate(
                full_text=full_text,
                semantic_messages=tuple(semantic_messages),
                automatic_texts=tuple(automatic_texts),
                hrefs=tuple(hrefs),
                article_label=article_label,
                author_name=author_name,
                is_collapsed_message=any(
                    message_text_requires_expansion(value) for value in semantic_messages
                ),
            ),
            group,
            min_length=self.settings.min_post_text_length,
        )

    async def _require_normal_page(self, page: Page, *, group_id: str) -> None:
        try:
            visible_text = await page.locator("body").inner_text(timeout=5000)
        except PlaywrightTimeoutError:
            await self._stop(
                page,
                group_id,
                FacebookPageState.UNEXPECTED,
                "Facebook page content did not become readable",
            )
        if is_browser_network_error_text(visible_text):
            raise TransientFacebookReadError(
                stage="navigation",
                kind="offline",
                screenshot_path=await self._capture_failure(
                    page,
                    group_id,
                    "transient-navigation-offline",
                ),
            )
        assessment = assess_facebook_page(page.url, visible_text)
        if assessment.state is not FacebookPageState.NORMAL:
            await self._stop(page, group_id, assessment.state, assessment.reason)

    async def _stop(
        self,
        page: Page,
        group_id: str,
        state: FacebookPageState,
        reason: str,
    ) -> NoReturn:
        screenshot_path = await self._capture_failure(page, group_id, state.value)
        raise FacebookSafetyStop(state, reason, screenshot_path=screenshot_path)

    async def _capture_failure(self, page: Page, group_id: str, kind: str) -> Path | None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_group = SAFE_FILENAME_PATTERN.sub("-", group_id).strip("-") or "unknown"
        safe_kind = SAFE_FILENAME_PATTERN.sub("-", kind).strip("-") or "failure"
        path = self.settings.screenshot_dir / f"{timestamp}-{safe_group}-{safe_kind}.png"
        try:
            await page.screenshot(path=path, full_page=False)
            path.chmod(0o600)
        except Exception:
            return None
        return path

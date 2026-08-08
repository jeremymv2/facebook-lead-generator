"""Strictly read-only Playwright adapter for visible Facebook group posts."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
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
from lead_agent.models import FacebookPost, canonicalize_facebook_url, normalize_post_text

FACEBOOK_HOME = "https://www.facebook.com/"
POST_PATH_PATTERN = re.compile(r"/(?:posts|permalink)/([^/?#]+)", re.IGNORECASE)
SAFE_FILENAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")
STORY_MESSAGE_SELECTOR = (
    '[data-ad-rendering-role="story_message"], '
    '[data-ad-preview="message"], '
    '[data-ad-comet-preview="message"]'
)


class FacebookBrowserError(RuntimeError):
    """Raised when the dedicated Playwright browser cannot be started safely."""


def is_facebook_comment_label(label: str | None) -> bool:
    """Identify Facebook's semantic labels for comment and reply articles."""
    normalized = normalize_post_text(label or "").casefold()
    return normalized.startswith(("comment by ", "reply by "))


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
            normalized = normalize_post_text(value)
            if len(normalized) >= min_length and normalized not in seen:
                seen.add(normalized)
                valid.append(normalized)
        return valid

    semantic = candidates(semantic_messages)
    if semantic:
        return max(semantic, key=len)
    automatic = candidates(automatic_texts)
    if automatic:
        return max(automatic, key=len)
    fallback = candidates([full_text])
    if fallback:
        return fallback[0]
    return None


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
                )
            else:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    str(self.settings.facebook_profile_path),
                    headless=self.settings.browser_headless,
                    accept_downloads=False,
                    locale="en-US",
                    channel=self.settings.browser_channel,
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
        page = await self._page()
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
    ) -> list[FacebookPost]:
        """Wait for and read visible posts from one allowlisted group."""
        page = await self._page()
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
        except PlaywrightTimeoutError:
            await self._require_normal_page(page, group_id=group.id)
            await self._stop(
                page,
                group.id,
                FacebookPageState.UNEXPECTED,
                "No visible Facebook posts appeared before the safety timeout",
            )
        except Error:
            await self._stop(
                page,
                group.id,
                FacebookPageState.UNEXPECTED,
                "Facebook could not be opened safely",
            )

        return await self._wait_for_readable_posts(page, group, max_posts=max_posts)

    async def _wait_for_readable_posts(
        self,
        page: Page,
        group: FacebookGroup,
        *,
        max_posts: int,
    ) -> list[FacebookPost]:
        """Retry through Facebook placeholders and transient feed re-renders."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.facebook_post_load_timeout_seconds
        initial_grace_deadline = loop.time() + 2
        visible_article_seen = False
        collected: dict[str, FacebookPost] = {}
        scrolls = 0

        while loop.time() < deadline:
            await self._require_normal_page(page, group_id=group.id)
            try:
                story_posts = await self._extract_story_posts(page, group, max_posts=max_posts)
            except Error:
                story_posts = []
            for story_post in story_posts:
                collected.setdefault(story_post.identity_key, story_post)
            if len(collected) >= max_posts:
                return list(collected.values())[:max_posts]

            if not story_posts:
                try:
                    articles = await self._post_articles(page)
                    count = min(await articles.count(), max(max_posts * 5, 50))
                except Error:
                    count = 0

                for index in range(count):
                    try:
                        article = articles.nth(index)
                        if not await article.is_visible(timeout=1000):
                            continue
                        visible_article_seen = True
                        if is_facebook_comment_label(
                            await article.get_attribute("aria-label", timeout=1000)
                        ):
                            continue
                        if await self._is_nested_article(article):
                            continue
                        legacy_post = await self._extract_article(article, group)
                    except Error:
                        # Facebook commonly replaces placeholder nodes while the feed hydrates.
                        continue
                    if legacy_post is not None:
                        collected.setdefault(legacy_post.identity_key, legacy_post)
                        if len(collected) >= max_posts:
                            return list(collected.values())[:max_posts]

            if not collected and loop.time() < initial_grace_deadline:
                await page.wait_for_timeout(250)
                continue
            if scrolls >= self.settings.facebook_max_scrolls:
                break

            await self._scroll_for_more(page)
            scrolls += 1
            remaining_seconds = max(deadline - loop.time(), 0)
            if remaining_seconds > 0:
                settle_milliseconds = min(
                    int(self.settings.facebook_scroll_settle_seconds * 1000),
                    max(int(remaining_seconds * 1000), 1),
                )
                await page.wait_for_timeout(settle_milliseconds)

        if collected:
            return list(collected.values())[:max_posts]

        await self._require_normal_page(page, group_id=group.id)
        reason = (
            "Facebook displayed post containers, but no readable post text loaded"
            if visible_article_seen
            else "No visible Facebook posts appeared before the safety timeout"
        )
        await self._stop(page, group.id, FacebookPageState.UNEXPECTED, reason)

    async def _scroll_for_more(self, page: Page) -> None:
        """Reach the last loaded story, then move less than one viewport without clicking."""
        messages = page.locator(STORY_MESSAGE_SELECTOR)
        if await messages.count():
            await messages.last.scroll_into_view_if_needed(timeout=1000)
        await page.evaluate(
            """
            () => {
                const distance = Math.max(Math.floor(window.innerHeight * 0.85), 600);
                window.scrollBy(0, distance);
            }
            """
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
        posts: list[FacebookPost] = []
        identities: set[str] = set()
        expected_group_key = facebook_group_key(group.url)

        for index in range(count):
            message = messages.nth(index)
            if not await message.is_visible(timeout=1000):
                continue
            comment_label = cast(
                str | None,
                await message.evaluate(
                    "node => node.closest('article, [role=article]')?.getAttribute('aria-label')"
                ),
            )
            if is_facebook_comment_label(comment_label):
                continue
            post_text = normalize_post_text(await message.inner_text(timeout=1000))
            if len(post_text) < self.settings.min_post_text_length:
                continue
            hrefs = await self._nearest_post_hrefs(message)
            post_url = select_facebook_permalink(hrefs, group.url)
            if post_url is not None and facebook_group_key(post_url) != expected_group_key:
                continue
            post = FacebookPost(
                external_post_id=extract_post_id(post_url) if post_url else None,
                post_url=post_url,
                group_id=group.id,
                group_name=group.name,
                post_text=post_text,
            )
            if post.identity_key in identities:
                continue
            identities.add(post.identity_key)
            posts.append(post)
            if len(posts) >= max_posts:
                break
        return posts

    async def _nearest_post_hrefs(self, message: Locator) -> list[str]:
        """Find permalink candidates on the smallest ancestor that owns the story message."""
        values = await message.evaluate(
            r"""
            node => {
                let owner = node;
                let depth = 0;
                while (owner && depth < 60 && owner.getAttribute('role') !== 'feed') {
                    const hrefs = Array.from(owner.querySelectorAll('a[href]'))
                        .map(link => link.href)
                        .filter(href => (
                            /\/groups\/[^/]+\/(posts|permalink)\//i.test(href)
                            || /[?&](story_fbid|multi_permalinks)=/i.test(href)
                        ));
                    if (hrefs.length) return hrefs;
                    owner = owner.parentElement;
                    depth += 1;
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

    async def _page(self) -> Page:
        if self._context is None:
            raise RuntimeError("Facebook browser context is not open")
        if self._context.pages:
            return self._context.pages[0]
        return await self._context.new_page()

    async def _extract_article(
        self,
        article: Locator,
        group: FacebookGroup,
    ) -> FacebookPost | None:
        if is_facebook_comment_label(await article.get_attribute("aria-label", timeout=1000)):
            return None
        full_text = await self._article_text_without_comments(article)
        semantic_messages = await self._owned_article_texts(
            article,
            STORY_MESSAGE_SELECTOR,
        )
        automatic_texts = await self._owned_article_texts(article, '[dir="auto"]')
        post_text = select_message_text(
            full_text,
            semantic_messages,
            automatic_texts,
            min_length=self.settings.min_post_text_length,
        )
        if post_text is None:
            return None

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

        post_url = select_facebook_permalink(hrefs, group.url)
        return FacebookPost(
            external_post_id=extract_post_id(post_url) if post_url else None,
            post_url=post_url,
            group_id=group.id,
            group_name=group.name,
            author_name=author_name,
            post_text=post_text,
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

"""Strictly read-only Playwright adapter for visible Facebook group posts."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NoReturn
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


class FacebookBrowserError(RuntimeError):
    """Raised when the dedicated Playwright browser cannot be started safely."""


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
        normalized = {normalize_post_text(value) for value in values}
        return [value for value in normalized if len(value) >= min_length]

    for source in (semantic_messages, automatic_texts, [full_text]):
        valid = candidates(source)
        if valid:
            return max(valid, key=len)
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
        """Read currently visible articles from one allowlisted group."""
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
            articles = page.get_by_role("article")
            await articles.first.wait_for(state="visible")
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

        count = min(await articles.count(), max_posts)
        posts: list[FacebookPost] = []
        for index in range(count):
            try:
                article = articles.nth(index)
                if not await article.is_visible():
                    continue
                post = await self._extract_article(article, group)
            except Error:
                await self._stop(
                    page,
                    group.id,
                    FacebookPageState.UNEXPECTED,
                    "A visible Facebook post could not be read safely",
                )
            if post is not None:
                posts.append(post)
        return posts

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
        full_text = await article.inner_text()
        semantic_messages = await article.locator(
            '[data-ad-preview="message"], [data-ad-comet-preview="message"]'
        ).all_inner_texts()
        automatic_texts = await article.locator('[dir="auto"]').all_inner_texts()
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
            href = await link.get_attribute("href")
            if href:
                hrefs.append(href)
            if author_name is None:
                candidate = normalize_post_text(await link.inner_text())
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

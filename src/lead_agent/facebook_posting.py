"""Fail-closed Playwright adapter for one exact, human-approved Facebook comment."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import NoReturn, cast
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from playwright.async_api import (
    BrowserContext,
    ElementHandle,
    Error,
    Locator,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from lead_agent.config import Settings
from lead_agent.facebook import (
    SAFE_FILENAME_PATTERN,
    STORY_MESSAGE_SELECTOR,
    FacebookBrowserError,
    browser_launch_arguments,
    clean_facebook_message_text,
    cleanup_old_screenshots,
    extract_post_id,
    facebook_group_key,
    is_facebook_comment_label,
)
from lead_agent.facebook_state import FacebookPageState, FacebookSafetyStop, assess_facebook_page
from lead_agent.models import PostingWorkItem, is_post_text_expansion, normalize_post_text
from lead_agent.posting import (
    PostingSourceTextExpandedError,
    PostingSubmissionResult,
    PostingSubmissionUncertainError,
    PostingValidation,
    PostingValidationError,
)

RESOLUTION_MARKERS = (
    "already hired",
    "already taken care of",
    "found a contractor",
    "found someone",
    "got it handled",
    "have someone now",
    "no longer looking",
    "no longer need",
    "position filled",
    "problem solved",
)

PENDING_COMMENT_STATUSES = frozenset({"posting...", "sending..."})


def post_text_is_safe_match(expected: str, rendered: str) -> bool:
    """Allow insignificant rendering drift while rejecting meaningful or resolved edits."""
    expected_normalized = normalize_post_text(expected).casefold()
    rendered_normalized = normalize_post_text(rendered).casefold()
    if expected_normalized == rendered_normalized:
        return True
    if any(
        marker in rendered_normalized and marker not in expected_normalized
        for marker in RESOLUTION_MARKERS
    ):
        return False
    longest = max(len(expected_normalized), len(rendered_normalized))
    allowed_delta = max(12, int(longest * 0.05))
    if abs(len(expected_normalized) - len(rendered_normalized)) > allowed_delta:
        return False
    return SequenceMatcher(None, expected_normalized, rendered_normalized).ratio() >= 0.96


def validate_post_snapshot(
    work: PostingWorkItem,
    *,
    current_url: str,
    rendered_post_texts: Sequence[str],
) -> str:
    """Return the one matching rendered text or stop when identity/content is uncertain."""
    expected_url = work.attempt.post_url
    expected_post_id = extract_post_id(expected_url)
    current_post_id = extract_post_id(current_url)
    if expected_post_id is None or work.post.external_post_id is None:
        raise PostingValidationError("The saved Facebook post lacks a stable post identifier")
    if expected_post_id != work.post.external_post_id:
        raise PostingValidationError("The saved post URL and Facebook post ID disagree")
    if current_post_id != expected_post_id:
        raise PostingValidationError("Facebook did not remain on the exact approved post")
    expected_group = facebook_group_key(expected_url)
    if expected_group is None or facebook_group_key(current_url) != expected_group:
        raise PostingValidationError("Facebook did not remain in the approved group")
    if work.post.text_hash != work.attempt.source_text_hash:
        raise PostingValidationError(
            "The local source-post snapshot changed after approval",
            code="source_text_updated",
        )
    response_hash = hashlib.sha256(work.attempt.approved_response.encode("utf-8")).hexdigest()
    if response_hash != work.attempt.approved_response_hash:
        raise PostingValidationError("The approved response snapshot failed its integrity check")

    unique_renderings = tuple(
        dict.fromkeys(
            clean_facebook_message_text(text)
            for text in rendered_post_texts
            if clean_facebook_message_text(text)
        )
    )
    matches = [
        text for text in unique_renderings if post_text_is_safe_match(work.post.post_text, text)
    ]
    if not matches:
        expansions = [
            text for text in unique_renderings if is_post_text_expansion(work.post.post_text, text)
        ]
        if len(expansions) == 1:
            raise PostingSourceTextExpandedError(
                "Facebook revealed more source-post text; fresh review is required",
                observed_post_text=expansions[0],
            )
        raise PostingValidationError(
            "The Facebook post text no longer matches the approved source",
            code="source_text_mismatch",
        )
    if len(matches) > 1:
        raise PostingValidationError("More than one Facebook post matches the approved source")
    return matches[0]


def select_comment_permalink(hrefs: Sequence[str], post_url: str) -> str | None:
    """Choose a same-post Facebook comment URL while discarding tracking parameters."""
    expected_post_id = extract_post_id(post_url)
    for href in hrefs:
        absolute = urljoin(post_url, href)
        parts = urlsplit(absolute)
        hostname = (parts.hostname or "").casefold()
        query = dict(parse_qsl(parts.query))
        if hostname != "facebook.com" and not hostname.endswith(".facebook.com"):
            continue
        if extract_post_id(absolute) != expected_post_id or "comment_id" not in query:
            continue
        identity_query = [
            (key, query[key]) for key in ("comment_id", "reply_comment_id") if key in query
        ]
        return urlunsplit(
            (
                "https",
                "www.facebook.com",
                parts.path.rstrip("/"),
                urlencode(identity_query),
                "",
            )
        )
    return None


def confirmed_comment_permalink(
    *,
    expected_response: str,
    rendered_comment_texts: Sequence[str],
    article_text: str,
    hrefs: Sequence[str],
    post_url: str,
) -> str | None:
    """Return durable comment identity, never Facebook's transient optimistic UI."""
    expected = normalize_post_text(expected_response)
    if not any(normalize_post_text(text) == expected for text in rendered_comment_texts):
        return None
    article_lines = {
        normalize_post_text(line).casefold().replace("…", "...")
        for line in article_text.splitlines()
        if normalize_post_text(line)
    }
    if article_lines & PENDING_COMMENT_STATUSES:
        return None
    return select_comment_permalink(hrefs, post_url)


def pending_content_url(post_url: str) -> str | None:
    """Return the authenticated group page that lists this user's pending content."""
    group_key = facebook_group_key(post_url)
    if group_key is None:
        return None
    return f"https://www.facebook.com/groups/{group_key}/my_pending_content"


@dataclass(slots=True)
class _ValidatedBrowserState:
    attempt_id: int
    page: Page
    composer: Locator


class FacebookCommentBrowser:  # pragma: no cover - requires an interactive Facebook session
    """Persistent browser with one narrowly gated comment-submission operation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._validated: _ValidatedBrowserState | None = None

    async def __aenter__(self) -> FacebookCommentBrowser:
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
        self._validated = None
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def validate(self, work: PostingWorkItem) -> PostingValidation:
        """Navigate, prove target identity/content, and locate one composer without clicking."""
        attempt_id = _attempt_id(work)
        page = await self._page()
        try:
            await page.goto(work.attempt.post_url, wait_until="domcontentloaded")
            rendered = await self._wait_for_top_level_post_messages(
                page,
                lead_id=work.lead.id or 0,
            )
            matched_text = validate_post_snapshot(
                work,
                current_url=page.url,
                rendered_post_texts=[text for text, _message in rendered],
            )
            matching_messages = [
                message
                for text, message in rendered
                if clean_facebook_message_text(text) == matched_text
            ]
            if len(matching_messages) != 1:
                raise PostingValidationError(
                    "The approved Facebook post could not be isolated to one visible story"
                )
            if await self._find_exact_comment(page, work) is not None:
                raise PostingValidationError(
                    "The exact approved response is already visible; manual review is required",
                    code="response_already_visible",
                )
            composer = await self._comment_composer(
                page,
                matching_messages[0],
                lead_id=work.lead.id or 0,
            )
            before = await self._capture(page, work.lead.id or 0, "before-posting")
            self._validated = _ValidatedBrowserState(
                attempt_id=attempt_id,
                page=page,
                composer=composer,
            )
            return PostingValidation(before_screenshot_path=before)
        except FacebookSafetyStop:
            raise
        except PostingValidationError as error:
            screenshot = error.screenshot_path or await self._capture(
                page, work.lead.id or 0, "validation-failed"
            )
            error.screenshot_path = screenshot
            raise
        except (Error, PlaywrightTimeoutError) as error:
            screenshot = await self._capture(page, work.lead.id or 0, "validation-failed")
            raise PostingValidationError(
                "Facebook posting controls did not become safely readable",
                screenshot_path=screenshot,
                code="posting_controls_unreadable",
            ) from error

    async def submit(
        self,
        work: PostingWorkItem,
        validation: PostingValidation,
        *,
        on_before_submit: Callable[[], None],
    ) -> PostingSubmissionResult:
        """Type the exact approved snapshot, cross the durable boundary, and press Enter once."""
        del validation
        self.settings.require_posting_allowed()
        validated = self._validated
        if validated is None or validated.attempt_id != _attempt_id(work):
            raise PostingValidationError("No matching browser validation is active")
        page = validated.page
        composer = validated.composer
        boundary_crossed = False
        try:
            await self._require_normal_page(page, lead_id=work.lead.id or 0)
            if extract_post_id(page.url) != extract_post_id(work.attempt.post_url):
                raise PostingValidationError("Facebook navigated away before submission")
            if not await composer.is_visible(timeout=1000):
                raise PostingValidationError("The validated comment composer disappeared")

            self.settings.require_posting_allowed()
            await composer.click()
            await composer.fill(work.attempt.approved_response)
            entered = normalize_post_text(await composer.inner_text(timeout=2000))
            if entered != normalize_post_text(work.attempt.approved_response):
                raise PostingValidationError("Facebook did not retain the exact approved response")

            await self._require_normal_page(page, lead_id=work.lead.id or 0)
            self.settings.require_posting_allowed()
            on_before_submit()
            boundary_crossed = True
            await composer.press("Enter")
            try:
                reply_url = await self._wait_for_exact_comment(page, work)
            except PostingSubmissionUncertainError:
                try:
                    pending_moderation = await self._pending_moderation_is_visible(page, work)
                except Exception:
                    pending_moderation = False
                if pending_moderation:
                    after = await self._capture(page, work.lead.id or 0, "pending-moderation")
                    self._validated = None
                    return PostingSubmissionResult(
                        pending_moderation=True,
                        after_screenshot_path=after,
                    )
                raise
            reply_url = await self._confirm_comment_survived_reload(page, work, reply_url)
            after = await self._capture(page, work.lead.id or 0, "posted")
            self._validated = None
            return PostingSubmissionResult(
                facebook_reply_url=reply_url,
                after_screenshot_path=after,
            )
        except PostingSubmissionUncertainError:
            raise
        except FacebookSafetyStop as error:
            if boundary_crossed:
                raise PostingSubmissionUncertainError(
                    "Facebook became abnormal after submission began",
                    screenshot_path=error.screenshot_path,
                ) from error
            raise
        except (Error, PlaywrightTimeoutError, PostingValidationError) as error:
            if boundary_crossed:
                recovered_reply_url: str | None = None
                try:
                    candidate_reply_url = await self._find_exact_comment(page, work)
                    if candidate_reply_url is not None:
                        recovered_reply_url = await self._confirm_comment_survived_reload(
                            page, work, candidate_reply_url
                        )
                except Exception:
                    pass
                if recovered_reply_url is not None:
                    after = await self._capture(page, work.lead.id or 0, "posted")
                    self._validated = None
                    return PostingSubmissionResult(
                        facebook_reply_url=recovered_reply_url,
                        after_screenshot_path=after,
                    )
                try:
                    pending_moderation = await self._pending_moderation_is_visible(page, work)
                except Exception:
                    pending_moderation = False
                if pending_moderation:
                    after = await self._capture(page, work.lead.id or 0, "pending-moderation")
                    self._validated = None
                    return PostingSubmissionResult(
                        pending_moderation=True,
                        after_screenshot_path=after,
                    )
                screenshot = await self._capture(page, work.lead.id or 0, "submission-uncertain")
                raise PostingSubmissionUncertainError(
                    "Facebook did not prove whether the comment was submitted",
                    screenshot_path=screenshot,
                ) from error
            if isinstance(error, PostingValidationError):
                raise
            screenshot = await self._capture(page, work.lead.id or 0, "validation-failed")
            raise PostingValidationError(
                "Facebook stopped before the submission boundary",
                screenshot_path=screenshot,
            ) from error

    async def _wait_for_top_level_post_messages(
        self,
        page: Page,
        *,
        lead_id: int,
    ) -> list[tuple[str, Locator]]:
        """Wait through Facebook's loading splash before validating source text."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.facebook_post_load_timeout_seconds
        while True:
            await self._require_normal_page(page, lead_id=lead_id)
            rendered = await self._top_level_post_messages(page)
            if rendered:
                return rendered
            if loop.time() >= deadline:
                raise PostingValidationError(
                    "Facebook did not finish rendering the source post before timeout",
                    code="source_post_load_timeout",
                )
            await page.wait_for_timeout(250)

    async def _top_level_post_messages(self, page: Page) -> list[tuple[str, Locator]]:
        scope = await self._story_message_scope(page)
        messages = scope.locator(STORY_MESSAGE_SELECTOR)
        visible_messages: list[Locator] = []
        for index in range(min(await messages.count(), 30)):
            message = messages.nth(index)
            if await message.is_visible(timeout=1000):
                visible_messages.append(message)
        rendered: list[tuple[str, Locator]] = []
        for message in await self._innermost_locators(visible_messages):
            label = cast(
                str | None,
                await message.evaluate(
                    "node => node.closest('article, [role=article]')?.getAttribute('aria-label')"
                ),
            )
            if is_facebook_comment_label(label):
                continue
            text = clean_facebook_message_text(await message.inner_text(timeout=1000))
            if text:
                rendered.append((text, message))
        return rendered

    async def _story_message_scope(self, page: Page) -> Page | Locator:
        """Prefer one foreground post dialog over its dimmed feed duplicate."""
        dialogs = page.get_by_role("dialog")
        candidates: list[Locator] = []
        for index in range(min(await dialogs.count(), 5)):
            dialog = dialogs.nth(index)
            if not await dialog.is_visible(timeout=1000):
                continue
            messages = dialog.locator(STORY_MESSAGE_SELECTOR)
            has_visible_message = False
            for message_index in range(min(await messages.count(), 30)):
                if await messages.nth(message_index).is_visible(timeout=1000):
                    has_visible_message = True
                    break
            if has_visible_message:
                candidates.append(dialog)
        innermost_candidates = await self._innermost_locators(candidates)
        if len(innermost_candidates) > 1:
            raise PostingValidationError("Facebook exposed more than one foreground post dialog")
        return innermost_candidates[0] if innermost_candidates else page

    @staticmethod
    async def _innermost_locators(candidates: Sequence[Locator]) -> list[Locator]:
        """Discard wrapper nodes while preserving ambiguity between separate candidates."""
        resolved: list[tuple[Locator, ElementHandle]] = []
        try:
            for candidate in candidates:
                handle = await candidate.element_handle()
                if handle is not None:
                    resolved.append((candidate, handle))
            innermost: list[Locator] = []
            for index, (candidate, handle) in enumerate(resolved):
                contains_another = False
                for other_index, (_other_candidate, other_handle) in enumerate(resolved):
                    if index == other_index:
                        continue
                    if await handle.evaluate("(node, other) => node.contains(other)", other_handle):
                        contains_another = True
                        break
                if not contains_another:
                    innermost.append(candidate)
            return innermost
        finally:
            for _candidate, handle in resolved:
                await handle.dispose()

    async def _comment_composer(
        self,
        page: Page,
        message: Locator,
        *,
        lead_id: int,
    ) -> Locator:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.facebook_post_load_timeout_seconds
        while True:
            owner = message.locator("xpath=ancestor::*[self::article or @role='article'][1]")
            scoped = owner.locator('[contenteditable="true"][role="textbox"]')
            candidates = await self._visible_comment_composers(scoped)
            if not candidates:
                candidates = await self._visible_comment_composers(
                    page.locator('[contenteditable="true"][role="textbox"]')
                )
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise PostingValidationError(
                    "Facebook exposed more than one recognizable comment composer",
                    code="comment_composer_ambiguous",
                )
            if loop.time() >= deadline:
                raise PostingValidationError(
                    "Facebook did not expose a recognizable comment composer before timeout",
                    code="comment_composer_missing",
                )
            await self._require_normal_page(page, lead_id=lead_id)
            await page.wait_for_timeout(250)

    async def _visible_comment_composers(self, locators: Locator) -> list[Locator]:
        visible: list[Locator] = []
        labeled: list[Locator] = []
        for index in range(min(await locators.count(), 12)):
            locator = locators.nth(index)
            if not await locator.is_visible(timeout=1000):
                continue
            visible.append(locator)
            labels = " ".join(
                filter(
                    None,
                    (
                        await locator.get_attribute("aria-label", timeout=1000),
                        await locator.get_attribute("aria-placeholder", timeout=1000),
                        await locator.get_attribute("data-placeholder", timeout=1000),
                    ),
                )
            ).casefold()
            if "comment" in labels or "reply" in labels:
                labeled.append(locator)
        return labeled if labeled else visible

    async def _wait_for_exact_comment(self, page: Page, work: PostingWorkItem) -> str:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.facebook_post_load_timeout_seconds
        while loop.time() < deadline:
            await self._require_normal_page(page, lead_id=work.lead.id or 0)
            result = await self._find_exact_comment(page, work)
            if result is not None:
                return result
            await page.wait_for_timeout(250)
        screenshot = await self._capture(page, work.lead.id or 0, "submission-uncertain")
        raise PostingSubmissionUncertainError(
            "The exact approved comment did not become visible after submission",
            screenshot_path=screenshot,
        )

    async def _confirm_comment_survived_reload(
        self,
        page: Page,
        work: PostingWorkItem,
        reply_url: str,
    ) -> str:
        """Reload through the stable identity and prove the exact comment still exists."""
        await page.goto(reply_url, wait_until="domcontentloaded")
        await self._require_normal_page(page, lead_id=work.lead.id or 0)
        confirmed_url = await self._wait_for_exact_comment(page, work)
        if confirmed_url != reply_url:
            raise PostingValidationError(
                "Facebook changed the comment identity after reloading its permalink"
            )
        return confirmed_url

    async def _pending_moderation_is_visible(
        self,
        page: Page,
        work: PostingWorkItem,
    ) -> bool:
        """Prove the exact response is listed in this group's private moderation queue."""
        url = pending_content_url(work.attempt.post_url)
        if url is None:
            return False
        await page.goto(url, wait_until="domcontentloaded")
        await self._require_normal_page(page, lead_id=work.lead.id or 0)
        expected = normalize_post_text(work.attempt.approved_response)
        texts = page.locator('[dir="auto"]')
        matches: list[Locator] = []
        for index in range(min(await texts.count(), 200)):
            text = texts.nth(index)
            if not await text.is_visible(timeout=1000):
                continue
            if normalize_post_text(await text.inner_text(timeout=1000)) == expected:
                matches.append(text)
        return len(await self._innermost_locators(matches)) == 1

    async def _find_exact_comment(self, page: Page, work: PostingWorkItem) -> str | None:
        expected = normalize_post_text(work.attempt.approved_response)
        articles = page.get_by_role("article")
        for index in range(min(await articles.count(), 100)):
            article = articles.nth(index)
            label = await article.get_attribute("aria-label", timeout=1000)
            if not is_facebook_comment_label(label):
                continue
            texts = article.locator('[dir="auto"]')
            rendered_texts: list[str] = []
            for text_index in range(min(await texts.count(), 30)):
                text = texts.nth(text_index)
                if not await text.is_visible(timeout=1000):
                    continue
                rendered_texts.append(await text.inner_text(timeout=1000))
            if not any(normalize_post_text(text) == expected for text in rendered_texts):
                continue
            links = article.get_by_role("link")
            hrefs: list[str] = []
            for link_index in range(min(await links.count(), 30)):
                href = await links.nth(link_index).get_attribute("href", timeout=1000)
                if href:
                    hrefs.append(href)
            reply_url = confirmed_comment_permalink(
                expected_response=work.attempt.approved_response,
                rendered_comment_texts=rendered_texts,
                article_text=await article.inner_text(timeout=1000),
                hrefs=hrefs,
                post_url=work.attempt.post_url,
            )
            if reply_url is not None:
                return reply_url
        return None

    async def _page(self) -> Page:
        if self._context is None:
            raise RuntimeError("Facebook browser context is not open")
        if self._context.pages:
            return self._context.pages[0]
        return await self._context.new_page()

    async def _require_normal_page(self, page: Page, *, lead_id: int) -> None:
        try:
            visible_text = await page.locator("body").inner_text(timeout=5000)
        except PlaywrightTimeoutError:
            await self._stop(
                page,
                lead_id,
                FacebookPageState.UNEXPECTED,
                "Facebook page content did not become readable",
            )
        assessment = assess_facebook_page(page.url, visible_text)
        if assessment.state is not FacebookPageState.NORMAL:
            await self._stop(page, lead_id, assessment.state, assessment.reason)

    async def _stop(
        self,
        page: Page,
        lead_id: int,
        state: FacebookPageState,
        reason: str,
    ) -> NoReturn:
        screenshot = await self._capture(page, lead_id, state.value)
        raise FacebookSafetyStop(state, reason, screenshot_path=screenshot)

    async def _capture(self, page: Page, lead_id: int, kind: str) -> Path | None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_kind = SAFE_FILENAME_PATTERN.sub("-", kind).strip("-") or "posting"
        path = self.settings.screenshot_dir / f"{timestamp}-lead-{lead_id}-{safe_kind}.png"
        try:
            await page.screenshot(path=path, full_page=False)
            path.chmod(0o600)
        except Exception:
            return None
        return path


def _attempt_id(work: PostingWorkItem) -> int:
    if work.attempt.id is None:  # pragma: no cover - database work always has an ID
        raise RuntimeError("Posting attempt is missing its database ID")
    return work.attempt.id


__all__ = [
    "FacebookCommentBrowser",
    "confirmed_comment_permalink",
    "pending_content_url",
    "post_text_is_safe_match",
    "select_comment_permalink",
    "validate_post_snapshot",
]

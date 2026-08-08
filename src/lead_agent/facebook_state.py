"""Fail-closed classification of Facebook browser pages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit


class FacebookPageState(StrEnum):
    """Page states that determine whether read-only scanning may continue."""

    NORMAL = "normal"
    LOGIN_REQUIRED = "login_required"
    CHECKPOINT = "checkpoint"
    CAPTCHA = "captcha"
    UNEXPECTED = "unexpected"


@dataclass(frozen=True, slots=True)
class PageAssessment:
    state: FacebookPageState
    reason: str


class FacebookSafetyStop(RuntimeError):
    """Raised when Facebook requires human attention or the page is uncertain."""

    def __init__(
        self,
        state: FacebookPageState,
        reason: str,
        *,
        screenshot_path: Path | None = None,
    ) -> None:
        super().__init__(reason)
        self.state = state
        self.reason = reason
        self.screenshot_path = screenshot_path


def assess_facebook_page(url: str, visible_text: str) -> PageAssessment:
    """Classify a Facebook page from its URL and visible text without mutating it."""
    parts = urlsplit(url)
    hostname = (parts.hostname or "").casefold()
    path_and_query = f"{parts.path}?{parts.query}".casefold()
    text = " ".join(visible_text.casefold().split())

    if hostname != "facebook.com" and not hostname.endswith(".facebook.com"):
        return PageAssessment(
            FacebookPageState.UNEXPECTED,
            "Browser left the facebook.com domain",
        )

    if "captcha" in path_and_query or any(
        marker in text
        for marker in (
            "enter the text you see",
            "security check required",
            "complete the security check",
        )
    ):
        return PageAssessment(
            FacebookPageState.CAPTCHA,
            "Facebook displayed a CAPTCHA or security check",
        )

    if any(
        marker in path_and_query
        for marker in (
            "/checkpoint",
            "/challenge",
            "/recover",
            "two_step_verification",
            "/identify",
        )
    ) or any(
        marker in text
        for marker in (
            "confirm your identity",
            "we need to confirm it's you",
            "account temporarily locked",
            "check your notifications on another device",
        )
    ):
        return PageAssessment(
            FacebookPageState.CHECKPOINT,
            "Facebook displayed an account checkpoint or identity challenge",
        )

    if parts.path.casefold().startswith("/login") or any(
        marker in text
        for marker in (
            "log into facebook",
            "log in to facebook",
            "email or phone number",
            "forgotten password?",
        )
    ):
        return PageAssessment(
            FacebookPageState.LOGIN_REQUIRED,
            "Facebook login is required",
        )

    return PageAssessment(FacebookPageState.NORMAL, "Facebook page appears normal")

"""Framework-independent domain models for discovered posts and leads."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


def normalize_post_text(text: str) -> str:
    """Normalize insignificant whitespace without changing the post's words."""
    return re.sub(r"\s+", " ", text).strip()


def hash_post_text(text: str) -> str:
    """Return a stable SHA-256 digest for normalized post text."""
    return hashlib.sha256(normalize_post_text(text).encode("utf-8")).hexdigest()


def canonicalize_facebook_url(url: str) -> str:
    """Remove tracking data while retaining query parameters that identify a post."""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/") or "/"
    identity_keys = {"fbid", "id", "multi_permalinks", "story_fbid"}
    identity_query = sorted(
        (key, value) for key, value in parse_qsl(parts.query) if key in identity_keys
    )
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            path,
            urlencode(identity_query),
            "",
        )
    )


class PostStatus(StrEnum):
    DISCOVERED = "discovered"
    PROCESSED = "processed"
    IGNORED = "ignored"
    FAILED = "failed"
    NEEDS_ATTENTION = "needs_attention"


class LeadStatus(StrEnum):
    DISCOVERED = "discovered"
    CLASSIFIED = "classified"
    IGNORED = "ignored"
    CANDIDATE = "candidate"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    POSTING = "posting"
    POSTED = "posted"
    EXPIRED = "expired"
    FAILED = "failed"
    NEEDS_ATTENTION = "needs_attention"


class LeadIntent(StrEnum):
    HIRING = "hiring"
    RECOMMENDATION = "recommendation"
    ADVICE = "advice"
    RESOLVED = "resolved"
    PRIVATE_CONTACT_ONLY = "private_contact_only"
    SELLING = "selling"
    COMPETITOR_ADVERTISEMENT = "competitor_advertisement"
    UNRELATED = "unrelated"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    EXPIRED = "expired"


class NotificationStatus(StrEnum):
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"


class PostingAttemptStatus(StrEnum):
    """Durable stages for one dry-run validation or live Facebook posting attempt."""

    VALIDATING = "validating"
    DRY_RUN_VALIDATED = "dry_run_validated"
    SUBMITTING = "submitting"
    POSTED = "posted"
    FAILED = "failed"
    NEEDS_ATTENTION = "needs_attention"


@dataclass(slots=True)
class FacebookPost:
    group_id: str
    group_name: str
    post_text: str
    external_post_id: str | None = None
    post_url: str | None = None
    author_name: str | None = None
    posted_at: datetime | None = None
    discovered_at: datetime = field(default_factory=utc_now)
    status: PostStatus = PostStatus.DISCOVERED
    error_state: str | None = None
    screenshot_path: str | None = None
    id: int | None = None
    identity_key: str = ""
    text_hash: str = ""

    def __post_init__(self) -> None:
        self.post_text = normalize_post_text(self.post_text)
        if not self.post_text:
            raise ValueError("post_text cannot be empty")
        if not self.group_id.strip():
            raise ValueError("group_id cannot be empty")
        self.group_id = self.group_id.strip()
        self.group_name = self.group_name.strip()
        self.text_hash = self.text_hash or hash_post_text(self.post_text)
        self.identity_key = self.identity_key or self.build_identity_key()

    def build_identity_key(self) -> str:
        """Prefer Facebook identifiers, then URLs, then stable visible content."""
        if self.external_post_id:
            return f"facebook-id:{self.external_post_id.strip()}"
        if self.post_url:
            return f"facebook-url:{canonicalize_facebook_url(self.post_url)}"
        author = (self.author_name or "unknown").strip().casefold()
        return f"content:{self.group_id.casefold()}:{author}:{self.text_hash}"

    def identity_aliases(self) -> tuple[str, ...]:
        """Return every durable identity currently visible for this post.

        Facebook can expose a story before its permalink has hydrated. Keeping the content
        identities alongside later IDs and URLs lets persistence join those renderings without
        changing the post's original primary identity.
        """
        aliases = [self.identity_key]
        if self.external_post_id:
            aliases.append(f"facebook-id:{self.external_post_id.strip()}")
        if self.post_url:
            aliases.append(f"facebook-url:{canonicalize_facebook_url(self.post_url)}")
        author = (self.author_name or "unknown").strip().casefold()
        aliases.extend(
            (
                f"content:{self.group_id.casefold()}:{author}:{self.text_hash}",
                f"content-text:{self.group_id.casefold()}:{self.text_hash}",
            )
        )
        return tuple(dict.fromkeys(aliases))


@dataclass(slots=True)
class Lead:
    facebook_post_id: int
    status: LeadStatus = LeadStatus.DISCOVERED
    service_category: str | None = None
    location: str | None = None
    intent: LeadIntent | None = None
    is_residential: bool | None = None
    is_spam: bool | None = None
    relevance_score: int | None = None
    geographic_score: int | None = None
    urgency_score: int | None = None
    overall_score: int | None = None
    confidence: float | None = None
    reasoning_summary: str | None = None
    drafted_response: str | None = None
    ai_provider: str | None = None
    ai_model: str | None = None
    classification_version: str | None = None
    approved_response: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    approval_timestamp: datetime | None = None
    approval_expires_at: datetime | None = None
    posting_timestamp: datetime | None = None
    facebook_reply_url: str | None = None
    error_state: str | None = None
    retry_count: int = 0
    screenshot_path: str | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        for name in ("relevance_score", "geographic_score", "urgency_score", "overall_score"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 100:
                raise ValueError(f"{name} must be between 0 and 100")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.retry_count < 0:
            raise ValueError("retry_count cannot be negative")


@dataclass(slots=True)
class ApprovalRequest:
    lead_id: int
    draft_response: str
    expires_at: datetime
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = field(default_factory=utc_now)
    decided_at: datetime | None = None
    decided_response: str | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        self.draft_response = self.draft_response.strip()
        if not self.draft_response:
            raise ValueError("draft_response cannot be empty")
        if self.expires_at <= self.requested_at:
            raise ValueError("approval expiration must follow its request time")


@dataclass(frozen=True, slots=True)
class ApprovalReview:
    request: ApprovalRequest
    lead: Lead
    post: FacebookPost


@dataclass(slots=True)
class ApprovalNotification:
    approval_request_id: int
    provider: str
    status: NotificationStatus
    attempt_count: int
    last_attempt_at: datetime
    sent_at: datetime | None = None
    provider_message_id: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.approval_request_id < 1:
            raise ValueError("approval_request_id must be positive")
        if not self.provider.strip():
            raise ValueError("provider cannot be empty")
        if self.attempt_count < 1:
            raise ValueError("attempt_count must be positive")


@dataclass(slots=True)
class PostingAttempt:
    """Immutable-input ledger plus mutable outcome for one posting workflow execution."""

    lead_id: int
    approval_request_id: int
    status: PostingAttemptStatus
    dry_run: bool
    approved_response: str
    approved_response_hash: str
    source_text_hash: str
    post_url: str
    group_id: str
    started_at: datetime
    validated_at: datetime | None = None
    submission_started_at: datetime | None = None
    completed_at: datetime | None = None
    facebook_reply_url: str | None = None
    before_screenshot_path: str | None = None
    after_screenshot_path: str | None = None
    error_code: str | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        if self.lead_id < 1 or self.approval_request_id < 1:
            raise ValueError("posting attempt IDs must be positive")
        self.approved_response = self.approved_response.strip()
        if not self.approved_response:
            raise ValueError("approved_response cannot be empty")
        if not self.post_url.strip() or not self.group_id.strip():
            raise ValueError("posting attempts require a post URL and group")


@dataclass(frozen=True, slots=True)
class PostingWorkItem:
    """A posting attempt joined to the exact approved lead and source post."""

    attempt: PostingAttempt
    lead: Lead
    post: FacebookPost


@dataclass(slots=True)
class AuditEvent:
    component: str
    action: str
    result: str
    lead_id: int | None = None
    post_id: int | None = None
    group_id: str | None = None
    details: dict[str, object] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=utc_now)
    id: int | None = None


@dataclass(slots=True)
class GroupScanState:
    group_id: str
    group_name: str
    group_url: str
    last_attempt_at: datetime
    last_success_at: datetime | None = None
    last_known_post_identity: str | None = None
    last_error: str | None = None
    posts_seen: int = 0
    posts_new: int = 0
    consecutive_failures: int = 0
    last_failure_at: datetime | None = None

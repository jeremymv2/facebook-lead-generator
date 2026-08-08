"""Idempotent, human-approved posting orchestration with durable no-retry boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

from lead_agent.config import Settings
from lead_agent.database import Database
from lead_agent.models import (
    AuditEvent,
    PostingAttemptStatus,
    PostingWorkItem,
    utc_now,
)


class PostingError(RuntimeError):
    """Base error for safe approved-posting failures."""

    code = "posting_error"


class PostingEligibilityError(PostingError):
    """Raised before browser use when a lead cannot safely enter posting."""

    code = "posting_ineligible"


class PostingValidationError(PostingError):
    """Raised when the exact Facebook target or comment composer is uncertain."""

    code = "facebook_validation_failed"

    def __init__(self, message: str, *, screenshot_path: Path | None = None) -> None:
        super().__init__(message)
        self.screenshot_path = screenshot_path


class PostingSubmissionUncertainError(PostingError):
    """Raised after the submission boundary when success cannot be proved."""

    code = "facebook_submission_uncertain"

    def __init__(self, message: str, *, screenshot_path: Path | None = None) -> None:
        super().__init__(message)
        self.screenshot_path = screenshot_path


@dataclass(frozen=True, slots=True)
class PostingValidation:
    """Evidence that the live page still matches the approved local snapshot."""

    before_screenshot_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PostingSubmissionResult:
    """Evidence returned only after the exact comment is visible on Facebook."""

    facebook_reply_url: str | None = None
    after_screenshot_path: Path | None = None


class FacebookPostingAdapter(Protocol):
    """Narrow browser boundary used by the independently testable posting service."""

    async def validate(self, work: PostingWorkItem) -> PostingValidation:
        """Validate the exact post and locate one comment composer without mutating it."""
        ...

    async def submit(
        self,
        work: PostingWorkItem,
        validation: PostingValidation,
        *,
        on_before_submit: Callable[[], None],
    ) -> PostingSubmissionResult:
        """Enter the snapshot response, persist the boundary callback, and submit once."""
        ...


@dataclass(frozen=True, slots=True)
class PostingExecutionResult:
    """Final durable state returned by one guarded command invocation."""

    work: PostingWorkItem
    created: bool


class ApprovedPostingService:
    """Coordinate one approved response without ever automatically retrying submission."""

    def __init__(
        self,
        database: Database,
        settings: Settings,
        *,
        enabled_group_ids: set[str],
    ) -> None:
        self.database = database
        self.settings = settings
        self.enabled_group_ids = enabled_group_ids

    async def execute(
        self,
        lead_id: int,
        adapter: FacebookPostingAdapter,
        *,
        dry_run: bool,
        now: datetime | None = None,
    ) -> PostingExecutionResult:
        """Validate repeatedly in dry-run, or reserve and submit one live attempt exactly once."""
        timestamp = now or utc_now()
        if not dry_run:
            self.settings.require_posting_allowed()

        lead = self.database.get_lead(lead_id)
        if lead is None:
            raise PostingEligibilityError(f"Lead {lead_id} does not exist")
        post = self.database.get_post(lead.facebook_post_id)
        if post is None:  # pragma: no cover - database foreign key contract
            raise PostingEligibilityError("Lead is missing its Facebook post")
        if post.group_id not in self.enabled_group_ids:
            self._record_event(
                lead_id=lead.id,
                post_id=post.id,
                group_id=post.group_id,
                action="posting.blocked",
                result="disabled_group",
                details={"dry_run": dry_run},
            )
            raise PostingEligibilityError("The lead's Facebook group is not currently enabled")

        day_started_at, next_day_started_at = self._posting_day_bounds(timestamp)
        try:
            claimed = self.database.begin_posting_attempt(
                lead_id,
                dry_run=dry_run,
                started_at=timestamp,
                oldest_approval_at=(
                    timestamp - timedelta(minutes=self.settings.posting_approval_max_age_minutes)
                ),
                day_started_at=day_started_at,
                next_day_started_at=next_day_started_at,
                daily_limit=self.settings.daily_posting_limit,
                per_group_daily_limit=self.settings.per_group_daily_posting_limit,
            )
        except (LookupError, ValueError) as error:
            self._record_event(
                lead_id=lead.id,
                post_id=post.id,
                group_id=post.group_id,
                action="posting.blocked",
                result="ineligible",
                details={"dry_run": dry_run, "reason": type(error).__name__},
            )
            raise PostingEligibilityError(str(error)) from error

        work = claimed.work
        if not claimed.created:
            self._record_work_event(
                work,
                action="posting.idempotent_stop",
                result=work.attempt.status.value,
            )
            return PostingExecutionResult(work=work, created=False)

        self._record_work_event(work, action="posting.started", result="validating")
        try:
            validation = await adapter.validate(work)
            validated_at = self._event_time(now)
            validated_attempt = self.database.complete_posting_validation(
                self._attempt_id(work),
                validated_at=validated_at,
                before_screenshot_path=_path_string(validation.before_screenshot_path),
            )
            work = PostingWorkItem(attempt=validated_attempt, lead=work.lead, post=work.post)
            self._record_work_event(
                work,
                action="posting.validated",
                result=("dry_run" if dry_run else "ready"),
            )
            if dry_run:
                return PostingExecutionResult(work=work, created=True)

            def mark_submission_boundary() -> None:
                nonlocal work
                submitting = self.database.mark_posting_submission_started(
                    self._attempt_id(work),
                    started_at=self._event_time(now),
                )
                work = PostingWorkItem(attempt=submitting, lead=work.lead, post=work.post)
                self._record_work_event(
                    work,
                    action="posting.submission_started",
                    result="submitting",
                )

            outcome = await adapter.submit(
                work,
                validation,
                on_before_submit=mark_submission_boundary,
            )
            completed = self.database.complete_posting_attempt(
                self._attempt_id(work),
                completed_at=self._event_time(now),
                facebook_reply_url=outcome.facebook_reply_url,
                after_screenshot_path=_path_string(outcome.after_screenshot_path),
            )
            self._record_work_event(completed, action="posting.succeeded", result="posted")
            return PostingExecutionResult(work=completed, created=True)
        except Exception as error:
            error_code = _safe_error_code(error)
            screenshot_path = _exception_screenshot_path(error)
            failed = self.database.fail_posting_attempt(
                self._attempt_id(work),
                failed_at=self._event_time(now),
                error_code=error_code,
                after_screenshot_path=_path_string(screenshot_path),
            )
            self._record_work_event(
                failed,
                action="posting.stopped",
                result=failed.attempt.status.value,
                details={"error_code": error_code},
            )
            if isinstance(error, PostingError):
                raise
            raise PostingError(
                "Posting stopped safely before success could be confirmed"
            ) from error

    def _posting_day_bounds(self, timestamp: datetime) -> tuple[datetime, datetime]:
        zone = ZoneInfo(self.settings.business_timezone)
        local = timestamp.astimezone(zone)
        local_start = datetime.combine(local.date(), time.min, tzinfo=zone)
        local_end = local_start + timedelta(days=1)
        return local_start.astimezone(UTC), local_end.astimezone(UTC)

    @staticmethod
    def _event_time(fixed: datetime | None) -> datetime:
        return fixed or utc_now()

    @staticmethod
    def _attempt_id(work: PostingWorkItem) -> int:
        if work.attempt.id is None:  # pragma: no cover - database work always has an ID
            raise RuntimeError("Posting attempt is missing its database ID")
        return work.attempt.id

    def _record_work_event(
        self,
        work: PostingWorkItem,
        *,
        action: str,
        result: str,
        details: dict[str, object] | None = None,
    ) -> None:
        self._record_event(
            lead_id=work.lead.id,
            post_id=work.post.id,
            group_id=work.post.group_id,
            action=action,
            result=result,
            details={
                "posting_attempt_id": work.attempt.id or 0,
                "dry_run": work.attempt.dry_run,
                "attempt_status": work.attempt.status.value,
                **(details or {}),
            },
        )

    def _record_event(
        self,
        *,
        lead_id: int | None,
        post_id: int | None,
        group_id: str,
        action: str,
        result: str,
        details: dict[str, object],
    ) -> None:
        self.database.record_audit_event(
            AuditEvent(
                component="posting",
                action=action,
                result=result,
                lead_id=lead_id,
                post_id=post_id,
                group_id=group_id,
                details=details,
            )
        )


def _safe_error_code(error: Exception) -> str:
    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        return code[:100]
    return type(error).__name__[:100]


def _exception_screenshot_path(error: Exception) -> Path | None:
    path = getattr(error, "screenshot_path", None)
    return path if isinstance(path, Path) else None


def _path_string(path: Path | None) -> str | None:
    return str(path) if path is not None else None


__all__ = [
    "ApprovedPostingService",
    "FacebookPostingAdapter",
    "PostingAttemptStatus",
    "PostingEligibilityError",
    "PostingError",
    "PostingExecutionResult",
    "PostingSubmissionResult",
    "PostingSubmissionUncertainError",
    "PostingValidation",
    "PostingValidationError",
]

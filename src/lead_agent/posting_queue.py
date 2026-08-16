"""Durable queue processing and outcome SMS for mobile-authorized Facebook posting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from lead_agent.database import Database
from lead_agent.models import (
    AuditEvent,
    NotificationStatus,
    PostingAttemptStatus,
    PostingJob,
    PostingJobStatus,
    utc_now,
)
from lead_agent.notifications import (
    SMS_BRAND_NAME,
    SMS_OPT_OUT_INSTRUCTION,
    SmsMessage,
    SmsProvider,
    SmsProviderError,
)
from lead_agent.posting import (
    ApprovedPostingService,
    FacebookPostingAdapter,
    PostingError,
)


@dataclass(frozen=True, slots=True)
class PostingQueueResult:
    """One queue processor invocation outcome."""

    job: PostingJob | None
    result: str


class PostingQueueProcessor:
    """Claim and execute at most one queued submission without automatic retries."""

    def __init__(
        self,
        database: Database,
        posting: ApprovedPostingService,
        *,
        approval_max_age_minutes: int,
    ) -> None:
        self.database = database
        self.posting = posting
        self.approval_max_age = timedelta(minutes=approval_max_age_minutes)

    def claim(self, *, now: datetime | None = None) -> PostingJob | None:
        return self.database.claim_next_posting_job(claimed_at=now or utc_now())

    def reconcile_stale_claims(
        self,
        *,
        now: datetime | None = None,
        stale_after_minutes: int = 10,
    ) -> int:
        """Recover only provably unstarted claims; terminalize all reserved attempts."""
        timestamp = now or utc_now()
        stale_before = timestamp - timedelta(minutes=stale_after_minutes)
        reconciled = 0
        for job in self.database.list_processing_posting_jobs():
            if job.claimed_at is None or job.claimed_at > stale_before or job.id is None:
                continue
            if (
                self.database.requeue_unstarted_posting_job(
                    job.id,
                    stale_before=stale_before,
                )
                is not None
            ):
                reconciled += 1
                continue
            attempts = [
                attempt
                for attempt in self.database.list_posting_attempts(lead_id=job.lead_id)
                if not attempt.dry_run
            ]
            if not attempts:
                continue
            attempt = attempts[-1]
            if attempt.status in {
                PostingAttemptStatus.VALIDATING,
                PostingAttemptStatus.SUBMITTING,
            }:
                if attempt.id is None:  # pragma: no cover - persisted attempt contract
                    raise RuntimeError("Posting attempt is missing its ID")
                attempt = self.database.fail_posting_attempt(
                    attempt.id,
                    failed_at=timestamp,
                    error_code="stale_posting_worker",
                ).attempt
            status_map = {
                PostingAttemptStatus.POSTED: PostingJobStatus.POSTED,
                PostingAttemptStatus.PENDING_MODERATION: PostingJobStatus.PENDING_MODERATION,
                PostingAttemptStatus.NEEDS_ATTENTION: PostingJobStatus.NEEDS_ATTENTION,
                PostingAttemptStatus.FAILED: PostingJobStatus.FAILED,
            }
            status = status_map.get(attempt.status, PostingJobStatus.NEEDS_ATTENTION)
            self.database.complete_posting_job(
                job.id,
                status=status,
                completed_at=timestamp,
                error_code="stale_posting_worker",
            )
            reconciled += 1
        return reconciled

    async def process(
        self,
        job: PostingJob,
        adapter: FacebookPostingAdapter,
        *,
        now: datetime | None = None,
    ) -> PostingQueueResult:
        timestamp = now or utc_now()
        if job.id is None:  # pragma: no cover - claimed jobs are persisted
            raise RuntimeError("Posting job is missing its database ID")
        if timestamp - job.requested_at >= self.approval_max_age:
            expired = self.database.expire_posting_job_for_rereview(
                job.id,
                expired_at=timestamp,
            )
            self._record(expired, result="expired")
            return PostingQueueResult(job=expired, result="expired")
        try:
            execution = await self.posting.execute(
                job.lead_id,
                adapter,
                dry_run=False,
                now=timestamp,
            )
        except PostingError as error:
            attempts = [
                attempt
                for attempt in self.database.list_posting_attempts(lead_id=job.lead_id)
                if not attempt.dry_run
            ]
            attempt = attempts[-1] if attempts else None
            status = (
                PostingJobStatus.NEEDS_ATTENTION
                if attempt is not None and attempt.status is PostingAttemptStatus.NEEDS_ATTENTION
                else PostingJobStatus.FAILED
            )
            failed = self.database.complete_posting_job(
                job.id,
                status=status,
                completed_at=timestamp,
                error_code=_posting_error_code(error),
            )
            self._record(failed, result=status.value)
            return PostingQueueResult(job=failed, result=status.value)
        attempt_status = execution.work.attempt.status
        status_map = {
            PostingAttemptStatus.POSTED: PostingJobStatus.POSTED,
            PostingAttemptStatus.PENDING_MODERATION: PostingJobStatus.PENDING_MODERATION,
            PostingAttemptStatus.NEEDS_ATTENTION: PostingJobStatus.NEEDS_ATTENTION,
            PostingAttemptStatus.FAILED: PostingJobStatus.FAILED,
        }
        status = status_map.get(attempt_status, PostingJobStatus.NEEDS_ATTENTION)
        completed = self.database.complete_posting_job(
            job.id,
            status=status,
            completed_at=timestamp,
            error_code=(
                None
                if status in {PostingJobStatus.POSTED, PostingJobStatus.PENDING_MODERATION}
                else "unexpected_posting_state"
            ),
        )
        self._record(completed, result=status.value)
        return PostingQueueResult(job=completed, result=status.value)

    def _record(self, job: PostingJob, *, result: str) -> None:
        lead = self.database.get_lead(job.lead_id)
        post = self.database.get_post(lead.facebook_post_id) if lead is not None else None
        self.database.record_audit_event(
            AuditEvent(
                component="posting_queue",
                action="posting_queue.completed",
                result=result,
                lead_id=job.lead_id,
                post_id=post.id if post is not None else None,
                group_id=post.group_id if post is not None else None,
                details={"posting_job_id": job.id or 0},
            )
        )


class PostingOutcomeNotificationService:
    """Send one non-retried SMS describing each terminal queued posting outcome."""

    def __init__(
        self,
        database: Database,
        provider: SmsProvider,
        *,
        recipient_number: str,
    ) -> None:
        self.database = database
        self.provider = provider
        self.recipient_number = recipient_number

    def notify_pending(self, *, limit: int = 10, now: datetime | None = None) -> int:
        timestamp = now or utc_now()
        sent = 0
        for job in self.database.list_unnotified_posting_jobs(limit=limit):
            if job.id is None:  # pragma: no cover - persisted job contract
                raise RuntimeError("Posting job is missing its database ID")
            if not self.database.claim_posting_outcome_notification(
                job.id,
                provider=self.provider.name,
                attempted_at=timestamp,
            ):
                continue
            message = SmsMessage(
                to=self.recipient_number,
                body=self._body(job),
                idempotency_key=f"posting-outcome:{job.id}",
            )
            try:
                receipt = self.provider.send(message)
            except SmsProviderError as error:
                self.database.complete_posting_outcome_notification(
                    job.id,
                    status=NotificationStatus.FAILED,
                    completed_at=timestamp,
                    error_code=type(error).__name__,
                )
                self._record(job, result="failed")
                continue
            self.database.complete_posting_outcome_notification(
                job.id,
                status=NotificationStatus.SENT,
                completed_at=timestamp,
                provider_message_id=receipt.provider_message_id,
            )
            self._record(job, result="sent")
            sent += 1
        return sent

    def _body(self, job: PostingJob) -> str:
        outcomes = {
            PostingJobStatus.POSTED: "posted publicly",
            PostingJobStatus.PENDING_MODERATION: "is pending group moderation",
            PostingJobStatus.EXPIRED: "expired and returned for fresh review",
            PostingJobStatus.FAILED: "stopped safely before confirmation",
            PostingJobStatus.NEEDS_ATTENTION: "needs manual Facebook review",
        }
        result = outcomes[job.status]
        failure_reasons = {
            "source_text_expanded": "Facebook revealed more post text; review it again",
            "source_text_mismatch": "the source post changed",
            "source_text_updated": "the saved source post changed; review it again",
            "comment_composer_missing": "Facebook did not expose a comment box",
            "comment_composer_ambiguous": "Facebook exposed ambiguous comment boxes",
            "response_already_visible": "the approved response may already be visible",
            "posting_controls_unreadable": "Facebook controls were not readable",
        }
        reason = failure_reasons.get(job.error_code or "")
        if job.status is PostingJobStatus.FAILED and reason is not None:
            result = f"stopped before sending: {reason}"
        body = f"{SMS_BRAND_NAME}: Lead {job.lead_id} {result}. {SMS_OPT_OUT_INSTRUCTION}"
        if len(body) > 160:  # pragma: no cover - fixed messages stay within one segment
            raise ValueError("Posting outcome SMS exceeds one segment")
        return body

    def _record(self, job: PostingJob, *, result: str) -> None:
        lead = self.database.get_lead(job.lead_id)
        post = self.database.get_post(lead.facebook_post_id) if lead is not None else None
        self.database.record_audit_event(
            AuditEvent(
                component="notification",
                action="posting_outcome.sms",
                result=result,
                lead_id=job.lead_id,
                post_id=post.id if post is not None else None,
                group_id=post.group_id if post is not None else None,
                details={
                    "posting_job_id": job.id or 0,
                    "provider": self.provider.name,
                },
            )
        )


def _posting_error_code(error: PostingError) -> str:
    code = getattr(error, "code", None)
    return code[:100] if isinstance(code, str) and code else type(error).__name__[:100]


__all__ = [
    "PostingOutcomeNotificationService",
    "PostingQueueProcessor",
    "PostingQueueResult",
]

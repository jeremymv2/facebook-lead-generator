from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lead_agent.approvals import ApprovalAction, LocalApprovalService
from lead_agent.config import Settings
from lead_agent.database import Database
from lead_agent.models import (
    FacebookPost,
    Lead,
    LeadIntent,
    LeadStatus,
    NotificationStatus,
    PostingAttemptStatus,
    PostingJobStatus,
)
from lead_agent.notifications import SmsDeliveryReceipt, SmsMessage, SmsProviderError
from lead_agent.posting import (
    ApprovedPostingService,
    PostingSubmissionResult,
    PostingSubmissionUncertainError,
    PostingValidation,
    PostingValidationError,
)
from lead_agent.posting_queue import (
    PostingOutcomeNotificationService,
    PostingQueueProcessor,
)

VALID_RESPONSE = (
    "JJ Miller & Co. handles deck repairs. Licensed & Insured. Free estimates. "
    "Text me at 502-528-0858. https://jjmillerco.com"
)


class FakePostingAdapter:
    def __init__(
        self,
        *,
        pending_moderation: bool = False,
        uncertain: bool = False,
        validation_error: PostingValidationError | None = None,
    ) -> None:
        self.pending_moderation = pending_moderation
        self.uncertain = uncertain
        self.validation_error = validation_error
        self.submit_calls = 0

    async def validate(self, work: object) -> PostingValidation:
        del work
        if self.validation_error is not None:
            raise self.validation_error
        return PostingValidation(before_screenshot_path=Path("before.png"))

    async def submit(
        self,
        work: object,
        validation: PostingValidation,
        *,
        on_before_submit: object,
    ) -> PostingSubmissionResult:
        del work, validation
        self.submit_calls += 1
        assert callable(on_before_submit)
        on_before_submit()
        if self.uncertain:
            raise PostingSubmissionUncertainError("Synthetic uncertainty")
        return PostingSubmissionResult(
            facebook_reply_url=(
                None
                if self.pending_moderation
                else "https://www.facebook.com/groups/111/posts/222?comment_id=333"
            ),
            pending_moderation=self.pending_moderation,
            after_screenshot_path=Path("after.png"),
        )


class FakeSmsProvider:
    name = "fake"

    def __init__(self) -> None:
        self.messages: list[SmsMessage] = []

    def send(self, message: SmsMessage) -> SmsDeliveryReceipt:
        self.messages.append(message)
        return SmsDeliveryReceipt(
            provider_message_id=f"message-{len(self.messages)}", status="sent"
        )


class FailingSmsProvider(FakeSmsProvider):
    def send(self, message: SmsMessage) -> SmsDeliveryReceipt:
        self.messages.append(message)
        raise SmsProviderError("Synthetic failure")


def live_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        posting_enabled=True,
        dry_run=False,
        posting_queue_enabled=True,
        database_path=tmp_path / "queue.sqlite3",
        facebook_profile_path=tmp_path / "profile",
        screenshot_dir=tmp_path / "screenshots",
        daily_posting_limit=1,
        per_group_daily_posting_limit=1,
    )


def queued_job(
    database: Database,
    *,
    now: datetime,
    queue_posting: bool = True,
) -> tuple[int, int]:
    post = database.save_post(
        FacebookPost(
            external_post_id="222",
            post_url="https://www.facebook.com/groups/111/posts/222",
            group_id="fixture-group",
            group_name="Fixture Group",
            post_text="Looking for someone in Louisville to repair a deck.",
        )
    ).post
    database.create_lead(
        Lead(
            facebook_post_id=post.id or 0,
            status=LeadStatus.CANDIDATE,
            service_category="decks",
            intent=LeadIntent.HIRING,
            overall_score=95,
            drafted_response=VALID_RESPONSE,
        )
    )
    approvals = LocalApprovalService(database, expiration_minutes=20)
    request = approvals.prepare_candidates(limit=1, now=now)[0].request
    review = approvals.decide(
        request.id or 0,
        ApprovalAction.APPROVE,
        queue_posting=queue_posting,
        now=now + timedelta(minutes=1),
    )
    if queue_posting:
        job = database.get_posting_job_for_approval(request.id or 0)
        assert job is not None
        return review.lead.id or 0, job.id or 0
    return review.lead.id or 0, 0


def processor(database: Database, settings: Settings) -> PostingQueueProcessor:
    posting = ApprovedPostingService(
        database,
        settings,
        posting_enabled_group_ids={"fixture-group"},
    )
    return PostingQueueProcessor(
        database,
        posting,
        approval_max_age_minutes=settings.posting_approval_max_age_minutes,
    )


def test_queue_posts_once_and_sends_one_outcome_sms(tmp_path: Path) -> None:
    settings = live_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    now = datetime(2026, 8, 14, 16, tzinfo=UTC)
    lead_id, job_id = queued_job(database, now=now)
    service = processor(database, settings)
    claimed = service.claim(now=now + timedelta(minutes=2))
    assert claimed is not None

    result = asyncio.run(
        service.process(claimed, FakePostingAdapter(), now=now + timedelta(minutes=2))
    )

    assert result.result == "posted"
    assert database.get_posting_job(job_id).status is PostingJobStatus.POSTED  # type: ignore[union-attr]
    assert database.get_lead(lead_id).status is LeadStatus.POSTED  # type: ignore[union-attr]
    provider = FakeSmsProvider()
    notifier = PostingOutcomeNotificationService(
        database,
        provider,
        recipient_number="+15025280858",
    )
    assert notifier.notify_pending(now=now + timedelta(minutes=3)) == 1
    assert "posted publicly" in provider.messages[0].body
    assert notifier.notify_pending(now=now + timedelta(minutes=4)) == 0
    assert len(provider.messages) == 1
    persisted = database.get_posting_job(job_id)
    assert persisted is not None
    assert persisted.outcome_notification_status is NotificationStatus.SENT


def test_queue_preserves_pending_moderation_and_uncertain_no_retry_states(
    tmp_path: Path,
) -> None:
    for pending, expected in (
        (True, PostingJobStatus.PENDING_MODERATION),
        (False, PostingJobStatus.NEEDS_ATTENTION),
    ):
        suffix = "pending" if pending else "uncertain"
        settings = live_settings(tmp_path / suffix)
        database = Database(settings.database_path)
        database.initialize()
        now = datetime(2026, 8, 14, 16, tzinfo=UTC)
        _, job_id = queued_job(database, now=now)
        service = processor(database, settings)
        claimed = service.claim(now=now + timedelta(minutes=2))
        assert claimed is not None
        adapter = FakePostingAdapter(
            pending_moderation=pending,
            uncertain=not pending,
        )

        result = asyncio.run(service.process(claimed, adapter, now=now + timedelta(minutes=2)))

        assert result.job is not None
        assert result.job.status is expected
        assert database.get_posting_job(job_id).status is expected  # type: ignore[union-attr]
        assert adapter.submit_calls == 1
        assert service.claim(now=now + timedelta(minutes=3)) is None


def test_expired_queue_job_returns_lead_for_fresh_review_without_browser(
    tmp_path: Path,
) -> None:
    settings = live_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    now = datetime(2026, 8, 14, 16, tzinfo=UTC)
    lead_id, job_id = queued_job(database, now=now)
    service = processor(database, settings)
    claimed = service.claim(now=now + timedelta(minutes=22))
    assert claimed is not None
    adapter = FakePostingAdapter()

    result = asyncio.run(service.process(claimed, adapter, now=now + timedelta(minutes=22)))

    assert result.result == "expired"
    assert adapter.submit_calls == 0
    assert database.get_posting_job(job_id).status is PostingJobStatus.EXPIRED  # type: ignore[union-attr]
    lead = database.get_lead(lead_id)
    assert lead is not None
    assert lead.status is LeadStatus.CANDIDATE
    assert lead.approved_response is None
    refreshed = LocalApprovalService(database, expiration_minutes=20).prepare_candidates(
        limit=1,
        now=now + timedelta(minutes=23),
    )
    assert len(refreshed) == 1
    assert refreshed[0].lead.id == lead_id


def test_blocked_unstarted_job_can_return_for_fresh_review(tmp_path: Path) -> None:
    settings = live_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    now = datetime(2026, 8, 14, 16, tzinfo=UTC)
    lead_id, job_id = queued_job(database, now=now)
    claimed = database.claim_next_posting_job(claimed_at=now + timedelta(minutes=2))
    assert claimed is not None
    database.complete_posting_job(
        job_id,
        status=PostingJobStatus.FAILED,
        completed_at=now + timedelta(minutes=2),
        error_code="posting_ineligible",
    )

    recovered = database.expire_posting_job_for_rereview(
        job_id,
        expired_at=now + timedelta(minutes=3),
    )

    assert recovered.status is PostingJobStatus.EXPIRED
    assert database.get_lead(lead_id).status is LeadStatus.CANDIDATE  # type: ignore[union-attr]


def test_queued_job_keeps_its_fresh_reservation_while_waiting_for_the_worker(
    tmp_path: Path,
) -> None:
    settings = live_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    now = datetime(2026, 8, 14, 16, tzinfo=UTC)
    lead_id, _ = queued_job(database, now=now, queue_posting=False)
    # Simulate a dashboard user approving first, then queuing eight minutes later.
    database.queue_approved_posting(
        lead_id,
        requested_at=now + timedelta(minutes=8),
        approval_max_age_minutes=settings.posting_approval_max_age_minutes,
    )
    # The original approval is now 26 minutes old, but the queue request is only 19 minutes old.
    process_time = now + timedelta(minutes=27)
    service = processor(database, settings)
    claimed = service.claim(now=process_time)
    assert claimed is not None

    result = asyncio.run(service.process(claimed, FakePostingAdapter(), now=process_time))

    assert result.result == "posted"
    assert database.get_lead(lead_id).status is LeadStatus.POSTED  # type: ignore[union-attr]


def test_stale_claim_without_a_live_attempt_is_safe_to_requeue(tmp_path: Path) -> None:
    settings = live_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    now = datetime(2026, 8, 14, 16, tzinfo=UTC)
    _, job_id = queued_job(database, now=now)
    service = processor(database, settings)
    assert service.claim(now=now + timedelta(minutes=2)) is not None

    assert service.reconcile_stale_claims(now=now + timedelta(minutes=13)) == 1

    job = database.get_posting_job(job_id)
    assert job is not None
    assert job.status is PostingJobStatus.QUEUED
    assert job.claimed_at is None


def test_stale_claim_with_reserved_attempt_is_terminalized_without_retry(
    tmp_path: Path,
) -> None:
    settings = live_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    now = datetime(2026, 8, 14, 16, tzinfo=UTC)
    lead_id, job_id = queued_job(database, now=now)
    service = processor(database, settings)
    assert service.claim(now=now + timedelta(minutes=2)) is not None
    claimed = database.begin_posting_attempt(
        lead_id,
        dry_run=False,
        started_at=now + timedelta(minutes=2),
        oldest_approval_at=now,
        day_started_at=datetime(2026, 8, 14, tzinfo=UTC),
        next_day_started_at=datetime(2026, 8, 15, tzinfo=UTC),
        daily_limit=1,
        per_group_daily_limit=1,
    )
    attempt_id = claimed.work.attempt.id or 0
    database.complete_posting_validation(
        attempt_id,
        validated_at=now + timedelta(minutes=2),
        before_screenshot_path="before.png",
    )

    assert service.reconcile_stale_claims(now=now + timedelta(minutes=13)) == 1

    job = database.get_posting_job(job_id)
    attempt = database.get_posting_attempt(attempt_id)
    assert job is not None
    assert attempt is not None
    assert job.status is PostingJobStatus.FAILED
    assert attempt.status is PostingAttemptStatus.FAILED
    assert database.get_lead(lead_id).status is LeadStatus.CANDIDATE  # type: ignore[union-attr]


def test_validation_failure_sms_names_the_safe_pre_submission_reason(tmp_path: Path) -> None:
    settings = live_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    now = datetime(2026, 8, 14, 16, tzinfo=UTC)
    lead_id, job_id = queued_job(database, now=now)
    service = processor(database, settings)
    claimed = service.claim(now=now + timedelta(minutes=2))
    assert claimed is not None

    result = asyncio.run(
        service.process(
            claimed,
            FakePostingAdapter(
                validation_error=PostingValidationError(
                    "Source changed",
                    code="source_text_mismatch",
                )
            ),
            now=now + timedelta(minutes=2),
        )
    )

    assert result.result == "failed"
    job = database.get_posting_job(job_id)
    assert job is not None
    assert job.error_code == "source_text_mismatch"
    assert database.get_lead(lead_id).status is LeadStatus.CANDIDATE  # type: ignore[union-attr]
    provider = FakeSmsProvider()
    notifier = PostingOutcomeNotificationService(
        database,
        provider,
        recipient_number="+15025280858",
    )
    assert notifier.notify_pending(now=now + timedelta(minutes=3)) == 1
    assert "stopped before sending: the source post changed" in provider.messages[0].body


def test_source_load_timeout_sms_does_not_claim_the_post_changed(tmp_path: Path) -> None:
    settings = live_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    now = datetime(2026, 8, 14, 16, tzinfo=UTC)
    _, job_id = queued_job(database, now=now)
    service = processor(database, settings)
    claimed = service.claim(now=now + timedelta(minutes=2))
    assert claimed is not None

    result = asyncio.run(
        service.process(
            claimed,
            FakePostingAdapter(
                validation_error=PostingValidationError(
                    "Source did not load",
                    code="source_post_load_timeout",
                )
            ),
            now=now + timedelta(minutes=2),
        )
    )

    assert result.result == "failed"
    job = database.get_posting_job(job_id)
    assert job is not None
    assert job.error_code == "source_post_load_timeout"
    provider = FakeSmsProvider()
    notifier = PostingOutcomeNotificationService(
        database,
        provider,
        recipient_number="+15025280858",
    )

    assert notifier.notify_pending(now=now + timedelta(minutes=3)) == 1
    assert "Facebook did not finish loading the source post" in provider.messages[0].body
    assert "source post changed" not in provider.messages[0].body


def test_outcome_sms_failure_is_recorded_without_automatic_retry(tmp_path: Path) -> None:
    settings = live_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    now = datetime(2026, 8, 14, 16, tzinfo=UTC)
    _, job_id = queued_job(database, now=now)
    service = processor(database, settings)
    claimed = service.claim(now=now + timedelta(minutes=2))
    assert claimed is not None
    asyncio.run(service.process(claimed, FakePostingAdapter(), now=now + timedelta(minutes=2)))
    provider = FailingSmsProvider()
    notifier = PostingOutcomeNotificationService(
        database,
        provider,
        recipient_number="+15025280858",
    )

    assert notifier.notify_pending(now=now + timedelta(minutes=3)) == 0
    assert notifier.notify_pending(now=now + timedelta(minutes=4)) == 0

    job = database.get_posting_job(job_id)
    assert job is not None
    assert job.outcome_notification_status is NotificationStatus.FAILED
    assert len(provider.messages) == 1

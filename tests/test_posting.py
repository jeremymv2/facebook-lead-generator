from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lead_agent.approvals import ApprovalAction, LocalApprovalService
from lead_agent.config import Settings
from lead_agent.database import Database
from lead_agent.facebook_posting import (
    FacebookCommentBrowser,
    post_text_is_safe_match,
    select_comment_permalink,
    validate_post_snapshot,
)
from lead_agent.models import (
    FacebookPost,
    Lead,
    LeadIntent,
    LeadStatus,
    PostingAttemptStatus,
    PostingWorkItem,
)
from lead_agent.posting import (
    ApprovedPostingService,
    PostingEligibilityError,
    PostingSubmissionResult,
    PostingSubmissionUncertainError,
    PostingValidation,
    PostingValidationError,
)

VALID_RESPONSE = (
    "JJ Miller & Co. handles deck repairs. Free estimates. "
    "Text me at 502-528-0858. https://jjmillerco.com"
)


def settings(tmp_path: Path, *, live: bool = False, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "facebook_profile_path": tmp_path / "facebook-profile",
        "screenshot_dir": tmp_path / "screenshots",
        "posting_enabled": live,
        "dry_run": not live,
        "posting_approval_max_age_minutes": 20,
        "daily_posting_limit": 5,
        "per_group_daily_posting_limit": 2,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def create_approved_lead(
    database: Database,
    *,
    post_id: str = "222",
    group_id: str = "fixture-group",
    group_path: str = "111",
    now: datetime,
) -> Lead:
    post = database.save_post(
        FacebookPost(
            external_post_id=post_id,
            post_url=f"https://www.facebook.com/groups/{group_path}/posts/{post_id}",
            group_id=group_id,
            group_name="Synthetic Homeowners",
            author_name="Fixture Customer",
            post_text="Looking for someone in Louisville to repair our deck this week.",
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
    request_id = approvals.prepare_candidates(limit=1, now=now)[0].request.id or 0
    return approvals.decide(
        request_id,
        ApprovalAction.APPROVE,
        now=now + timedelta(minutes=1),
    ).lead


class FakePostingAdapter:
    def __init__(
        self,
        *,
        validation_error: Exception | None = None,
        submission_error: Exception | None = None,
        cross_boundary: bool = True,
    ) -> None:
        self.validation_error = validation_error
        self.submission_error = submission_error
        self.cross_boundary = cross_boundary
        self.validate_calls = 0
        self.submit_calls = 0
        self.responses: list[str] = []

    async def validate(self, work: PostingWorkItem) -> PostingValidation:
        self.validate_calls += 1
        if self.validation_error is not None:
            raise self.validation_error
        assert work.attempt.approved_response == VALID_RESPONSE
        return PostingValidation(before_screenshot_path=Path("before.png"))

    async def submit(
        self,
        work: PostingWorkItem,
        validation: PostingValidation,
        *,
        on_before_submit: Callable[[], None],
    ) -> PostingSubmissionResult:
        del validation
        self.submit_calls += 1
        self.responses.append(work.attempt.approved_response)
        if self.cross_boundary:
            on_before_submit()
        if self.submission_error is not None:
            raise self.submission_error
        return PostingSubmissionResult(
            facebook_reply_url=f"{work.attempt.post_url}?comment_id=999",
            after_screenshot_path=Path("after.png"),
        )


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "posting.sqlite3")
    database.initialize()
    return database


def test_dry_run_validates_without_submit_and_can_repeat(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    lead = create_approved_lead(database, now=approved_at)
    service = ApprovedPostingService(
        database,
        settings(tmp_path),
        enabled_group_ids={"fixture-group"},
    )
    adapter = FakePostingAdapter()

    first = asyncio.run(
        service.execute(
            lead.id or 0,
            adapter,
            dry_run=True,
            now=approved_at + timedelta(minutes=2),
        )
    )
    second = asyncio.run(
        service.execute(
            lead.id or 0,
            adapter,
            dry_run=True,
            now=approved_at + timedelta(minutes=3),
        )
    )

    assert first.work.attempt.status is PostingAttemptStatus.DRY_RUN_VALIDATED
    assert second.work.attempt.status is PostingAttemptStatus.DRY_RUN_VALIDATED
    assert adapter.validate_calls == 2
    assert adapter.submit_calls == 0
    assert len(database.list_posting_attempts(lead_id=lead.id)) == 2
    assert database.get_lead(lead.id or 0).status is LeadStatus.APPROVED  # type: ignore[union-attr]


def test_live_post_uses_snapshot_once_and_second_invocation_is_idempotent(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    lead = create_approved_lead(database, now=approved_at)
    service = ApprovedPostingService(
        database,
        settings(tmp_path, live=True),
        enabled_group_ids={"fixture-group"},
    )
    first_adapter = FakePostingAdapter()

    first = asyncio.run(
        service.execute(
            lead.id or 0,
            first_adapter,
            dry_run=False,
            now=approved_at + timedelta(minutes=2),
        )
    )
    second_adapter = FakePostingAdapter()
    second = asyncio.run(
        service.execute(
            lead.id or 0,
            second_adapter,
            dry_run=False,
            now=approved_at + timedelta(minutes=3),
        )
    )

    assert first.created is True
    assert first.work.attempt.status is PostingAttemptStatus.POSTED
    assert first_adapter.responses == [VALID_RESPONSE]
    assert second.created is False
    assert second.work.attempt.id == first.work.attempt.id
    assert second_adapter.validate_calls == 0
    assert second_adapter.submit_calls == 0
    persisted = database.get_lead(lead.id or 0)
    assert persisted is not None
    assert persisted.status is LeadStatus.POSTED
    assert persisted.facebook_reply_url is not None


def test_stale_approval_stops_before_creating_an_attempt(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    lead = create_approved_lead(database, now=approved_at)
    service = ApprovedPostingService(
        database,
        settings(tmp_path),
        enabled_group_ids={"fixture-group"},
    )
    adapter = FakePostingAdapter()

    with pytest.raises(PostingEligibilityError, match="stale"):
        asyncio.run(
            service.execute(
                lead.id or 0,
                adapter,
                dry_run=True,
                now=approved_at + timedelta(minutes=22),
            )
        )

    assert database.list_posting_attempts(lead_id=lead.id) == []
    assert adapter.validate_calls == 0


def test_disabled_group_stops_before_claim_or_browser(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    lead = create_approved_lead(database, now=approved_at)
    service = ApprovedPostingService(database, settings(tmp_path), enabled_group_ids=set())
    adapter = FakePostingAdapter()

    with pytest.raises(PostingEligibilityError, match="not currently enabled"):
        asyncio.run(
            service.execute(
                lead.id or 0,
                adapter,
                dry_run=True,
                now=approved_at + timedelta(minutes=2),
            )
        )

    assert database.list_posting_attempts(lead_id=lead.id) == []
    assert adapter.validate_calls == 0


def test_live_validation_failure_never_submits_and_requires_attention(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    lead = create_approved_lead(database, now=approved_at)
    service = ApprovedPostingService(
        database,
        settings(tmp_path, live=True),
        enabled_group_ids={"fixture-group"},
    )
    adapter = FakePostingAdapter(
        validation_error=PostingValidationError("Post changed", screenshot_path=Path("bad.png"))
    )

    with pytest.raises(PostingValidationError, match="Post changed"):
        asyncio.run(
            service.execute(
                lead.id or 0,
                adapter,
                dry_run=False,
                now=approved_at + timedelta(minutes=2),
            )
        )

    attempt = database.list_posting_attempts(lead_id=lead.id)[0]
    assert attempt.status is PostingAttemptStatus.FAILED
    assert attempt.submission_started_at is None
    assert attempt.error_code == "facebook_validation_failed"
    assert adapter.submit_calls == 0
    assert database.get_lead(lead.id or 0).status is LeadStatus.NEEDS_ATTENTION  # type: ignore[union-attr]


def test_uncertain_submission_is_never_retried(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    lead = create_approved_lead(database, now=approved_at)
    service = ApprovedPostingService(
        database,
        settings(tmp_path, live=True),
        enabled_group_ids={"fixture-group"},
    )
    adapter = FakePostingAdapter(
        submission_error=PostingSubmissionUncertainError(
            "Result unknown", screenshot_path=Path("unknown.png")
        )
    )

    with pytest.raises(PostingSubmissionUncertainError, match="unknown"):
        asyncio.run(
            service.execute(
                lead.id or 0,
                adapter,
                dry_run=False,
                now=approved_at + timedelta(minutes=2),
            )
        )

    attempt = database.list_posting_attempts(lead_id=lead.id)[0]
    assert attempt.status is PostingAttemptStatus.NEEDS_ATTENTION
    assert attempt.submission_started_at is not None
    retry_adapter = FakePostingAdapter()
    retry = asyncio.run(
        service.execute(
            lead.id or 0,
            retry_adapter,
            dry_run=False,
            now=approved_at + timedelta(minutes=3),
        )
    )
    assert retry.created is False
    assert retry.work.attempt.status is PostingAttemptStatus.NEEDS_ATTENTION
    assert retry_adapter.validate_calls == 0


def test_daily_limit_reserves_live_attempts_transactionally(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    first = create_approved_lead(database, post_id="222", now=approved_at)
    second = create_approved_lead(database, post_id="333", now=approved_at)
    service = ApprovedPostingService(
        database,
        settings(tmp_path, live=True, daily_posting_limit=1),
        enabled_group_ids={"fixture-group"},
    )
    asyncio.run(
        service.execute(
            first.id or 0,
            FakePostingAdapter(),
            dry_run=False,
            now=approved_at + timedelta(minutes=2),
        )
    )

    with pytest.raises(PostingEligibilityError, match="Global daily"):
        asyncio.run(
            service.execute(
                second.id or 0,
                FakePostingAdapter(),
                dry_run=False,
                now=approved_at + timedelta(minutes=3),
            )
        )

    assert database.list_posting_attempts(lead_id=second.id) == []


def test_per_group_limit_is_independent_from_global_limit(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    first = create_approved_lead(database, post_id="222", now=approved_at)
    second = create_approved_lead(database, post_id="333", now=approved_at)
    service = ApprovedPostingService(
        database,
        settings(tmp_path, live=True, daily_posting_limit=5, per_group_daily_posting_limit=1),
        enabled_group_ids={"fixture-group"},
    )
    asyncio.run(
        service.execute(
            first.id or 0,
            FakePostingAdapter(),
            dry_run=False,
            now=approved_at + timedelta(minutes=2),
        )
    )

    with pytest.raises(PostingEligibilityError, match="Per-group daily"):
        asyncio.run(
            service.execute(
                second.id or 0,
                FakePostingAdapter(),
                dry_run=False,
                now=approved_at + timedelta(minutes=3),
            )
        )


def test_post_text_matching_rejects_resolved_or_materially_changed_posts() -> None:
    expected = "Looking for someone to repair our deck in Louisville this week."

    assert post_text_is_safe_match(expected, expected)
    assert post_text_is_safe_match(expected, expected.replace(".", "!"))
    assert not post_text_is_safe_match(expected, f"{expected} Update: found someone.")
    assert not post_text_is_safe_match(expected, "Looking for someone to mow our lawn today.")


def test_snapshot_validation_requires_exact_post_group_and_integrity(
    database: Database,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    lead = create_approved_lead(database, now=approved_at)
    claimed = database.begin_posting_attempt(
        lead.id or 0,
        dry_run=True,
        started_at=approved_at + timedelta(minutes=2),
        oldest_approval_at=approved_at,
        day_started_at=datetime(2026, 8, 8, 4, 0, tzinfo=UTC),
        next_day_started_at=datetime(2026, 8, 9, 4, 0, tzinfo=UTC),
        daily_limit=5,
        per_group_daily_limit=2,
    )

    matched = validate_post_snapshot(
        claimed.work,
        current_url="https://www.facebook.com/groups/111/posts/222",
        rendered_post_texts=[claimed.work.post.post_text],
    )
    assert matched == claimed.work.post.post_text

    with pytest.raises(PostingValidationError, match="exact approved post"):
        validate_post_snapshot(
            claimed.work,
            current_url="https://www.facebook.com/groups/111/posts/999",
            rendered_post_texts=[claimed.work.post.post_text],
        )
    with pytest.raises(PostingValidationError, match="approved group"):
        validate_post_snapshot(
            claimed.work,
            current_url="https://www.facebook.com/groups/999/posts/222",
            rendered_post_texts=[claimed.work.post.post_text],
        )


def test_comment_permalink_keeps_only_same_post_comment_identity() -> None:
    post_url = "https://www.facebook.com/groups/111/posts/222"

    assert (
        select_comment_permalink(
            [
                "https://example.com/groups/111/posts/222?comment_id=777",
                "/groups/111/posts/222?comment_id=777&ref=share",
            ],
            post_url,
        )
        == "https://www.facebook.com/groups/111/posts/222?comment_id=777"
    )
    assert (
        select_comment_permalink(
            ["/groups/111/posts/999?comment_id=777"],
            post_url,
        )
        is None
    )


def test_dry_run_browser_validation_contains_no_write_actions() -> None:
    source = inspect.getsource(FacebookCommentBrowser.validate)

    for forbidden_call in (".click(", ".fill(", ".type(", ".press(", ".check("):
        assert forbidden_call not in source

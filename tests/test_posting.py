from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lead_agent.approvals import ApprovalAction, LocalApprovalService
from lead_agent.config import Settings
from lead_agent.database import Database
from lead_agent.facebook import STORY_MESSAGE_SELECTOR
from lead_agent.facebook_posting import (
    FacebookCommentBrowser,
    confirmed_comment_permalink,
    pending_content_url,
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
    PostingSourceTextExpandedError,
    PostingSubmissionResult,
    PostingSubmissionUncertainError,
    PostingValidation,
    PostingValidationError,
)

VALID_RESPONSE = (
    "JJ Miller & Co. handles deck repairs. Licensed & Insured. Free estimates. "
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
            post_text=f"Looking for someone in Louisville to repair deck project {post_id}.",
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
        return_permalink: bool = True,
        pending_moderation: bool = False,
    ) -> None:
        self.validation_error = validation_error
        self.submission_error = submission_error
        self.cross_boundary = cross_boundary
        self.return_permalink = return_permalink
        self.pending_moderation = pending_moderation
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
            facebook_reply_url=(
                f"{work.attempt.post_url}?comment_id=999"
                if self.return_permalink and not self.pending_moderation
                else None
            ),
            pending_moderation=self.pending_moderation,
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
        posting_enabled_group_ids={"fixture-group"},
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
        posting_enabled_group_ids={"fixture-group"},
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
        posting_enabled_group_ids={"fixture-group"},
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
    service = ApprovedPostingService(database, settings(tmp_path), posting_enabled_group_ids=set())
    adapter = FakePostingAdapter()

    with pytest.raises(PostingEligibilityError, match="explicitly enabled for posting"):
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


def test_live_validation_failure_returns_for_fresh_review_and_can_retry(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    lead = create_approved_lead(database, now=approved_at)
    service = ApprovedPostingService(
        database,
        settings(tmp_path, live=True),
        posting_enabled_group_ids={"fixture-group"},
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
    returned = database.get_lead(lead.id or 0)
    assert returned is not None
    assert returned.status is LeadStatus.CANDIDATE
    assert returned.approved_response is None
    assert returned.approval_timestamp is None

    approvals = LocalApprovalService(database, expiration_minutes=20)
    request = approvals.prepare_candidates(
        limit=1,
        now=approved_at + timedelta(minutes=3),
    )[0].request
    approvals.decide(
        request.id or 0,
        ApprovalAction.APPROVE,
        now=approved_at + timedelta(minutes=4),
    )
    retried = asyncio.run(
        service.execute(
            lead.id or 0,
            FakePostingAdapter(),
            dry_run=False,
            now=approved_at + timedelta(minutes=5),
        )
    )

    assert retried.created is True
    assert retried.work.attempt.status is PostingAttemptStatus.POSTED
    assert [value.status for value in database.list_posting_attempts(lead_id=lead.id)] == [
        PostingAttemptStatus.FAILED,
        PostingAttemptStatus.POSTED,
    ]


def test_source_prefix_expansion_is_saved_for_fresh_review(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    lead = create_approved_lead(database, now=approved_at)
    original = database.get_post(lead.facebook_post_id)
    assert original is not None
    expanded = f"{original.post_text} Pickup is in Shawnee."
    service = ApprovedPostingService(
        database,
        settings(tmp_path, live=True),
        posting_enabled_group_ids={"fixture-group"},
    )

    with pytest.raises(PostingSourceTextExpandedError):
        asyncio.run(
            service.execute(
                lead.id or 0,
                FakePostingAdapter(
                    validation_error=PostingSourceTextExpandedError(
                        "More text",
                        observed_post_text=expanded,
                    )
                ),
                dry_run=False,
                now=approved_at + timedelta(minutes=2),
            )
        )

    refreshed = database.get_post(lead.facebook_post_id)
    assert refreshed is not None
    assert refreshed.post_text == expanded
    assert database.get_lead(lead.id or 0).status is LeadStatus.CANDIDATE  # type: ignore[union-attr]
    assert database.list_posting_attempts(lead_id=lead.id)[0].error_code == ("source_text_expanded")


def test_uncertain_submission_is_never_retried(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    lead = create_approved_lead(database, now=approved_at)
    service = ApprovedPostingService(
        database,
        settings(tmp_path, live=True),
        posting_enabled_group_ids={"fixture-group"},
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


def test_uncertain_submission_can_be_manually_reconciled_as_pending_moderation(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    lead = create_approved_lead(database, now=approved_at)
    service = ApprovedPostingService(
        database,
        settings(tmp_path, live=True),
        posting_enabled_group_ids={"fixture-group"},
    )

    with pytest.raises(PostingSubmissionUncertainError):
        asyncio.run(
            service.execute(
                lead.id or 0,
                FakePostingAdapter(
                    submission_error=PostingSubmissionUncertainError("Result unknown")
                ),
                dry_run=False,
                now=approved_at + timedelta(minutes=2),
            )
        )

    attempt = database.list_posting_attempts(lead_id=lead.id)[0]
    reconciled_at = approved_at + timedelta(minutes=10)
    work = database.reconcile_posting_moderation(
        attempt.id or 0,
        reconciled_at=reconciled_at,
        after_screenshot_path="manual-pending-proof.png",
    )

    assert work.attempt.status is PostingAttemptStatus.PENDING_MODERATION
    assert work.attempt.error_code is None
    assert work.lead.status is LeadStatus.PENDING_MODERATION
    assert work.lead.posting_timestamp == attempt.submission_started_at
    assert work.lead.error_state is None
    assert work.lead.screenshot_path == "manual-pending-proof.png"
    assert (
        database.reconcile_posting_moderation(
            attempt.id or 0,
            reconciled_at=reconciled_at + timedelta(minutes=1),
        ).attempt.status
        is PostingAttemptStatus.PENDING_MODERATION
    )


def test_permalinkless_submission_requires_attention(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    lead = create_approved_lead(database, now=approved_at)
    service = ApprovedPostingService(
        database,
        settings(tmp_path, live=True),
        posting_enabled_group_ids={"fixture-group"},
    )

    with pytest.raises(PostingSubmissionUncertainError, match="stable comment permalink"):
        asyncio.run(
            service.execute(
                lead.id or 0,
                FakePostingAdapter(return_permalink=False),
                dry_run=False,
                now=approved_at + timedelta(minutes=2),
            )
        )

    attempt = database.list_posting_attempts(lead_id=lead.id)[0]
    assert attempt.status is PostingAttemptStatus.NEEDS_ATTENTION
    assert attempt.submission_started_at is not None
    assert database.get_lead(lead.id or 0).status is LeadStatus.NEEDS_ATTENTION  # type: ignore[union-attr]


def test_pending_moderation_is_durable_and_never_retried(
    database: Database,
    tmp_path: Path,
) -> None:
    approved_at = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    lead = create_approved_lead(database, now=approved_at)
    service = ApprovedPostingService(
        database,
        settings(tmp_path, live=True),
        posting_enabled_group_ids={"fixture-group"},
    )

    first = asyncio.run(
        service.execute(
            lead.id or 0,
            FakePostingAdapter(pending_moderation=True),
            dry_run=False,
            now=approved_at + timedelta(minutes=2),
        )
    )

    assert first.work.attempt.status is PostingAttemptStatus.PENDING_MODERATION
    persisted = database.get_lead(lead.id or 0)
    assert persisted is not None
    assert persisted.status is LeadStatus.PENDING_MODERATION
    assert persisted.facebook_reply_url is None
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
    assert retry.work.attempt.status is PostingAttemptStatus.PENDING_MODERATION
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
        posting_enabled_group_ids={"fixture-group"},
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
        posting_enabled_group_ids={"fixture-group"},
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

    expanded = f"{claimed.work.post.post_text} Pickup is in Shawnee."
    with pytest.raises(PostingSourceTextExpandedError) as expansion:
        validate_post_snapshot(
            claimed.work,
            current_url="https://www.facebook.com/groups/111/posts/222",
            rendered_post_texts=[expanded],
        )
    assert expansion.value.code == "source_text_expanded"
    assert expansion.value.observed_post_text == expanded


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


def test_comment_confirmation_rejects_transient_or_permalinkless_ui() -> None:
    post_url = "https://www.facebook.com/groups/111/posts/222"
    comment_url = f"{post_url}?comment_id=777"

    assert (
        confirmed_comment_permalink(
            expected_response=VALID_RESPONSE,
            rendered_comment_texts=[VALID_RESPONSE],
            article_text=f"Jeremy Miller\n{VALID_RESPONSE}\nPosting…",
            hrefs=[comment_url],
            post_url=post_url,
        )
        is None
    )
    assert (
        confirmed_comment_permalink(
            expected_response=VALID_RESPONSE,
            rendered_comment_texts=[VALID_RESPONSE],
            article_text=f"Jeremy Miller\n{VALID_RESPONSE}",
            hrefs=[post_url],
            post_url=post_url,
        )
        is None
    )
    assert (
        confirmed_comment_permalink(
            expected_response=VALID_RESPONSE,
            rendered_comment_texts=[VALID_RESPONSE],
            article_text=f"Jeremy Miller\n{VALID_RESPONSE}\nJust now",
            hrefs=[comment_url],
            post_url=post_url,
        )
        == comment_url
    )


def test_pending_content_url_is_scoped_to_the_exact_group() -> None:
    assert (
        pending_content_url("https://www.facebook.com/groups/111/posts/222")
        == "https://www.facebook.com/groups/111/my_pending_content"
    )
    assert pending_content_url("https://www.facebook.com/story.php?story_fbid=222") is None


def test_pending_moderation_requires_exact_response_in_group_queue(
    database: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    browser = FacebookCommentBrowser(settings(tmp_path))
    page = MagicMock()
    page.goto = AsyncMock()
    texts = MagicMock()
    exact_text = MagicMock()
    page.locator.return_value = texts
    texts.count = AsyncMock(return_value=1)
    texts.nth.return_value = exact_text
    exact_text.is_visible = AsyncMock(return_value=True)
    exact_text.inner_text = AsyncMock(return_value=VALID_RESPONSE)
    require_normal_page = AsyncMock()
    innermost_locators = AsyncMock(return_value=[exact_text])
    monkeypatch.setattr(browser, "_require_normal_page", require_normal_page)
    monkeypatch.setattr(browser, "_innermost_locators", innermost_locators)

    pending = asyncio.run(browser._pending_moderation_is_visible(page, claimed.work))

    assert pending is True
    page.goto.assert_awaited_once_with(
        "https://www.facebook.com/groups/111/my_pending_content",
        wait_until="domcontentloaded",
    )
    require_normal_page.assert_awaited_once_with(page, lead_id=lead.id or 0)
    innermost_locators.assert_awaited_once_with([exact_text])


def test_comment_confirmation_reloads_the_stable_permalink(
    database: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    browser = FacebookCommentBrowser(settings(tmp_path))
    page = AsyncMock()
    expected_url = f"{claimed.work.attempt.post_url}?comment_id=999"
    require_normal_page = AsyncMock()
    wait_for_exact_comment = AsyncMock(return_value=expected_url)
    monkeypatch.setattr(browser, "_require_normal_page", require_normal_page)
    monkeypatch.setattr(browser, "_wait_for_exact_comment", wait_for_exact_comment)

    confirmed_url = asyncio.run(
        browser._confirm_comment_survived_reload(page, claimed.work, expected_url)
    )

    assert confirmed_url == expected_url
    page.goto.assert_awaited_once_with(expected_url, wait_until="domcontentloaded")
    require_normal_page.assert_awaited_once_with(page, lead_id=lead.id or 0)
    wait_for_exact_comment.assert_awaited_once_with(page, claimed.work)


def test_story_validation_prefers_one_visible_foreground_dialog(tmp_path: Path) -> None:
    browser = FacebookCommentBrowser(settings(tmp_path))
    page = MagicMock()
    dialogs = MagicMock()
    dialog = MagicMock()
    messages = MagicMock()
    message = MagicMock()
    page.get_by_role.return_value = dialogs
    dialogs.count = AsyncMock(return_value=1)
    dialogs.nth.return_value = dialog
    dialog.is_visible = AsyncMock(return_value=True)
    dialog.locator.return_value = messages
    messages.count = AsyncMock(return_value=1)
    messages.nth.return_value = message
    message.is_visible = AsyncMock(return_value=True)
    dialog_handle = MagicMock()
    dialog.element_handle = AsyncMock(return_value=dialog_handle)
    dialog_handle.dispose = AsyncMock()

    scope = asyncio.run(browser._story_message_scope(page))

    assert scope is dialog
    dialog.locator.assert_called_once_with(STORY_MESSAGE_SELECTOR)


def test_story_validation_rejects_multiple_foreground_dialogs(tmp_path: Path) -> None:
    browser = FacebookCommentBrowser(settings(tmp_path))
    page = MagicMock()
    dialogs = MagicMock()
    page.get_by_role.return_value = dialogs
    dialogs.count = AsyncMock(return_value=2)
    configured_dialogs = []
    configured_handles = []
    for _index in range(2):
        dialog = MagicMock()
        messages = MagicMock()
        message = MagicMock()
        dialog.is_visible = AsyncMock(return_value=True)
        dialog.locator.return_value = messages
        messages.count = AsyncMock(return_value=1)
        messages.nth.return_value = message
        message.is_visible = AsyncMock(return_value=True)
        dialog_handle = MagicMock()
        dialog.element_handle = AsyncMock(return_value=dialog_handle)
        dialog_handle.evaluate = AsyncMock(return_value=False)
        dialog_handle.dispose = AsyncMock()
        configured_dialogs.append(dialog)
        configured_handles.append(dialog_handle)
    dialogs.nth.side_effect = configured_dialogs

    with pytest.raises(PostingValidationError, match="more than one foreground"):
        asyncio.run(browser._story_message_scope(page))


def test_story_validation_discards_nested_wrapper_dialog(tmp_path: Path) -> None:
    browser = FacebookCommentBrowser(settings(tmp_path))
    page = MagicMock()
    dialogs = MagicMock()
    page.get_by_role.return_value = dialogs
    dialogs.count = AsyncMock(return_value=2)
    outer = MagicMock()
    inner = MagicMock()
    outer_handle = MagicMock()
    inner_handle = MagicMock()
    for dialog, handle in ((outer, outer_handle), (inner, inner_handle)):
        messages = MagicMock()
        message = MagicMock()
        dialog.is_visible = AsyncMock(return_value=True)
        dialog.locator.return_value = messages
        messages.count = AsyncMock(return_value=1)
        messages.nth.return_value = message
        message.is_visible = AsyncMock(return_value=True)
        dialog.element_handle = AsyncMock(return_value=handle)
        handle.dispose = AsyncMock()
    outer_handle.evaluate = AsyncMock(side_effect=lambda _expression, other: other is inner_handle)
    inner_handle.evaluate = AsyncMock(return_value=False)
    dialogs.nth.side_effect = [outer, inner]

    scope = asyncio.run(browser._story_message_scope(page))

    assert scope is inner


def test_comment_composer_waits_for_facebook_to_finish_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = FacebookCommentBrowser(settings(tmp_path))
    page = MagicMock()
    message = MagicMock()
    owner = MagicMock()
    scoped = MagicMock()
    global_composers = MagicMock()
    composer = MagicMock()
    message.locator.return_value = owner
    owner.locator.return_value = scoped
    page.locator.return_value = global_composers
    page.wait_for_timeout = AsyncMock()
    visible_composers = AsyncMock(side_effect=[[], [], [composer]])
    require_normal_page = AsyncMock()
    monkeypatch.setattr(browser, "_visible_comment_composers", visible_composers)
    monkeypatch.setattr(browser, "_require_normal_page", require_normal_page)

    selected = asyncio.run(browser._comment_composer(page, message, lead_id=1989))

    assert selected is composer
    assert visible_composers.await_count == 3
    page.wait_for_timeout.assert_awaited_once_with(250)
    require_normal_page.assert_awaited_once_with(page, lead_id=1989)


def test_dry_run_browser_validation_contains_no_write_actions() -> None:
    source = inspect.getsource(FacebookCommentBrowser.validate)

    for forbidden_call in (".click(", ".fill(", ".type(", ".press(", ".check("):
        assert forbidden_call not in source

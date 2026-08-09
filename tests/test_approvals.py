from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lead_agent.approvals import (
    ApprovalAction,
    ApprovalExpiredError,
    ApprovalStateError,
    ApprovalValidationError,
    LocalApprovalService,
)
from lead_agent.database import Database
from lead_agent.models import (
    ApprovalStatus,
    FacebookPost,
    Lead,
    LeadIntent,
    LeadStatus,
    RejectionReason,
)

VALID_DRAFT = (
    "JJ Miller & Co. can help with your deck project. Free estimates. "
    "Text me at 502-528-0858 or visit https://jjmillerco.com."
)
VALID_EDIT = (
    "JJ Miller & Co. handles deck repairs. Free estimates. "
    "Text me at 502-528-0858. https://jjmillerco.com"
)


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "approvals.sqlite3")
    database.initialize()
    return database


def create_candidate(
    database: Database,
    *,
    draft: str | None = VALID_DRAFT,
    external_post_id: str = "222",
    group_id: str = "fixture-group",
) -> Lead:
    post = database.save_post(
        FacebookPost(
            external_post_id=external_post_id,
            post_url=f"https://www.facebook.com/groups/111/posts/{external_post_id}",
            group_id=group_id,
            group_name="Synthetic Fixture Group",
            author_name="Fixture Customer",
            post_text="Looking for someone in Louisville to repair our deck this week.",
        )
    ).post
    return database.create_lead(
        Lead(
            facebook_post_id=post.id or 0,
            status=LeadStatus.CANDIDATE,
            service_category="decks",
            location="Louisville",
            intent=LeadIntent.HIRING,
            is_residential=True,
            is_spam=False,
            overall_score=95,
            confidence=0.95,
            drafted_response=draft,
        )
    )


def test_prepare_candidates_snapshots_draft_and_starts_expiration(database: Database) -> None:
    lead = create_candidate(database)
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    service = LocalApprovalService(database, expiration_minutes=20)

    reviews = service.prepare_candidates(limit=10, now=now)

    assert len(reviews) == 1
    review = reviews[0]
    assert review.request.lead_id == lead.id
    assert review.request.status is ApprovalStatus.PENDING
    assert review.request.draft_response == VALID_DRAFT
    assert review.request.expires_at == now + timedelta(minutes=20)
    assert review.lead.status is LeadStatus.PENDING_APPROVAL
    assert review.lead.approval_expires_at == now + timedelta(minutes=20)
    assert [event.action for event in database.list_audit_events()] == ["approval.requested"]


def test_approve_is_one_time_and_preserves_exact_draft(database: Database) -> None:
    create_candidate(database)
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    service = LocalApprovalService(database, expiration_minutes=20)
    request_id = service.prepare_candidates(limit=10, now=now)[0].request.id or 0

    approved = service.decide(
        request_id,
        ApprovalAction.APPROVE,
        now=now + timedelta(minutes=1),
    )

    assert approved.request.status is ApprovalStatus.APPROVED
    assert approved.request.decided_response == VALID_DRAFT
    assert approved.lead.status is LeadStatus.APPROVED
    assert approved.lead.approved_response == VALID_DRAFT
    feedback = database.approval_feedback_summary()
    assert feedback.reviewed == 1
    assert feedback.accepted == 1
    assert feedback.acceptance_percent == 100.0
    with pytest.raises(ApprovalStateError, match="already been decided"):
        service.decide(
            request_id,
            ApprovalAction.REJECT,
            now=now + timedelta(minutes=2),
        )
    assert database.get_approval_request(request_id) == approved.request


def test_edit_requires_a_locally_valid_response(database: Database) -> None:
    create_candidate(database)
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    service = LocalApprovalService(database, expiration_minutes=20)
    request_id = service.prepare_candidates(limit=10, now=now)[0].request.id or 0

    with pytest.raises(ApprovalValidationError, match="company identity"):
        service.decide(
            request_id,
            ApprovalAction.EDIT,
            edited_response="Call me.",
            now=now + timedelta(minutes=1),
        )
    assert database.get_approval_request(request_id).status is ApprovalStatus.PENDING  # type: ignore[union-attr]

    edited = service.decide(
        request_id,
        ApprovalAction.EDIT,
        edited_response=VALID_EDIT,
        now=now + timedelta(minutes=2),
    )

    assert edited.request.status is ApprovalStatus.EDITED
    assert edited.lead.status is LeadStatus.EDITED
    assert edited.lead.approved_response == VALID_EDIT


def test_reject_never_creates_an_approved_response(database: Database) -> None:
    create_candidate(database)
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    service = LocalApprovalService(database, expiration_minutes=20)
    request_id = service.prepare_candidates(limit=10, now=now)[0].request.id or 0

    with pytest.raises(ApprovalValidationError, match="rejection reason"):
        service.decide(request_id, ApprovalAction.REJECT, now=now + timedelta(seconds=30))

    rejected = service.decide(
        request_id,
        ApprovalAction.REJECT,
        rejection_reason=RejectionReason.PROVIDER_ADVERTISEMENT,
        now=now + timedelta(minutes=1),
    )

    assert rejected.request.status is ApprovalStatus.REJECTED
    assert rejected.request.decided_response is None
    assert rejected.request.rejection_reason is RejectionReason.PROVIDER_ADVERTISEMENT
    assert rejected.lead.status is LeadStatus.REJECTED
    assert rejected.lead.approved_response is None
    feedback = database.approval_feedback_summary()
    assert feedback.rejected == 1
    assert feedback.rejection_reasons == (("provider_advertisement", 1),)


def test_expired_approval_requires_re_review_and_cannot_be_decided(database: Database) -> None:
    create_candidate(database)
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    service = LocalApprovalService(database, expiration_minutes=20)
    request_id = service.prepare_candidates(limit=10, now=now)[0].request.id or 0

    pending = service.list_pending(now=now + timedelta(minutes=20))

    assert pending == []
    request = database.get_approval_request(request_id)
    assert request is not None
    assert request.status is ApprovalStatus.EXPIRED
    lead = database.get_lead(request.lead_id)
    assert lead is not None
    assert lead.status is LeadStatus.EXPIRED
    with pytest.raises(ApprovalExpiredError, match="re-review"):
        service.decide(
            request_id,
            ApprovalAction.APPROVE,
            now=now + timedelta(minutes=21),
        )


def test_local_backlog_restores_expired_candidate_and_decides_at_click_time(
    database: Database,
) -> None:
    lead = create_candidate(database)
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    service = LocalApprovalService(database, expiration_minutes=20)
    expired_request_id = service.prepare_candidates(limit=10, now=now)[0].request.id or 0

    backlog = service.list_local_backlog(now=now + timedelta(hours=2))

    assert len(backlog) == 1
    assert backlog[0].lead.id == lead.id
    assert backlog[0].request is None
    expired_request = database.get_approval_request(expired_request_id)
    assert expired_request is not None
    assert expired_request.status is ApprovalStatus.EXPIRED
    restored = database.get_lead(lead.id or 0)
    assert restored is not None
    assert restored.status is LeadStatus.CANDIDATE

    approved = service.decide_local_lead(
        lead.id or 0,
        ApprovalAction.APPROVE,
        now=now + timedelta(hours=2, minutes=1),
    )

    assert approved.request.id != expired_request_id
    assert approved.request.status is ApprovalStatus.APPROVED
    assert approved.lead.status is LeadStatus.APPROVED
    assert Counter(event.action for event in database.list_audit_events()) == Counter(
        {
            "approval.approved": 1,
            "approval.requested": 2,
            "approval.restored": 1,
            "approval.expired": 1,
        }
    )


def test_local_backlog_does_not_start_expiration_for_new_candidates(database: Database) -> None:
    lead = create_candidate(database)
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    service = LocalApprovalService(database, expiration_minutes=20)

    first = service.list_local_backlog(now=now)
    later = service.list_local_backlog(now=now + timedelta(days=30))

    assert [item.lead.id for item in first] == [lead.id]
    assert [item.lead.id for item in later] == [lead.id]
    assert database.list_pending_approval_reviews() == []
    persisted = database.get_lead(lead.id or 0)
    assert persisted is not None
    assert persisted.status is LeadStatus.CANDIDATE


def test_local_backlog_includes_exact_reposts_for_training(database: Database) -> None:
    first = create_candidate(database)
    second = create_candidate(
        database,
        external_post_id="333",
        group_id="second-fixture-group",
    )
    service = LocalApprovalService(database, expiration_minutes=20)

    backlog = service.list_local_backlog()

    assert {item.lead.id for item in backlog} == {first.id, second.id}


def test_candidates_without_drafts_do_not_enter_approval(database: Database) -> None:
    create_candidate(database, draft=None)
    service = LocalApprovalService(database, expiration_minutes=20)

    assert service.prepare_candidates(limit=10) == []
    assert database.list_pending_approval_reviews() == []

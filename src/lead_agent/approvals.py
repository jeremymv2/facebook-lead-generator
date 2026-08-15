"""Local, one-time human approval workflow with no Facebook posting capability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import ValidationError

from lead_agent.ai import DraftResponse
from lead_agent.database import Database
from lead_agent.models import (
    ApprovalRequest,
    ApprovalReview,
    ApprovalStatus,
    AuditEvent,
    FacebookPost,
    Lead,
    LeadStatus,
    RejectionReason,
    utc_now,
)


class ApprovalError(RuntimeError):
    """Base error for safe local approval failures."""


class ApprovalStateError(ApprovalError):
    """Raised when an approval is missing or no longer pending."""


class ApprovalExpiredError(ApprovalError):
    """Raised when a decision arrives after its review window."""


class ApprovalValidationError(ApprovalError):
    """Raised when an edited response fails the local business rules."""


class ApprovalAction(StrEnum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class LocalReviewItem:
    """A durable local backlog item, optionally backed by a pending remote request."""

    lead: Lead
    post: FacebookPost
    request: ApprovalRequest | None = None

    @property
    def draft_response(self) -> str:
        draft: str | None
        if self.request is not None:
            draft = self.request.draft_response
        else:
            draft = self.lead.drafted_response
        if draft is None:  # pragma: no cover - query contract
            raise RuntimeError("Local review item is missing its draft")
        return draft


class LocalApprovalService:
    """Prepare candidates and apply exactly one local human decision per request."""

    def __init__(
        self,
        database: Database,
        *,
        expiration_minutes: int,
        duplicate_window_hours: int = 72,
        classification_version: str | None = None,
    ) -> None:
        if expiration_minutes < 1:
            raise ValueError("expiration_minutes must be positive")
        if duplicate_window_hours < 1:
            raise ValueError("duplicate_window_hours must be positive")
        self.database = database
        self.expiration = timedelta(minutes=expiration_minutes)
        self.duplicate_window_hours = duplicate_window_hours
        self.classification_version = classification_version

    def prepare_candidates(
        self,
        *,
        limit: int,
        now: datetime | None = None,
    ) -> list[ApprovalReview]:
        timestamp = now or utc_now()
        self.expire_pending(now=timestamp)
        for lead in self.database.list_candidate_leads(
            limit=limit,
            duplicate_window_hours=self.duplicate_window_hours,
            classification_version=self.classification_version,
        ):
            self._prepare_lead(lead, requested_at=timestamp)
        return self.database.list_pending_approval_reviews()

    def list_local_backlog(
        self,
        *,
        limit: int | None = None,
        now: datetime | None = None,
    ) -> list[LocalReviewItem]:
        """Return unreviewed leads without starting their approval-expiration clocks."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        timestamp = now or utc_now()
        self.expire_pending(now=timestamp)
        restored = self.database.restore_expired_candidate_leads(
            restored_at=timestamp,
            classification_version=self.classification_version,
        )
        for lead in restored:
            self._record_restored_event(lead)

        pending = [
            review
            for review in self.database.list_pending_approval_reviews()
            if self.classification_version is None
            or review.lead.classification_version == self.classification_version
        ]
        remaining = None if limit is None else max(0, limit - len(pending))
        candidates = (
            self.database.list_candidate_leads(
                limit=remaining,
                duplicate_window_hours=self.duplicate_window_hours,
                classification_version=self.classification_version,
            )
            if remaining is None or remaining > 0
            else []
        )
        items = [
            LocalReviewItem(lead=review.lead, post=review.post, request=review.request)
            for review in pending
        ]
        for lead in candidates:
            post = self.database.get_post(lead.facebook_post_id)
            if post is None:  # pragma: no cover - protected by foreign key
                raise RuntimeError("Approval candidate is missing its Facebook post")
            items.append(LocalReviewItem(lead=lead, post=post))
        return sorted(
            items,
            key=lambda item: (
                -(item.lead.overall_score or 0),
                item.lead.created_at,
                item.lead.id or 0,
            ),
        )

    def decide_local_lead(
        self,
        lead_id: int,
        action: ApprovalAction,
        *,
        edited_response: str | None = None,
        rejection_reason: RejectionReason | str | None = None,
        now: datetime | None = None,
    ) -> ApprovalReview:
        """Create a fresh snapshot at click time and immediately apply the local decision."""
        timestamp = now or utc_now()
        self.expire_pending(now=timestamp)
        for restored in self.database.restore_expired_candidate_leads(
            restored_at=timestamp,
            classification_version=self.classification_version,
        ):
            self._record_restored_event(restored)

        lead = self.database.get_lead(lead_id)
        if lead is None:
            raise ApprovalStateError("Lead does not exist")
        if (
            self.classification_version is not None
            and lead.classification_version != self.classification_version
        ):
            raise ApprovalStateError("Lead is not on the current classifier version")

        if lead.status is LeadStatus.PENDING_APPROVAL:
            review = next(
                (
                    value
                    for value in self.database.list_pending_approval_reviews()
                    if value.lead.id == lead_id
                ),
                None,
            )
            if review is None:  # pragma: no cover - state invariant
                raise ApprovalStateError("Lead has no pending approval request")
        elif lead.status is LeadStatus.CANDIDATE:
            try:
                review = self._prepare_lead(lead, requested_at=timestamp)
            except (LookupError, ValueError) as error:
                raise ApprovalStateError("Lead is no longer awaiting review") from error
        else:
            raise ApprovalStateError("Lead has already been decided")

        request_id = review.request.id
        if request_id is None:  # pragma: no cover - persisted review contract
            raise RuntimeError("Approval request is missing its ID")
        return self.decide(
            request_id,
            action,
            edited_response=edited_response,
            rejection_reason=rejection_reason,
            now=timestamp,
        )

    def list_pending(self, *, now: datetime | None = None) -> list[ApprovalReview]:
        self.expire_pending(now=now)
        return self.database.list_pending_approval_reviews()

    def expire_pending(self, *, now: datetime | None = None) -> list[ApprovalReview]:
        expired = self.database.expire_approval_requests(expired_at=now or utc_now())
        for review in expired:
            self._record_event(review, action="approval.expired", result="expired")
        return expired

    def decide(
        self,
        request_id: int,
        action: ApprovalAction,
        *,
        edited_response: str | None = None,
        rejection_reason: RejectionReason | str | None = None,
        queue_posting: bool = False,
        now: datetime | None = None,
    ) -> ApprovalReview:
        timestamp = now or utc_now()
        self.expire_pending(now=timestamp)
        request = self.database.get_approval_request(request_id)
        if request is None:
            raise ApprovalStateError("Approval request does not exist")
        if request.status is ApprovalStatus.EXPIRED:
            raise ApprovalExpiredError("Approval request has expired; re-review is required")
        if request.status is not ApprovalStatus.PENDING:
            raise ApprovalStateError("Approval request has already been decided")

        selected_rejection_reason: RejectionReason | None = None
        if action is ApprovalAction.APPROVE:
            decision = ApprovalStatus.APPROVED
            response = self._validate_response(request.draft_response)
        elif action is ApprovalAction.EDIT:
            decision = ApprovalStatus.EDITED
            response = self._validate_response(edited_response or "")
        else:
            decision = ApprovalStatus.REJECTED
            response = None
            try:
                selected_rejection_reason = RejectionReason(rejection_reason or "")
            except ValueError as error:
                raise ApprovalValidationError("Select a valid rejection reason") from error

        review, changed = self.database.decide_approval_request(
            request_id,
            decision,
            decided_at=timestamp,
            edited_response=response,
            rejection_reason=(
                selected_rejection_reason if action is ApprovalAction.REJECT else None
            ),
            enqueue_posting=queue_posting,
        )
        if not changed:
            if review.request.status is ApprovalStatus.EXPIRED:
                self._record_event(review, action="approval.expired", result="expired")
                raise ApprovalExpiredError("Approval request has expired; re-review is required")
            raise ApprovalStateError("Approval request has already been decided")
        self._record_event(
            review,
            action=f"approval.{review.request.status.value}",
            result=review.request.status.value,
            details={
                "edited": action is ApprovalAction.EDIT,
                "posting_queued": queue_posting,
                "rejection_reason": (
                    review.request.rejection_reason.value
                    if review.request.rejection_reason is not None
                    else None
                ),
            },
        )
        return review

    @staticmethod
    def _validate_response(response: str) -> str:
        try:
            return DraftResponse(response=response).response
        except (ValidationError, ValueError) as error:
            raise ApprovalValidationError(
                "Response must satisfy the company identity, Licensed & Insured, free-estimate, "
                "website, text, and length rules"
            ) from error

    def _record_event(
        self,
        review: ApprovalReview,
        *,
        action: str,
        result: str,
        details: dict[str, object] | None = None,
    ) -> None:
        self.database.record_audit_event(
            AuditEvent(
                component="approval",
                action=action,
                result=result,
                lead_id=review.lead.id,
                post_id=review.post.id,
                group_id=review.post.group_id,
                details={
                    "approval_request_id": review.request.id or 0,
                    "lead_status": review.lead.status.value,
                    **(details or {}),
                },
            )
        )

    def _prepare_lead(self, lead: Lead, *, requested_at: datetime) -> ApprovalReview:
        if lead.id is None or lead.drafted_response is None:  # pragma: no cover - query contract
            raise RuntimeError("Approval candidate is missing its ID or draft")
        review = self.database.create_approval_request(
            ApprovalRequest(
                lead_id=lead.id,
                draft_response=lead.drafted_response,
                requested_at=requested_at,
                expires_at=requested_at + self.expiration,
            )
        )
        self._record_event(review, action="approval.requested", result="pending")
        return review

    def _record_restored_event(self, lead: Lead) -> None:
        post = self.database.get_post(lead.facebook_post_id)
        if post is None:  # pragma: no cover - protected by foreign key
            raise RuntimeError("Restored lead is missing its Facebook post")
        self.database.record_audit_event(
            AuditEvent(
                component="approval",
                action="approval.restored",
                result="candidate",
                lead_id=lead.id,
                post_id=post.id,
                group_id=post.group_id,
                details={"lead_status": lead.status.value},
            )
        )

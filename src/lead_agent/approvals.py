"""Local, one-time human approval workflow with no Facebook posting capability."""

from __future__ import annotations

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
            if (
                lead.id is None or lead.drafted_response is None
            ):  # pragma: no cover - query contract
                raise RuntimeError("Approval candidate is missing its ID or draft")
            review = self.database.create_approval_request(
                ApprovalRequest(
                    lead_id=lead.id,
                    draft_response=lead.drafted_response,
                    requested_at=timestamp,
                    expires_at=timestamp + self.expiration,
                )
            )
            self._record_event(review, action="approval.requested", result="pending")
        return self.database.list_pending_approval_reviews()

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
                "Response must satisfy the company identity, free-estimate, website, text, and "
                "length rules"
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

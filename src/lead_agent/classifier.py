"""Idempotent lead classification and drafting orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from lead_agent.ai import (
    CLASSIFICATION_VERSION,
    AIProvider,
    AIResponseError,
    ClassificationContext,
    LeadClassification,
)
from lead_agent.database import Database
from lead_agent.models import (
    AuditEvent,
    FacebookPost,
    Lead,
    LeadIntent,
    LeadStatus,
    PostStatus,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class ClassificationSummary:
    posts_considered: int
    candidates: tuple[Lead, ...]
    ignored: tuple[Lead, ...]

    @property
    def leads_created(self) -> int:
        return len(self.candidates) + len(self.ignored)


@dataclass(frozen=True, slots=True)
class ReclassificationChange:
    lead_id: int
    post_id: int
    prior_status: LeadStatus
    status: LeadStatus
    prior_version: str | None
    version: str
    service_category: str | None
    intent: LeadIntent
    overall_score: int


@dataclass(frozen=True, slots=True)
class ReclassificationSummary:
    leads_considered: int
    changes: tuple[ReclassificationChange, ...]


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    lead_id: int
    post_id: int
    current_status: LeadStatus
    replay_status: LeadStatus
    current_service: str | None
    replay_service: str | None
    current_intent: LeadIntent | None
    replay_intent: LeadIntent
    current_score: int | None
    replay_score: int
    changed: bool


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    leads_considered: int
    outcomes: tuple[ReplayOutcome, ...]

    @property
    def changed(self) -> int:
        return sum(outcome.changed for outcome in self.outcomes)


class LeadClassificationService:
    """Classify unprocessed posts without performing any Facebook action."""

    def __init__(
        self,
        database: Database,
        provider: AIProvider,
        context: ClassificationContext,
    ) -> None:
        self.database = database
        self.provider = provider
        self.context = context

    def classify_posts(
        self,
        *,
        limit: int,
        post_id: int | None = None,
    ) -> ClassificationSummary:
        posts = self.database.list_unclassified_posts(limit=limit, post_id=post_id)
        candidates: list[Lead] = []
        ignored: list[Lead] = []

        for post in posts:
            if post.id is None:  # pragma: no cover - persisted query contract
                raise RuntimeError("Persisted Facebook post is missing its database ID")
            try:
                lead_to_save, classification = self._classify_saved_post(post, draft=True)
            except Exception as error:
                safe_error = type(error).__name__
                self.database.update_post_status(
                    post.id,
                    PostStatus.FAILED,
                    error_state=safe_error,
                )
                self.database.record_audit_event(
                    AuditEvent(
                        component="classifier",
                        action="post.classify",
                        result="failed",
                        post_id=post.id,
                        group_id=post.group_id,
                        details={
                            "error": safe_error,
                            "provider": self.provider.provider_name,
                            "model": self.provider.model_name,
                        },
                    )
                )
                raise

            save_result = self.database.save_classified_lead(lead_to_save)
            if not save_result.created:
                continue
            lead = save_result.lead
            drafted_response = lead.drafted_response
            destination = candidates if lead.status is LeadStatus.CANDIDATE else ignored
            destination.append(lead)
            self.database.record_audit_event(
                AuditEvent(
                    component="classifier",
                    action="lead.classified",
                    result=lead.status.value,
                    lead_id=lead.id,
                    post_id=post.id,
                    group_id=post.group_id,
                    details={
                        "intent": classification.intent.value,
                        "service": classification.service_category,
                        "overall_score": classification.overall_score,
                        "provider": self.provider.provider_name,
                        "model": self.provider.model_name,
                        "drafted": drafted_response is not None,
                        "classification_version": CLASSIFICATION_VERSION,
                    },
                )
            )
            if drafted_response is not None:
                self.database.record_audit_event(
                    AuditEvent(
                        component="classifier",
                        action="response.drafted",
                        result="success",
                        lead_id=lead.id,
                        post_id=post.id,
                        group_id=post.group_id,
                        details={
                            "provider": self.provider.provider_name,
                            "model": self.provider.model_name,
                            "classification_version": CLASSIFICATION_VERSION,
                        },
                    )
                )

        return ClassificationSummary(
            posts_considered=len(posts),
            candidates=tuple(candidates),
            ignored=tuple(ignored),
        )

    def reclassify_leads(
        self,
        *,
        limit: int,
        lead_id: int | None = None,
    ) -> ReclassificationSummary:
        """Reclassify only unreviewed candidate/ignored leads; targeted IDs may be current."""
        items = self.database.list_classification_work_items(
            limit=limit,
            lead_id=lead_id,
            current_version=CLASSIFICATION_VERSION,
            reclassifiable_only=True,
        )
        changes: list[ReclassificationChange] = []
        for item in items:
            if item.lead.id is None or item.post.id is None:  # pragma: no cover - query contract
                raise RuntimeError("Reclassification work item is missing an ID")
            replacement, classification = self._classify_saved_post(
                item.post,
                draft=True,
                lead_id=item.lead.id,
                created_at=item.lead.created_at,
            )
            persisted = self.database.replace_unreviewed_classification(replacement)
            change = ReclassificationChange(
                lead_id=item.lead.id,
                post_id=item.post.id,
                prior_status=item.lead.status,
                status=persisted.status,
                prior_version=item.lead.classification_version,
                version=CLASSIFICATION_VERSION,
                service_category=classification.service_category,
                intent=classification.intent,
                overall_score=classification.overall_score,
            )
            changes.append(change)
            self.database.record_audit_event(
                AuditEvent(
                    component="classifier",
                    action="lead.reclassified",
                    result=persisted.status.value,
                    lead_id=persisted.id,
                    post_id=item.post.id,
                    group_id=item.post.group_id,
                    details={
                        "prior_status": item.lead.status.value,
                        "status": persisted.status.value,
                        "prior_version": item.lead.classification_version,
                        "classification_version": CLASSIFICATION_VERSION,
                        "intent": classification.intent.value,
                        "service": classification.service_category,
                        "overall_score": classification.overall_score,
                    },
                )
            )
        return ReclassificationSummary(leads_considered=len(items), changes=tuple(changes))

    def replay_history(
        self,
        *,
        limit: int,
        lead_id: int | None = None,
    ) -> ReplaySummary:
        """Evaluate saved posts with current rules without drafting or changing local state."""
        items = self.database.list_classification_work_items(limit=limit, lead_id=lead_id)
        outcomes: list[ReplayOutcome] = []
        for item in items:
            if item.lead.id is None or item.post.id is None:  # pragma: no cover - query contract
                raise RuntimeError("Replay work item is missing an ID")
            replay, classification = self._classify_saved_post(item.post, draft=False)
            changed = any(
                (
                    item.lead.status is not replay.status,
                    item.lead.service_category != replay.service_category,
                    item.lead.intent is not replay.intent,
                    item.lead.overall_score != replay.overall_score,
                )
            )
            outcomes.append(
                ReplayOutcome(
                    lead_id=item.lead.id,
                    post_id=item.post.id,
                    current_status=item.lead.status,
                    replay_status=replay.status,
                    current_service=item.lead.service_category,
                    replay_service=replay.service_category,
                    current_intent=item.lead.intent,
                    replay_intent=classification.intent,
                    current_score=item.lead.overall_score,
                    replay_score=classification.overall_score,
                    changed=changed,
                )
            )
        return ReplaySummary(leads_considered=len(items), outcomes=tuple(outcomes))

    def _classify_saved_post(
        self,
        post: FacebookPost,
        *,
        draft: bool,
        lead_id: int | None = None,
        created_at: datetime | None = None,
    ) -> tuple[Lead, LeadClassification]:
        if post.id is None:
            raise ValueError("Saved classification requires a persisted post")
        classification = self.provider.classify_post(post, self.context)
        self._validate_service_allowlist(classification)
        status = self._lead_status(classification)
        drafted_response = None
        if draft and status is LeadStatus.CANDIDATE:
            drafted_response = self.provider.draft_response(
                post,
                classification,
                self.context,
            ).response
        created = created_at or utc_now()
        return (
            Lead(
                id=lead_id,
                facebook_post_id=post.id,
                status=status,
                service_category=classification.service_category,
                location=classification.location,
                intent=classification.intent,
                is_residential=classification.is_residential,
                is_spam=classification.is_spam,
                relevance_score=classification.relevance_score,
                geographic_score=classification.geographic_score,
                urgency_score=classification.urgency_score,
                overall_score=classification.overall_score,
                confidence=classification.confidence,
                reasoning_summary=classification.reasoning_summary,
                drafted_response=drafted_response,
                ai_provider=self.provider.provider_name,
                ai_model=self.provider.model_name,
                classification_version=CLASSIFICATION_VERSION,
                created_at=created,
            ),
            classification,
        )

    def _validate_service_allowlist(self, classification: LeadClassification) -> None:
        if (
            classification.service_category is not None
            and classification.service_category not in self.context.enabled_services
        ):
            raise AIResponseError("Provider returned a service outside the enabled allowlist")

    def _lead_status(self, classification: LeadClassification) -> LeadStatus:
        eligible_intent = classification.intent in {
            LeadIntent.HIRING,
            LeadIntent.RECOMMENDATION,
        }
        is_candidate = (
            classification.overall_score >= self.context.lead_threshold
            and classification.service_category is not None
            and classification.is_residential
            and not classification.is_spam
            and eligible_intent
        )
        return LeadStatus.CANDIDATE if is_candidate else LeadStatus.IGNORED

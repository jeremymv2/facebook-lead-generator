"""Idempotent lead classification and drafting orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from lead_agent.ai import (
    CLASSIFICATION_VERSION,
    AIProvider,
    AIResponseError,
    ClassificationContext,
    LeadClassification,
)
from lead_agent.database import Database
from lead_agent.models import AuditEvent, Lead, LeadIntent, LeadStatus, PostStatus


@dataclass(frozen=True, slots=True)
class ClassificationSummary:
    posts_considered: int
    candidates: tuple[Lead, ...]
    ignored: tuple[Lead, ...]

    @property
    def leads_created(self) -> int:
        return len(self.candidates) + len(self.ignored)


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
                classification = self.provider.classify_post(post, self.context)
                self._validate_service_allowlist(classification)
                status = self._lead_status(classification)
                drafted_response = None
                if status is LeadStatus.CANDIDATE:
                    drafted_response = self.provider.draft_response(
                        post,
                        classification,
                        self.context,
                    ).response
                lead_to_save = Lead(
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
                )
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

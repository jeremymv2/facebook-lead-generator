from pathlib import Path

import pytest

from lead_agent.ai import (
    AIResponseError,
    ClassificationContext,
    DraftResponse,
    HeuristicAIProvider,
    LeadClassification,
)
from lead_agent.classifier import LeadClassificationService
from lead_agent.config import DEFAULT_SERVICES
from lead_agent.database import Database
from lead_agent.models import FacebookPost, LeadIntent, LeadStatus, PostStatus


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "classifier.sqlite3")
    database.initialize()
    return database


def context(*, threshold: int = 75) -> ClassificationContext:
    return ClassificationContext(
        service_area="Louisville, Kentucky",
        service_radius_miles=50,
        enabled_services=tuple(DEFAULT_SERVICES),
        lead_threshold=threshold,
        max_input_characters=5000,
    )


def save_post(database: Database, external_id: str, text: str) -> FacebookPost:
    return database.save_post(
        FacebookPost(
            external_post_id=external_id,
            group_id="fixture-group",
            group_name="Synthetic Fixture Group",
            author_name="Sarah Example",
            post_text=text,
        )
    ).post


def test_service_persists_candidate_and_ignored_results_with_audit_history(
    database: Database,
) -> None:
    strong = save_post(
        database,
        "strong",
        "Looking for someone in Louisville to repair and stain our deck this week.",
    )
    weak = save_post(
        database,
        "weak",
        "Does anyone know what stain color looks good on cedar?",
    )
    service = LeadClassificationService(database, HeuristicAIProvider(), context())

    summary = service.classify_posts(limit=10)

    assert summary.posts_considered == 2
    assert len(summary.candidates) == 1
    assert len(summary.ignored) == 1
    candidate = database.get_lead_for_post(strong.id or 0)
    ignored = database.get_lead_for_post(weak.id or 0)
    assert candidate is not None
    assert candidate.status is LeadStatus.CANDIDATE
    assert candidate.intent is LeadIntent.HIRING
    assert candidate.drafted_response is not None
    assert candidate.ai_provider == "heuristic"
    assert candidate.ai_model == "heuristic-v1"
    assert candidate.classification_version is not None
    assert ignored is not None
    assert ignored.status is LeadStatus.IGNORED
    assert ignored.drafted_response is None
    assert database.get_post(strong.id or 0).status is PostStatus.PROCESSED  # type: ignore[union-attr]
    assert database.get_post(weak.id or 0).status is PostStatus.PROCESSED  # type: ignore[union-attr]
    actions = [event.action for event in database.list_audit_events()]
    assert actions.count("lead.classified") == 2
    assert actions.count("response.drafted") == 1


@pytest.mark.parametrize(
    ("external_id", "text", "expected_service", "expected_intent"),
    [
        (
            "structural-crawl-space",
            "Need estimates for structural repairs in crawl space",
            "structural_repairs",
            LeadIntent.HIRING,
        ),
        (
            "investor-paint-flooring",
            "INVESTORS: Who are you using for interior paint jobs and flooring installation?",
            "flooring",
            LeadIntent.RECOMMENDATION,
        ),
    ],
)
def test_realistic_service_requests_become_candidates(
    database: Database,
    external_id: str,
    text: str,
    expected_service: str,
    expected_intent: LeadIntent,
) -> None:
    source = save_post(database, external_id, text)
    service = LeadClassificationService(database, HeuristicAIProvider(), context())

    summary = service.classify_posts(limit=1)

    lead = database.get_lead_for_post(source.id or 0)
    assert len(summary.candidates) == 1
    assert lead is not None
    assert lead.status is LeadStatus.CANDIDATE
    assert lead.service_category == expected_service
    assert lead.intent is expected_intent
    assert lead.drafted_response is not None


def test_classification_is_idempotent_and_does_not_redraft(database: Database) -> None:
    save_post(
        database,
        "strong",
        "Looking for someone in Louisville to repair and stain our deck this week.",
    )
    service = LeadClassificationService(database, HeuristicAIProvider(), context())

    first = service.classify_posts(limit=10)
    second = service.classify_posts(limit=10)

    assert first.leads_created == 1
    assert second.posts_considered == 0
    assert second.leads_created == 0
    assert len(database.list_audit_events()) == 2


def test_configured_threshold_controls_drafting(database: Database) -> None:
    source = save_post(
        database,
        "strong",
        "Looking for someone in Louisville to repair and stain our deck this week.",
    )
    service = LeadClassificationService(
        database,
        HeuristicAIProvider(),
        context(threshold=99),
    )

    summary = service.classify_posts(limit=10)

    lead = database.get_lead_for_post(source.id or 0)
    assert summary.candidates == ()
    assert len(summary.ignored) == 1
    assert lead is not None
    assert lead.status is LeadStatus.IGNORED
    assert lead.drafted_response is None


def test_resolved_yard_request_is_landscaping_but_never_drafted(database: Database) -> None:
    source = save_post(
        database,
        "resolved-yard-work",
        (
            "Need someone to mow two yards today, trim bushes, and remove weeds along the fence. "
            "Update: I have found someone."
        ),
    )
    service = LeadClassificationService(database, HeuristicAIProvider(), context())

    summary = service.classify_posts(limit=10)

    lead = database.get_lead_for_post(source.id or 0)
    assert summary.candidates == ()
    assert len(summary.ignored) == 1
    assert lead is not None
    assert lead.status is LeadStatus.IGNORED
    assert lead.service_category == "landscaping"
    assert lead.intent is LeadIntent.RESOLVED
    assert lead.overall_score is not None and lead.overall_score <= 10
    assert lead.drafted_response is None


def test_provider_failure_records_only_safe_error_type(database: Database) -> None:
    source = save_post(
        database,
        "failure",
        "Looking for someone in Louisville to repair our deck.",
    )

    class FailingProvider:
        provider_name = "fixture"
        model_name = "fixture-model"

        def classify_post(
            self,
            post: FacebookPost,
            context: ClassificationContext,
        ) -> LeadClassification:
            del post, context
            raise RuntimeError("raw response and customer post must stay out of state")

        def draft_response(
            self,
            post: FacebookPost,
            classification: LeadClassification,
            context: ClassificationContext,
        ) -> DraftResponse:
            del post, classification, context
            raise AssertionError("drafting must not run")

    service = LeadClassificationService(database, FailingProvider(), context())

    with pytest.raises(RuntimeError, match="raw response"):
        service.classify_posts(limit=10)

    failed_post = database.get_post(source.id or 0)
    assert failed_post is not None
    assert failed_post.status is PostStatus.FAILED
    assert failed_post.error_state == "RuntimeError"
    event = database.list_audit_events()[0]
    assert event.details["error"] == "RuntimeError"
    assert "raw response" not in str(event.details)


def test_service_rejects_provider_service_outside_allowlist(database: Database) -> None:
    save_post(database, "roof", "Need a roofer in Louisville.")

    class UnapprovedServiceProvider:
        provider_name = "fixture"
        model_name = "fixture-model"

        def classify_post(
            self,
            post: FacebookPost,
            context: ClassificationContext,
        ) -> LeadClassification:
            del post, context
            return LeadClassification(
                service_category="roofing",
                location="Louisville",
                intent=LeadIntent.HIRING,
                is_residential=True,
                is_spam=False,
                relevance_score=95,
                geographic_score=100,
                urgency_score=70,
                overall_score=90,
                confidence=0.95,
                reasoning_summary="Fixture classification outside the service allowlist.",
            )

        def draft_response(
            self,
            post: FacebookPost,
            classification: LeadClassification,
            context: ClassificationContext,
        ) -> DraftResponse:
            del post, classification, context
            raise AssertionError("drafting must not run")

    service = LeadClassificationService(database, UnapprovedServiceProvider(), context())

    with pytest.raises(AIResponseError, match="allowlist"):
        service.classify_posts(limit=10)

    assert database.get_lead_for_post(1) is None


def test_post_id_limits_classification_to_one_saved_post(database: Database) -> None:
    first = save_post(database, "first", "Need drywall repair in Louisville this week.")
    save_post(database, "second", "Need a deck repair in Louisville this week.")
    service = LeadClassificationService(database, HeuristicAIProvider(), context())

    summary = service.classify_posts(limit=10, post_id=first.id)

    assert summary.posts_considered == 1
    assert database.get_lead_for_post(first.id or 0) is not None
    assert len(database.list_unclassified_posts(limit=10)) == 1

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lead_agent.approvals import ApprovalAction, LocalApprovalService
from lead_agent.database import Database
from lead_agent.feedback import export_regression_fixtures, sanitize_fixture_text
from lead_agent.models import FacebookPost, Lead, LeadIntent, LeadStatus, RejectionReason

VALID_DRAFT = (
    "JJ Miller & Co. provides free estimates for carpentry. We'd be happy to help. "
    "Text me at 502-528-0858. https://jjmillerco.com"
)


def create_candidate(database: Database, *, external_id: str = "feedback") -> Lead:
    post = database.save_post(
        FacebookPost(
            external_post_id=external_id,
            group_id="fixture-group",
            group_name="Synthetic Group",
            author_name="Sarah Example",
            post_text=(
                "Sarah Example at Example Trim Carpentry LLC builds stairs. Contact "
                "sarah@example.com or 502-555-0199 at 123 Main Street. "
                "See https://example.com."
            ),
        )
    ).post
    return database.create_lead(
        Lead(
            facebook_post_id=post.id or 0,
            status=LeadStatus.CANDIDATE,
            service_category="carpentry",
            intent=LeadIntent.HIRING,
            overall_score=90,
            drafted_response=VALID_DRAFT,
        )
    )


def test_structured_feedback_exports_sanitized_regression_fixture(tmp_path: Path) -> None:
    database = Database(tmp_path / "feedback.sqlite3")
    database.initialize()
    lead = create_candidate(database)
    service = LocalApprovalService(database, expiration_minutes=20)
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)
    request_id = service.prepare_candidates(limit=10, now=now)[0].request.id or 0
    service.decide(
        request_id,
        ApprovalAction.REJECT,
        rejection_reason=RejectionReason.PROVIDER_ADVERTISEMENT,
        now=now + timedelta(minutes=1),
    )
    output = tmp_path / "private" / "regressions.json"

    summary = export_regression_fixtures(database, output, limit=10, lead_id=lead.id)

    fixtures = json.loads(output.read_text(encoding="utf-8"))
    assert summary.fixtures_exported == 1
    assert summary.feedback_skipped == 0
    assert output.stat().st_mode & 0o777 == 0o600
    assert fixtures[0]["expected"] == {
        "intent": "competitor_advertisement",
        "maximum_score": 10,
    }
    exported_text = fixtures[0]["text"]
    assert "[name]" in exported_text
    assert "[business]" in exported_text
    assert "[email]" in exported_text
    assert "[phone]" in exported_text
    assert "[address]" in exported_text
    assert "[url]" in exported_text
    assert "Sarah Example" not in exported_text
    assert "502-555-0199" not in exported_text


def test_non_classifier_rejection_reason_is_skipped(tmp_path: Path) -> None:
    database = Database(tmp_path / "skip.sqlite3")
    database.initialize()
    create_candidate(database)
    service = LocalApprovalService(database, expiration_minutes=20)
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)
    request_id = service.prepare_candidates(limit=10, now=now)[0].request.id or 0
    service.decide(
        request_id,
        ApprovalAction.REJECT,
        rejection_reason=RejectionReason.DUPLICATE_OR_REPOST,
        now=now + timedelta(minutes=1),
    )

    summary = export_regression_fixtures(database, tmp_path / "fixtures.json", limit=10)

    assert summary.feedback_considered == 1
    assert summary.fixtures_exported == 0
    assert summary.feedback_skipped == 1


def test_duplicate_rejection_feedback_exports_one_training_example(tmp_path: Path) -> None:
    database = Database(tmp_path / "deduplicated.sqlite3")
    database.initialize()
    first = create_candidate(database, external_id="first-copy")
    second = create_candidate(database, external_id="second-copy")
    service = LocalApprovalService(database, expiration_minutes=20)
    for lead in (first, second):
        service.decide_local_lead(
            lead.id or 0,
            ApprovalAction.REJECT,
            rejection_reason=RejectionReason.PROVIDER_ADVERTISEMENT,
        )

    output = tmp_path / "fixtures.json"
    summary = export_regression_fixtures(database, output, limit=10)

    fixtures = json.loads(output.read_text(encoding="utf-8"))
    assert summary.feedback_considered == 2
    assert summary.fixtures_exported == 1
    assert summary.feedback_skipped == 1
    assert len(fixtures) == 1


def test_sanitizer_normalizes_without_direct_identifiers() -> None:
    sanitized = sanitize_fixture_text(
        "At Community Garage Door Services call (502)501-5060 or name@example.com. "
        "Visit www.example.com today."
    )

    assert sanitized == "[business] call [phone] or [email]. Visit [url] today."


def test_sanitizer_redacts_named_providers_in_calls_to_action() -> None:
    sanitized = sanitize_fixture_text(
        "Message Example\u2019s Touch Services today. "
        "Contact Sample Home Experts for an estimate. "
        "A big thank you to Fixture Hall for choosing us. "
        "#SampleHomeExperts #Example\u2019sLawnCareService"
    )

    assert sanitized == (
        "Message [business] today. Contact [business] for an estimate. "
        "A big thank you to [business] for choosing us. [business] [business]"
    )

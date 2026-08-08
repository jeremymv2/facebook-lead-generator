import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lead_agent.database import SCHEMA_VERSION, Database
from lead_agent.models import AuditEvent, FacebookPost, Lead, LeadStatus, PostStatus


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "runtime" / "lead-agent.sqlite3")
    database.initialize()
    return database


def make_post(**overrides: object) -> FacebookPost:
    values: dict[str, object] = {
        "external_post_id": "post-123",
        "post_url": "https://www.facebook.com/groups/123/posts/post-123",
        "group_id": "group-123",
        "group_name": "Louisville Homeowners",
        "author_name": "Sarah",
        "post_text": "Looking for someone to repair and stain our deck.",
    }
    values.update(overrides)
    return FacebookPost(**values)  # type: ignore[arg-type]


def test_initialize_is_idempotent_and_records_schema_version(database: Database) -> None:
    database.initialize()

    with database.connection() as connection:
        row = connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()

    assert row is not None
    assert row["value"] == str(SCHEMA_VERSION)
    assert database.path.stat().st_mode & 0o777 == 0o600


def test_post_discovery_is_deduplicated(database: Database) -> None:
    first = database.save_post(make_post())
    duplicate = database.save_post(
        make_post(post_url="https://facebook.com/different-link", post_text="Changed rendering")
    )

    assert first.created is True
    assert duplicate.created is False
    assert duplicate.post.id == first.post.id
    assert duplicate.post.post_text == first.post.post_text
    assert len(database.list_posts()) == 1


def test_content_fallback_deduplicates_posts_without_ids_or_urls(database: Database) -> None:
    first = database.save_post(
        make_post(external_post_id=None, post_url=None, post_text="Need a fence estimate.")
    )
    second = database.save_post(
        make_post(
            external_post_id=None,
            post_url=None,
            post_text=" Need  a fence estimate. ",
        )
    )

    assert first.created is True
    assert second.created is False
    assert second.post.id == first.post.id


def test_saved_post_survives_new_database_instance(database: Database) -> None:
    saved = database.save_post(make_post()).post

    reopened = Database(database.path)
    loaded = reopened.get_post(saved.id or 0)

    assert loaded == saved


def test_post_status_updates_are_durable(database: Database) -> None:
    post = database.save_post(make_post()).post

    updated = database.update_post_status(
        post.id or 0,
        PostStatus.NEEDS_ATTENTION,
        error_state="Facebook checkpoint detected",
    )

    assert updated.status is PostStatus.NEEDS_ATTENTION
    assert updated.error_state == "Facebook checkpoint detected"
    assert database.get_post(post.id or 0) == updated


def test_updating_missing_post_raises(database: Database) -> None:
    with pytest.raises(LookupError, match="does not exist"):
        database.update_post_status(404, PostStatus.FAILED)


def test_one_lead_is_allowed_per_post(database: Database) -> None:
    post = database.save_post(make_post()).post
    lead = database.create_lead(
        Lead(
            facebook_post_id=post.id or 0,
            status=LeadStatus.CANDIDATE,
            service_category="decks",
            location="Louisville",
            relevance_score=98,
            geographic_score=100,
            urgency_score=85,
            overall_score=96,
            confidence=0.97,
            reasoning_summary="Local customer actively seeking deck repair.",
        )
    )

    assert lead.id is not None
    assert database.get_lead(lead.id) == lead
    assert database.get_lead_for_post(post.id or 0) == lead

    with pytest.raises(sqlite3.IntegrityError):
        database.create_lead(Lead(facebook_post_id=post.id or 0))


def test_lead_status_updates_are_durable(database: Database) -> None:
    post = database.save_post(make_post()).post
    lead = database.create_lead(Lead(facebook_post_id=post.id or 0))

    updated = database.update_lead_status(
        lead.id or 0,
        LeadStatus.NEEDS_ATTENTION,
        error_state="Manual review required",
    )

    assert updated.status is LeadStatus.NEEDS_ATTENTION
    assert updated.error_state == "Manual review required"
    assert updated.updated_at >= lead.updated_at


def test_updating_missing_lead_raises(database: Database) -> None:
    with pytest.raises(LookupError, match="does not exist"):
        database.update_lead_status(404, LeadStatus.FAILED)


def test_audit_events_append_structured_details(database: Database) -> None:
    post = database.save_post(make_post()).post
    lead = database.create_lead(Lead(facebook_post_id=post.id or 0))
    recorded = database.record_audit_event(
        AuditEvent(
            component="classifier",
            action="lead.scored",
            result="candidate",
            lead_id=lead.id,
            post_id=post.id,
            group_id=post.group_id,
            details={"overall_score": 96, "service": "decks"},
        )
    )

    all_events = database.list_audit_events()
    lead_events = database.list_audit_events(lead_id=lead.id)

    assert recorded.id is not None
    assert all_events == lead_events
    assert all_events[0].details == {"overall_score": 96, "service": "decks"}


def test_list_posts_rejects_non_positive_limit(database: Database) -> None:
    with pytest.raises(ValueError, match="positive"):
        database.list_posts(limit=0)


def test_group_scan_state_preserves_success_across_failure(database: Database) -> None:
    success_at = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    failure_at = datetime(2026, 8, 7, 12, 5, tzinfo=UTC)
    database.record_group_scan_success(
        group_id="group-123",
        group_name="Louisville Homeowners",
        group_url="https://www.facebook.com/groups/123",
        posts_seen=8,
        posts_new=3,
        last_known_post_identity="facebook-id:123",
        occurred_at=success_at,
    )

    failed = database.record_group_scan_failure(
        group_id="group-123",
        group_name="Louisville Homeowners",
        group_url="https://www.facebook.com/groups/123",
        error="FacebookSafetyStop:checkpoint",
        occurred_at=failure_at,
    )

    assert failed.last_attempt_at == failure_at
    assert failed.last_success_at == success_at
    assert failed.last_known_post_identity == "facebook-id:123"
    assert failed.last_error == "FacebookSafetyStop:checkpoint"
    assert failed.posts_seen == 0
    assert failed.posts_new == 0


def test_empty_success_preserves_last_known_post(database: Database) -> None:
    database.record_group_scan_success(
        group_id="group-123",
        group_name="Louisville Homeowners",
        group_url="https://www.facebook.com/groups/123",
        posts_seen=1,
        posts_new=1,
        last_known_post_identity="facebook-id:123",
    )

    state = database.record_group_scan_success(
        group_id="group-123",
        group_name="Louisville Homeowners",
        group_url="https://www.facebook.com/groups/123",
        posts_seen=0,
        posts_new=0,
        last_known_post_identity=None,
    )

    assert state.last_known_post_identity == "facebook-id:123"
    assert state.last_error is None

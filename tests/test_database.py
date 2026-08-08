import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lead_agent.database import SCHEMA_VERSION, Database
from lead_agent.models import AuditEvent, FacebookPost, Lead, LeadIntent, LeadStatus, PostStatus


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


def test_permalink_hydration_enriches_one_content_discovery(database: Database) -> None:
    first = database.save_post(
        make_post(
            external_post_id=None,
            post_url=None,
            author_name=None,
            post_text="Need a fence estimate in Louisville.",
        )
    )
    hydrated = database.save_post(
        make_post(
            external_post_id="post-456",
            post_url="https://www.facebook.com/groups/group-123/posts/post-456",
            author_name="Fixture Account",
            post_text="Need a fence estimate in Louisville.",
        )
    )

    assert first.created is True
    assert hydrated.created is False
    assert hydrated.post.id == first.post.id
    assert hydrated.post.identity_key == first.post.identity_key
    assert hydrated.post.external_post_id == "post-456"
    assert hydrated.post.post_url == ("https://www.facebook.com/groups/group-123/posts/post-456")
    assert hydrated.post.author_name == "Fixture Account"
    assert len(database.list_posts()) == 1

    distinct_stable_post = database.save_post(
        make_post(
            external_post_id="post-999",
            post_url="https://www.facebook.com/groups/group-123/posts/post-999",
            author_name="Another Fixture Account",
            post_text="Need a fence estimate in Louisville.",
        )
    )
    assert distinct_stable_post.created is True
    assert len(database.list_posts()) == 2


def test_content_only_rendering_matches_one_stable_post(database: Database) -> None:
    stable = database.save_post(
        make_post(post_text="Looking for a painter in Louisville next week.")
    )
    content_only = database.save_post(
        make_post(
            external_post_id=None,
            post_url=None,
            author_name=None,
            post_text="Looking for a painter in Louisville next week.",
        )
    )

    assert content_only.created is False
    assert content_only.post.id == stable.post.id


def test_distinct_stable_ids_are_not_merged_only_because_text_matches(
    database: Database,
) -> None:
    first = database.save_post(make_post(post_text="Synthetic repeated group announcement."))
    second = database.save_post(
        make_post(
            external_post_id="post-456",
            post_url="https://www.facebook.com/groups/123/posts/post-456",
            post_text="Synthetic repeated group announcement.",
        )
    )

    assert first.created is True
    assert second.created is True
    assert second.post.id != first.post.id
    assert len(database.list_posts()) == 2


def test_content_aliases_never_cross_group_boundaries(database: Database) -> None:
    first = database.save_post(
        make_post(
            external_post_id=None,
            post_url=None,
            author_name=None,
            post_text="Need a deck repair estimate.",
        )
    )
    second = database.save_post(
        make_post(
            external_post_id=None,
            post_url=None,
            group_id="another-group",
            group_name="Another Group",
            author_name=None,
            post_text="Need a deck repair estimate.",
        )
    )

    assert first.created is True
    assert second.created is True


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
            intent=LeadIntent.HIRING,
            is_residential=True,
            is_spam=False,
            relevance_score=98,
            geographic_score=100,
            urgency_score=85,
            overall_score=96,
            confidence=0.97,
            reasoning_summary="Local customer actively seeking deck repair.",
            ai_provider="heuristic",
            ai_model="heuristic-v1",
            classification_version="fixture-v1",
        )
    )

    assert lead.id is not None
    assert database.get_lead(lead.id) == lead
    assert database.get_lead_for_post(post.id or 0) == lead
    assert lead.intent is LeadIntent.HIRING
    assert lead.is_residential is True
    assert lead.is_spam is False

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


def test_save_classified_lead_is_atomic_and_idempotent(database: Database) -> None:
    first_post = database.save_post(make_post()).post
    second_post = database.save_post(
        make_post(
            external_post_id="post-456",
            post_url="https://www.facebook.com/groups/123/posts/post-456",
        )
    ).post
    assert [post.id for post in database.list_unclassified_posts(limit=10)] == [
        second_post.id,
        first_post.id,
    ]
    lead = Lead(
        facebook_post_id=first_post.id or 0,
        status=LeadStatus.CANDIDATE,
        intent=LeadIntent.HIRING,
        is_residential=True,
        is_spam=False,
        overall_score=90,
    )

    saved = database.save_classified_lead(lead)
    duplicate = database.save_classified_lead(lead)

    assert saved.created is True
    assert duplicate.created is False
    assert duplicate.lead == saved.lead
    persisted_post = database.get_post(first_post.id or 0)
    assert persisted_post is not None
    assert persisted_post.status is PostStatus.PROCESSED
    assert database.list_unclassified_posts(limit=10) == [second_post]
    assert database.list_unclassified_posts(limit=10, post_id=first_post.id) == []


def test_list_unclassified_posts_rejects_non_positive_limit(database: Database) -> None:
    with pytest.raises(ValueError, match="positive"):
        database.list_unclassified_posts(limit=0)


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
    assert failed.consecutive_failures == 1
    assert failed.last_failure_at == failure_at


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


def test_group_scan_failure_streak_resets_after_recovery(database: Database) -> None:
    first_failure = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    second_failure = datetime(2026, 8, 7, 12, 5, tzinfo=UTC)
    recovery = datetime(2026, 8, 7, 12, 10, tzinfo=UTC)
    for occurred_at in (first_failure, second_failure):
        failed = database.record_group_scan_failure(
            group_id="group-123",
            group_name="Louisville Homeowners",
            group_url="https://www.facebook.com/groups/123",
            error="FacebookBrowserError",
            occurred_at=occurred_at,
        )

    assert failed.consecutive_failures == 2
    assert failed.last_failure_at == second_failure

    recovered = database.record_group_scan_success(
        group_id="group-123",
        group_name="Louisville Homeowners",
        group_url="https://www.facebook.com/groups/123",
        posts_seen=10,
        posts_new=0,
        last_known_post_identity="facebook-id:123",
        occurred_at=recovery,
    )

    assert recovered.consecutive_failures == 0
    assert recovered.last_failure_at == second_failure
    assert recovered.last_error is None
    assert database.list_group_scan_states() == [recovered]


def test_initialize_migrates_v2_state_and_backfills_post_aliases(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    legacy_post = make_post(
        external_post_id=None,
        post_url=None,
        author_name=None,
        post_text="Need a synthetic fence estimate.",
    )
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE facebook_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_key TEXT NOT NULL UNIQUE,
                external_post_id TEXT,
                post_url TEXT,
                group_id TEXT NOT NULL,
                group_name TEXT NOT NULL,
                author_name TEXT,
                post_text TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                discovered_at TEXT NOT NULL,
                posted_at TEXT,
                status TEXT NOT NULL,
                error_state TEXT,
                screenshot_path TEXT
            );
            CREATE TABLE group_scan_state (
                group_id TEXT PRIMARY KEY,
                group_name TEXT NOT NULL,
                group_url TEXT NOT NULL,
                last_attempt_at TEXT NOT NULL,
                last_success_at TEXT,
                last_known_post_identity TEXT,
                last_error TEXT,
                posts_seen INTEGER NOT NULL DEFAULT 0 CHECK(posts_seen >= 0),
                posts_new INTEGER NOT NULL DEFAULT 0 CHECK(posts_new >= 0)
            );
            CREATE TABLE leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                facebook_post_id INTEGER NOT NULL UNIQUE,
                status TEXT NOT NULL,
                service_category TEXT,
                location TEXT,
                relevance_score INTEGER CHECK(relevance_score BETWEEN 0 AND 100),
                geographic_score INTEGER CHECK(geographic_score BETWEEN 0 AND 100),
                urgency_score INTEGER CHECK(urgency_score BETWEEN 0 AND 100),
                overall_score INTEGER CHECK(overall_score BETWEEN 0 AND 100),
                confidence REAL CHECK(confidence BETWEEN 0 AND 1),
                reasoning_summary TEXT,
                drafted_response TEXT,
                approved_response TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                approval_timestamp TEXT,
                approval_expires_at TEXT,
                posting_timestamp TEXT,
                facebook_reply_url TEXT,
                error_state TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0 CHECK(retry_count >= 0),
                screenshot_path TEXT,
                FOREIGN KEY(facebook_post_id) REFERENCES facebook_posts(id) ON DELETE RESTRICT
            );
            """
        )
        connection.execute(
            """
            INSERT INTO facebook_posts (
                identity_key, group_id, group_name, post_text, text_hash,
                discovered_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                legacy_post.identity_key,
                legacy_post.group_id,
                legacy_post.group_name,
                legacy_post.post_text,
                legacy_post.text_hash,
                legacy_post.discovered_at.isoformat(),
                legacy_post.status.value,
            ),
        )
        connection.execute(
            """
            INSERT INTO leads (
                facebook_post_id, status, created_at, updated_at, retry_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                1,
                LeadStatus.IGNORED.value,
                legacy_post.discovered_at.isoformat(),
                legacy_post.discovered_at.isoformat(),
                0,
            ),
        )
        connection.execute(
            """
            INSERT INTO group_scan_state (
                group_id, group_name, group_url, last_attempt_at, posts_seen, posts_new
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "group-123",
                "Louisville Homeowners",
                "https://www.facebook.com/groups/123",
                datetime(2026, 8, 7, 12, 0, tzinfo=UTC).isoformat(),
                1,
                1,
            ),
        )

    migrated = Database(path)
    migrated.initialize()
    state = migrated.get_group_scan_state("group-123")
    legacy_lead = migrated.get_lead_for_post(1)
    hydrated = migrated.save_post(
        make_post(
            external_post_id="post-789",
            post_url="https://www.facebook.com/groups/123/posts/post-789",
            author_name="Fixture Account",
            post_text=legacy_post.post_text,
        )
    )

    assert state is not None
    assert state.consecutive_failures == 0
    assert state.last_failure_at is None
    assert legacy_lead is not None
    assert legacy_lead.intent is None
    assert legacy_lead.is_residential is None
    assert legacy_lead.is_spam is None
    assert legacy_lead.ai_provider is None
    assert hydrated.created is False
    assert len(migrated.list_posts()) == 1
    with migrated.connection() as connection:
        version = connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()
    assert version is not None
    assert version["value"] == str(SCHEMA_VERSION)
    with migrated.connection() as connection:
        approval_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'approval_requests'"
        ).fetchone()
    assert approval_table is not None
    with migrated.connection() as connection:
        approval_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(approval_requests)")
        }
        notification_table = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'approval_notifications'
            """
        ).fetchone()
    assert "remote_token_hash" in approval_columns
    assert notification_table is not None

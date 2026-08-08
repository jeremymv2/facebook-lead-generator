"""SQLite persistence with durable uniqueness constraints and audit history."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lead_agent.models import (
    AuditEvent,
    FacebookPost,
    GroupScanState,
    Lead,
    LeadStatus,
    PostStatus,
    utc_now,
)

SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class SaveResult:
    """The persisted post and whether this call inserted it."""

    post: FacebookPost
    created: bool


class Database:
    """Small repository layer that opens a fresh SQLite connection per operation."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create runtime directories and apply the initial idempotent schema."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS facebook_posts (
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

                CREATE INDEX IF NOT EXISTS idx_facebook_posts_group_discovered
                    ON facebook_posts(group_id, discovered_at DESC);
                CREATE INDEX IF NOT EXISTS idx_facebook_posts_text_hash
                    ON facebook_posts(text_hash);

                CREATE TABLE IF NOT EXISTS facebook_post_identity_aliases (
                    identity_key TEXT NOT NULL,
                    facebook_post_id INTEGER NOT NULL,
                    identity_kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(identity_key, facebook_post_id),
                    FOREIGN KEY(facebook_post_id) REFERENCES facebook_posts(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_facebook_post_aliases_post
                    ON facebook_post_identity_aliases(facebook_post_id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_facebook_post_aliases_stable
                    ON facebook_post_identity_aliases(identity_key)
                    WHERE identity_kind IN ('facebook_id', 'facebook_url');

                CREATE TABLE IF NOT EXISTS leads (
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

                CREATE INDEX IF NOT EXISTS idx_leads_status_score
                    ON leads(status, overall_score DESC);

                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    component TEXT NOT NULL,
                    action TEXT NOT NULL,
                    result TEXT NOT NULL,
                    lead_id INTEGER,
                    post_id INTEGER,
                    group_id TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(lead_id) REFERENCES leads(id) ON DELETE SET NULL,
                    FOREIGN KEY(post_id) REFERENCES facebook_posts(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_audit_events_occurred
                    ON audit_events(occurred_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_events_lead
                    ON audit_events(lead_id, occurred_at DESC);

                CREATE TABLE IF NOT EXISTS group_scan_state (
                    group_id TEXT PRIMARY KEY,
                    group_name TEXT NOT NULL,
                    group_url TEXT NOT NULL,
                    last_attempt_at TEXT NOT NULL,
                    last_success_at TEXT,
                    last_known_post_identity TEXT,
                    last_error TEXT,
                    posts_seen INTEGER NOT NULL DEFAULT 0 CHECK(posts_seen >= 0),
                    posts_new INTEGER NOT NULL DEFAULT 0 CHECK(posts_new >= 0),
                    consecutive_failures INTEGER NOT NULL DEFAULT 0
                        CHECK(consecutive_failures >= 0),
                    last_failure_at TEXT
                );
                """
            )
            _add_column_if_missing(
                connection,
                "group_scan_state",
                "consecutive_failures",
                "INTEGER NOT NULL DEFAULT 0 CHECK(consecutive_failures >= 0)",
            )
            _add_column_if_missing(
                connection,
                "group_scan_state",
                "last_failure_at",
                "TEXT",
            )
            self._backfill_post_identity_aliases(connection)
            connection.execute(
                """
                INSERT INTO schema_metadata(key, value) VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )
        self.path.chmod(0o600)

    def _backfill_post_identity_aliases(self, connection: sqlite3.Connection) -> None:
        """Populate schema-v3 aliases for posts written by earlier schema versions."""
        rows = connection.execute("SELECT * FROM facebook_posts ORDER BY id").fetchall()
        for row in rows:
            persisted = _post_from_row(row)
            if persisted.id is None:  # pragma: no cover - persisted rows always have IDs
                raise RuntimeError("Saved post is missing its database ID")
            _insert_identity_aliases(
                connection,
                post_id=persisted.id,
                aliases=persisted.identity_aliases(),
                created_at=persisted.discovered_at,
            )

    def save_post(self, post: FacebookPost) -> SaveResult:
        """Insert a post once across IDs, URLs, and unambiguous content aliases."""
        with self.connection() as connection:
            # Serialize the alias lookup and insert across scanner processes.
            connection.execute("BEGIN IMMEDIATE")
            aliases = post.identity_aliases()
            existing_post_id = _find_duplicate_post_id(connection, post, aliases)
            created = existing_post_id is None
            if existing_post_id is None:
                cursor = connection.execute(
                    """
                    INSERT INTO facebook_posts (
                        identity_key, external_post_id, post_url, group_id, group_name,
                        author_name, post_text, text_hash, discovered_at, posted_at,
                        status, error_state, screenshot_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        post.identity_key,
                        post.external_post_id,
                        post.post_url,
                        post.group_id,
                        post.group_name,
                        post.author_name,
                        post.post_text,
                        post.text_hash,
                        _serialize_datetime(post.discovered_at),
                        _serialize_datetime(post.posted_at),
                        post.status.value,
                        post.error_state,
                        post.screenshot_path,
                    ),
                )
                inserted_post_id = cursor.lastrowid
                if inserted_post_id is None:  # pragma: no cover - SQLite insert contract
                    raise RuntimeError("Failed to retrieve inserted post ID")
                existing_post_id = inserted_post_id
            else:
                connection.execute(
                    """
                    UPDATE facebook_posts SET
                        external_post_id = COALESCE(external_post_id, ?),
                        post_url = COALESCE(post_url, ?),
                        author_name = COALESCE(author_name, ?),
                        posted_at = COALESCE(posted_at, ?)
                    WHERE id = ?
                    """,
                    (
                        post.external_post_id,
                        post.post_url,
                        post.author_name,
                        _serialize_datetime(post.posted_at),
                        existing_post_id,
                    ),
                )
            _insert_identity_aliases(
                connection,
                post_id=existing_post_id,
                aliases=aliases,
                created_at=post.discovered_at,
            )
            row = connection.execute(
                "SELECT * FROM facebook_posts WHERE id = ?", (existing_post_id,)
            ).fetchone()
            if row is None:  # pragma: no cover - protected by the insert/select transaction
                raise RuntimeError("Failed to retrieve saved post")
            return SaveResult(post=_post_from_row(row), created=created)

    def get_post(self, post_id: int) -> FacebookPost | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM facebook_posts WHERE id = ?", (post_id,)
            ).fetchone()
        return _post_from_row(row) if row is not None else None

    def list_posts(self, *, limit: int = 100) -> list[FacebookPost]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM facebook_posts ORDER BY discovered_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_post_from_row(row) for row in rows]

    def update_post_status(
        self,
        post_id: int,
        status: PostStatus,
        *,
        error_state: str | None = None,
    ) -> FacebookPost:
        """Durably update the processing state of a discovered post."""
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE facebook_posts SET status = ?, error_state = ? WHERE id = ?",
                (status.value, error_state, post_id),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"Post {post_id} does not exist")
            row = connection.execute(
                "SELECT * FROM facebook_posts WHERE id = ?", (post_id,)
            ).fetchone()
            if row is None:  # pragma: no cover - protected by rowcount check and transaction
                raise RuntimeError("Failed to retrieve updated post")
            return _post_from_row(row)

    def record_group_scan_success(
        self,
        *,
        group_id: str,
        group_name: str,
        group_url: str,
        posts_seen: int,
        posts_new: int,
        last_known_post_identity: str | None,
        occurred_at: datetime | None = None,
    ) -> GroupScanState:
        """Record a successful group scan while preserving a durable last-known post."""
        timestamp = occurred_at or utc_now()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO group_scan_state (
                    group_id, group_name, group_url, last_attempt_at, last_success_at,
                    last_known_post_identity, last_error, posts_seen, posts_new,
                    consecutive_failures
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, 0)
                ON CONFLICT(group_id) DO UPDATE SET
                    group_name = excluded.group_name,
                    group_url = excluded.group_url,
                    last_attempt_at = excluded.last_attempt_at,
                    last_success_at = excluded.last_success_at,
                    last_known_post_identity = COALESCE(
                        excluded.last_known_post_identity,
                        group_scan_state.last_known_post_identity
                    ),
                    last_error = NULL,
                    posts_seen = excluded.posts_seen,
                    posts_new = excluded.posts_new,
                    consecutive_failures = 0
                """,
                (
                    group_id,
                    group_name,
                    group_url,
                    _serialize_datetime(timestamp),
                    _serialize_datetime(timestamp),
                    last_known_post_identity,
                    posts_seen,
                    posts_new,
                ),
            )
        state = self.get_group_scan_state(group_id)
        if state is None:  # pragma: no cover - protected by the upsert
            raise RuntimeError("Failed to retrieve successful group scan state")
        return state

    def record_group_scan_failure(
        self,
        *,
        group_id: str,
        group_name: str,
        group_url: str,
        error: str,
        occurred_at: datetime | None = None,
    ) -> GroupScanState:
        """Record a failed scan without erasing the last successful scan metadata."""
        timestamp = occurred_at or utc_now()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO group_scan_state (
                    group_id, group_name, group_url, last_attempt_at, last_error,
                    consecutive_failures, last_failure_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(group_id) DO UPDATE SET
                    group_name = excluded.group_name,
                    group_url = excluded.group_url,
                    last_attempt_at = excluded.last_attempt_at,
                    last_error = excluded.last_error,
                    posts_seen = 0,
                    posts_new = 0,
                    consecutive_failures = group_scan_state.consecutive_failures + 1,
                    last_failure_at = excluded.last_failure_at
                """,
                (
                    group_id,
                    group_name,
                    group_url,
                    _serialize_datetime(timestamp),
                    error,
                    _serialize_datetime(timestamp),
                ),
            )
        state = self.get_group_scan_state(group_id)
        if state is None:  # pragma: no cover - protected by the upsert
            raise RuntimeError("Failed to retrieve failed group scan state")
        return state

    def get_group_scan_state(self, group_id: str) -> GroupScanState | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM group_scan_state WHERE group_id = ?", (group_id,)
            ).fetchone()
        return _group_scan_state_from_row(row) if row is not None else None

    def list_group_scan_states(self) -> list[GroupScanState]:
        """Return persisted group health without exposing discovered post content."""
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM group_scan_state ORDER BY group_name, group_id"
            ).fetchall()
        return [_group_scan_state_from_row(row) for row in rows]

    def create_lead(self, lead: Lead) -> Lead:
        """Create at most one lead per Facebook post."""
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO leads (
                    facebook_post_id, status, service_category, location,
                    relevance_score, geographic_score, urgency_score, overall_score,
                    confidence, reasoning_summary, drafted_response, approved_response,
                    created_at, updated_at, approval_timestamp, approval_expires_at,
                    posting_timestamp, facebook_reply_url, error_state, retry_count,
                    screenshot_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lead.facebook_post_id,
                    lead.status.value,
                    lead.service_category,
                    lead.location,
                    lead.relevance_score,
                    lead.geographic_score,
                    lead.urgency_score,
                    lead.overall_score,
                    lead.confidence,
                    lead.reasoning_summary,
                    lead.drafted_response,
                    lead.approved_response,
                    _serialize_datetime(lead.created_at),
                    _serialize_datetime(lead.updated_at),
                    _serialize_datetime(lead.approval_timestamp),
                    _serialize_datetime(lead.approval_expires_at),
                    _serialize_datetime(lead.posting_timestamp),
                    lead.facebook_reply_url,
                    lead.error_state,
                    lead.retry_count,
                    lead.screenshot_path,
                ),
            )
            row = connection.execute(
                "SELECT * FROM leads WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            if row is None:  # pragma: no cover - protected by the transaction
                raise RuntimeError("Failed to retrieve created lead")
            return _lead_from_row(row)

    def get_lead(self, lead_id: int) -> Lead | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
        return _lead_from_row(row) if row is not None else None

    def get_lead_for_post(self, post_id: int) -> Lead | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM leads WHERE facebook_post_id = ?", (post_id,)
            ).fetchone()
        return _lead_from_row(row) if row is not None else None

    def update_lead_status(
        self,
        lead_id: int,
        status: LeadStatus,
        *,
        error_state: str | None = None,
    ) -> Lead:
        """Durably update workflow state without replacing the rest of the lead."""
        updated_at = utc_now()
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE leads SET status = ?, error_state = ?, updated_at = ? WHERE id = ?",
                (status.value, error_state, _serialize_datetime(updated_at), lead_id),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"Lead {lead_id} does not exist")
            row = connection.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
            if row is None:  # pragma: no cover - protected by rowcount check and transaction
                raise RuntimeError("Failed to retrieve updated lead")
            return _lead_from_row(row)

    def record_audit_event(self, event: AuditEvent) -> AuditEvent:
        """Append an immutable structured event to the audit trail."""
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO audit_events (
                    occurred_at, component, action, result, lead_id, post_id, group_id, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _serialize_datetime(event.occurred_at),
                    event.component,
                    event.action,
                    event.result,
                    event.lead_id,
                    event.post_id,
                    event.group_id,
                    json.dumps(event.details, sort_keys=True, separators=(",", ":")),
                ),
            )
            return AuditEvent(
                id=cursor.lastrowid,
                occurred_at=event.occurred_at,
                component=event.component,
                action=event.action,
                result=event.result,
                lead_id=event.lead_id,
                post_id=event.post_id,
                group_id=event.group_id,
                details=event.details,
            )

    def list_audit_events(self, *, lead_id: int | None = None) -> list[AuditEvent]:
        with self.connection() as connection:
            if lead_id is None:
                rows = connection.execute(
                    "SELECT * FROM audit_events ORDER BY occurred_at, id"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM audit_events WHERE lead_id = ? ORDER BY occurred_at, id",
                    (lead_id,),
                ).fetchall()
        return [_audit_event_from_row(row) for row in rows]


def _add_column_if_missing(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _identity_kind(identity_key: str) -> str:
    if identity_key.startswith("facebook-id:"):
        return "facebook_id"
    if identity_key.startswith("facebook-url:"):
        return "facebook_url"
    if identity_key.startswith("content-text:"):
        return "content_text"
    if identity_key.startswith("content:"):
        return "content_author"
    return "primary"


def _alias_post_ids(
    connection: sqlite3.Connection,
    aliases: tuple[str, ...],
    *,
    without_stable_identity: bool = False,
) -> set[int]:
    if not aliases:
        return set()
    placeholders = ",".join("?" for _ in aliases)
    stable_filter = (
        """
        AND NOT EXISTS (
            SELECT 1
            FROM facebook_post_identity_aliases AS stable
            WHERE stable.facebook_post_id = posts.id
              AND stable.identity_kind IN ('facebook_id', 'facebook_url')
        )
        """
        if without_stable_identity
        else ""
    )
    rows = connection.execute(
        f"""
        SELECT DISTINCT aliases.facebook_post_id
        FROM facebook_post_identity_aliases AS aliases
        JOIN facebook_posts AS posts ON posts.id = aliases.facebook_post_id
        WHERE aliases.identity_key IN ({placeholders}) {stable_filter}
        """,
        aliases,
    ).fetchall()
    return {int(row["facebook_post_id"]) for row in rows}


def _find_duplicate_post_id(
    connection: sqlite3.Connection,
    post: FacebookPost,
    aliases: tuple[str, ...],
) -> int | None:
    direct = connection.execute(
        "SELECT id FROM facebook_posts WHERE identity_key = ?", (post.identity_key,)
    ).fetchone()
    if direct is not None:
        return int(direct["id"])

    stable_aliases = tuple(
        alias for alias in aliases if _identity_kind(alias) in {"facebook_id", "facebook_url"}
    )
    stable_matches = _alias_post_ids(connection, stable_aliases)
    if len(stable_matches) > 1:
        raise RuntimeError("Facebook post ID and URL resolve to conflicting saved posts")
    if stable_matches:
        return next(iter(stable_matches))

    content_aliases = tuple(
        alias for alias in aliases if _identity_kind(alias) in {"content_author", "content_text"}
    )
    content_matches = _alias_post_ids(
        connection,
        content_aliases,
        without_stable_identity=bool(stable_aliases),
    )
    return next(iter(content_matches)) if len(content_matches) == 1 else None


def _insert_identity_aliases(
    connection: sqlite3.Connection,
    *,
    post_id: int,
    aliases: tuple[str, ...],
    created_at: datetime,
) -> None:
    connection.executemany(
        """
        INSERT OR IGNORE INTO facebook_post_identity_aliases (
            identity_key, facebook_post_id, identity_kind, created_at
        ) VALUES (?, ?, ?, ?)
        """,
        (
            (alias, post_id, _identity_kind(alias), _serialize_datetime(created_at))
            for alias in aliases
        ),
    )


def _serialize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _post_from_row(row: sqlite3.Row) -> FacebookPost:
    return FacebookPost(
        id=row["id"],
        identity_key=row["identity_key"],
        external_post_id=row["external_post_id"],
        post_url=row["post_url"],
        group_id=row["group_id"],
        group_name=row["group_name"],
        author_name=row["author_name"],
        post_text=row["post_text"],
        text_hash=row["text_hash"],
        discovered_at=datetime.fromisoformat(row["discovered_at"]),
        posted_at=_parse_datetime(row["posted_at"]),
        status=PostStatus(row["status"]),
        error_state=row["error_state"],
        screenshot_path=row["screenshot_path"],
    )


def _lead_from_row(row: sqlite3.Row) -> Lead:
    return Lead(
        id=row["id"],
        facebook_post_id=row["facebook_post_id"],
        status=LeadStatus(row["status"]),
        service_category=row["service_category"],
        location=row["location"],
        relevance_score=row["relevance_score"],
        geographic_score=row["geographic_score"],
        urgency_score=row["urgency_score"],
        overall_score=row["overall_score"],
        confidence=row["confidence"],
        reasoning_summary=row["reasoning_summary"],
        drafted_response=row["drafted_response"],
        approved_response=row["approved_response"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        approval_timestamp=_parse_datetime(row["approval_timestamp"]),
        approval_expires_at=_parse_datetime(row["approval_expires_at"]),
        posting_timestamp=_parse_datetime(row["posting_timestamp"]),
        facebook_reply_url=row["facebook_reply_url"],
        error_state=row["error_state"],
        retry_count=row["retry_count"],
        screenshot_path=row["screenshot_path"],
    )


def _audit_event_from_row(row: sqlite3.Row) -> AuditEvent:
    details = json.loads(row["details_json"])
    if not isinstance(
        details, dict
    ):  # pragma: no cover - database constraint by application writes
        raise ValueError("Audit event details must decode to an object")
    return AuditEvent(
        id=row["id"],
        occurred_at=datetime.fromisoformat(row["occurred_at"]),
        component=row["component"],
        action=row["action"],
        result=row["result"],
        lead_id=row["lead_id"],
        post_id=row["post_id"],
        group_id=row["group_id"],
        details=details,
    )


def _group_scan_state_from_row(row: sqlite3.Row) -> GroupScanState:
    return GroupScanState(
        group_id=row["group_id"],
        group_name=row["group_name"],
        group_url=row["group_url"],
        last_attempt_at=datetime.fromisoformat(row["last_attempt_at"]),
        last_success_at=_parse_datetime(row["last_success_at"]),
        last_known_post_identity=row["last_known_post_identity"],
        last_error=row["last_error"],
        posts_seen=row["posts_seen"],
        posts_new=row["posts_new"],
        consecutive_failures=row["consecutive_failures"],
        last_failure_at=_parse_datetime(row["last_failure_at"]),
    )

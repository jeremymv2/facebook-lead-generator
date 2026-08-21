"""SQLite persistence with durable uniqueness constraints and audit history."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from lead_agent.models import (
    ApprovalNotification,
    ApprovalRequest,
    ApprovalReview,
    ApprovalStatus,
    AuditEvent,
    FacebookPost,
    GroupQuality,
    GroupScanState,
    Lead,
    LeadIntent,
    LeadStatus,
    NotificationStatus,
    PostingAttempt,
    PostingAttemptStatus,
    PostingJob,
    PostingJobStatus,
    PostingWorkItem,
    PostStatus,
    RejectionReason,
    hash_post_text,
    is_exact_facebook_post_url,
    is_post_text_expansion,
    normalize_post_text,
    utc_now,
)

SCHEMA_VERSION = 11
_REVIEW_DEDUPE_REPEATED_FRAGMENT_MINIMUM_TOKENS = 7


@lru_cache(maxsize=8_192)
def _review_deduplication_key(text: str) -> str:
    """Collapse formatting changes and repeated extraction fragments for review deduplication."""
    tokens = re.findall(r"[^\W_]+", text.casefold())
    minimum = _REVIEW_DEDUPE_REPEATED_FRAGMENT_MINIMUM_TOKENS
    separator = "\x1f"
    while len(tokens) >= minimum * 2:
        for suffix_length in range(len(tokens) // 2, minimum - 1, -1):
            prefix = separator + separator.join(tokens[:-suffix_length]) + separator
            suffix = separator + separator.join(tokens[-suffix_length:]) + separator
            if suffix in prefix:
                tokens = tokens[:-suffix_length]
                break
        else:
            break
    material = " ".join(tokens)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@lru_cache(maxsize=16_384)
def _review_posts_are_duplicates(first: str, second: str) -> int:
    """Conservatively match minor edits without collapsing distinct reordered requests."""
    if _review_deduplication_key(first) == _review_deduplication_key(second):
        return 1
    first_tokens = re.findall(r"[^\W_]+", first.casefold())
    second_tokens = re.findall(r"[^\W_]+", second.casefold())
    if min(len(first_tokens), len(second_tokens)) < 8:
        return 0
    sequence_ratio = SequenceMatcher(
        None,
        first_tokens,
        second_tokens,
        autojunk=False,
    ).ratio()
    first_numbers = set(re.findall(r"\d+", first))
    second_numbers = set(re.findall(r"\d+", second))
    conflicting_numbers = bool(first_numbers or second_numbers) and first_numbers != second_numbers
    if sequence_ratio >= 0.9 and not conflicting_numbers:
        return 1
    if min(len(first_tokens), len(second_tokens)) < 20:
        return 0
    first_set = set(first_tokens)
    second_set = set(second_tokens)
    containment = len(first_set & second_set) / min(len(first_set), len(second_set))
    shared_numbers = first_numbers & second_numbers
    return int(containment >= 0.85 and bool(shared_numbers))


@dataclass(frozen=True, slots=True)
class SaveResult:
    """The persisted post and whether this call inserted it."""

    post: FacebookPost
    created: bool


@dataclass(frozen=True, slots=True)
class LeadSaveResult:
    """The persisted lead and whether this call inserted it."""

    lead: Lead
    created: bool


@dataclass(frozen=True, slots=True)
class ClassificationWorkItem:
    """A saved classification joined to its immutable source post."""

    lead: Lead
    post: FacebookPost


@dataclass(frozen=True, slots=True)
class ApprovalFeedbackSummary:
    reviewed: int
    accepted: int
    rejected: int
    rejection_reasons: tuple[tuple[str, int], ...]

    @property
    def acceptance_percent(self) -> float:
        return round(self.accepted * 100 / self.reviewed, 1) if self.reviewed else 0.0


@dataclass(frozen=True, slots=True)
class PostingAttemptSaveResult:
    """The claimed posting work and whether this call created its attempt."""

    work: PostingWorkItem
    created: bool


class Database:
    """Small repository layer that opens a fresh SQLite connection per operation."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.create_function(
            "review_deduplication_key",
            1,
            _review_deduplication_key,
            deterministic=True,
        )
        connection.create_function(
            "review_posts_are_duplicates",
            2,
            _review_posts_are_duplicates,
            deterministic=True,
        )
        connection.create_function(
            "is_exact_facebook_post_url",
            1,
            lambda value: int(is_exact_facebook_post_url(value)),
            deterministic=True,
        )
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
                    intent TEXT,
                    is_residential INTEGER CHECK(is_residential IN (0, 1)),
                    is_spam INTEGER CHECK(is_spam IN (0, 1)),
                    relevance_score INTEGER CHECK(relevance_score BETWEEN 0 AND 100),
                    geographic_score INTEGER CHECK(geographic_score BETWEEN 0 AND 100),
                    urgency_score INTEGER CHECK(urgency_score BETWEEN 0 AND 100),
                    overall_score INTEGER CHECK(overall_score BETWEEN 0 AND 100),
                    confidence REAL CHECK(confidence BETWEEN 0 AND 1),
                    reasoning_summary TEXT,
                    drafted_response TEXT,
                    ai_provider TEXT,
                    ai_model TEXT,
                    classification_version TEXT,
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

                CREATE TABLE IF NOT EXISTS approval_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lead_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    draft_response TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    decided_at TEXT,
                    decided_response TEXT,
                    rejection_reason TEXT,
                    remote_token_hash TEXT,
                    FOREIGN KEY(lead_id) REFERENCES leads(id) ON DELETE RESTRICT
                );

                CREATE INDEX IF NOT EXISTS idx_approval_requests_status_expiration
                    ON approval_requests(status, expires_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_approval_requests_one_pending_per_lead
                    ON approval_requests(lead_id)
                    WHERE status = 'pending';
                CREATE UNIQUE INDEX IF NOT EXISTS idx_approval_requests_remote_token
                    ON approval_requests(remote_token_hash)
                    WHERE remote_token_hash IS NOT NULL;

                CREATE TABLE IF NOT EXISTS approval_notifications (
                    approval_request_id INTEGER PRIMARY KEY,
                    provider TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
                    last_attempt_at TEXT NOT NULL,
                    sent_at TEXT,
                    provider_message_id TEXT,
                    error_code TEXT,
                    delivery_status TEXT,
                    delivery_checked_at TEXT,
                    delivery_error_code TEXT,
                    FOREIGN KEY(approval_request_id)
                        REFERENCES approval_requests(id) ON DELETE RESTRICT
                );

                CREATE TABLE IF NOT EXISTS posting_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lead_id INTEGER NOT NULL,
                    approval_request_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    dry_run INTEGER NOT NULL CHECK(dry_run IN (0, 1)),
                    approved_response TEXT NOT NULL,
                    approved_response_hash TEXT NOT NULL,
                    source_text_hash TEXT NOT NULL,
                    post_url TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    validated_at TEXT,
                    submission_started_at TEXT,
                    completed_at TEXT,
                    facebook_reply_url TEXT,
                    before_screenshot_path TEXT,
                    after_screenshot_path TEXT,
                    error_code TEXT,
                    FOREIGN KEY(lead_id) REFERENCES leads(id) ON DELETE RESTRICT,
                    FOREIGN KEY(approval_request_id)
                        REFERENCES approval_requests(id) ON DELETE RESTRICT
                );

                CREATE INDEX IF NOT EXISTS idx_posting_attempts_started
                    ON posting_attempts(started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_posting_attempts_group_started
                    ON posting_attempts(group_id, started_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_posting_attempts_one_live_per_lead
                    ON posting_attempts(lead_id)
                    WHERE dry_run = 0 AND status IN (
                        'validating', 'submitting', 'posted',
                        'pending_moderation', 'needs_attention'
                    );

                CREATE TABLE IF NOT EXISTS posting_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lead_id INTEGER NOT NULL,
                    approval_request_id INTEGER NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    claimed_at TEXT,
                    completed_at TEXT,
                    error_code TEXT,
                    outcome_notification_status TEXT,
                    outcome_notification_attempted_at TEXT,
                    outcome_notification_sent_at TEXT,
                    outcome_provider TEXT,
                    outcome_provider_message_id TEXT,
                    outcome_notification_error_code TEXT,
                    FOREIGN KEY(lead_id) REFERENCES leads(id) ON DELETE RESTRICT,
                    FOREIGN KEY(approval_request_id)
                        REFERENCES approval_requests(id) ON DELETE RESTRICT
                );

                CREATE INDEX IF NOT EXISTS idx_posting_jobs_status_requested
                    ON posting_jobs(status, requested_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_posting_jobs_one_active_per_lead
                    ON posting_jobs(lead_id)
                    WHERE status IN ('queued', 'processing');

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
                    posts_requested INTEGER NOT NULL DEFAULT 0 CHECK(posts_requested >= 0),
                    last_scan_partial INTEGER NOT NULL DEFAULT 0
                        CHECK(last_scan_partial IN (0, 1)),
                    consecutive_failures INTEGER NOT NULL DEFAULT 0
                        CHECK(consecutive_failures >= 0),
                    last_failure_at TEXT
                );
                """
            )
            _add_column_if_missing(
                connection,
                "group_scan_state",
                "posts_requested",
                "INTEGER NOT NULL DEFAULT 0 CHECK(posts_requested >= 0)",
            )
            _add_column_if_missing(
                connection,
                "group_scan_state",
                "last_scan_partial",
                "INTEGER NOT NULL DEFAULT 0 CHECK(last_scan_partial IN (0, 1))",
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
            _add_column_if_missing(connection, "leads", "intent", "TEXT")
            _add_column_if_missing(
                connection,
                "leads",
                "is_residential",
                "INTEGER CHECK(is_residential IN (0, 1))",
            )
            _add_column_if_missing(
                connection,
                "leads",
                "is_spam",
                "INTEGER CHECK(is_spam IN (0, 1))",
            )
            _add_column_if_missing(connection, "leads", "ai_provider", "TEXT")
            _add_column_if_missing(connection, "leads", "ai_model", "TEXT")
            _add_column_if_missing(connection, "leads", "classification_version", "TEXT")
            _add_column_if_missing(connection, "approval_requests", "remote_token_hash", "TEXT")
            _add_column_if_missing(connection, "approval_requests", "rejection_reason", "TEXT")
            _add_column_if_missing(connection, "approval_notifications", "delivery_status", "TEXT")
            _add_column_if_missing(
                connection, "approval_notifications", "delivery_checked_at", "TEXT"
            )
            _add_column_if_missing(
                connection, "approval_notifications", "delivery_error_code", "TEXT"
            )
            connection.execute(
                """
                UPDATE approval_requests SET rejection_reason = ?
                WHERE status = ? AND rejection_reason IS NULL
                """,
                (RejectionReason.OTHER.value, ApprovalStatus.REJECTED.value),
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_approval_requests_remote_token
                ON approval_requests(remote_token_hash)
                WHERE remote_token_hash IS NOT NULL
                """
            )
            connection.execute("DROP INDEX IF EXISTS idx_posting_attempts_one_live_per_lead")
            connection.execute(
                """
                CREATE UNIQUE INDEX idx_posting_attempts_one_live_per_lead
                ON posting_attempts(lead_id)
                WHERE dry_run = 0 AND status IN (
                    'validating', 'submitting', 'posted',
                    'pending_moderation', 'needs_attention'
                )
                """
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
                existing_row = connection.execute(
                    "SELECT post_text FROM facebook_posts WHERE id = ?",
                    (existing_post_id,),
                ).fetchone()
                should_expand_text = existing_row is not None and is_post_text_expansion(
                    str(existing_row["post_text"]),
                    post.post_text,
                )
                connection.execute(
                    """
                    UPDATE facebook_posts SET
                        external_post_id = COALESCE(external_post_id, ?),
                        post_url = COALESCE(post_url, ?),
                        author_name = COALESCE(author_name, ?),
                        posted_at = COALESCE(posted_at, ?),
                        post_text = CASE WHEN ? THEN ? ELSE post_text END,
                        text_hash = CASE WHEN ? THEN ? ELSE text_hash END
                    WHERE id = ?
                    """,
                    (
                        post.external_post_id,
                        post.post_url,
                        post.author_name,
                        _serialize_datetime(post.posted_at),
                        should_expand_text,
                        post.post_text,
                        should_expand_text,
                        post.text_hash,
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

    def expand_post_text_for_rereview(
        self,
        post_id: int,
        *,
        expected_text_hash: str,
        expanded_text: str,
    ) -> FacebookPost:
        """Persist one verified prefix expansion without overwriting unrelated source edits."""
        normalized = normalize_post_text(expanded_text)
        expanded_hash = hash_post_text(normalized)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM facebook_posts WHERE id = ?",
                (post_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"Facebook post {post_id} does not exist")
            post = _post_from_row(row)
            if post.text_hash != expected_text_hash:
                if post.post_text == normalized:
                    return post
                raise ValueError("Facebook post changed before its expansion could be saved")
            if not is_post_text_expansion(post.post_text, normalized):
                raise ValueError("Replacement text is not a source-prefix expansion")
            connection.execute(
                "UPDATE facebook_posts SET post_text = ?, text_hash = ? WHERE id = ?",
                (normalized, expanded_hash, post_id),
            )
            updated_row = connection.execute(
                "SELECT * FROM facebook_posts WHERE id = ?",
                (post_id,),
            ).fetchone()
            if updated_row is None:  # pragma: no cover - protected by transaction
                raise RuntimeError("Failed to retrieve expanded Facebook post")
            updated = _post_from_row(updated_row)
            _insert_identity_aliases(
                connection,
                post_id=post_id,
                aliases=updated.identity_aliases(),
                created_at=updated.discovered_at,
            )
            return updated

    def list_posts(self, *, limit: int = 100) -> list[FacebookPost]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM facebook_posts ORDER BY discovered_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_post_from_row(row) for row in rows]

    def list_unclassified_posts(
        self,
        *,
        limit: int,
        post_id: int | None = None,
    ) -> list[FacebookPost]:
        """Return posts that do not yet have a persisted lead classification."""
        if limit < 1:
            raise ValueError("limit must be positive")
        with self.connection() as connection:
            if post_id is None:
                rows = connection.execute(
                    """
                    SELECT posts.*
                    FROM facebook_posts AS posts
                    LEFT JOIN leads ON leads.facebook_post_id = posts.id
                    WHERE leads.id IS NULL
                    ORDER BY posts.discovered_at DESC, posts.id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT posts.*
                    FROM facebook_posts AS posts
                    LEFT JOIN leads ON leads.facebook_post_id = posts.id
                    WHERE leads.id IS NULL AND posts.id = ?
                    LIMIT 1
                    """,
                    (post_id,),
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
        posts_requested: int = 0,
        is_partial: bool = False,
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
                    posts_requested, last_scan_partial, consecutive_failures
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, 0)
                ON CONFLICT(group_id) DO UPDATE SET
                    group_name = excluded.group_name,
                    group_url = excluded.group_url,
                    last_attempt_at = excluded.last_attempt_at,
                    last_success_at = COALESCE(
                        excluded.last_success_at,
                        group_scan_state.last_success_at
                    ),
                    last_known_post_identity = COALESCE(
                        excluded.last_known_post_identity,
                        group_scan_state.last_known_post_identity
                    ),
                    last_error = NULL,
                    posts_seen = excluded.posts_seen,
                    posts_new = excluded.posts_new,
                    posts_requested = excluded.posts_requested,
                    last_scan_partial = excluded.last_scan_partial,
                    consecutive_failures = 0
                """,
                (
                    group_id,
                    group_name,
                    group_url,
                    _serialize_datetime(timestamp),
                    _serialize_datetime(timestamp) if not is_partial else None,
                    last_known_post_identity,
                    posts_seen,
                    posts_new,
                    posts_requested,
                    int(is_partial),
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
        posts_requested: int = 0,
        occurred_at: datetime | None = None,
    ) -> GroupScanState:
        """Record a failed scan without erasing the last successful scan metadata."""
        timestamp = occurred_at or utc_now()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO group_scan_state (
                    group_id, group_name, group_url, last_attempt_at, last_error,
                    posts_requested, last_scan_partial, consecutive_failures, last_failure_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, 1, ?)
                ON CONFLICT(group_id) DO UPDATE SET
                    group_name = excluded.group_name,
                    group_url = excluded.group_url,
                    last_attempt_at = excluded.last_attempt_at,
                    last_error = excluded.last_error,
                    posts_seen = 0,
                    posts_new = 0,
                    posts_requested = excluded.posts_requested,
                    last_scan_partial = 0,
                    consecutive_failures = group_scan_state.consecutive_failures + 1,
                    last_failure_at = excluded.last_failure_at
                """,
                (
                    group_id,
                    group_name,
                    group_url,
                    _serialize_datetime(timestamp),
                    error,
                    posts_requested,
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

    def list_group_quality(self) -> list[GroupQuality]:
        """Return content-free lead yield and noise metrics grouped by Facebook group."""
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    posts.group_id,
                    MAX(posts.group_name) AS group_name,
                    COUNT(posts.id) AS posts_discovered,
                    COUNT(leads.id) AS posts_classified,
                    SUM(CASE WHEN leads.drafted_response IS NOT NULL THEN 1 ELSE 0 END)
                        AS candidates_created,
                    SUM(CASE WHEN leads.intent = ? THEN 1 ELSE 0 END)
                        AS provider_advertisements,
                    COUNT(posts.id) - COUNT(DISTINCT posts.text_hash)
                        AS exact_text_duplicates,
                    SUM(CASE WHEN EXISTS (
                        SELECT 1 FROM facebook_posts AS other_posts
                        WHERE other_posts.text_hash = posts.text_hash
                          AND other_posts.group_id != posts.group_id
                    ) THEN 1 ELSE 0 END) AS cross_group_reposts,
                    MAX(posts.discovered_at) AS last_discovered_at
                FROM facebook_posts AS posts
                LEFT JOIN leads ON leads.facebook_post_id = posts.id
                GROUP BY posts.group_id
                ORDER BY candidates_created DESC, posts_classified DESC, posts.group_id
                """,
                (LeadIntent.COMPETITOR_ADVERTISEMENT.value,),
            ).fetchall()
        return [_group_quality_from_row(row) for row in rows]

    def create_lead(self, lead: Lead) -> Lead:
        """Create at most one lead per Facebook post."""
        with self.connection() as connection:
            return _insert_lead(connection, lead)

    def save_classified_lead(self, lead: Lead) -> LeadSaveResult:
        """Atomically save one classification and mark its source post processed."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM leads WHERE facebook_post_id = ?", (lead.facebook_post_id,)
            ).fetchone()
            if row is not None:
                return LeadSaveResult(lead=_lead_from_row(row), created=False)
            persisted = _insert_lead(connection, lead)
            cursor = connection.execute(
                "UPDATE facebook_posts SET status = ?, error_state = NULL WHERE id = ?",
                (PostStatus.PROCESSED.value, lead.facebook_post_id),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"Post {lead.facebook_post_id} does not exist")
            return LeadSaveResult(lead=persisted, created=True)

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

    def list_classification_work_items(
        self,
        *,
        limit: int,
        lead_id: int | None = None,
        current_version: str | None = None,
        reclassifiable_only: bool = False,
    ) -> list[ClassificationWorkItem]:
        """Return bounded lead/post pairs for replay or safe unreviewed reclassification."""
        if limit < 1:
            raise ValueError("limit must be positive")
        filters: list[str] = []
        parameters: list[object] = []
        if lead_id is not None:
            filters.append("leads.id = ?")
            parameters.append(lead_id)
        if reclassifiable_only:
            filters.extend(
                (
                    "leads.status IN (?, ?)",
                    "NOT EXISTS (SELECT 1 FROM approval_requests "
                    "WHERE approval_requests.lead_id = leads.id)",
                )
            )
            parameters.extend((LeadStatus.CANDIDATE.value, LeadStatus.IGNORED.value))
            if lead_id is None and current_version is not None:
                filters.append(
                    "(leads.classification_version IS NULL OR leads.classification_version != ?)"
                )
                parameters.append(current_version)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        parameters.append(limit)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT leads.id AS selected_lead_id, posts.id AS selected_post_id
                FROM leads
                JOIN facebook_posts AS posts ON posts.id = leads.facebook_post_id
                {where_clause}
                ORDER BY leads.updated_at, leads.id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            return [
                _classification_work_item_from_connection(
                    connection,
                    lead_id=int(row["selected_lead_id"]),
                    post_id=int(row["selected_post_id"]),
                )
                for row in rows
            ]

    def replace_unreviewed_classification(self, lead: Lead) -> Lead:
        """Replace only candidate/ignored classifications that never entered human review."""
        if lead.id is None:
            raise ValueError("Replacement classification requires a persisted lead ID")
        if lead.status not in {LeadStatus.CANDIDATE, LeadStatus.IGNORED}:
            raise ValueError("Replacement classification must be candidate or ignored")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT * FROM leads WHERE id = ?", (lead.id,)).fetchone()
            if current is None:
                raise LookupError(f"Lead {lead.id} does not exist")
            current_lead = _lead_from_row(current)
            reviewed = connection.execute(
                "SELECT 1 FROM approval_requests WHERE lead_id = ? LIMIT 1",
                (lead.id,),
            ).fetchone()
            if current_lead.status not in {LeadStatus.CANDIDATE, LeadStatus.IGNORED} or reviewed:
                raise ValueError("Reviewed or terminal leads cannot be reclassified")
            if current_lead.facebook_post_id != lead.facebook_post_id:
                raise ValueError("Replacement classification cannot change its source post")
            connection.execute(
                """
                UPDATE leads SET
                    status = ?, service_category = ?, location = ?, intent = ?,
                    is_residential = ?, is_spam = ?, relevance_score = ?,
                    geographic_score = ?, urgency_score = ?, overall_score = ?,
                    confidence = ?, reasoning_summary = ?, drafted_response = ?,
                    ai_provider = ?, ai_model = ?, classification_version = ?,
                    updated_at = ?, error_state = NULL
                WHERE id = ?
                """,
                (
                    lead.status.value,
                    lead.service_category,
                    lead.location,
                    lead.intent.value if lead.intent else None,
                    lead.is_residential,
                    lead.is_spam,
                    lead.relevance_score,
                    lead.geographic_score,
                    lead.urgency_score,
                    lead.overall_score,
                    lead.confidence,
                    lead.reasoning_summary,
                    lead.drafted_response,
                    lead.ai_provider,
                    lead.ai_model,
                    lead.classification_version,
                    _serialize_datetime(lead.updated_at),
                    lead.id,
                ),
            )
            row = connection.execute("SELECT * FROM leads WHERE id = ?", (lead.id,)).fetchone()
            if row is None:  # pragma: no cover - protected by transaction
                raise RuntimeError("Failed to retrieve reclassified lead")
            return _lead_from_row(row)

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

    def list_candidate_leads(
        self,
        *,
        limit: int | None,
        duplicate_window_hours: int = 72,
        classification_version: str | None = None,
        suppress_duplicates: bool = True,
    ) -> list[Lead]:
        """Return review-ready candidates, suppressing nearby exact and extraction duplicates."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        if duplicate_window_hours < 1:
            raise ValueError("duplicate_window_hours must be positive")
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT leads.*
                FROM leads
                JOIN facebook_posts AS posts ON posts.id = leads.facebook_post_id
                WHERE leads.status = ? AND leads.drafted_response IS NOT NULL
                  AND is_exact_facebook_post_url(posts.post_url) = 1
                  AND (? IS NULL OR leads.classification_version = ?)
                  AND (? = 0 OR NOT EXISTS (
                    SELECT 1
                    FROM leads AS prior_leads
                    JOIN facebook_posts AS prior_posts
                      ON prior_posts.id = prior_leads.facebook_post_id
                    WHERE prior_leads.drafted_response IS NOT NULL
                      AND review_posts_are_duplicates(
                        prior_posts.post_text, posts.post_text
                      ) = 1
                      AND (
                        prior_posts.discovered_at < posts.discovered_at
                        OR (
                          prior_posts.discovered_at = posts.discovered_at
                          AND prior_leads.id < leads.id
                        )
                      )
                      AND (
                        julianday(posts.discovered_at) - julianday(prior_posts.discovered_at)
                      ) * 24 <= ?
                  ))
                ORDER BY leads.overall_score DESC, leads.created_at, leads.id
                LIMIT COALESCE(?, -1)
                """,
                (
                    LeadStatus.CANDIDATE.value,
                    classification_version,
                    classification_version,
                    int(suppress_duplicates),
                    duplicate_window_hours,
                    limit,
                ),
            ).fetchall()
        return [_lead_from_row(row) for row in rows]

    def restore_expired_candidate_leads(
        self,
        *,
        restored_at: datetime,
        classification_version: str | None = None,
    ) -> list[Lead]:
        """Return undecided expired approvals to the candidate backlog."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT DISTINCT leads.id
                FROM leads
                JOIN approval_requests AS requests ON requests.lead_id = leads.id
                WHERE leads.status = ? AND requests.status = ?
                  AND leads.drafted_response IS NOT NULL
                  AND (? IS NULL OR leads.classification_version = ?)
                ORDER BY leads.id
                """,
                (
                    LeadStatus.EXPIRED.value,
                    ApprovalStatus.EXPIRED.value,
                    classification_version,
                    classification_version,
                ),
            ).fetchall()
            lead_ids = [int(row["id"]) for row in rows]
            if not lead_ids:
                return []
            placeholders = ",".join("?" for _ in lead_ids)
            connection.execute(
                f"""
                UPDATE leads SET
                    status = ?, updated_at = ?, approval_expires_at = NULL,
                    error_state = NULL
                WHERE id IN ({placeholders}) AND status = ?
                """,
                (
                    LeadStatus.CANDIDATE.value,
                    _serialize_datetime(restored_at),
                    *lead_ids,
                    LeadStatus.EXPIRED.value,
                ),
            )
            restored_rows = connection.execute(
                f"SELECT * FROM leads WHERE id IN ({placeholders}) ORDER BY id",
                lead_ids,
            ).fetchall()
            return [_lead_from_row(row) for row in restored_rows]

    def create_approval_request(self, request: ApprovalRequest) -> ApprovalReview:
        """Atomically snapshot a candidate draft and move its lead into pending review."""
        if request.status is not ApprovalStatus.PENDING:
            raise ValueError("New approval requests must be pending")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lead_row = connection.execute(
                "SELECT * FROM leads WHERE id = ?", (request.lead_id,)
            ).fetchone()
            if lead_row is None:
                raise LookupError(f"Lead {request.lead_id} does not exist")
            lead = _lead_from_row(lead_row)
            if lead.status is not LeadStatus.CANDIDATE or lead.drafted_response is None:
                raise ValueError("Only draft-bearing candidate leads can enter approval")
            post_row = connection.execute(
                "SELECT post_url FROM facebook_posts WHERE id = ?",
                (lead.facebook_post_id,),
            ).fetchone()
            if post_row is None or not is_exact_facebook_post_url(post_row["post_url"]):
                raise ValueError(
                    "Only candidates with an exact Facebook post URL can enter approval"
                )
            existing = connection.execute(
                "SELECT id FROM approval_requests WHERE lead_id = ? AND status = ?",
                (request.lead_id, ApprovalStatus.PENDING.value),
            ).fetchone()
            if existing is not None:
                raise ValueError("Lead already has a pending approval request")
            cursor = connection.execute(
                """
                INSERT INTO approval_requests (
                    lead_id, status, draft_response, requested_at, expires_at,
                    decided_at, decided_response, rejection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.lead_id,
                    request.status.value,
                    request.draft_response,
                    _serialize_datetime(request.requested_at),
                    _serialize_datetime(request.expires_at),
                    _serialize_datetime(request.decided_at),
                    request.decided_response,
                    request.rejection_reason.value if request.rejection_reason else None,
                ),
            )
            request_id = cursor.lastrowid
            if request_id is None:  # pragma: no cover - SQLite insert contract
                raise RuntimeError("Failed to retrieve approval request ID")
            connection.execute(
                """
                UPDATE leads SET
                    status = ?, updated_at = ?, approval_expires_at = ?,
                    approval_timestamp = NULL, approved_response = NULL, error_state = NULL
                WHERE id = ?
                """,
                (
                    LeadStatus.PENDING_APPROVAL.value,
                    _serialize_datetime(request.requested_at),
                    _serialize_datetime(request.expires_at),
                    request.lead_id,
                ),
            )
            return _approval_review_from_connection(connection, int(request_id))

    def get_approval_request(self, request_id: int) -> ApprovalRequest | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM approval_requests WHERE id = ?", (request_id,)
            ).fetchone()
        return _approval_request_from_row(row) if row is not None else None

    def list_pending_approval_reviews(self) -> list[ApprovalReview]:
        with self.connection() as connection:
            request_ids = [
                int(row["id"])
                for row in connection.execute(
                    """
                    SELECT id FROM approval_requests
                    WHERE status = ?
                    ORDER BY expires_at, id
                    """,
                    (ApprovalStatus.PENDING.value,),
                ).fetchall()
            ]
            return [
                _approval_review_from_connection(connection, request_id)
                for request_id in request_ids
            ]

    def list_rejected_approval_reviews(
        self,
        *,
        limit: int,
        lead_id: int | None = None,
    ) -> list[ApprovalReview]:
        """Return bounded reason-bearing feedback records for sanitized fixture export."""
        if limit < 1:
            raise ValueError("limit must be positive")
        filters = ["status = ?", "rejection_reason IS NOT NULL"]
        parameters: list[object] = [ApprovalStatus.REJECTED.value]
        if lead_id is not None:
            filters.append("lead_id = ?")
            parameters.append(lead_id)
        parameters.append(limit)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT id FROM approval_requests
                WHERE {" AND ".join(filters)}
                ORDER BY decided_at, id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            return [_approval_review_from_connection(connection, int(row["id"])) for row in rows]

    def approval_feedback_summary(self) -> ApprovalFeedbackSummary:
        """Return content-free human-review accuracy and rejection-reason counts."""
        with self.connection() as connection:
            counts = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN status IN (?, ?, ?) THEN 1 ELSE 0 END) AS reviewed,
                    SUM(CASE WHEN status IN (?, ?) THEN 1 ELSE 0 END) AS accepted,
                    SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS rejected
                FROM approval_requests
                """,
                (
                    ApprovalStatus.APPROVED.value,
                    ApprovalStatus.EDITED.value,
                    ApprovalStatus.REJECTED.value,
                    ApprovalStatus.APPROVED.value,
                    ApprovalStatus.EDITED.value,
                    ApprovalStatus.REJECTED.value,
                ),
            ).fetchone()
            reason_rows = connection.execute(
                """
                SELECT rejection_reason, COUNT(*) AS reason_count
                FROM approval_requests
                WHERE status = ? AND rejection_reason IS NOT NULL
                GROUP BY rejection_reason
                ORDER BY reason_count DESC, rejection_reason
                """,
                (ApprovalStatus.REJECTED.value,),
            ).fetchall()
        return ApprovalFeedbackSummary(
            reviewed=int(counts["reviewed"] or 0),
            accepted=int(counts["accepted"] or 0),
            rejected=int(counts["rejected"] or 0),
            rejection_reasons=tuple(
                (str(row["rejection_reason"]), int(row["reason_count"])) for row in reason_rows
            ),
        )

    def list_notifiable_approval_reviews(
        self,
        *,
        include_failed: bool = False,
    ) -> list[ApprovalReview]:
        """Return pending approvals that have not been sent by an SMS provider."""
        with self.connection() as connection:
            if include_failed:
                notification_filter = (
                    "notifications.approval_request_id IS NULL OR notifications.status = ?"
                )
                parameters: tuple[object, ...] = (
                    ApprovalStatus.PENDING.value,
                    NotificationStatus.FAILED.value,
                )
            else:
                notification_filter = "notifications.approval_request_id IS NULL"
                parameters = (ApprovalStatus.PENDING.value,)
            rows = connection.execute(
                f"""
                SELECT requests.id
                FROM approval_requests AS requests
                LEFT JOIN approval_notifications AS notifications
                    ON notifications.approval_request_id = requests.id
                WHERE requests.status = ? AND ({notification_filter})
                ORDER BY requests.expires_at, requests.id
                """,
                parameters,
            ).fetchall()
            return [_approval_review_from_connection(connection, int(row["id"])) for row in rows]

    def claim_approval_notification(
        self,
        request_id: int,
        *,
        provider: str,
        remote_token_hash: str,
        attempted_at: datetime,
        retry_failed: bool = False,
    ) -> bool:
        """Atomically claim one pending approval for a single outbound SMS attempt."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            request_row = connection.execute(
                "SELECT * FROM approval_requests WHERE id = ?", (request_id,)
            ).fetchone()
            if request_row is None:
                raise LookupError(f"Approval request {request_id} does not exist")
            request = _approval_request_from_row(request_row)
            if request.status is not ApprovalStatus.PENDING or attempted_at >= request.expires_at:
                return False
            notification_row = connection.execute(
                "SELECT * FROM approval_notifications WHERE approval_request_id = ?",
                (request_id,),
            ).fetchone()
            if notification_row is None:
                connection.execute(
                    """
                    INSERT INTO approval_notifications (
                        approval_request_id, provider, status, attempt_count,
                        last_attempt_at, sent_at, provider_message_id, error_code
                    ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)
                    """,
                    (
                        request_id,
                        provider,
                        NotificationStatus.SENDING.value,
                        1,
                        _serialize_datetime(attempted_at),
                    ),
                )
            else:
                notification = _approval_notification_from_row(notification_row)
                if not retry_failed or notification.status is not NotificationStatus.FAILED:
                    return False
                connection.execute(
                    """
                    UPDATE approval_notifications SET
                        provider = ?, status = ?, attempt_count = attempt_count + 1,
                        last_attempt_at = ?, sent_at = NULL,
                        provider_message_id = NULL, error_code = NULL
                    WHERE approval_request_id = ?
                    """,
                    (
                        provider,
                        NotificationStatus.SENDING.value,
                        _serialize_datetime(attempted_at),
                        request_id,
                    ),
                )
            connection.execute(
                "UPDATE approval_requests SET remote_token_hash = ? WHERE id = ?",
                (remote_token_hash, request_id),
            )
            return True

    def complete_approval_notification(
        self,
        request_id: int,
        *,
        status: NotificationStatus,
        completed_at: datetime,
        provider_message_id: str | None = None,
        error_code: str | None = None,
    ) -> ApprovalNotification:
        """Persist a claimed SMS attempt result without storing its body or phone number."""
        if status not in {NotificationStatus.SENT, NotificationStatus.FAILED}:
            raise ValueError("Notification completion must be sent or failed")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM approval_notifications WHERE approval_request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"Approval notification {request_id} does not exist")
            notification = _approval_notification_from_row(row)
            if notification.status is not NotificationStatus.SENDING:
                raise ValueError("Approval notification is not awaiting completion")
            connection.execute(
                """
                UPDATE approval_notifications SET
                    status = ?, sent_at = ?, provider_message_id = ?, error_code = ?
                WHERE approval_request_id = ?
                """,
                (
                    status.value,
                    _serialize_datetime(completed_at)
                    if status is NotificationStatus.SENT
                    else None,
                    provider_message_id,
                    error_code,
                    request_id,
                ),
            )
            if status is NotificationStatus.FAILED:
                connection.execute(
                    "UPDATE approval_requests SET remote_token_hash = NULL WHERE id = ?",
                    (request_id,),
                )
            completed_row = connection.execute(
                "SELECT * FROM approval_notifications WHERE approval_request_id = ?",
                (request_id,),
            ).fetchone()
            if completed_row is None:  # pragma: no cover - protected by the transaction
                raise RuntimeError("Failed to retrieve completed approval notification")
            return _approval_notification_from_row(completed_row)

    def get_approval_notification(self, request_id: int) -> ApprovalNotification | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM approval_notifications WHERE approval_request_id = ?",
                (request_id,),
            ).fetchone()
        return _approval_notification_from_row(row) if row is not None else None

    def get_approval_review(self, request_id: int) -> ApprovalReview | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT id FROM approval_requests WHERE id = ?", (request_id,)
            ).fetchone()
            return (
                _approval_review_from_connection(connection, int(row["id"]))
                if row is not None
                else None
            )

    def list_approval_notifications_due_delivery_check(
        self,
        *,
        checked_before: datetime,
        limit: int,
    ) -> list[ApprovalNotification]:
        """Return recent, accepted SMS notifications whose final delivery state is unknown."""
        if limit < 1:
            raise ValueError("limit must be positive")
        terminal_statuses = ("delivered", "delivery_failed", "sending_failed")
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM approval_notifications
                WHERE provider = ?
                  AND status = ?
                  AND provider_message_id IS NOT NULL
                  AND (delivery_status IS NULL OR delivery_status NOT IN (?, ?, ?))
                  AND (delivery_checked_at IS NULL OR delivery_checked_at <= ?)
                ORDER BY sent_at, approval_request_id
                LIMIT ?
                """,
                (
                    "telnyx",
                    NotificationStatus.SENT.value,
                    *terminal_statuses,
                    _serialize_datetime(checked_before),
                    limit,
                ),
            ).fetchall()
        return [_approval_notification_from_row(row) for row in rows]

    def record_approval_delivery_status(
        self,
        request_id: int,
        *,
        delivery_status: str | None,
        checked_at: datetime,
        error_code: str | None = None,
    ) -> ApprovalNotification:
        """Persist Telnyx delivery state separately from API acceptance."""
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE approval_notifications
                SET delivery_status = ?, delivery_checked_at = ?, delivery_error_code = ?
                WHERE approval_request_id = ? AND status = ?
                """,
                (
                    delivery_status,
                    _serialize_datetime(checked_at),
                    error_code,
                    request_id,
                    NotificationStatus.SENT.value,
                ),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"Sent approval notification {request_id} does not exist")
            row = connection.execute(
                "SELECT * FROM approval_notifications WHERE approval_request_id = ?", (request_id,)
            ).fetchone()
        if row is None:  # pragma: no cover - protected by the update above
            raise RuntimeError("Failed to retrieve approval delivery status")
        return _approval_notification_from_row(row)

    def get_approval_review_by_remote_token_hash(
        self,
        remote_token_hash: str,
    ) -> ApprovalReview | None:
        """Resolve one opaque remote token hash without exposing a list endpoint."""
        with self.connection() as connection:
            row = connection.execute(
                "SELECT id FROM approval_requests WHERE remote_token_hash = ?",
                (remote_token_hash,),
            ).fetchone()
            if row is None:
                return None
            return _approval_review_from_connection(connection, int(row["id"]))

    def decide_approval_request(
        self,
        request_id: int,
        decision: ApprovalStatus,
        *,
        decided_at: datetime,
        edited_response: str | None = None,
        rejection_reason: RejectionReason | None = None,
        enqueue_posting: bool = False,
    ) -> tuple[ApprovalReview, bool]:
        """Apply one terminal review decision; return false when already terminal or expired."""
        allowed = {ApprovalStatus.APPROVED, ApprovalStatus.EDITED, ApprovalStatus.REJECTED}
        if decision not in allowed:
            raise ValueError("Approval decision must be approved, edited, or rejected")
        if enqueue_posting and decision is ApprovalStatus.REJECTED:
            raise ValueError("Rejected approvals cannot be queued for posting")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM approval_requests WHERE id = ?", (request_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Approval request {request_id} does not exist")
            request = _approval_request_from_row(row)
            if request.status is not ApprovalStatus.PENDING:
                return _approval_review_from_connection(connection, request_id), False
            if enqueue_posting:
                post_row = connection.execute(
                    """
                    SELECT posts.post_url
                    FROM leads
                    JOIN facebook_posts AS posts ON posts.id = leads.facebook_post_id
                    WHERE leads.id = ?
                    """,
                    (request.lead_id,),
                ).fetchone()
                if post_row is None or not is_exact_facebook_post_url(post_row["post_url"]):
                    raise ValueError("Queued posting requires an exact Facebook post URL")
            if decided_at >= request.expires_at:
                connection.execute(
                    "UPDATE approval_requests SET status = ? WHERE id = ?",
                    (ApprovalStatus.EXPIRED.value, request_id),
                )
                connection.execute(
                    """
                    UPDATE leads SET status = ?, updated_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        LeadStatus.EXPIRED.value,
                        _serialize_datetime(decided_at),
                        request.lead_id,
                        LeadStatus.PENDING_APPROVAL.value,
                    ),
                )
                return _approval_review_from_connection(connection, request_id), False
            if decision is ApprovalStatus.EDITED:
                decided_response = (edited_response or "").strip()
                if not decided_response:
                    raise ValueError("Edited approvals require a response")
                lead_status = LeadStatus.EDITED
            elif decision is ApprovalStatus.APPROVED:
                decided_response = request.draft_response
                lead_status = LeadStatus.APPROVED
            else:
                decided_response = None
                lead_status = LeadStatus.REJECTED
                if rejection_reason is None:
                    raise ValueError("Rejected approvals require a rejection reason")
            connection.execute(
                """
                UPDATE approval_requests SET
                    status = ?, decided_at = ?, decided_response = ?, rejection_reason = ?
                WHERE id = ?
                """,
                (
                    decision.value,
                    _serialize_datetime(decided_at),
                    decided_response,
                    rejection_reason.value if rejection_reason is not None else None,
                    request_id,
                ),
            )
            connection.execute(
                """
                UPDATE leads SET
                    status = ?, approved_response = ?, approval_timestamp = ?,
                    updated_at = ?, error_state = NULL
                WHERE id = ?
                """,
                (
                    lead_status.value,
                    decided_response,
                    _serialize_datetime(decided_at),
                    _serialize_datetime(decided_at),
                    request.lead_id,
                ),
            )
            if enqueue_posting:
                connection.execute(
                    """
                    INSERT INTO posting_jobs (
                        lead_id, approval_request_id, status, requested_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        request.lead_id,
                        request_id,
                        PostingJobStatus.QUEUED.value,
                        _serialize_datetime(decided_at),
                    ),
                )
            return _approval_review_from_connection(connection, request_id), True

    def get_posting_job(self, job_id: int) -> PostingJob | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM posting_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return _posting_job_from_row(row) if row is not None else None

    def get_posting_job_for_approval(self, request_id: int) -> PostingJob | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM posting_jobs WHERE approval_request_id = ?", (request_id,)
            ).fetchone()
        return _posting_job_from_row(row) if row is not None else None

    def list_approved_unposted_reviews(self, *, limit: int = 50) -> list[ApprovalReview]:
        """Return terminal approvals that have not yet entered the posting queue."""
        if limit < 1:
            raise ValueError("limit must be positive")
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT requests.id
                FROM approval_requests AS requests
                JOIN leads ON leads.id = requests.lead_id
                WHERE requests.status IN (?, ?)
                  AND leads.status IN (?, ?)
                  AND NOT EXISTS (
                    SELECT 1 FROM posting_jobs
                    WHERE posting_jobs.approval_request_id = requests.id
                  )
                  AND requests.decided_at = leads.approval_timestamp
                  AND requests.decided_response = leads.approved_response
                ORDER BY requests.decided_at DESC, requests.id DESC
                LIMIT ?
                """,
                (
                    ApprovalStatus.APPROVED.value,
                    ApprovalStatus.EDITED.value,
                    LeadStatus.APPROVED.value,
                    LeadStatus.EDITED.value,
                    limit,
                ),
            ).fetchall()
            return [_approval_review_from_connection(connection, int(row["id"])) for row in rows]

    def queue_approved_posting(
        self,
        lead_id: int,
        *,
        requested_at: datetime,
        approval_max_age_minutes: int,
    ) -> PostingJob:
        """Queue one fresh-enough terminal approval without changing its decision."""
        if approval_max_age_minutes < 1:
            raise ValueError("approval_max_age_minutes must be positive")
        oldest_approval_at = requested_at - timedelta(minutes=approval_max_age_minutes)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lead_row = connection.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
            if lead_row is None:
                raise LookupError(f"Lead {lead_id} does not exist")
            lead = _lead_from_row(lead_row)
            if lead.status not in {LeadStatus.APPROVED, LeadStatus.EDITED}:
                raise ValueError("Only approved or edited leads can enter posting")
            if lead.approval_timestamp is None or lead.approval_timestamp < oldest_approval_at:
                raise ValueError("Approval is stale; re-review is required before posting")
            if lead.approved_response is None:
                raise ValueError("Approved posting requires a response")
            post_row = connection.execute(
                "SELECT * FROM facebook_posts WHERE id = ?", (lead.facebook_post_id,)
            ).fetchone()
            if post_row is None or not is_exact_facebook_post_url(post_row["post_url"]):
                raise ValueError("Approved posting requires an exact Facebook post URL")
            approval_row = connection.execute(
                """
                SELECT * FROM approval_requests
                WHERE lead_id = ? AND status IN (?, ?)
                ORDER BY decided_at DESC, id DESC LIMIT 1
                """,
                (
                    lead_id,
                    ApprovalStatus.APPROVED.value,
                    ApprovalStatus.EDITED.value,
                ),
            ).fetchone()
            if approval_row is None:
                raise ValueError("Approved lead is missing its terminal approval request")
            approval = _approval_request_from_row(approval_row)
            if (
                approval.id is None
                or approval.decided_at != lead.approval_timestamp
                or approval.decided_response != lead.approved_response
            ):
                raise ValueError("Lead approval does not match its immutable review decision")
            existing = connection.execute(
                "SELECT * FROM posting_jobs WHERE approval_request_id = ?", (approval.id,)
            ).fetchone()
            if existing is not None:
                raise ValueError("This approval already has a posting job")
            cursor = connection.execute(
                """
                INSERT INTO posting_jobs (lead_id, approval_request_id, status, requested_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    lead_id,
                    approval.id,
                    PostingJobStatus.QUEUED.value,
                    _serialize_datetime(requested_at),
                ),
            )
            row = connection.execute(
                "SELECT * FROM posting_jobs WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            if row is None:  # pragma: no cover - insert contract
                raise RuntimeError("Failed to retrieve queued posting job")
            return _posting_job_from_row(row)

    def reopen_approved_lead(self, lead_id: int, *, reopened_at: datetime) -> Lead:
        """Return one unposted approval to candidate status for an explicit fresh review."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
            if row is None:
                raise LookupError(f"Lead {lead_id} does not exist")
            lead = _lead_from_row(row)
            if lead.status not in {LeadStatus.APPROVED, LeadStatus.EDITED}:
                raise ValueError("Only approved or edited leads can be reopened")
            existing = connection.execute(
                "SELECT 1 FROM posting_jobs WHERE lead_id = ?", (lead_id,)
            ).fetchone()
            if existing is not None:
                raise ValueError("A posting job already exists for this lead")
            connection.execute(
                """
                UPDATE leads SET status = ?, approved_response = NULL,
                    approval_timestamp = NULL, updated_at = ?
                WHERE id = ?
                """,
                (
                    LeadStatus.CANDIDATE.value,
                    _serialize_datetime(reopened_at),
                    lead_id,
                ),
            )
            updated = connection.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
            if updated is None:  # pragma: no cover - update contract
                raise RuntimeError("Failed to retrieve reopened lead")
            return _lead_from_row(updated)

    def claim_next_posting_job(self, *, claimed_at: datetime) -> PostingJob | None:
        """Atomically claim the oldest queued mobile-authorized submission."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM posting_jobs
                WHERE status = ?
                ORDER BY requested_at, id
                LIMIT 1
                """,
                (PostingJobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            job = _posting_job_from_row(row)
            if job.id is None:  # pragma: no cover - persisted job contract
                raise RuntimeError("Posting job is missing its ID")
            connection.execute(
                """
                UPDATE posting_jobs SET status = ?, claimed_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    PostingJobStatus.PROCESSING.value,
                    _serialize_datetime(claimed_at),
                    job.id,
                    PostingJobStatus.QUEUED.value,
                ),
            )
            claimed = connection.execute(
                "SELECT * FROM posting_jobs WHERE id = ?", (job.id,)
            ).fetchone()
            if claimed is None:  # pragma: no cover - protected by transaction
                raise RuntimeError("Failed to retrieve claimed posting job")
            return _posting_job_from_row(claimed)

    def has_queued_posting_job(self) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM posting_jobs WHERE status = ? LIMIT 1",
                (PostingJobStatus.QUEUED.value,),
            ).fetchone()
        return row is not None

    def list_processing_posting_jobs(self) -> list[PostingJob]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM posting_jobs
                WHERE status = ?
                ORDER BY claimed_at, id
                """,
                (PostingJobStatus.PROCESSING.value,),
            ).fetchall()
        return [_posting_job_from_row(row) for row in rows]

    def requeue_unstarted_posting_job(
        self,
        job_id: int,
        *,
        stale_before: datetime,
    ) -> PostingJob | None:
        """Safely requeue a stale claim only when no live attempt was ever reserved."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM posting_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Posting job {job_id} does not exist")
            job = _posting_job_from_row(row)
            if (
                job.status is not PostingJobStatus.PROCESSING
                or job.claimed_at is None
                or job.claimed_at > stale_before
            ):
                return None
            attempt = connection.execute(
                "SELECT 1 FROM posting_attempts WHERE lead_id = ? AND dry_run = 0 LIMIT 1",
                (job.lead_id,),
            ).fetchone()
            if attempt is not None:
                return None
            connection.execute(
                """
                UPDATE posting_jobs SET status = ?, claimed_at = NULL
                WHERE id = ?
                """,
                (PostingJobStatus.QUEUED.value, job_id),
            )
            requeued = connection.execute(
                "SELECT * FROM posting_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if requeued is None:  # pragma: no cover - protected by transaction
                raise RuntimeError("Failed to retrieve requeued posting job")
            return _posting_job_from_row(requeued)

    def complete_posting_job(
        self,
        job_id: int,
        *,
        status: PostingJobStatus,
        completed_at: datetime,
        error_code: str | None = None,
    ) -> PostingJob:
        terminal = {
            PostingJobStatus.POSTED,
            PostingJobStatus.PENDING_MODERATION,
            PostingJobStatus.EXPIRED,
            PostingJobStatus.FAILED,
            PostingJobStatus.NEEDS_ATTENTION,
        }
        if status not in terminal:
            raise ValueError("Posting job completion requires a terminal status")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM posting_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Posting job {job_id} does not exist")
            job = _posting_job_from_row(row)
            if job.status in terminal:
                return job
            if job.status is not PostingJobStatus.PROCESSING:
                raise ValueError("Posting job is not being processed")
            connection.execute(
                """
                UPDATE posting_jobs SET status = ?, completed_at = ?, error_code = ?
                WHERE id = ?
                """,
                (status.value, _serialize_datetime(completed_at), error_code, job_id),
            )
            completed = connection.execute(
                "SELECT * FROM posting_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if completed is None:  # pragma: no cover - protected by transaction
                raise RuntimeError("Failed to retrieve completed posting job")
            return _posting_job_from_row(completed)

    def expire_posting_job_for_rereview(
        self,
        job_id: int,
        *,
        expired_at: datetime,
    ) -> PostingJob:
        """Expire an unsubmitted queue job and return its lead to the review backlog."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM posting_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Posting job {job_id} does not exist")
            job = _posting_job_from_row(row)
            if job.status is PostingJobStatus.EXPIRED:
                return job
            if job.status is not PostingJobStatus.PROCESSING:
                raise ValueError("Posting job is not being processed")
            live_attempt = connection.execute(
                "SELECT 1 FROM posting_attempts WHERE lead_id = ? AND dry_run = 0 LIMIT 1",
                (job.lead_id,),
            ).fetchone()
            if live_attempt is not None:
                raise ValueError("A started posting attempt cannot return to review")
            connection.execute(
                """
                UPDATE posting_jobs SET status = ?, completed_at = ?, error_code = ?
                WHERE id = ?
                """,
                (
                    PostingJobStatus.EXPIRED.value,
                    _serialize_datetime(expired_at),
                    "posting_approval_expired",
                    job_id,
                ),
            )
            connection.execute(
                """
                UPDATE leads SET status = ?, approved_response = NULL,
                    approval_timestamp = NULL, posting_timestamp = NULL,
                    updated_at = ?, error_state = NULL
                WHERE id = ? AND status IN (?, ?)
                """,
                (
                    LeadStatus.CANDIDATE.value,
                    _serialize_datetime(expired_at),
                    job.lead_id,
                    LeadStatus.APPROVED.value,
                    LeadStatus.EDITED.value,
                ),
            )
            expired = connection.execute(
                "SELECT * FROM posting_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if expired is None:  # pragma: no cover - protected by transaction
                raise RuntimeError("Failed to retrieve expired posting job")
            return _posting_job_from_row(expired)

    def list_unnotified_posting_jobs(self, *, limit: int) -> list[PostingJob]:
        if limit < 1:
            raise ValueError("limit must be positive")
        terminal_values = tuple(
            status.value
            for status in (
                PostingJobStatus.POSTED,
                PostingJobStatus.PENDING_MODERATION,
                PostingJobStatus.EXPIRED,
                PostingJobStatus.FAILED,
                PostingJobStatus.NEEDS_ATTENTION,
            )
        )
        placeholders = ",".join("?" for _ in terminal_values)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM posting_jobs
                WHERE status IN ({placeholders})
                  AND outcome_notification_status IS NULL
                ORDER BY completed_at, id
                LIMIT ?
                """,
                (*terminal_values, limit),
            ).fetchall()
        return [_posting_job_from_row(row) for row in rows]

    def claim_posting_outcome_notification(
        self,
        job_id: int,
        *,
        provider: str,
        attempted_at: datetime,
    ) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE posting_jobs SET outcome_notification_status = ?,
                    outcome_notification_attempted_at = ?, outcome_provider = ?
                WHERE id = ? AND outcome_notification_status IS NULL
                """,
                (
                    NotificationStatus.SENDING.value,
                    _serialize_datetime(attempted_at),
                    provider,
                    job_id,
                ),
            )
            return cursor.rowcount == 1

    def complete_posting_outcome_notification(
        self,
        job_id: int,
        *,
        status: NotificationStatus,
        completed_at: datetime,
        provider_message_id: str | None = None,
        error_code: str | None = None,
    ) -> PostingJob:
        if status not in {NotificationStatus.SENT, NotificationStatus.FAILED}:
            raise ValueError("Outcome notification completion must be sent or failed")
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM posting_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Posting job {job_id} does not exist")
            job = _posting_job_from_row(row)
            if job.outcome_notification_status is not NotificationStatus.SENDING:
                raise ValueError("Posting outcome notification is not being sent")
            connection.execute(
                """
                UPDATE posting_jobs SET outcome_notification_status = ?,
                    outcome_notification_sent_at = ?, outcome_provider_message_id = ?,
                    outcome_notification_error_code = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    _serialize_datetime(completed_at)
                    if status is NotificationStatus.SENT
                    else None,
                    provider_message_id,
                    error_code,
                    job_id,
                ),
            )
            completed = connection.execute(
                "SELECT * FROM posting_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if completed is None:  # pragma: no cover - protected by transaction
                raise RuntimeError("Failed to retrieve completed posting notification")
            return _posting_job_from_row(completed)

    def expire_approval_requests(self, *, expired_at: datetime) -> list[ApprovalReview]:
        """Atomically expire every pending request whose review window has closed."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT id, lead_id FROM approval_requests
                WHERE status = ? AND expires_at <= ?
                ORDER BY expires_at, id
                """,
                (ApprovalStatus.PENDING.value, _serialize_datetime(expired_at)),
            ).fetchall()
            if not rows:
                return []
            request_ids = [int(row["id"]) for row in rows]
            lead_ids = [int(row["lead_id"]) for row in rows]
            request_placeholders = ",".join("?" for _ in request_ids)
            lead_placeholders = ",".join("?" for _ in lead_ids)
            connection.execute(
                f"UPDATE approval_requests SET status = ? WHERE id IN ({request_placeholders})",
                (ApprovalStatus.EXPIRED.value, *request_ids),
            )
            connection.execute(
                f"""
                UPDATE leads SET status = ?, updated_at = ?
                WHERE id IN ({lead_placeholders}) AND status = ?
                """,
                (
                    LeadStatus.EXPIRED.value,
                    _serialize_datetime(expired_at),
                    *lead_ids,
                    LeadStatus.PENDING_APPROVAL.value,
                ),
            )
            return [
                _approval_review_from_connection(connection, request_id)
                for request_id in request_ids
            ]

    def begin_posting_attempt(
        self,
        lead_id: int,
        *,
        dry_run: bool,
        started_at: datetime,
        oldest_approval_at: datetime,
        day_started_at: datetime,
        next_day_started_at: datetime,
        daily_limit: int,
        per_group_daily_limit: int,
    ) -> PostingAttemptSaveResult:
        """Atomically snapshot approved work and reserve one live-attempt slot."""
        if daily_limit < 1 or per_group_daily_limit < 1:
            raise ValueError("posting limits must be positive")
        if not day_started_at <= started_at < next_day_started_at:
            raise ValueError("posting day bounds must contain the attempt timestamp")

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not dry_run:
                existing = connection.execute(
                    """
                    SELECT id FROM posting_attempts
                    WHERE lead_id = ? AND dry_run = 0 AND status IN (?, ?, ?, ?, ?)
                    """,
                    (
                        lead_id,
                        PostingAttemptStatus.VALIDATING.value,
                        PostingAttemptStatus.SUBMITTING.value,
                        PostingAttemptStatus.POSTED.value,
                        PostingAttemptStatus.PENDING_MODERATION.value,
                        PostingAttemptStatus.NEEDS_ATTENTION.value,
                    ),
                ).fetchone()
                if existing is not None:
                    return PostingAttemptSaveResult(
                        work=_posting_work_from_connection(connection, int(existing["id"])),
                        created=False,
                    )

            lead_row = connection.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
            if lead_row is None:
                raise LookupError(f"Lead {lead_id} does not exist")
            lead = _lead_from_row(lead_row)
            if lead.status not in {LeadStatus.APPROVED, LeadStatus.EDITED}:
                raise ValueError("Only approved or edited leads can enter posting")
            if lead.approved_response is None or lead.approval_timestamp is None:
                raise ValueError("Approved posting requires a response and approval timestamp")
            if lead.approval_timestamp < oldest_approval_at:
                raise ValueError("Approval is stale; re-review is required before posting")

            post_row = connection.execute(
                "SELECT * FROM facebook_posts WHERE id = ?", (lead.facebook_post_id,)
            ).fetchone()
            if post_row is None:  # pragma: no cover - foreign key constraint
                raise RuntimeError("Lead is missing its Facebook post")
            post = _post_from_row(post_row)
            if not is_exact_facebook_post_url(post.post_url):
                raise ValueError("Approved posting requires an exact Facebook post URL")

            approval_row = connection.execute(
                """
                SELECT * FROM approval_requests
                WHERE lead_id = ? AND status IN (?, ?)
                ORDER BY decided_at DESC, id DESC
                LIMIT 1
                """,
                (
                    lead_id,
                    ApprovalStatus.APPROVED.value,
                    ApprovalStatus.EDITED.value,
                ),
            ).fetchone()
            if approval_row is None:
                raise ValueError("Approved lead is missing its terminal approval request")
            approval = _approval_request_from_row(approval_row)
            if (
                approval.id is None
                or approval.decided_at != lead.approval_timestamp
                or approval.decided_response != lead.approved_response
            ):
                raise ValueError("Lead approval does not match its immutable review decision")

            if not dry_run:
                live_today = connection.execute(
                    """
                    SELECT COUNT(*) AS total FROM posting_attempts
                    WHERE dry_run = 0 AND submission_started_at IS NOT NULL
                      AND started_at >= ? AND started_at < ?
                    """,
                    (
                        _serialize_datetime(day_started_at),
                        _serialize_datetime(next_day_started_at),
                    ),
                ).fetchone()
                if live_today is None or int(live_today["total"]) >= daily_limit:
                    raise ValueError("Global daily posting limit has been reached")
                group_today = connection.execute(
                    """
                    SELECT COUNT(*) AS total FROM posting_attempts
                    WHERE dry_run = 0 AND submission_started_at IS NOT NULL AND group_id = ?
                      AND started_at >= ? AND started_at < ?
                    """,
                    (
                        post.group_id,
                        _serialize_datetime(day_started_at),
                        _serialize_datetime(next_day_started_at),
                    ),
                ).fetchone()
                if group_today is None or int(group_today["total"]) >= per_group_daily_limit:
                    raise ValueError("Per-group daily posting limit has been reached")

            response_hash = hashlib.sha256(lead.approved_response.encode("utf-8")).hexdigest()
            cursor = connection.execute(
                """
                INSERT INTO posting_attempts (
                    lead_id, approval_request_id, status, dry_run,
                    approved_response, approved_response_hash, source_text_hash,
                    post_url, group_id, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lead_id,
                    approval.id,
                    PostingAttemptStatus.VALIDATING.value,
                    dry_run,
                    lead.approved_response,
                    response_hash,
                    post.text_hash,
                    post.post_url,
                    post.group_id,
                    _serialize_datetime(started_at),
                ),
            )
            attempt_id = cursor.lastrowid
            if attempt_id is None:  # pragma: no cover - SQLite insert contract
                raise RuntimeError("Failed to retrieve posting attempt ID")
            if not dry_run:
                connection.execute(
                    """
                    UPDATE leads SET status = ?, posting_timestamp = ?, updated_at = ?,
                        error_state = NULL
                    WHERE id = ?
                    """,
                    (
                        LeadStatus.POSTING.value,
                        _serialize_datetime(started_at),
                        _serialize_datetime(started_at),
                        lead_id,
                    ),
                )
            return PostingAttemptSaveResult(
                work=_posting_work_from_connection(connection, int(attempt_id)),
                created=True,
            )

    def get_posting_attempt(self, attempt_id: int) -> PostingAttempt | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM posting_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
        return _posting_attempt_from_row(row) if row is not None else None

    def list_posting_attempts(self, *, lead_id: int | None = None) -> list[PostingAttempt]:
        with self.connection() as connection:
            if lead_id is None:
                rows = connection.execute(
                    "SELECT * FROM posting_attempts ORDER BY started_at, id"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM posting_attempts
                    WHERE lead_id = ? ORDER BY started_at, id
                    """,
                    (lead_id,),
                ).fetchall()
        return [_posting_attempt_from_row(row) for row in rows]

    def posting_attempt_status_counts(self) -> dict[PostingAttemptStatus, int]:
        """Return content-free outcome counts for live Facebook posting attempts."""
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS attempt_count
                FROM posting_attempts
                WHERE dry_run = 0
                GROUP BY status
                """
            ).fetchall()
        return {PostingAttemptStatus(str(row["status"])): int(row["attempt_count"]) for row in rows}

    def complete_posting_validation(
        self,
        attempt_id: int,
        *,
        validated_at: datetime,
        before_screenshot_path: str | None,
    ) -> PostingAttempt:
        """Record exact-post and composer validation before any Facebook write action."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM posting_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Posting attempt {attempt_id} does not exist")
            attempt = _posting_attempt_from_row(row)
            if attempt.status is not PostingAttemptStatus.VALIDATING:
                raise ValueError("Posting attempt is not awaiting validation")
            status = (
                PostingAttemptStatus.DRY_RUN_VALIDATED
                if attempt.dry_run
                else PostingAttemptStatus.VALIDATING
            )
            connection.execute(
                """
                UPDATE posting_attempts SET status = ?, validated_at = ?,
                    completed_at = ?, before_screenshot_path = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    _serialize_datetime(validated_at),
                    _serialize_datetime(validated_at) if attempt.dry_run else None,
                    before_screenshot_path,
                    attempt_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM posting_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if updated is None:  # pragma: no cover - protected by transaction
                raise RuntimeError("Failed to retrieve validated posting attempt")
            return _posting_attempt_from_row(updated)

    def mark_posting_submission_started(
        self,
        attempt_id: int,
        *,
        started_at: datetime,
    ) -> PostingAttempt:
        """Durably cross the no-retry boundary immediately before browser submission."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM posting_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Posting attempt {attempt_id} does not exist")
            attempt = _posting_attempt_from_row(row)
            if (
                attempt.dry_run
                or attempt.status is not PostingAttemptStatus.VALIDATING
                or attempt.validated_at is None
                or attempt.submission_started_at is not None
            ):
                raise ValueError("Posting attempt is not ready for submission")
            connection.execute(
                """
                UPDATE posting_attempts SET status = ?, submission_started_at = ?
                WHERE id = ?
                """,
                (
                    PostingAttemptStatus.SUBMITTING.value,
                    _serialize_datetime(started_at),
                    attempt_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM posting_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if updated is None:  # pragma: no cover - protected by transaction
                raise RuntimeError("Failed to retrieve submitting attempt")
            return _posting_attempt_from_row(updated)

    def complete_posting_attempt(
        self,
        attempt_id: int,
        *,
        completed_at: datetime,
        facebook_reply_url: str | None,
        after_screenshot_path: str | None,
    ) -> PostingWorkItem:
        """Atomically mark a verified Facebook comment and its lead posted."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM posting_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Posting attempt {attempt_id} does not exist")
            attempt = _posting_attempt_from_row(row)
            if attempt.status is not PostingAttemptStatus.SUBMITTING or attempt.dry_run:
                raise ValueError("Posting attempt is not awaiting verified completion")
            connection.execute(
                """
                UPDATE posting_attempts SET status = ?, completed_at = ?,
                    facebook_reply_url = ?, after_screenshot_path = ?, error_code = NULL
                WHERE id = ?
                """,
                (
                    PostingAttemptStatus.POSTED.value,
                    _serialize_datetime(completed_at),
                    facebook_reply_url,
                    after_screenshot_path,
                    attempt_id,
                ),
            )
            connection.execute(
                """
                UPDATE leads SET status = ?, posting_timestamp = ?, facebook_reply_url = ?,
                    screenshot_path = ?, updated_at = ?, error_state = NULL
                WHERE id = ?
                """,
                (
                    LeadStatus.POSTED.value,
                    _serialize_datetime(completed_at),
                    facebook_reply_url,
                    after_screenshot_path,
                    _serialize_datetime(completed_at),
                    attempt.lead_id,
                ),
            )
            return _posting_work_from_connection(connection, attempt_id)

    def complete_posting_moderation(
        self,
        attempt_id: int,
        *,
        completed_at: datetime,
        after_screenshot_path: str | None,
    ) -> PostingWorkItem:
        """Atomically record that Facebook accepted a comment for group review."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM posting_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Posting attempt {attempt_id} does not exist")
            attempt = _posting_attempt_from_row(row)
            if attempt.status is not PostingAttemptStatus.SUBMITTING or attempt.dry_run:
                raise ValueError("Posting attempt is not awaiting moderation confirmation")
            connection.execute(
                """
                UPDATE posting_attempts SET status = ?, completed_at = ?,
                    facebook_reply_url = NULL, after_screenshot_path = ?, error_code = NULL
                WHERE id = ?
                """,
                (
                    PostingAttemptStatus.PENDING_MODERATION.value,
                    _serialize_datetime(completed_at),
                    after_screenshot_path,
                    attempt_id,
                ),
            )
            connection.execute(
                """
                UPDATE leads SET status = ?, posting_timestamp = ?, facebook_reply_url = NULL,
                    screenshot_path = ?, updated_at = ?, error_state = NULL
                WHERE id = ?
                """,
                (
                    LeadStatus.PENDING_MODERATION.value,
                    _serialize_datetime(completed_at),
                    after_screenshot_path,
                    _serialize_datetime(completed_at),
                    attempt.lead_id,
                ),
            )
            return _posting_work_from_connection(connection, attempt_id)

    def reconcile_posting_moderation(
        self,
        attempt_id: int,
        *,
        reconciled_at: datetime,
        after_screenshot_path: str | None = None,
    ) -> PostingWorkItem:
        """Reclassify one uncertain submitted comment after manual moderation proof."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM posting_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Posting attempt {attempt_id} does not exist")
            attempt = _posting_attempt_from_row(row)
            if attempt.status is PostingAttemptStatus.PENDING_MODERATION:
                return _posting_work_from_connection(connection, attempt_id)
            if (
                attempt.dry_run
                or attempt.status is not PostingAttemptStatus.NEEDS_ATTENTION
                or attempt.submission_started_at is None
            ):
                raise ValueError("Posting attempt cannot be reconciled as pending moderation")
            connection.execute(
                """
                UPDATE posting_attempts SET status = ?, completed_at = ?,
                    facebook_reply_url = NULL,
                    after_screenshot_path = COALESCE(?, after_screenshot_path),
                    error_code = NULL
                WHERE id = ?
                """,
                (
                    PostingAttemptStatus.PENDING_MODERATION.value,
                    _serialize_datetime(reconciled_at),
                    after_screenshot_path,
                    attempt_id,
                ),
            )
            connection.execute(
                """
                UPDATE leads SET status = ?, posting_timestamp = ?, facebook_reply_url = NULL,
                    screenshot_path = COALESCE(?, screenshot_path),
                    updated_at = ?, error_state = NULL
                WHERE id = ?
                """,
                (
                    LeadStatus.PENDING_MODERATION.value,
                    _serialize_datetime(attempt.submission_started_at),
                    after_screenshot_path,
                    _serialize_datetime(reconciled_at),
                    attempt.lead_id,
                ),
            )
            return _posting_work_from_connection(connection, attempt_id)

    def fail_posting_attempt(
        self,
        attempt_id: int,
        *,
        failed_at: datetime,
        error_code: str,
        after_screenshot_path: str | None = None,
    ) -> PostingWorkItem:
        """Fail closed; a post-submit uncertainty is permanently marked for attention."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM posting_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Posting attempt {attempt_id} does not exist")
            attempt = _posting_attempt_from_row(row)
            if attempt.status in {
                PostingAttemptStatus.POSTED,
                PostingAttemptStatus.PENDING_MODERATION,
                PostingAttemptStatus.DRY_RUN_VALIDATED,
                PostingAttemptStatus.FAILED,
                PostingAttemptStatus.NEEDS_ATTENTION,
            }:
                return _posting_work_from_connection(connection, attempt_id)
            status = (
                PostingAttemptStatus.NEEDS_ATTENTION
                if attempt.submission_started_at is not None
                else PostingAttemptStatus.FAILED
            )
            connection.execute(
                """
                UPDATE posting_attempts SET status = ?, completed_at = ?,
                    after_screenshot_path = COALESCE(?, after_screenshot_path),
                    error_code = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    _serialize_datetime(failed_at),
                    after_screenshot_path,
                    error_code,
                    attempt_id,
                ),
            )
            if not attempt.dry_run:
                if status is PostingAttemptStatus.FAILED:
                    connection.execute(
                        """
                        UPDATE leads SET status = ?, approved_response = NULL,
                            approval_timestamp = NULL, posting_timestamp = NULL,
                            updated_at = ?, error_state = ?,
                            screenshot_path = COALESCE(?, screenshot_path)
                        WHERE id = ?
                        """,
                        (
                            LeadStatus.CANDIDATE.value,
                            _serialize_datetime(failed_at),
                            error_code,
                            after_screenshot_path,
                            attempt.lead_id,
                        ),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE leads SET status = ?, updated_at = ?, error_state = ?,
                            screenshot_path = COALESCE(?, screenshot_path)
                        WHERE id = ?
                        """,
                        (
                            LeadStatus.NEEDS_ATTENTION.value,
                            _serialize_datetime(failed_at),
                            error_code,
                            after_screenshot_path,
                            attempt.lead_id,
                        ),
                    )
            return _posting_work_from_connection(connection, attempt_id)

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

    def list_audit_events(
        self,
        *,
        lead_id: int | None = None,
        component: str | None = None,
        action: str | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[AuditEvent]:
        """List filtered audit events without loading unrelated history."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        filters: list[str] = []
        parameters: list[object] = []
        if lead_id is not None:
            filters.append("lead_id = ?")
            parameters.append(lead_id)
        if component is not None:
            filters.append("component = ?")
            parameters.append(component)
        if action is not None:
            filters.append("action = ?")
            parameters.append(action)
        where_clause = f" WHERE {' AND '.join(filters)}" if filters else ""
        direction = "DESC" if newest_first else "ASC"
        limit_clause = " LIMIT ?" if limit is not None else ""
        if limit is not None:
            parameters.append(limit)
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM audit_events{where_clause} "
                f"ORDER BY occurred_at {direction}, id {direction}{limit_clause}",
                parameters,
            ).fetchall()
        return [_audit_event_from_row(row) for row in rows]


def _insert_lead(connection: sqlite3.Connection, lead: Lead) -> Lead:
    cursor = connection.execute(
        """
        INSERT INTO leads (
            facebook_post_id, status, service_category, location,
            intent, is_residential, is_spam,
            relevance_score, geographic_score, urgency_score, overall_score,
            confidence, reasoning_summary, drafted_response, ai_provider, ai_model,
            classification_version, approved_response,
            created_at, updated_at, approval_timestamp, approval_expires_at,
            posting_timestamp, facebook_reply_url, error_state, retry_count,
            screenshot_path
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?
        )
        """,
        (
            lead.facebook_post_id,
            lead.status.value,
            lead.service_category,
            lead.location,
            lead.intent.value if lead.intent is not None else None,
            lead.is_residential,
            lead.is_spam,
            lead.relevance_score,
            lead.geographic_score,
            lead.urgency_score,
            lead.overall_score,
            lead.confidence,
            lead.reasoning_summary,
            lead.drafted_response,
            lead.ai_provider,
            lead.ai_model,
            lead.classification_version,
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
    row = connection.execute("SELECT * FROM leads WHERE id = ?", (cursor.lastrowid,)).fetchone()
    if row is None:  # pragma: no cover - protected by the transaction
        raise RuntimeError("Failed to retrieve created lead")
    return _lead_from_row(row)


def _classification_work_item_from_connection(
    connection: sqlite3.Connection,
    *,
    lead_id: int,
    post_id: int,
) -> ClassificationWorkItem:
    lead_row = connection.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
    post_row = connection.execute(
        "SELECT * FROM facebook_posts WHERE id = ?", (post_id,)
    ).fetchone()
    if lead_row is None or post_row is None:  # pragma: no cover - joined query contract
        raise RuntimeError("Classification work item is incomplete")
    return ClassificationWorkItem(
        lead=_lead_from_row(lead_row),
        post=_post_from_row(post_row),
    )


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


def _parse_optional_bool(value: int | None) -> bool | None:
    return bool(value) if value is not None else None


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
        intent=LeadIntent(row["intent"]) if row["intent"] is not None else None,
        is_residential=_parse_optional_bool(row["is_residential"]),
        is_spam=_parse_optional_bool(row["is_spam"]),
        relevance_score=row["relevance_score"],
        geographic_score=row["geographic_score"],
        urgency_score=row["urgency_score"],
        overall_score=row["overall_score"],
        confidence=row["confidence"],
        reasoning_summary=row["reasoning_summary"],
        drafted_response=row["drafted_response"],
        ai_provider=row["ai_provider"],
        ai_model=row["ai_model"],
        classification_version=row["classification_version"],
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


def _approval_request_from_row(row: sqlite3.Row) -> ApprovalRequest:
    expires_at = _parse_datetime(row["expires_at"])
    if expires_at is None:  # pragma: no cover - database constraint
        raise ValueError("Approval request is missing its expiration")
    return ApprovalRequest(
        id=row["id"],
        lead_id=row["lead_id"],
        status=ApprovalStatus(row["status"]),
        draft_response=row["draft_response"],
        requested_at=datetime.fromisoformat(row["requested_at"]),
        expires_at=expires_at,
        decided_at=_parse_datetime(row["decided_at"]),
        decided_response=row["decided_response"],
        rejection_reason=(
            RejectionReason(row["rejection_reason"])
            if row["rejection_reason"] is not None
            else None
        ),
    )


def _approval_notification_from_row(row: sqlite3.Row) -> ApprovalNotification:
    return ApprovalNotification(
        approval_request_id=row["approval_request_id"],
        provider=row["provider"],
        status=NotificationStatus(row["status"]),
        attempt_count=row["attempt_count"],
        last_attempt_at=datetime.fromisoformat(row["last_attempt_at"]),
        sent_at=_parse_datetime(row["sent_at"]),
        provider_message_id=row["provider_message_id"],
        error_code=row["error_code"],
        delivery_status=row["delivery_status"],
        delivery_checked_at=_parse_datetime(row["delivery_checked_at"]),
        delivery_error_code=row["delivery_error_code"],
    )


def _posting_attempt_from_row(row: sqlite3.Row) -> PostingAttempt:
    return PostingAttempt(
        id=row["id"],
        lead_id=row["lead_id"],
        approval_request_id=row["approval_request_id"],
        status=PostingAttemptStatus(row["status"]),
        dry_run=bool(row["dry_run"]),
        approved_response=row["approved_response"],
        approved_response_hash=row["approved_response_hash"],
        source_text_hash=row["source_text_hash"],
        post_url=row["post_url"],
        group_id=row["group_id"],
        started_at=datetime.fromisoformat(row["started_at"]),
        validated_at=_parse_datetime(row["validated_at"]),
        submission_started_at=_parse_datetime(row["submission_started_at"]),
        completed_at=_parse_datetime(row["completed_at"]),
        facebook_reply_url=row["facebook_reply_url"],
        before_screenshot_path=row["before_screenshot_path"],
        after_screenshot_path=row["after_screenshot_path"],
        error_code=row["error_code"],
    )


def _posting_job_from_row(row: sqlite3.Row) -> PostingJob:
    notification_status = row["outcome_notification_status"]
    requested_at = _parse_datetime(row["requested_at"])
    if requested_at is None:  # pragma: no cover - schema requires a timestamp
        raise ValueError("Posting job is missing its requested timestamp")
    return PostingJob(
        id=row["id"],
        lead_id=row["lead_id"],
        approval_request_id=row["approval_request_id"],
        status=PostingJobStatus(row["status"]),
        requested_at=requested_at,
        claimed_at=_parse_datetime(row["claimed_at"]),
        completed_at=_parse_datetime(row["completed_at"]),
        error_code=row["error_code"],
        outcome_notification_status=(
            NotificationStatus(notification_status) if notification_status is not None else None
        ),
        outcome_notification_attempted_at=_parse_datetime(row["outcome_notification_attempted_at"]),
        outcome_notification_sent_at=_parse_datetime(row["outcome_notification_sent_at"]),
        outcome_provider=row["outcome_provider"],
        outcome_provider_message_id=row["outcome_provider_message_id"],
        outcome_notification_error_code=row["outcome_notification_error_code"],
    )


def _posting_work_from_connection(
    connection: sqlite3.Connection,
    attempt_id: int,
) -> PostingWorkItem:
    attempt_row = connection.execute(
        "SELECT * FROM posting_attempts WHERE id = ?", (attempt_id,)
    ).fetchone()
    if attempt_row is None:  # pragma: no cover - protected by caller query
        raise RuntimeError("Failed to retrieve posting attempt")
    attempt = _posting_attempt_from_row(attempt_row)
    lead_row = connection.execute("SELECT * FROM leads WHERE id = ?", (attempt.lead_id,)).fetchone()
    if lead_row is None:  # pragma: no cover - foreign key constraint
        raise RuntimeError("Posting attempt is missing its lead")
    lead = _lead_from_row(lead_row)
    post_row = connection.execute(
        "SELECT * FROM facebook_posts WHERE id = ?", (lead.facebook_post_id,)
    ).fetchone()
    if post_row is None:  # pragma: no cover - foreign key constraint
        raise RuntimeError("Posting attempt lead is missing its Facebook post")
    return PostingWorkItem(attempt=attempt, lead=lead, post=_post_from_row(post_row))


def _approval_review_from_connection(
    connection: sqlite3.Connection,
    request_id: int,
) -> ApprovalReview:
    request_row = connection.execute(
        "SELECT * FROM approval_requests WHERE id = ?", (request_id,)
    ).fetchone()
    if request_row is None:  # pragma: no cover - protected by caller query
        raise RuntimeError("Failed to retrieve approval request")
    request = _approval_request_from_row(request_row)
    lead_row = connection.execute("SELECT * FROM leads WHERE id = ?", (request.lead_id,)).fetchone()
    if lead_row is None:  # pragma: no cover - foreign key constraint
        raise RuntimeError("Approval request is missing its lead")
    lead = _lead_from_row(lead_row)
    post_row = connection.execute(
        "SELECT * FROM facebook_posts WHERE id = ?", (lead.facebook_post_id,)
    ).fetchone()
    if post_row is None:  # pragma: no cover - foreign key constraint
        raise RuntimeError("Approval request lead is missing its Facebook post")
    return ApprovalReview(request=request, lead=lead, post=_post_from_row(post_row))


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
        posts_requested=row["posts_requested"],
        last_scan_partial=bool(row["last_scan_partial"]),
        consecutive_failures=row["consecutive_failures"],
        last_failure_at=_parse_datetime(row["last_failure_at"]),
    )


def _group_quality_from_row(row: sqlite3.Row) -> GroupQuality:
    return GroupQuality(
        group_id=row["group_id"],
        group_name=row["group_name"],
        posts_discovered=row["posts_discovered"],
        posts_classified=row["posts_classified"],
        candidates_created=row["candidates_created"],
        provider_advertisements=row["provider_advertisements"],
        exact_text_duplicates=row["exact_text_duplicates"],
        cross_group_reposts=row["cross_group_reposts"],
        last_discovered_at=datetime.fromisoformat(row["last_discovered_at"]),
    )

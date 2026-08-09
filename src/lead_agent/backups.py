"""Private SQLite backup creation, retention, integrity checks, and restore drills."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lead_agent.database import SCHEMA_VERSION

BACKUP_PREFIX = "lead-agent-"
BACKUP_SUFFIX = ".sqlite3"


class BackupError(RuntimeError):
    """Raised when a backup cannot be created or verified safely."""


@dataclass(frozen=True, slots=True)
class BackupSummary:
    created: int
    verified: int
    removed: int
    backup_path: Path | None = None


class DatabaseBackupService:
    """Create private online backups and prove they restore into a disposable database."""

    def __init__(
        self,
        database_path: Path,
        backup_dir: Path,
        *,
        retention_days: int,
        interval_hours: int,
    ) -> None:
        if retention_days < 1:
            raise ValueError("retention_days must be positive")
        if interval_hours < 1:
            raise ValueError("interval_hours must be positive")
        self.database_path = database_path
        self.backup_dir = backup_dir
        self.retention_days = retention_days
        self.interval = timedelta(hours=interval_hours)

    def run(self, *, force: bool = False, now: datetime | None = None) -> BackupSummary:
        timestamp = now or datetime.now(UTC)
        self._ensure_backup_dir()
        removed = self._remove_expired(timestamp)
        newest = self.latest_backup()
        if not force and newest is not None:
            modified = datetime.fromtimestamp(newest.stat().st_mtime, tz=UTC)
            if timestamp - modified < self.interval:
                return BackupSummary(created=0, verified=0, removed=removed)
        backup_path = self._create(timestamp)
        self.verify_restore(backup_path)
        return BackupSummary(
            created=1,
            verified=1,
            removed=removed,
            backup_path=backup_path,
        )

    def latest_backup(self) -> Path | None:
        if not self.backup_dir.is_dir():
            return None
        backups = sorted(
            path
            for path in self.backup_dir.iterdir()
            if path.is_file()
            and path.name.startswith(BACKUP_PREFIX)
            and path.suffix == BACKUP_SUFFIX
        )
        return backups[-1] if backups else None

    def verify_restore(self, backup_path: Path | None = None) -> Path:
        """Restore a selected backup to a disposable file, verify it, then remove it."""
        selected = backup_path or self.latest_backup()
        if selected is None:
            raise BackupError("No database backup is available for restore testing")
        selected = selected.resolve()
        backup_root = self.backup_dir.resolve()
        if (
            selected.parent != backup_root
            or not selected.name.startswith(BACKUP_PREFIX)
            or selected.suffix != BACKUP_SUFFIX
        ):
            raise BackupError(
                "Restore tests are limited to the configured private backup directory"
            )
        restore_path = self.backup_dir / f".restore-test.{os.getpid()}.sqlite3"
        try:
            try:
                with (
                    sqlite3.connect(selected) as source,
                    sqlite3.connect(restore_path) as restored,
                ):
                    source.backup(restored)
                    restored.execute("PRAGMA journal_mode=DELETE")
                restore_path.chmod(0o600)
                _verify_database(restore_path)
            except (OSError, sqlite3.Error, BackupError) as error:
                raise BackupError("Database restore test failed") from error
        finally:
            _remove_sqlite_temporary_files(restore_path)
        return selected

    def _create(self, timestamp: datetime) -> Path:
        if not self.database_path.is_file():
            raise BackupError("The source database does not exist")
        name = timestamp.astimezone(UTC).strftime(f"{BACKUP_PREFIX}%Y%m%dT%H%M%S%fZ{BACKUP_SUFFIX}")
        destination = self.backup_dir / name
        temporary = self.backup_dir / f".{name}.{os.getpid()}.tmp"
        try:
            with (
                sqlite3.connect(self.database_path) as source,
                sqlite3.connect(temporary) as backup,
            ):
                source.backup(backup)
                backup.execute("PRAGMA journal_mode=DELETE")
            temporary.chmod(0o600)
            _verify_database(temporary)
            temporary.replace(destination)
            destination.chmod(0o600)
        except (OSError, sqlite3.Error, BackupError) as error:
            raise BackupError("Database backup creation or verification failed") from error
        finally:
            _remove_sqlite_temporary_files(temporary)
        return destination

    def _remove_expired(self, timestamp: datetime) -> int:
        cutoff = timestamp - timedelta(days=self.retention_days)
        removed = 0
        for path in self.backup_dir.iterdir():
            if (
                not path.is_file()
                or not path.name.startswith(BACKUP_PREFIX)
                or path.suffix != BACKUP_SUFFIX
            ):
                continue
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            if modified < cutoff:
                path.unlink()
                removed += 1
        return removed

    def _ensure_backup_dir(self) -> None:
        self.backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.backup_dir.chmod(0o700)


def _verify_database(path: Path) -> None:
    try:
        uri = f"file:{path.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            version = connection.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()
    except sqlite3.Error as error:
        raise BackupError("Restored database could not be read") from error
    if integrity is None or integrity[0] != "ok":
        raise BackupError("Restored database failed SQLite integrity verification")
    if version is None or int(version[0]) != SCHEMA_VERSION:
        raise BackupError("Restored database schema version does not match this application")


def _remove_sqlite_temporary_files(path: Path) -> None:
    for candidate in (
        path,
        path.with_name(f"{path.name}-wal"),
        path.with_name(f"{path.name}-shm"),
        path.with_name(f"{path.name}-journal"),
    ):
        candidate.unlink(missing_ok=True)

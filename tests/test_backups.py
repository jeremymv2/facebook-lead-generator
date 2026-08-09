import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lead_agent.backups import BackupError, DatabaseBackupService
from lead_agent.database import Database
from lead_agent.models import FacebookPost


def prepared_service(tmp_path: Path) -> tuple[Database, DatabaseBackupService]:
    database = Database(tmp_path / "data" / "lead-agent.sqlite3")
    database.initialize()
    database.save_post(
        FacebookPost(
            external_post_id="backup-fixture",
            group_id="fixture-group",
            group_name="Synthetic Group",
            post_text="Synthetic post retained in a verified database backup.",
        )
    )
    service = DatabaseBackupService(
        database.path,
        tmp_path / "data" / "backups",
        retention_days=14,
        interval_hours=24,
    )
    return database, service


def test_backup_is_private_verified_and_restore_test_is_disposable(tmp_path: Path) -> None:
    database, service = prepared_service(tmp_path)
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)

    summary = service.run(force=True, now=now)

    assert summary.created == 1
    assert summary.verified == 1
    assert summary.backup_path is not None
    assert summary.backup_path.stat().st_mode & 0o777 == 0o600
    assert summary.backup_path.parent.stat().st_mode & 0o777 == 0o700
    assert service.verify_restore(summary.backup_path) == summary.backup_path.resolve()
    assert not list(summary.backup_path.parent.glob(".restore-test.*"))
    restored = Database(summary.backup_path)
    assert len(restored.list_posts()) == len(database.list_posts()) == 1


def test_scheduled_backup_respects_interval_and_removes_only_expired_backups(
    tmp_path: Path,
) -> None:
    _, service = prepared_service(tmp_path)
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)
    first = service.run(force=True, now=now)
    assert first.backup_path is not None
    unrelated = service.backup_dir / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")

    skipped = service.run(now=now + timedelta(hours=1))

    assert skipped.created == 0
    old_time = (now - timedelta(days=20)).timestamp()
    os.utime(first.backup_path, (old_time, old_time))
    replaced = service.run(now=now + timedelta(days=1))
    assert replaced.created == 1
    assert replaced.removed == 1
    assert unrelated.exists()


def test_restore_test_rejects_corrupt_or_external_files(tmp_path: Path) -> None:
    _, service = prepared_service(tmp_path)
    service.backup_dir.mkdir(parents=True, exist_ok=True)
    corrupt = service.backup_dir / "lead-agent-corrupt.sqlite3"
    corrupt.write_text("not sqlite", encoding="utf-8")
    outside = tmp_path / "lead-agent-outside.sqlite3"
    outside.write_text("not sqlite", encoding="utf-8")

    with pytest.raises(BackupError):
        service.verify_restore(corrupt)
    with pytest.raises(BackupError, match="configured private backup directory"):
        service.verify_restore(outside)

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lead_agent.dashboard_metrics import DashboardMetricsService
from lead_agent.database import Database
from lead_agent.models import AuditEvent


def test_snapshot_reconstructs_legacy_and_rich_cycle_history(tmp_path: Path) -> None:
    database = Database(tmp_path / "trends.sqlite3")
    database.initialize()
    now = datetime(2026, 8, 8, 12, tzinfo=UTC)
    database.record_audit_event(
        AuditEvent(
            component="classifier",
            action="lead.scored",
            result="ignored",
            occurred_at=now - timedelta(minutes=10),
        )
    )
    database.record_audit_event(
        AuditEvent(
            component="operations",
            action="cycle.run",
            result="success",
            occurred_at=now,
            details={
                "groups_scanned": 8,
                "posts_seen": 80,
                "posts_new": 5,
                "posts_classified": 5,
                "candidates_created": 1,
            },
        )
    )
    database.record_audit_event(
        AuditEvent(
            component="operations",
            action="cycle.run",
            result="degraded",
            occurred_at=now + timedelta(minutes=15),
            details={
                "groups_scanned": 5,
                "groups_failed": 3,
                "groups_partial": 2,
                "groups_retried": 4,
                "groups_recovered": 1,
                "posts_seen": 39,
                "posts_new": 5,
                "duplicates": 34,
                "posts_classified": 5,
                "candidates_created": 0,
                "posts_ignored": 5,
            },
        )
    )

    snapshot = DashboardMetricsService(database).snapshot()

    assert [cycle.status for cycle in snapshot.cycles] == ["success", "degraded"]
    assert snapshot.cycles[0].duplicates == 75
    assert snapshot.cycles[0].posts_ignored == 4
    assert snapshot.cycles[1].groups_retried == 4
    assert snapshot.groups_scanned == 13
    assert snapshot.groups_failed == 3
    assert snapshot.groups_partial == 2
    assert snapshot.group_success_percent == 68.8
    assert snapshot.posts_seen == 119
    assert snapshot.posts_new == 10
    assert snapshot.candidates_created == 1
    assert snapshot.retries == 4
    assert snapshot.recoveries == 1


def test_snapshot_is_bounded_to_newest_cycles_and_sanitizes_counters(tmp_path: Path) -> None:
    database = Database(tmp_path / "bounded.sqlite3")
    database.initialize()
    now = datetime(2026, 8, 8, 12, tzinfo=UTC)
    for index in range(3):
        database.record_audit_event(
            AuditEvent(
                component="operations",
                action="cycle.run",
                result="unexpected" if index == 2 else "success",
                occurred_at=now + timedelta(minutes=index),
                details={"posts_seen": -1 if index == 2 else index},
            )
        )

    snapshot = DashboardMetricsService(database, history_limit=2).snapshot()

    assert [cycle.occurred_at for cycle in snapshot.cycles] == [
        now + timedelta(minutes=1),
        now + timedelta(minutes=2),
    ]
    assert snapshot.cycles[-1].status == "unknown"
    assert snapshot.cycles[-1].posts_seen == 0


def test_history_limits_must_be_positive(tmp_path: Path) -> None:
    database = Database(tmp_path / "invalid.sqlite3")
    database.initialize()

    with pytest.raises(ValueError, match="positive"):
        DashboardMetricsService(database, history_limit=0)
    with pytest.raises(ValueError, match="positive"):
        database.list_audit_events(limit=0)

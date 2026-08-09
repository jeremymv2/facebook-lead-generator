import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from lead_agent.classifier import ClassificationSummary
from lead_agent.database import Database
from lead_agent.models import FacebookPost, Lead, LeadIntent, LeadStatus
from lead_agent.notifications import NotificationSummary
from lead_agent.operations import (
    CycleAlreadyRunningError,
    CycleLock,
    OperationPaths,
    OperationsCycleRunner,
    OperationsState,
    RetentionService,
    RetentionSummary,
    ScanCycleSummary,
)


def operation_paths(tmp_path: Path) -> OperationPaths:
    return OperationPaths(
        state_dir=tmp_path / "state",
        log_dir=tmp_path / "logs",
        screenshot_dir=tmp_path / "screenshots",
    )


def test_cycle_lock_rejects_an_overlapping_process(tmp_path: Path) -> None:
    path = operation_paths(tmp_path).lock_path

    with (
        CycleLock(path),
        pytest.raises(CycleAlreadyRunningError, match="already running"),
        CycleLock(path),
    ):
        raise AssertionError("second lock must never be acquired")

    with CycleLock(path):
        assert path.read_text(encoding="utf-8").strip().isdigit()


def test_pause_resume_and_stale_health_are_content_free(tmp_path: Path) -> None:
    paths = operation_paths(tmp_path)
    state = OperationsState(paths)
    now = datetime(2026, 8, 8, 12, tzinfo=UTC)

    state.pause(now=now)
    state.mark_paused(now=now)
    payload = state.status_payload(
        stale_after_seconds=300,
        now=now + timedelta(minutes=6),
    )

    assert payload["status"] == "paused"
    assert payload["paused"] is True
    assert payload["stale"] is True
    assert "post_text" not in paths.health_path.read_text(encoding="utf-8")
    assert state.resume() is True
    assert state.resume() is False
    assert state.paused is False


def test_cycle_runner_records_success_and_safe_failure_type(tmp_path: Path) -> None:
    state = OperationsState(operation_paths(tmp_path))
    times = iter(
        [
            datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
            datetime(2026, 8, 8, 12, 1, tzinfo=UTC),
        ]
    )
    runner = OperationsCycleRunner(state)

    result = runner.run(
        scan=lambda: ScanCycleSummary(2, 0, 20, 3, 17),
        classify=lambda: ClassificationSummary(3, (), ()),
        notify=lambda: NotificationSummary(considered=1, sent=1, failed=0),
        retain=lambda: RetentionSummary(screenshots_removed=2),
        now=lambda: next(times),
    )

    assert result is not None
    assert result.status == "success"
    assert result.scan.posts_new == 3
    health = state.read_health()
    assert health["status"] == "success"
    assert health["consecutive_failures"] == 0

    failure_times = iter(
        [
            datetime(2026, 8, 8, 12, 2, tzinfo=UTC),
            datetime(2026, 8, 8, 12, 3, tzinfo=UTC),
        ]
    )

    with pytest.raises(RuntimeError, match="private customer text"):
        runner.run(
            scan=lambda: (_ for _ in ()).throw(RuntimeError("private customer text")),
            classify=lambda: ClassificationSummary(0, (), ()),
            notify=None,
            retain=RetentionSummary,
            now=lambda: next(failure_times),
        )

    failed_health = state.read_health()
    assert failed_health["status"] == "failed"
    assert failed_health["last_error"] == "RuntimeError"
    assert "private customer text" not in paths_text(state.paths.health_path)


def test_repeated_incomplete_cycles_trip_pause_circuit_breaker(tmp_path: Path) -> None:
    state = OperationsState(operation_paths(tmp_path))
    runner = OperationsCycleRunner(
        state,
        degraded_cycle_limit=2,
        incomplete_group_rate_threshold=0.25,
    )
    times = iter(datetime(2026, 8, 8, 12, minute, tzinfo=UTC) for minute in range(4))

    first = runner.run(
        scan=lambda: ScanCycleSummary(
            groups_scanned=7,
            groups_failed=1,
            posts_seen=60,
            posts_new=2,
            duplicates=58,
            groups_partial=1,
            posts_requested=80,
        ),
        classify=lambda: ClassificationSummary(2, (), ()),
        notify=None,
        retain=RetentionSummary,
        now=lambda: next(times),
    )
    second = runner.run(
        scan=lambda: ScanCycleSummary(
            groups_scanned=6,
            groups_failed=2,
            posts_seen=50,
            posts_new=1,
            duplicates=49,
            posts_requested=80,
        ),
        classify=lambda: ClassificationSummary(1, (), ()),
        notify=None,
        retain=RetentionSummary,
        now=lambda: next(times),
    )

    assert first is not None
    assert first.status == "degraded"
    assert first.circuit_breaker_tripped is False
    assert second is not None
    assert second.circuit_breaker_tripped is True
    assert state.paused is True
    health = state.read_health()
    assert health["consecutive_degraded_cycles"] == 2
    assert health["circuit_breaker_reason"] == "incomplete_group_rate"
    summary = cast(dict[str, object], health["summary"])
    scan = cast(dict[str, object], summary["scan"])
    assert scan["groups_partial"] == 0


def test_repeated_fatal_cycles_trip_pause_circuit_breaker(tmp_path: Path) -> None:
    state = OperationsState(operation_paths(tmp_path))
    runner = OperationsCycleRunner(state, degraded_cycle_limit=2)
    times = iter(datetime(2026, 8, 8, 12, minute, tzinfo=UTC) for minute in range(4))

    for _ in range(2):
        with pytest.raises(RuntimeError, match="private failure details"):
            runner.run(
                scan=lambda: (_ for _ in ()).throw(RuntimeError("private failure details")),
                classify=lambda: ClassificationSummary(0, (), ()),
                notify=None,
                retain=RetentionSummary,
                now=lambda: next(times),
            )

    health = state.read_health()
    assert state.paused is True
    assert health["consecutive_failures"] == 2
    assert health["circuit_breaker_tripped"] is True
    assert health["circuit_breaker_reason"] == "consecutive_cycle_failures"
    assert "private failure details" not in paths_text(state.paths.health_path)


def test_retention_removes_only_expired_known_artifacts_and_rotates_logs(
    tmp_path: Path,
) -> None:
    paths = operation_paths(tmp_path)
    paths.screenshot_dir.mkdir()
    paths.log_dir.mkdir()
    old_screenshot = paths.screenshot_dir / "old.png"
    recent_screenshot = paths.screenshot_dir / "recent.png"
    unrelated = paths.screenshot_dir / "notes.txt"
    keep = paths.screenshot_dir / ".gitkeep"
    old_log = paths.log_dir / "old.log"
    large_log = paths.log_dir / "cycle.stdout.log"
    for path in (old_screenshot, recent_screenshot, unrelated, keep, old_log):
        path.write_text("fixture", encoding="utf-8")
    large_log.write_text("x" * 100, encoding="utf-8")
    old_time = datetime(2026, 7, 1, tzinfo=UTC).timestamp()
    for path in (old_screenshot, unrelated, keep, old_log):
        os.utime(path, (old_time, old_time))
    now = datetime(2026, 8, 8, tzinfo=UTC)
    service = RetentionService(
        paths,
        screenshot_retention_days=14,
        log_retention_days=14,
        log_max_bytes=50,
    )

    result = service.cleanup(now=now)

    assert result.screenshots_removed == 1
    assert result.logs_removed == 1
    assert result.logs_rotated == 1
    assert not old_screenshot.exists()
    assert recent_screenshot.exists()
    assert unrelated.exists()
    assert keep.exists()
    assert not large_log.exists()
    assert len(list(paths.log_dir.glob("cycle.stdout.*.log"))) == 1


def test_candidate_review_deduplicates_exact_text_across_groups_within_window(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "dedupe.sqlite3")
    database.initialize()
    now = datetime(2026, 8, 8, 12, tzinfo=UTC)
    first = save_candidate(database, "first", "group-a", now, score=80)
    save_candidate(database, "second", "group-a", now + timedelta(hours=1), score=95)
    save_candidate(
        database,
        "third",
        "group-b",
        now + timedelta(hours=1),
        score=90,
    )
    late_repost = save_candidate(
        database,
        "fourth",
        "group-a",
        now + timedelta(hours=74),
        score=85,
    )

    reviewable = database.list_candidate_leads(limit=10, duplicate_window_hours=72)

    assert {lead.id for lead in reviewable} == {
        first.id,
        late_repost.id,
    }


def test_candidate_review_can_require_the_current_classification_version(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "versions.sqlite3")
    database.initialize()
    now = datetime(2026, 8, 8, 12, tzinfo=UTC)
    stale = save_candidate(
        database,
        "stale",
        "group-a",
        now,
        score=90,
        classification_version="fixture-v3",
        post_text="Need a deck repair estimate.",
    )
    current = save_candidate(
        database,
        "current",
        "group-a",
        now,
        score=85,
        classification_version="fixture-v4",
        post_text="Need a drywall repair estimate.",
    )

    reviewable = database.list_candidate_leads(
        limit=10,
        classification_version="fixture-v4",
    )

    assert [lead.id for lead in reviewable] == [current.id]
    assert stale.id not in {lead.id for lead in reviewable}


def test_group_quality_reports_yield_ads_and_exact_duplicates_without_content(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "quality.sqlite3")
    database.initialize()
    now = datetime(2026, 8, 8, 12, tzinfo=UTC)
    save_candidate(database, "first", "group-a", now, score=90)
    advertisement = database.save_post(
        FacebookPost(
            external_post_id="ad",
            group_id="group-a",
            group_name="Group A",
            post_text="We offer deck work. Call us for a quote.",
            discovered_at=now + timedelta(minutes=1),
        )
    ).post
    database.create_lead(
        Lead(
            facebook_post_id=advertisement.id or 0,
            status=LeadStatus.IGNORED,
            intent=LeadIntent.COMPETITOR_ADVERTISEMENT,
            overall_score=10,
        )
    )
    database.save_post(
        FacebookPost(
            external_post_id="duplicate",
            group_id="group-a",
            group_name="Group A",
            post_text="Need someone for the same deck repair.",
            discovered_at=now + timedelta(minutes=2),
        )
    )
    database.save_post(
        FacebookPost(
            external_post_id="cross-group-repost",
            group_id="group-b",
            group_name="Group B",
            post_text="Need someone for the same deck repair.",
            discovered_at=now + timedelta(minutes=3),
        )
    )

    quality = database.list_group_quality()[0]

    assert quality.group_id == "group-a"
    assert quality.posts_discovered == 3
    assert quality.posts_classified == 2
    assert quality.candidates_created == 1
    assert quality.provider_advertisements == 1
    assert quality.exact_text_duplicates == 1
    assert quality.cross_group_reposts == 2
    assert quality.candidate_yield_percent == 50.0


def save_candidate(
    database: Database,
    external_id: str,
    group_id: str,
    discovered_at: datetime,
    *,
    score: int,
    classification_version: str | None = None,
    post_text: str = "Need someone for the same deck repair.",
) -> Lead:
    post = database.save_post(
        FacebookPost(
            external_post_id=external_id,
            group_id=group_id,
            group_name=group_id,
            post_text=post_text,
            discovered_at=discovered_at,
        )
    ).post
    return database.create_lead(
        Lead(
            facebook_post_id=post.id or 0,
            status=LeadStatus.CANDIDATE,
            service_category="decks",
            intent=LeadIntent.HIRING,
            overall_score=score,
            drafted_response="Fixture draft",
            classification_version=classification_version,
            created_at=discovered_at,
            updated_at=discovered_at,
        )
    )


def paths_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

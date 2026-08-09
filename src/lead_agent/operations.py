"""Safe local runtime controls for unattended scan and review cycles."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import TextIO

from lead_agent.classifier import ClassificationSummary
from lead_agent.notifications import NotificationSummary

HEALTH_SCHEMA_VERSION = 1
MAX_HEALTH_FILE_BYTES = 65_536
SCREENSHOT_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}
LOG_SUFFIXES = {".jsonl", ".log"}


class OperationsError(RuntimeError):
    """Base error for safe local operations."""


class CycleAlreadyRunningError(OperationsError):
    """Raised when a second cycle tries to overlap an active cycle."""


@dataclass(frozen=True, slots=True)
class OperationPaths:
    """Non-secret files used by the local operations process."""

    state_dir: Path
    log_dir: Path
    screenshot_dir: Path

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "cycle.lock"

    @property
    def pause_path(self) -> Path:
        return self.state_dir / "PAUSED"

    @property
    def health_path(self) -> Path:
        return self.state_dir / "health.json"


@dataclass(frozen=True, slots=True)
class ScanCycleSummary:
    """Aggregate scan counts that are safe to persist in operational health."""

    groups_scanned: int
    groups_failed: int
    posts_seen: int
    posts_new: int
    duplicates: int


@dataclass(frozen=True, slots=True)
class RetentionSummary:
    """Counts of expired local artifacts removed or rotated."""

    screenshots_removed: int = 0
    logs_removed: int = 0
    logs_rotated: int = 0


@dataclass(frozen=True, slots=True)
class CycleSummary:
    """One complete scan, classify, notification, and retention outcome."""

    status: str
    scan: ScanCycleSummary
    posts_classified: int
    candidates_created: int
    posts_ignored: int
    notifications_considered: int
    notifications_sent: int
    notifications_failed: int
    retention: RetentionSummary


class CycleLock:
    """Non-blocking process lock that prevents overlapping browser cycles."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: TextIO | None = None

    def __enter__(self) -> CycleLock:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        handle = self.path.open("a+", encoding="utf-8")
        self.path.chmod(0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise CycleAlreadyRunningError("A lead-agent cycle is already running") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._handle is None:
            return
        handle = self._handle
        self._handle = None
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class OperationsState:
    """Pause control plus atomic, content-free health snapshots."""

    def __init__(self, paths: OperationPaths) -> None:
        self.paths = paths

    def pause(self, *, now: datetime | None = None) -> None:
        timestamp = now or datetime.now(UTC)
        self._ensure_state_dir()
        _atomic_write_json(
            self.paths.pause_path,
            {"paused_at": timestamp.isoformat(), "schema_version": HEALTH_SCHEMA_VERSION},
        )

    def resume(self) -> bool:
        try:
            self.paths.pause_path.unlink()
        except FileNotFoundError:
            return False
        return True

    @property
    def paused(self) -> bool:
        return self.paths.pause_path.is_file()

    def read_health(self) -> dict[str, object]:
        if not self.paths.health_path.exists():
            return {"schema_version": HEALTH_SCHEMA_VERSION, "status": "never_run"}
        if self.paths.health_path.stat().st_size > MAX_HEALTH_FILE_BYTES:
            return {"schema_version": HEALTH_SCHEMA_VERSION, "status": "invalid"}
        try:
            payload = json.loads(self.paths.health_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {"schema_version": HEALTH_SCHEMA_VERSION, "status": "invalid"}
        if not isinstance(payload, dict):
            return {"schema_version": HEALTH_SCHEMA_VERSION, "status": "invalid"}
        return {str(key): value for key, value in payload.items()}

    def status_payload(
        self,
        *,
        stale_after_seconds: int,
        now: datetime | None = None,
    ) -> dict[str, object]:
        timestamp = now or datetime.now(UTC)
        payload = self.read_health()
        updated_at = _parse_health_timestamp(payload.get("updated_at"))
        stale = updated_at is not None and timestamp - updated_at > timedelta(
            seconds=stale_after_seconds
        )
        return {**payload, "paused": self.paused, "stale": stale}

    def mark_paused(self, *, now: datetime | None = None) -> None:
        timestamp = now or datetime.now(UTC)
        previous = self.read_health()
        self._write_health(
            {
                **_preserved_health(previous),
                "status": "paused",
                "updated_at": timestamp.isoformat(),
            }
        )

    def mark_running(self, *, started_at: datetime) -> None:
        previous = self.read_health()
        self._write_health(
            {
                **_preserved_health(previous),
                "status": "running",
                "cycle_started_at": started_at.isoformat(),
                "updated_at": started_at.isoformat(),
            }
        )

    def mark_completed(
        self,
        summary: CycleSummary,
        *,
        started_at: datetime,
        completed_at: datetime,
    ) -> None:
        previous = self.read_health()
        self._write_health(
            {
                **_preserved_health(previous),
                "status": summary.status,
                "cycle_started_at": started_at.isoformat(),
                "cycle_completed_at": completed_at.isoformat(),
                "updated_at": completed_at.isoformat(),
                "last_success_at": completed_at.isoformat(),
                "consecutive_failures": 0,
                "summary": asdict(summary),
                "last_error": None,
            }
        )

    def mark_failed(
        self,
        error: Exception,
        *,
        started_at: datetime,
        failed_at: datetime,
    ) -> None:
        previous = self.read_health()
        previous_failures = previous.get("consecutive_failures", 0)
        failure_count = previous_failures if isinstance(previous_failures, int) else 0
        self._write_health(
            {
                **_preserved_health(previous),
                "status": "failed",
                "cycle_started_at": started_at.isoformat(),
                "cycle_completed_at": failed_at.isoformat(),
                "updated_at": failed_at.isoformat(),
                "consecutive_failures": failure_count + 1,
                "last_error": type(error).__name__,
            }
        )

    def _ensure_state_dir(self) -> None:
        self.paths.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.paths.state_dir.chmod(0o700)

    def _write_health(self, payload: dict[str, object]) -> None:
        self._ensure_state_dir()
        _atomic_write_json(
            self.paths.health_path,
            {"schema_version": HEALTH_SCHEMA_VERSION, **payload},
        )


class RetentionService:
    """Delete only expired known artifacts and rotate oversized operation logs."""

    def __init__(
        self,
        paths: OperationPaths,
        *,
        screenshot_retention_days: int,
        log_retention_days: int,
        log_max_bytes: int,
    ) -> None:
        self.paths = paths
        self.screenshot_retention_days = screenshot_retention_days
        self.log_retention_days = log_retention_days
        self.log_max_bytes = log_max_bytes

    def cleanup(self, *, now: datetime | None = None) -> RetentionSummary:
        timestamp = now or datetime.now(UTC)
        screenshots_removed = _remove_expired_files(
            self.paths.screenshot_dir,
            suffixes=SCREENSHOT_SUFFIXES,
            older_than=timestamp - timedelta(days=self.screenshot_retention_days),
        )
        logs_removed = _remove_expired_files(
            self.paths.log_dir,
            suffixes=LOG_SUFFIXES,
            older_than=timestamp - timedelta(days=self.log_retention_days),
        )
        logs_rotated = _rotate_large_logs(
            self.paths.log_dir,
            max_bytes=self.log_max_bytes,
            now=timestamp,
        )
        return RetentionSummary(
            screenshots_removed=screenshots_removed,
            logs_removed=logs_removed,
            logs_rotated=logs_rotated,
        )


class OperationsCycleRunner:
    """Run one non-overlapping, pausable, health-tracked local cycle."""

    def __init__(self, state: OperationsState) -> None:
        self.state = state

    def run(
        self,
        *,
        scan: Callable[[], ScanCycleSummary],
        classify: Callable[[], ClassificationSummary],
        notify: Callable[[], NotificationSummary] | None,
        retain: Callable[[], RetentionSummary],
        now: Callable[[], datetime] | None = None,
    ) -> CycleSummary | None:
        clock = now or (lambda: datetime.now(UTC))
        with CycleLock(self.state.paths.lock_path):
            if self.state.paused:
                self.state.mark_paused(now=clock())
                return None
            started_at = clock()
            self.state.mark_running(started_at=started_at)
            try:
                scan_summary = scan()
                classification = classify()
                notification = (
                    notify()
                    if notify is not None
                    else NotificationSummary(considered=0, sent=0, failed=0)
                )
                retention = retain()
                status = (
                    "degraded" if scan_summary.groups_failed or notification.failed else "success"
                )
                summary = CycleSummary(
                    status=status,
                    scan=scan_summary,
                    posts_classified=classification.leads_created,
                    candidates_created=len(classification.candidates),
                    posts_ignored=len(classification.ignored),
                    notifications_considered=notification.considered,
                    notifications_sent=notification.sent,
                    notifications_failed=notification.failed,
                    retention=retention,
                )
            except Exception as error:
                self.state.mark_failed(error, started_at=started_at, failed_at=clock())
                raise
            self.state.mark_completed(
                summary,
                started_at=started_at,
                completed_at=clock(),
            )
            return summary


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        temporary.write_text(data, encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _preserved_health(payload: dict[str, object]) -> dict[str, object]:
    keys = ("last_success_at", "consecutive_failures")
    return {key: payload[key] for key in keys if key in payload}


def _parse_health_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _remove_expired_files(
    directory: Path,
    *,
    suffixes: set[str],
    older_than: datetime,
) -> int:
    if not directory.is_dir():
        return 0
    removed = 0
    cutoff = older_than.timestamp()
    for path in directory.rglob("*"):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.name == ".gitkeep"
            or path.suffix.casefold() not in suffixes
        ):
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def _rotate_large_logs(directory: Path, *, max_bytes: int, now: datetime) -> int:
    if not directory.is_dir():
        return 0
    rotated = 0
    timestamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    for path in directory.glob("*.log"):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= max_bytes:
            continue
        destination = path.with_name(f"{path.stem}.{timestamp}.{os.getpid()}.log")
        path.replace(destination)
        rotated += 1
    return rotated

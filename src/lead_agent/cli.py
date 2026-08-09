"""Local command-line interface for database and read-only Facebook discovery."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from lead_agent.ai import (
    CLASSIFICATION_VERSION,
    AIProviderError,
    build_ai_provider,
    classification_context,
)
from lead_agent.approval_web import run_local_approval_dashboard
from lead_agent.approvals import ApprovalError, LocalApprovalService
from lead_agent.backups import BackupError, DatabaseBackupService
from lead_agent.classifier import ClassificationSummary, LeadClassificationService
from lead_agent.config import (
    NotificationConfigurationError,
    PostingDisabledError,
    Settings,
    UnsafeReadOnlyModeError,
    load_settings,
)
from lead_agent.database import Database
from lead_agent.facebook import FacebookBrowserError, FacebookReadOnlyBrowser
from lead_agent.facebook_posting import FacebookCommentBrowser
from lead_agent.facebook_state import FacebookSafetyStop
from lead_agent.feedback import export_regression_fixtures
from lead_agent.groups import FacebookGroup, GroupsConfigError, load_group_catalog
from lead_agent.logging_config import configure_logging
from lead_agent.models import AuditEvent, GroupScanState
from lead_agent.notifications import (
    ApprovalNotificationService,
    NotificationSummary,
    build_sms_provider,
)
from lead_agent.operations import (
    CycleAlreadyRunningError,
    CycleSummary,
    OperationPaths,
    OperationsCycleRunner,
    OperationsState,
    QuietHoursActiveError,
    RetentionService,
    RetentionSummary,
    ScanCycleSummary,
    is_severe_post_shortfall,
    is_within_quiet_hours,
)
from lead_agent.posting import ApprovedPostingService, PostingError, PostingExecutionResult
from lead_agent.remote_approval_web import (
    RemoteApprovalController,
    relay_is_healthy,
    run_remote_approval_server,
)
from lead_agent.scanner import (
    ReadOnlyScanService,
    ScanSummary,
    TransientFacebookReadError,
    safe_scan_error_code,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lead-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="Create or upgrade the local SQLite database")
    subparsers.add_parser("doctor", help="Validate configuration and show safety state")
    status_parser = subparsers.add_parser(
        "scan-status",
        help="Show persisted per-group scan health without post content",
    )
    status_parser.add_argument("--group-id", help="Show health for one previously scanned group")
    subparsers.add_parser(
        "operations-status",
        help="Show content-free health for the unattended local workflow",
    )
    subparsers.add_parser(
        "operations-pause",
        help="Pause future unattended cycles without stopping approval servers",
    )
    subparsers.add_parser(
        "operations-resume",
        help="Resume unattended cycles",
    )
    subparsers.add_parser(
        "group-report",
        help="Show content-free per-group lead yield and noise metrics",
    )
    classify_parser = subparsers.add_parser(
        "classify-posts",
        help="Classify unprocessed posts and draft candidate replies without using Facebook",
    )
    classify_parser.add_argument(
        "--post-id",
        type=_positive_int,
        help="Classify one saved post ID instead of the newest unclassified posts",
    )
    classify_parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Maximum posts to classify (defaults to AI_MAX_POSTS_PER_RUN)",
    )
    reclassify_parser = subparsers.add_parser(
        "reclassify-leads",
        help="Safely update stale unreviewed classifications without touching reviewed leads",
    )
    reclassify_parser.add_argument(
        "--lead-id",
        type=_positive_int,
        help="Reclassify one unreviewed lead, including a lead already on the current version",
    )
    reclassify_parser.add_argument("--limit", type=_positive_int, help="Maximum leads to update")
    replay_parser = subparsers.add_parser(
        "classification-replay",
        help="Compare current classifier rules with saved results without changing state",
    )
    replay_parser.add_argument("--lead-id", type=_positive_int, help="Replay one saved lead")
    replay_parser.add_argument("--limit", type=_positive_int, help="Maximum saved leads to replay")
    replay_parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Print only results whose status, service, intent, or score would change",
    )
    export_parser = subparsers.add_parser(
        "export-regression-fixtures",
        help="Export sanitized classifier fixtures from structured human rejection feedback",
    )
    export_parser.add_argument("--lead-id", type=_positive_int, help="Export one rejected lead")
    export_parser.add_argument("--limit", type=_positive_int, help="Maximum feedback records")
    export_parser.add_argument(
        "--output",
        type=Path,
        help="Private JSON output path (defaults under DATA_DIR)",
    )
    subparsers.add_parser(
        "database-backup",
        help="Create and verify a private SQLite backup immediately",
    )
    restore_parser = subparsers.add_parser(
        "database-restore-test",
        help="Restore a private backup into a disposable database and verify integrity",
    )
    restore_parser.add_argument(
        "--backup-path",
        type=Path,
        help="Backup inside DATABASE_BACKUP_DIR (defaults to latest)",
    )
    approval_parser = subparsers.add_parser(
        "approval-dashboard",
        help="Review candidate drafts on a loopback-only local web page",
    )
    approval_parser.add_argument(
        "--port",
        type=_local_port,
        help="Local loopback port (defaults to APPROVAL_LOCAL_PORT)",
    )
    approval_parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Maximum new candidates to prepare for review",
    )
    remote_parser = subparsers.add_parser(
        "remote-approval",
        help="Send Telnyx alerts and serve tokenized reviews through a secure relay",
    )
    remote_parser.add_argument(
        "--port",
        type=_local_port,
        help="Loopback origin port (defaults to REMOTE_APPROVAL_PORT)",
    )
    remote_parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Maximum candidates to prepare per notification cycle",
    )
    remote_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry earlier failed SMS attempts once at startup using new review links",
    )
    subparsers.add_parser(
        "facebook-login",
        help="Open the dedicated browser profile for a manual Facebook login",
    )
    scan_parser = subparsers.add_parser(
        "scan-facebook",
        help="Read visible posts from explicitly enabled Facebook groups",
    )
    scan_parser.add_argument(
        "--group-id",
        help="Scan one enabled group ID instead of every enabled group",
    )
    scan_parser.add_argument(
        "--max-posts",
        type=_positive_int,
        help="Maximum visible posts per group (defaults to MAX_POSTS_PER_GROUP)",
    )
    scan_parser.add_argument(
        "--pause-after-scan",
        action="store_true",
        help="Keep the browser open for inspection until Enter is pressed",
    )
    cycle_parser = subparsers.add_parser(
        "run-cycle",
        help="Run one locked scan, classify, notify, and retention cycle",
    )
    cycle_parser.add_argument(
        "--max-posts",
        type=_positive_int,
        help="Maximum visible posts per group (defaults to MAX_POSTS_PER_GROUP)",
    )
    cycle_parser.add_argument(
        "--classification-limit",
        type=_positive_int,
        help="Maximum posts to classify in this cycle",
    )
    cycle_parser.add_argument(
        "--skip-notifications",
        action="store_true",
        help="Never send SMS in this manually invoked cycle",
    )
    cycle_parser.add_argument(
        "--ignore-quiet-hours",
        action="store_true",
        help="Explicitly allow this manual cycle during configured quiet hours",
    )
    posting_parser = subparsers.add_parser(
        "post-approved",
        help="Validate one approved lead; submit only with --submit and both safety flags",
    )
    posting_parser.add_argument(
        "--lead-id",
        type=_positive_int,
        required=True,
        help="Approved or edited lead to validate against its exact Facebook post",
    )
    posting_parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit once; requires POSTING_ENABLED=true and DRY_RUN=false",
    )
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _local_port(value: str) -> int:
    parsed = int(value)
    if not 1024 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 1024 and 65535")
    return parsed


def _doctor_payload(settings: Settings) -> dict[str, object]:
    return {
        "database_path": str(settings.database_path),
        "dry_run": settings.dry_run,
        "posting_enabled": settings.posting_enabled,
        "posting_allowed": settings.posting_allowed,
        "read_only_mode_ready": not settings.posting_enabled and settings.dry_run,
        "service_area": settings.service_area,
        "lead_threshold": settings.lead_threshold,
        "approval_expiration_minutes": settings.approval_expiration_minutes,
        "approval_local_url": f"http://127.0.0.1:{settings.approval_local_port}",
        "facebook_profile_path": str(settings.facebook_profile_path),
        "browser_headless": settings.browser_headless,
        "groups_config_path": str(settings.groups_config_path),
        "ai_provider": settings.ai_provider,
        "ai_model": settings.ai_model,
        "ai_ready": settings.ai_provider == "heuristic"
        or (
            settings.ai_provider == "gemini"
            and settings.gemini_api_key is not None
            and bool(settings.ai_model)
        ),
        "notifications_enabled": settings.notifications_enabled,
        "sms_provider": settings.sms_provider,
        "remote_approval_ready": settings.remote_approval_ready,
        "remote_approval_port": settings.remote_approval_port,
        "posting_approval_max_age_minutes": settings.posting_approval_max_age_minutes,
        "daily_posting_limit": settings.daily_posting_limit,
        "per_group_daily_posting_limit": settings.per_group_daily_posting_limit,
        "business_timezone": settings.business_timezone,
        "scan_interval_seconds": settings.scan_interval_seconds,
        "facebook_group_max_retries": settings.facebook_group_max_retries,
        "facebook_group_retry_backoff_seconds": settings.facebook_group_retry_backoff_seconds,
        "facebook_group_delay_seconds": settings.facebook_group_delay_seconds,
        "max_posts_per_group": settings.max_posts_per_group,
        "operations_degraded_cycle_limit": settings.operations_degraded_cycle_limit,
        "operations_incomplete_group_rate_threshold": (
            settings.operations_incomplete_group_rate_threshold
        ),
        "operations_minimum_group_post_yield_rate": (
            settings.operations_minimum_group_post_yield_rate
        ),
        "cycle_classification_limit": settings.cycle_classification_limit,
        "candidate_duplicate_window_hours": settings.candidate_duplicate_window_hours,
        "operations_state_dir": str(settings.operations_state_dir),
        "operations_log_dir": str(settings.operations_log_dir),
        "database_backup_dir": str(settings.database_backup_dir),
        "database_backup_retention_days": settings.database_backup_retention_days,
        "database_backup_interval_hours": settings.database_backup_interval_hours,
        "operations_quiet_hours_enabled": settings.operations_quiet_hours_enabled,
        "operations_quiet_hours_active": _quiet_hours_active(settings),
        "operations_quiet_hours_start": settings.operations_quiet_hours_start.strftime("%H:%M"),
        "operations_quiet_hours_end": settings.operations_quiet_hours_end.strftime("%H:%M"),
    }


def _quiet_hours_active(settings: Settings, *, now: datetime | None = None) -> bool:
    if not settings.operations_quiet_hours_enabled:
        return False
    return is_within_quiet_hours(
        now or datetime.now(UTC),
        timezone=settings.business_timezone,
        start=settings.operations_quiet_hours_start,
        end=settings.operations_quiet_hours_end,
    )


def _scan_state_payload(state: GroupScanState) -> dict[str, object]:
    health = (
        "degraded"
        if state.last_error is not None
        else "partial"
        if state.last_scan_partial
        else "healthy"
    )
    return {
        "group_id": state.group_id,
        "group_name": state.group_name,
        "health": health,
        "consecutive_failures": state.consecutive_failures,
        "last_attempt_at": state.last_attempt_at.isoformat(),
        "last_success_at": state.last_success_at.isoformat() if state.last_success_at else None,
        "last_failure_at": state.last_failure_at.isoformat() if state.last_failure_at else None,
        "last_error": state.last_error,
        "posts_seen": state.posts_seen,
        "posts_new": state.posts_new,
        "posts_requested": state.posts_requested,
        "last_scan_partial": state.last_scan_partial,
    }


def _operation_paths(settings: Settings) -> OperationPaths:
    return OperationPaths(
        state_dir=settings.operations_state_dir,
        log_dir=settings.operations_log_dir,
        screenshot_dir=settings.screenshot_dir,
    )


async def _manual_login(settings: Settings) -> None:
    settings.require_read_only_mode()
    async with FacebookReadOnlyBrowser(settings) as browser:
        await browser.manual_login()


async def _scan_groups(
    settings: Settings,
    groups: Sequence[FacebookGroup],
    *,
    max_posts: int,
    pause_after_scan: bool,
) -> list[ScanSummary]:
    settings.require_read_only_mode()
    database = Database(settings.database_path)
    database.initialize()
    summaries: list[ScanSummary] = []
    async with FacebookReadOnlyBrowser(settings) as browser:
        try:
            scanner = ReadOnlyScanService(database, browser)
            for group in groups:
                summaries.append(await scanner.scan_group(group, max_posts=max_posts))
        finally:
            if pause_after_scan:
                print("Browser paused for inspection. No Facebook actions will be taken.")
                with suppress(EOFError):
                    await asyncio.to_thread(input, "Press Enter to close the browser... ")
    return summaries


def _print_scan_results(summaries: Sequence[ScanSummary]) -> None:
    for summary in summaries:
        counts = (
            f"[{summary.group.name}] seen={summary.posts_seen} "
            f"new={len(summary.new_posts)} duplicates={summary.duplicates}"
        )
        if summary.posts_requested:
            counts += f" requested={summary.posts_requested}"
        if summary.partial:
            counts += " PARTIAL"
        print(counts)
        for post in summary.new_posts:
            print(f"NEW {post.post_url or '(no permalink)'}")
            print(post.post_text[:500])


def _print_classification_results(summary: ClassificationSummary) -> None:
    print(
        f"considered={summary.posts_considered} classified={summary.leads_created} "
        f"candidates={len(summary.candidates)} ignored={len(summary.ignored)}"
    )
    for lead in summary.candidates:
        print(
            f"CANDIDATE lead={lead.id} post={lead.facebook_post_id} "
            f"score={lead.overall_score} service={lead.service_category or 'unknown'}"
        )
        if lead.drafted_response:
            print(f"DRAFT {lead.drafted_response}")


async def _scan_groups_for_cycle(
    settings: Settings,
    groups: Sequence[FacebookGroup],
    *,
    max_posts: int,
) -> ScanCycleSummary:
    """Scan every group while isolating ordinary per-group failures."""
    settings.require_read_only_mode()
    database = Database(settings.database_path)
    database.initialize()
    summaries: list[ScanSummary] = []
    failures = 0
    retried = 0
    recovered = 0
    logger = logging.getLogger("lead_agent.operations")
    async with FacebookReadOnlyBrowser(settings) as browser:
        scanner = ReadOnlyScanService(database, browser)
        for group_index, group in enumerate(groups):
            if group_index:
                await asyncio.sleep(settings.facebook_group_delay_seconds)
            retry_count = 0
            try:
                while True:
                    try:
                        summary = await scanner.scan_group(group, max_posts=max_posts)
                        summaries.append(summary)
                        if retry_count:
                            recovered += 1
                        break
                    except FacebookSafetyStop:
                        raise
                    except TransientFacebookReadError as error:
                        if retry_count >= settings.facebook_group_max_retries:
                            raise
                        retry_count += 1
                        retried += 1
                        logger.warning(
                            "Transient group scan failure; one bounded retry is scheduled",
                            extra={
                                "action": "cycle.group_scan.retry",
                                "result": "retrying",
                                "group_id": group.id,
                                "attempt": retry_count + 1,
                                "error_code": error.safe_code,
                                "retry_in_seconds": (settings.facebook_group_retry_backoff_seconds),
                            },
                        )
                        await asyncio.sleep(settings.facebook_group_retry_backoff_seconds)
            except FacebookSafetyStop:
                raise
            except Exception as error:
                failures += 1
                logger.warning(
                    "Group scan failed; cycle will continue",
                    extra={
                        "action": "cycle.group_scan",
                        "result": "failed",
                        "group_id": group.id,
                        "attempt": retry_count + 1,
                        "error_code": safe_scan_error_code(error),
                    },
                )
    return ScanCycleSummary(
        groups_scanned=len(summaries),
        groups_failed=failures,
        posts_seen=sum(summary.posts_seen for summary in summaries),
        posts_new=sum(len(summary.new_posts) for summary in summaries),
        duplicates=sum(summary.duplicates for summary in summaries),
        groups_retried=retried,
        groups_recovered=recovered,
        groups_partial=sum(summary.partial for summary in summaries),
        groups_severely_partial=sum(
            is_severe_post_shortfall(
                posts_seen=summary.posts_seen,
                posts_requested=summary.posts_requested,
                minimum_yield_rate=settings.operations_minimum_group_post_yield_rate,
            )
            for summary in summaries
        ),
        posts_requested=sum(summary.posts_requested for summary in summaries)
        + failures * max_posts,
    )


def _build_approval_notifier(
    settings: Settings,
    database: Database,
) -> ApprovalNotificationService:
    settings.require_remote_approval_ready()
    approvals = LocalApprovalService(
        database,
        expiration_minutes=settings.approval_expiration_minutes,
        duplicate_window_hours=settings.candidate_duplicate_window_hours,
        classification_version=CLASSIFICATION_VERSION,
    )
    sms_provider = build_sms_provider(settings)
    if settings.remote_approval_base_url is None:
        raise RuntimeError("Validated remote approval URL is unexpectedly missing")
    if settings.sms_recipient_number is None:
        raise RuntimeError("Validated SMS recipient is unexpectedly missing")
    return ApprovalNotificationService(
        database,
        approvals,
        sms_provider,
        public_base_url=str(settings.remote_approval_base_url),
        recipient_number=settings.sms_recipient_number,
        relay_healthcheck=lambda: relay_is_healthy(
            str(settings.remote_approval_base_url),
            timeout_seconds=5,
        ),
    )


def _run_operations_cycle(
    settings: Settings,
    *,
    max_posts: int,
    classification_limit: int,
    skip_notifications: bool,
    ignore_quiet_hours: bool = False,
    now: datetime | None = None,
) -> tuple[CycleSummary | None, OperationsState]:
    settings.require_read_only_mode()
    state = OperationsState(_operation_paths(settings))
    if state.paused:
        state.mark_paused()
        return None, state
    if max_posts > 50:
        raise GroupsConfigError("--max-posts cannot exceed the read-only safety cap of 50")
    if classification_limit > 1_000:
        raise AIProviderError("--classification-limit cannot exceed 1000")
    if not ignore_quiet_hours and _quiet_hours_active(settings, now=now):
        window = (
            f"{settings.operations_quiet_hours_start.strftime('%H:%M')}-"
            f"{settings.operations_quiet_hours_end.strftime('%H:%M')} "
            f"{settings.business_timezone}"
        )
        raise QuietHoursActiveError(f"overnight quiet hours are active ({window})")
    catalog = load_group_catalog(settings.groups_config_path)
    groups = catalog.enabled_groups()
    if not groups:
        raise GroupsConfigError("No Facebook groups are enabled in the group allowlist")

    database = Database(settings.database_path)
    database.initialize()
    provider = build_ai_provider(settings)
    classifier = LeadClassificationService(
        database,
        provider,
        classification_context(settings),
    )
    retention = RetentionService(
        state.paths,
        screenshot_retention_days=settings.screenshot_retention_days,
        log_retention_days=settings.operations_log_retention_days,
        log_max_bytes=settings.operations_log_max_bytes,
    )
    backups = DatabaseBackupService(
        settings.database_path,
        settings.database_backup_dir,
        retention_days=settings.database_backup_retention_days,
        interval_hours=settings.database_backup_interval_hours,
    )
    notifier = None
    if settings.notifications_enabled and not skip_notifications:
        notifier = _build_approval_notifier(settings, database)

    runner = OperationsCycleRunner(
        state,
        degraded_cycle_limit=settings.operations_degraded_cycle_limit,
        incomplete_group_rate_threshold=(settings.operations_incomplete_group_rate_threshold),
    )

    def notify_cycle() -> NotificationSummary:
        if notifier is None:  # pragma: no cover - callback is installed only when configured
            raise RuntimeError("Notification callback is not configured")
        return notifier.notify_candidates(limit=settings.ai_max_posts_per_run)

    notify_callback = notify_cycle if notifier is not None else None

    def retain_cycle() -> RetentionSummary:
        retained = retention.cleanup()
        backup = backups.run()
        return RetentionSummary(
            screenshots_removed=retained.screenshots_removed,
            logs_removed=retained.logs_removed,
            logs_rotated=retained.logs_rotated,
            backups_created=backup.created,
            backups_verified=backup.verified,
            backups_removed=backup.removed,
        )

    try:
        result = runner.run(
            scan=lambda: asyncio.run(_scan_groups_for_cycle(settings, groups, max_posts=max_posts)),
            classify=lambda: classifier.classify_posts(limit=classification_limit),
            notify=notify_callback,
            retain=retain_cycle,
        )
    except Exception as error:
        health = state.read_health()
        database.record_audit_event(
            AuditEvent(
                component="operations",
                action="cycle.run",
                result="failed",
                details={
                    "error_code": type(error).__name__,
                    "circuit_breaker_tripped": health.get("circuit_breaker_tripped", False),
                },
            )
        )
        raise
    if result is not None:
        database.record_audit_event(
            AuditEvent(
                component="operations",
                action="cycle.run",
                result=result.status,
                details={
                    "groups_scanned": result.scan.groups_scanned,
                    "groups_failed": result.scan.groups_failed,
                    "groups_partial": result.scan.groups_partial,
                    "groups_severely_partial": result.scan.groups_severely_partial,
                    "groups_retried": result.scan.groups_retried,
                    "groups_recovered": result.scan.groups_recovered,
                    "posts_seen": result.scan.posts_seen,
                    "posts_requested": result.scan.posts_requested,
                    "posts_new": result.scan.posts_new,
                    "duplicates": result.scan.duplicates,
                    "posts_classified": result.posts_classified,
                    "candidates_created": result.candidates_created,
                    "posts_ignored": result.posts_ignored,
                    "notifications_considered": result.notifications_considered,
                    "notifications_sent": result.notifications_sent,
                    "notifications_failed": result.notifications_failed,
                    "circuit_breaker_tripped": result.circuit_breaker_tripped,
                    "backups_created": result.retention.backups_created,
                    "backups_verified": result.retention.backups_verified,
                    "backups_removed": result.retention.backups_removed,
                },
            )
        )
    return result, state


async def _execute_approved_posting(
    settings: Settings,
    service: ApprovedPostingService,
    *,
    lead_id: int,
    dry_run: bool,
) -> PostingExecutionResult:
    async with FacebookCommentBrowser(settings) as browser:
        return await service.execute(lead_id, browser, dry_run=dry_run)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    configure_logging(settings)
    logger = logging.getLogger("lead_agent.cli")

    if args.command == "init-db":
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.screenshot_dir.mkdir(parents=True, exist_ok=True)
        Database(settings.database_path).initialize()
        logger.info(
            "Database initialized",
            extra={"action": "database.initialize", "result": "success"},
        )
        return 0

    if args.command == "doctor":
        print(json.dumps(_doctor_payload(settings), indent=2, sort_keys=True))
        return 0

    if args.command == "scan-status":
        database = Database(settings.database_path)
        database.initialize()
        if args.group_id:
            selected_state = database.get_group_scan_state(args.group_id)
            states = [selected_state] if selected_state is not None else []
        else:
            states = database.list_group_scan_states()
        print(json.dumps([_scan_state_payload(state) for state in states], indent=2))
        return 0

    if args.command == "operations-status":
        state = OperationsState(_operation_paths(settings))
        stale_after = max(600, settings.scan_interval_seconds * 3)
        payload = state.status_payload(stale_after_seconds=stale_after)
        quiet_hours_active = _quiet_hours_active(settings)
        if quiet_hours_active:
            payload["stale"] = False
        payload.update(
            {
                "ai_ready": _doctor_payload(settings)["ai_ready"],
                "notifications_enabled": settings.notifications_enabled,
                "remote_approval_ready": settings.remote_approval_ready,
                "read_only_mode_ready": not settings.posting_enabled and settings.dry_run,
                "quiet_hours_enabled": settings.operations_quiet_hours_enabled,
                "quiet_hours_active": quiet_hours_active,
                "quiet_hours_start": settings.operations_quiet_hours_start.strftime("%H:%M"),
                "quiet_hours_end": settings.operations_quiet_hours_end.strftime("%H:%M"),
                "quiet_hours_timezone": settings.business_timezone,
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "operations-pause":
        state = OperationsState(_operation_paths(settings))
        state.pause()
        print("Unattended lead-agent cycles are paused")
        return 0

    if args.command == "operations-resume":
        state = OperationsState(_operation_paths(settings))
        changed = state.resume()
        print("Unattended lead-agent cycles resumed" if changed else "Cycles were not paused")
        return 0

    if args.command == "group-report":
        database = Database(settings.database_path)
        database.initialize()
        catalog = load_group_catalog(settings.groups_config_path)
        priorities = {group.id: group.priority for group in catalog.enabled_groups()}
        report_payload = [
            {
                "group_id": quality.group_id,
                "group_name": quality.group_name,
                "priority": priorities.get(quality.group_id),
                "posts_discovered": quality.posts_discovered,
                "posts_classified": quality.posts_classified,
                "candidates_created": quality.candidates_created,
                "candidate_yield_percent": quality.candidate_yield_percent,
                "provider_advertisements": quality.provider_advertisements,
                "exact_text_duplicates": quality.exact_text_duplicates,
                "cross_group_reposts": quality.cross_group_reposts,
                "last_discovered_at": quality.last_discovered_at.isoformat(),
            }
            for quality in database.list_group_quality()
        ]
        print(json.dumps(report_payload, indent=2))
        return 0

    if args.command == "run-cycle":
        try:
            cycle_result, _state = _run_operations_cycle(
                settings,
                max_posts=args.max_posts or settings.max_posts_per_group,
                classification_limit=(
                    args.classification_limit or settings.cycle_classification_limit
                ),
                skip_notifications=args.skip_notifications,
                ignore_quiet_hours=args.ignore_quiet_hours,
            )
        except CycleAlreadyRunningError as error:
            print(f"No action taken: {error}")
            return 0
        except QuietHoursActiveError as error:
            print(f"No action taken: {error}")
            return 0
        except Exception as error:
            logger.error(
                "Operations cycle stopped safely",
                extra={
                    "action": "cycle.run",
                    "result": "failed",
                    "error": type(error).__name__,
                },
            )
            print(f"Stopped safely: {type(error).__name__}", file=sys.stderr)
            return 2
        if cycle_result is None:
            print("No action taken: unattended cycles are paused")
        else:
            print(json.dumps(asdict(cycle_result), sort_keys=True))
        return 0

    if args.command == "classify-posts":
        try:
            settings.require_read_only_mode()
            database = Database(settings.database_path)
            database.initialize()
            provider = build_ai_provider(settings)
            classifier = LeadClassificationService(
                database,
                provider,
                classification_context(settings),
            )
            summary = classifier.classify_posts(
                limit=args.limit or settings.ai_max_posts_per_run,
                post_id=args.post_id,
            )
        except (AIProviderError, UnsafeReadOnlyModeError) as error:
            print(f"Stopped safely: {error}", file=sys.stderr)
            return 2
        _print_classification_results(summary)
        return 0

    if args.command in {"reclassify-leads", "classification-replay"}:
        try:
            settings.require_read_only_mode()
            database = Database(settings.database_path)
            database.initialize()
            classifier = LeadClassificationService(
                database,
                build_ai_provider(settings),
                classification_context(settings),
            )
            limit = args.limit or settings.ai_max_posts_per_run
            if args.command == "reclassify-leads":
                reclassification_result = classifier.reclassify_leads(
                    limit=limit, lead_id=args.lead_id
                )
                print(json.dumps(asdict(reclassification_result), indent=2, sort_keys=True))
            else:
                replay_result = classifier.replay_history(limit=limit, lead_id=args.lead_id)
                outcomes = [
                    asdict(outcome)
                    for outcome in replay_result.outcomes
                    if not args.changed_only or outcome.changed
                ]
                print(
                    json.dumps(
                        {
                            "leads_considered": replay_result.leads_considered,
                            "changed": replay_result.changed,
                            "outcomes": outcomes,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
        except (AIProviderError, UnsafeReadOnlyModeError, ValueError) as error:
            print(f"Stopped safely: {error}", file=sys.stderr)
            return 2
        return 0

    if args.command == "export-regression-fixtures":
        database = Database(settings.database_path)
        database.initialize()
        output_path = args.output or settings.data_dir / "regression-fixtures.json"
        export_summary = export_regression_fixtures(
            database,
            output_path,
            limit=args.limit or 100,
            lead_id=args.lead_id,
        )
        print(
            json.dumps(
                {
                    **asdict(export_summary),
                    "output_path": str(export_summary.output_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command in {"database-backup", "database-restore-test"}:
        try:
            database = Database(settings.database_path)
            database.initialize()
            backups = DatabaseBackupService(
                settings.database_path,
                settings.database_backup_dir,
                retention_days=settings.database_backup_retention_days,
                interval_hours=settings.database_backup_interval_hours,
            )
            if args.command == "database-backup":
                backup_summary = backups.run(force=True)
                payload = {
                    "created": backup_summary.created,
                    "verified": backup_summary.verified,
                    "removed": backup_summary.removed,
                    "backup_path": (
                        str(backup_summary.backup_path) if backup_summary.backup_path else None
                    ),
                }
            else:
                verified_path = backups.verify_restore(args.backup_path)
                payload = {"verified": True, "backup_path": str(verified_path)}
        except BackupError as error:
            print(f"Stopped safely: {error}", file=sys.stderr)
            return 2
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "approval-dashboard":
        try:
            settings.require_read_only_mode()
            database = Database(settings.database_path)
            database.initialize()
            service = LocalApprovalService(
                database,
                expiration_minutes=settings.approval_expiration_minutes,
                duplicate_window_hours=settings.candidate_duplicate_window_hours,
                classification_version=CLASSIFICATION_VERSION,
            )
            run_local_approval_dashboard(
                service,
                port=args.port or settings.approval_local_port,
                candidate_limit=args.limit,
                business_timezone=settings.business_timezone,
            )
        except (ApprovalError, OSError, UnsafeReadOnlyModeError) as error:
            print(f"Stopped safely: {error}", file=sys.stderr)
            return 2
        return 0

    if args.command == "remote-approval":
        try:
            settings.require_read_only_mode()
            settings.require_remote_approval_ready()
            database = Database(settings.database_path)
            database.initialize()
            approvals = LocalApprovalService(
                database,
                expiration_minutes=settings.approval_expiration_minutes,
                duplicate_window_hours=settings.candidate_duplicate_window_hours,
                classification_version=CLASSIFICATION_VERSION,
            )
            sms_provider = build_sms_provider(settings)
            if settings.remote_approval_base_url is None:
                raise RuntimeError("Validated remote approval URL is unexpectedly missing")
            if settings.sms_recipient_number is None or settings.approval_signing_key is None:
                raise RuntimeError("Validated remote approval secrets are unexpectedly missing")
            notifier = ApprovalNotificationService(
                database,
                approvals,
                sms_provider,
                public_base_url=str(settings.remote_approval_base_url),
                recipient_number=settings.sms_recipient_number,
                relay_healthcheck=lambda: relay_is_healthy(
                    str(settings.remote_approval_base_url),
                    timeout_seconds=5,
                ),
            )
            candidate_limit = args.limit or settings.ai_max_posts_per_run
            controller = RemoteApprovalController(
                approvals,
                signing_key=settings.approval_signing_key.get_secret_value(),
            )

            retry_failed = args.retry_failed

            def notify_cycle() -> None:
                nonlocal retry_failed
                summary = notifier.notify_candidates(
                    limit=candidate_limit,
                    retry_failed=retry_failed,
                )
                retry_failed = False
                if summary.considered:
                    print(
                        f"notifications considered={summary.considered} "
                        f"sent={summary.sent} failed={summary.failed}"
                    )

            run_remote_approval_server(
                controller,
                port=args.port or settings.remote_approval_port,
                public_base_url=str(settings.remote_approval_base_url).rstrip("/"),
                periodic_callback=notify_cycle,
                callback_interval_seconds=settings.notification_poll_interval_seconds,
            )
        except (
            ApprovalError,
            NotificationConfigurationError,
            OSError,
            UnsafeReadOnlyModeError,
        ) as error:
            print(f"Stopped safely: {error}", file=sys.stderr)
            return 2
        return 0

    if args.command == "facebook-login":
        try:
            asyncio.run(_manual_login(settings))
        except (FacebookBrowserError, FacebookSafetyStop, UnsafeReadOnlyModeError) as error:
            print(f"Stopped safely: {error}", file=sys.stderr)
            return 2
        print("Manual Facebook session verified and saved in the dedicated profile.")
        return 0

    if args.command == "scan-facebook":
        try:
            catalog = load_group_catalog(settings.groups_config_path)
            groups = (
                [catalog.enabled_group(args.group_id)]
                if args.group_id
                else catalog.enabled_groups()
            )
            if not groups:
                raise GroupsConfigError("No Facebook groups are enabled in the group allowlist")
            max_posts = args.max_posts or settings.max_posts_per_group
            if max_posts > 50:
                raise GroupsConfigError("--max-posts cannot exceed the read-only safety cap of 50")
            summaries = asyncio.run(
                _scan_groups(
                    settings,
                    groups,
                    max_posts=max_posts,
                    pause_after_scan=args.pause_after_scan,
                )
            )
        except (
            FacebookBrowserError,
            FacebookSafetyStop,
            GroupsConfigError,
            UnsafeReadOnlyModeError,
        ) as error:
            print(f"Stopped safely: {error}", file=sys.stderr)
            if isinstance(error, FacebookSafetyStop) and error.screenshot_path is not None:
                print(f"Diagnostic screenshot: {error.screenshot_path}", file=sys.stderr)
            return 2
        _print_scan_results(summaries)
        return 0

    if args.command == "post-approved":
        dry_run = not args.submit
        try:
            if dry_run:
                settings.require_read_only_mode()
            else:
                settings.require_posting_allowed()
            catalog = load_group_catalog(settings.groups_config_path)
            enabled_group_ids = {group.id for group in catalog.enabled_groups()}
            if not enabled_group_ids:
                raise GroupsConfigError("No Facebook groups are enabled in the group allowlist")
            database = Database(settings.database_path)
            database.initialize()
            posting_service = ApprovedPostingService(
                database,
                settings,
                enabled_group_ids=enabled_group_ids,
            )
            posting_result = asyncio.run(
                _execute_approved_posting(
                    settings,
                    posting_service,
                    lead_id=args.lead_id,
                    dry_run=dry_run,
                )
            )
        except (
            FacebookBrowserError,
            FacebookSafetyStop,
            GroupsConfigError,
            PostingDisabledError,
            PostingError,
            UnsafeReadOnlyModeError,
        ) as error:
            print(f"Stopped safely: {error}", file=sys.stderr)
            screenshot_path = getattr(error, "screenshot_path", None)
            if screenshot_path is not None:
                print(f"Diagnostic screenshot: {screenshot_path}", file=sys.stderr)
            return 2
        attempt = posting_result.work.attempt
        if not posting_result.created:
            print(
                f"No action taken: live posting attempt already exists "
                f"with status={attempt.status.value}"
            )
            return 0 if attempt.status.value == "posted" else 2
        if dry_run:
            print(
                f"DRY RUN lead={args.lead_id} validated; "
                "no Facebook comment was entered or submitted"
            )
        else:
            print(f"POSTED lead={args.lead_id} exactly once")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

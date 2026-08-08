"""Local command-line interface for database and read-only Facebook discovery."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from contextlib import suppress

from lead_agent.ai import AIProviderError, build_ai_provider, classification_context
from lead_agent.classifier import ClassificationSummary, LeadClassificationService
from lead_agent.config import Settings, UnsafeReadOnlyModeError, load_settings
from lead_agent.database import Database
from lead_agent.facebook import FacebookBrowserError, FacebookReadOnlyBrowser
from lead_agent.facebook_state import FacebookSafetyStop
from lead_agent.groups import FacebookGroup, GroupsConfigError, load_group_catalog
from lead_agent.logging_config import configure_logging
from lead_agent.models import GroupScanState
from lead_agent.scanner import ReadOnlyScanService, ScanSummary


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
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
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
    }


def _scan_state_payload(state: GroupScanState) -> dict[str, object]:
    return {
        "group_id": state.group_id,
        "group_name": state.group_name,
        "health": "healthy" if state.last_error is None else "degraded",
        "consecutive_failures": state.consecutive_failures,
        "last_attempt_at": state.last_attempt_at.isoformat(),
        "last_success_at": state.last_success_at.isoformat() if state.last_success_at else None,
        "last_failure_at": state.last_failure_at.isoformat() if state.last_failure_at else None,
        "last_error": state.last_error,
        "posts_seen": state.posts_seen,
        "posts_new": state.posts_new,
    }


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
        print(
            f"[{summary.group.name}] seen={summary.posts_seen} "
            f"new={len(summary.new_posts)} duplicates={summary.duplicates}"
        )
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

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

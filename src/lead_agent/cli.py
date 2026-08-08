"""Bootstrap command-line interface. Facebook browser actions are intentionally absent."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from lead_agent.config import Settings, load_settings
from lead_agent.database import Database
from lead_agent.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lead-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="Create or upgrade the local SQLite database")
    subparsers.add_parser("doctor", help="Validate configuration and show safety state")
    return parser


def _doctor_payload(settings: Settings) -> dict[str, object]:
    return {
        "database_path": str(settings.database_path),
        "dry_run": settings.dry_run,
        "posting_enabled": settings.posting_enabled,
        "posting_allowed": settings.posting_allowed,
        "service_area": settings.service_area,
        "lead_threshold": settings.lead_threshold,
        "facebook_profile_path": str(settings.facebook_profile_path),
        "ai_provider": settings.ai_provider,
        "notifications_enabled": settings.notifications_enabled,
    }


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

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

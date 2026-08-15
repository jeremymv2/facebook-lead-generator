"""Install or remove JJ Miller & Co. user launch agents on macOS."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
from pathlib import Path

from lead_agent.config import load_settings
from lead_agent.launchd import (
    CYCLE_AGENT_LABEL,
    POSTING_AGENT_LABEL,
    REMOTE_AGENT_LABEL,
    build_launch_agents,
    write_launch_agents,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manage_launchd.py")
    subparsers = parser.add_subparsers(dest="command", required=True)
    install = subparsers.add_parser("install")
    install.add_argument(
        "--include-remote-approval",
        action="store_true",
        help="Also install the Telnyx/tunnel review server; requires ready configuration",
    )
    install.add_argument(
        "--include-posting-worker",
        action="store_true",
        help="Install the guarded queued-posting worker with process-local live flags",
    )
    subparsers.add_parser("uninstall")
    subparsers.add_parser("status")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if platform.system() != "Darwin":
        raise SystemExit("launchd management is supported only on macOS")

    repository = Path.cwd().resolve()
    executable = repository / ".venv" / "bin" / "lead-agent"
    if not executable.is_file():
        raise SystemExit("Run this command from the repository after creating .venv")
    destination = Path.home() / "Library" / "LaunchAgents"
    domain = f"gui/{os.getuid()}"
    labels = (CYCLE_AGENT_LABEL, REMOTE_AGENT_LABEL, POSTING_AGENT_LABEL)

    if args.command == "install":
        settings = load_settings()
        settings.require_read_only_mode()
        settings.operations_state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        settings.operations_log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        definitions = build_launch_agents(
            settings,
            executable=executable,
            working_directory=repository,
            include_remote_approval=args.include_remote_approval,
            include_posting_worker=args.include_posting_worker,
        )
        paths = write_launch_agents(definitions, destination=destination)
        for path in paths:
            _bootout(domain, path)
            _run_launchctl("bootstrap", domain, str(path))
        print("Installed: " + ", ".join(definition.label for definition in definitions))
        return 0

    if args.command == "uninstall":
        for label in labels:
            path = destination / f"{label}.plist"
            _bootout(domain, path)
            path.unlink(missing_ok=True)
        print("Removed JJ Miller & Co. lead-agent launch agents")
        return 0

    for label in labels:
        result = subprocess.run(
            ["launchctl", "print", f"{domain}/{label}"],
            check=False,
            capture_output=True,
            text=True,
        )
        print(f"{label}: {'loaded' if result.returncode == 0 else 'not loaded'}")
    return 0


def _bootout(domain: str, path: Path) -> None:
    if not path.exists():
        return
    subprocess.run(
        ["launchctl", "bootout", domain, str(path)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _run_launchctl(*arguments: str) -> None:
    try:
        subprocess.run(["launchctl", *arguments], check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"launchctl {arguments[0]} failed") from error


if __name__ == "__main__":
    raise SystemExit(main())

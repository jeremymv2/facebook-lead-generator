#!/usr/bin/env python3
"""Reject tracked paths likely to contain credentials or private runtime data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import PurePosixPath

_SENSITIVE_DIRECTORY_NAMES = {
    ".auth",
    ".cloudflared",
    "browser-profile",
    "facebook-profile",
    "logs",
    "secrets",
}
_SENSITIVE_EXACT_PATHS = {
    "config/groups.yaml",
}
_SENSITIVE_BASENAMES = {
    ".envrc",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
_SENSITIVE_SUFFIXES = {
    ".har",
    ".jks",
    ".kdbx",
    ".key",
    ".keystore",
    ".log",
    ".mobileprovision",
    ".ovpn",
    ".p12",
    ".p8",
    ".pem",
    ".pfx",
    ".sqlite-journal",
    ".sqlite",
    ".sqlite3",
    ".sqlite3-shm",
    ".sqlite3-wal",
}


def sensitive_path_reason(raw_path: str) -> str | None:
    """Explain why a Git path is unsafe, or return ``None`` when allowed."""
    normalized = raw_path.replace("\\", "/").removeprefix("./")
    path = PurePosixPath(normalized)
    lower_path = normalized.casefold()
    lower_name = path.name.casefold()
    lower_parts = {part.casefold() for part in path.parts}

    if lower_path in _SENSITIVE_EXACT_PATHS or lower_name in _SENSITIVE_BASENAMES:
        return "local configuration or credential file"
    if lower_name == ".env" or (lower_name.startswith(".env.") and lower_name != ".env.example"):
        return "environment file"
    if lower_name.startswith(("client_secret", "cookies", "service-account")):
        return "exported credential file"
    if lower_name.startswith(("storage-state", "storage_state")) and lower_name.endswith(".json"):
        return "browser authentication state"
    if lower_parts & _SENSITIVE_DIRECTORY_NAMES:
        return "private runtime directory"
    if "screenshots" in lower_parts and lower_name != ".gitkeep":
        return "runtime screenshot"
    if any(lower_name.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES):
        return "credential, trace, log, or database file"
    return None


def tracked_paths() -> list[str]:
    """Return paths currently present in the Git index."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [path for path in result.stdout.split("\0") if path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Paths supplied by pre-commit")
    parser.add_argument(
        "--all-tracked",
        action="store_true",
        help="Check every path in the Git index in addition to supplied paths",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = set(args.paths)
    if args.all_tracked or not paths:
        paths.update(tracked_paths())

    failures = [
        (path, reason)
        for path in sorted(paths)
        if (reason := sensitive_path_reason(path)) is not None
    ]
    if not failures:
        return 0

    print("Refusing to commit sensitive runtime or credential paths:", file=sys.stderr)
    for path, reason in failures:
        print(f"  - {path}: {reason}", file=sys.stderr)
    print(
        "Keep these files outside Git. If a real secret was already committed, rotate it first.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

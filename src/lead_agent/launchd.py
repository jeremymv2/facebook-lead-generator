"""Generate credential-free macOS launch agent definitions."""

from __future__ import annotations

import plistlib
from dataclasses import dataclass
from pathlib import Path

from lead_agent.config import Settings

CYCLE_AGENT_LABEL = "com.jjmillerco.lead-agent-cycle"
REMOTE_AGENT_LABEL = "com.jjmillerco.lead-agent-remote-approval"


@dataclass(frozen=True, slots=True)
class LaunchAgentDefinition:
    label: str
    payload: dict[str, object]

    @property
    def filename(self) -> str:
        return f"{self.label}.plist"

    def to_bytes(self) -> bytes:
        return plistlib.dumps(self.payload, fmt=plistlib.FMT_XML, sort_keys=True)


def build_launch_agents(
    settings: Settings,
    *,
    executable: Path,
    working_directory: Path,
    include_remote_approval: bool,
) -> tuple[LaunchAgentDefinition, ...]:
    """Build launch agents without embedding the contents of ``.env``."""
    executable = executable.resolve()
    working_directory = working_directory.resolve()
    log_directory = settings.operations_log_dir.resolve()
    common: dict[str, object] = {
        "WorkingDirectory": str(working_directory),
        "ProcessType": "Background",
        "LowPriorityIO": True,
        "Nice": 10,
    }
    cycle = LaunchAgentDefinition(
        label=CYCLE_AGENT_LABEL,
        payload={
            **common,
            "Label": CYCLE_AGENT_LABEL,
            "ProgramArguments": [str(executable), "run-cycle"],
            "RunAtLoad": True,
            "StartInterval": settings.scan_interval_seconds,
            "StandardOutPath": str(log_directory / "cycle.stdout.log"),
            "StandardErrorPath": str(log_directory / "cycle.stderr.log"),
        },
    )
    agents = [cycle]
    if include_remote_approval:
        settings.require_remote_approval_ready()
        agents.append(
            LaunchAgentDefinition(
                label=REMOTE_AGENT_LABEL,
                payload={
                    **common,
                    "Label": REMOTE_AGENT_LABEL,
                    "ProgramArguments": [str(executable), "remote-approval"],
                    "RunAtLoad": True,
                    "KeepAlive": {"SuccessfulExit": False},
                    "ThrottleInterval": 30,
                    "StandardOutPath": str(log_directory / "remote.stdout.log"),
                    "StandardErrorPath": str(log_directory / "remote.stderr.log"),
                },
            )
        )
    return tuple(agents)


def write_launch_agents(
    definitions: tuple[LaunchAgentDefinition, ...],
    *,
    destination: Path,
) -> tuple[Path, ...]:
    """Atomically write private launch agent files to an explicit directory."""
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    written: list[Path] = []
    for definition in definitions:
        target = destination / definition.filename
        temporary = destination / f".{definition.filename}.tmp"
        try:
            temporary.write_bytes(definition.to_bytes())
            temporary.chmod(0o600)
            temporary.replace(target)
            target.chmod(0o600)
        finally:
            temporary.unlink(missing_ok=True)
        written.append(target)
    return tuple(written)

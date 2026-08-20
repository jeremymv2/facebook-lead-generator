import plistlib
import stat
from pathlib import Path

from lead_agent.config import Settings
from lead_agent.launchd import (
    CYCLE_AGENT_LABEL,
    DASHBOARD_AGENT_LABEL,
    POSTING_AGENT_LABEL,
    REMOTE_AGENT_LABEL,
    build_launch_agents,
    write_launch_agents,
)


def settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": tmp_path / "data",
        "database_path": tmp_path / "data" / "agent.sqlite3",
        "screenshot_dir": tmp_path / "screenshots",
        "groups_config_path": tmp_path / "groups.yaml",
        "facebook_profile_path": tmp_path.parent / "facebook-profile",
    }
    values.update(overrides)
    return Settings.model_validate(values)


def test_cycle_launch_agent_uses_fixed_cadence_and_contains_no_environment_secrets(
    tmp_path: Path,
) -> None:
    configured = settings(tmp_path)
    definitions = build_launch_agents(
        configured,
        executable=tmp_path / ".venv" / "bin" / "lead-agent",
        working_directory=tmp_path,
        include_remote_approval=False,
    )

    assert {definition.label for definition in definitions} == {
        CYCLE_AGENT_LABEL,
        DASHBOARD_AGENT_LABEL,
    }
    cycle = next(value for value in definitions if value.label == CYCLE_AGENT_LABEL)
    assert cycle.payload["StartCalendarInterval"] == [
        {"Minute": 0},
        {"Minute": 8},
        {"Minute": 15},
        {"Minute": 23},
        {"Minute": 30},
        {"Minute": 38},
        {"Minute": 45},
        {"Minute": 53},
    ]
    assert "StartInterval" not in cycle.payload
    assert cycle.payload["ProgramArguments"][-1] == "run-cycle"  # type: ignore[index]
    assert "EnvironmentVariables" not in cycle.payload

    dashboard = next(value for value in definitions if value.label == DASHBOARD_AGENT_LABEL)
    assert dashboard.payload["ProgramArguments"][-1] == "approval-dashboard"  # type: ignore[index]
    assert dashboard.payload["KeepAlive"] == {"SuccessfulExit": False}
    assert dashboard.payload["RunAtLoad"] is True
    assert "EnvironmentVariables" not in dashboard.payload


def test_remote_launch_agent_requires_ready_config_but_embeds_no_credentials(
    tmp_path: Path,
) -> None:
    configured = settings(
        tmp_path,
        notifications_enabled=True,
        sms_provider="telnyx",
        remote_approval_base_url="https://approve.example",
        approval_signing_key="signing-secret-" * 4,
        sms_recipient_number="+15025550101",
        telnyx_api_key="telnyx-secret",
        telnyx_from_number="+15025550100",
    )

    definitions = build_launch_agents(
        configured,
        executable=tmp_path / ".venv" / "bin" / "lead-agent",
        working_directory=tmp_path,
        include_remote_approval=True,
    )
    serialized = b"".join(definition.to_bytes() for definition in definitions)

    assert {definition.label for definition in definitions} == {
        CYCLE_AGENT_LABEL,
        DASHBOARD_AGENT_LABEL,
        REMOTE_AGENT_LABEL,
    }
    assert b"telnyx-secret" not in serialized
    assert b"signing-secret" not in serialized
    assert b"+15025550101" not in serialized
    assert b"+15025550100" not in serialized


def test_posting_worker_has_process_local_live_flags_and_no_secrets(tmp_path: Path) -> None:
    configured = settings(
        tmp_path,
        posting_queue_enabled=True,
        posting_queue_poll_interval_seconds=60,
        notifications_enabled=True,
        sms_provider="telnyx",
        remote_approval_base_url="https://approve.example",
        approval_signing_key="signing-secret-" * 4,
        sms_recipient_number="+15025550101",
        telnyx_api_key="telnyx-secret",
        telnyx_from_number="+15025550100",
    )

    definitions = build_launch_agents(
        configured,
        executable=tmp_path / ".venv" / "bin" / "lead-agent",
        working_directory=tmp_path,
        include_remote_approval=True,
        include_posting_worker=True,
    )

    worker = next(value for value in definitions if value.label == POSTING_AGENT_LABEL)
    assert worker.payload["ProgramArguments"][-1] == "process-posting-queue"  # type: ignore[index]
    assert worker.payload["StartInterval"] == 60
    assert worker.payload["EnvironmentVariables"] == {
        "POSTING_ENABLED": "true",
        "DRY_RUN": "false",
    }
    assert b"telnyx-secret" not in worker.to_bytes()
    assert b"signing-secret" not in worker.to_bytes()


def test_launch_agent_files_are_written_privately(tmp_path: Path) -> None:
    definitions = build_launch_agents(
        settings(tmp_path),
        executable=tmp_path / ".venv" / "bin" / "lead-agent",
        working_directory=tmp_path,
        include_remote_approval=False,
    )

    paths = write_launch_agents(definitions, destination=tmp_path / "LaunchAgents")

    assert len(paths) == 2
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in paths)
    payloads = [plistlib.loads(path.read_bytes()) for path in paths]
    assert {payload["Label"] for payload in payloads} == {
        CYCLE_AGENT_LABEL,
        DASHBOARD_AGENT_LABEL,
    }

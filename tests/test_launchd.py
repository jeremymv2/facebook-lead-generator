import plistlib
import stat
from pathlib import Path

from lead_agent.config import Settings
from lead_agent.launchd import (
    CYCLE_AGENT_LABEL,
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


def test_cycle_launch_agent_is_bounded_and_contains_no_environment_secrets(
    tmp_path: Path,
) -> None:
    configured = settings(tmp_path, scan_interval_seconds=600)
    definitions = build_launch_agents(
        configured,
        executable=tmp_path / ".venv" / "bin" / "lead-agent",
        working_directory=tmp_path,
        include_remote_approval=False,
    )

    assert len(definitions) == 1
    definition = definitions[0]
    assert definition.label == CYCLE_AGENT_LABEL
    assert definition.payload["StartInterval"] == 600
    assert definition.payload["ProgramArguments"][-1] == "run-cycle"  # type: ignore[index]
    assert "EnvironmentVariables" not in definition.payload


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
        REMOTE_AGENT_LABEL,
    }
    assert b"telnyx-secret" not in serialized
    assert b"signing-secret" not in serialized
    assert b"+15025550101" not in serialized
    assert b"+15025550100" not in serialized


def test_launch_agent_files_are_written_privately(tmp_path: Path) -> None:
    definitions = build_launch_agents(
        settings(tmp_path),
        executable=tmp_path / ".venv" / "bin" / "lead-agent",
        working_directory=tmp_path,
        include_remote_approval=False,
    )

    paths = write_launch_agents(definitions, destination=tmp_path / "LaunchAgents")

    assert len(paths) == 1
    assert stat.S_IMODE(paths[0].stat().st_mode) == 0o600
    payload = plistlib.loads(paths[0].read_bytes())
    assert payload["Label"] == CYCLE_AGENT_LABEL

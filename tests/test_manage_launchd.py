import os
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.manage_launchd as manage_launchd
from lead_agent.config import NotificationConfigurationError
from lead_agent.launchd import CYCLE_AGENT_LABEL, REMOTE_AGENT_LABEL


def prepare_macos_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, list[tuple[str, ...]]]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "facebook-profile"))
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    executable = tmp_path / ".venv" / "bin" / "lead-agent"
    executable.parent.mkdir(parents=True)
    executable.write_text("fixture", encoding="utf-8")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(manage_launchd, "_bootout", lambda domain, path: None)
    monkeypatch.setattr(
        manage_launchd,
        "_run_launchctl",
        lambda *arguments: calls.append(arguments),
    )
    return tmp_path / "home" / "Library" / "LaunchAgents", calls


def test_install_and_uninstall_cycle_launch_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination, calls = prepare_macos_fixture(monkeypatch, tmp_path)

    assert manage_launchd.main(["install"]) == 0
    cycle_path = destination / f"{CYCLE_AGENT_LABEL}.plist"
    assert cycle_path.exists()
    assert calls[0][:2] == ("bootstrap", f"gui/{os.getuid()}")
    assert CYCLE_AGENT_LABEL in capsys.readouterr().out

    assert manage_launchd.main(["uninstall"]) == 0
    assert not cycle_path.exists()
    assert "Removed" in capsys.readouterr().out


def test_remote_install_fails_closed_until_notifications_are_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepare_macos_fixture(monkeypatch, tmp_path)

    with pytest.raises(NotificationConfigurationError, match="NOTIFICATIONS_ENABLED"):
        manage_launchd.main(["install", "--include-remote-approval"])


def test_status_reports_loaded_state_without_printing_launchctl_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_macos_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    assert manage_launchd.main(["status"]) == 0
    output = capsys.readouterr().out
    assert f"{CYCLE_AGENT_LABEL}: loaded" in output
    assert f"{REMOTE_AGENT_LABEL}: loaded" in output


def test_launchd_manager_rejects_non_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    with pytest.raises(SystemExit, match="only on macOS"):
        manage_launchd.main(["status"])


def test_run_launchctl_converts_command_failure_to_safe_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "launchctl")

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(SystemExit, match="bootstrap failed"):
        manage_launchd._run_launchctl("bootstrap", "gui/501", "fixture.plist")

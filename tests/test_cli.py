import json
from pathlib import Path

import pytest

from lead_agent.cli import build_parser, main
from lead_agent.database import Database


def test_doctor_reports_safe_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))

    result = main(["doctor"])
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["posting_enabled"] is False
    assert payload["dry_run"] is True
    assert payload["posting_allowed"] is False
    assert payload["read_only_mode_ready"] is True


def test_init_db_creates_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    database_path = tmp_path / "data" / "test.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))

    result = main(["init-db"])

    assert result == 0
    assert database_path.exists()
    assert Database(database_path).list_posts() == []


def test_scan_status_reports_health_without_post_or_url_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    database_path = tmp_path / "data" / "test.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    database = Database(database_path)
    database.initialize()
    database.record_group_scan_failure(
        group_id="fixture-group",
        group_name="Synthetic Fixture Group",
        group_url="https://www.facebook.com/groups/111",
        error="FacebookBrowserError",
    )

    result = main(["scan-status", "--group-id", "fixture-group"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert result == 0
    assert payload[0]["group_id"] == "fixture-group"
    assert payload[0]["health"] == "degraded"
    assert payload[0]["consecutive_failures"] == 1
    assert payload[0]["last_error"] == "FacebookBrowserError"
    assert "group_url" not in output
    assert "last_known_post_identity" not in output


def test_scan_parser_rejects_non_positive_max_posts() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["scan-facebook", "--max-posts", "0"])


def test_scan_parser_accepts_pause_for_visual_inspection() -> None:
    args = build_parser().parse_args(
        ["scan-facebook", "--group-id", "approved-group", "--pause-after-scan"]
    )

    assert args.pause_after_scan is True


def test_scan_command_fails_closed_when_no_group_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    groups_path = tmp_path / "groups.yaml"
    groups_path.write_text("groups: []\n", encoding="utf-8")
    monkeypatch.setenv("GROUPS_CONFIG_PATH", str(groups_path))

    result = main(["scan-facebook"])

    assert result == 2
    assert "No Facebook groups are enabled" in capsys.readouterr().err

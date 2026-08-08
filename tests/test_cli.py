import json
from pathlib import Path

import pytest

import lead_agent.cli as cli_module
from lead_agent.cli import build_parser, main
from lead_agent.database import Database
from lead_agent.models import FacebookPost, LeadStatus


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
    assert payload["ai_provider"] == "disabled"
    assert payload["ai_ready"] is False


def test_doctor_reports_ai_readiness_without_printing_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "placeholder")

    result = main(["doctor"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert result == 0
    assert payload["ai_provider"] == "gemini"
    assert payload["ai_ready"] is True
    assert "placeholder" not in output


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


def test_classify_parser_accepts_bounded_post_selection() -> None:
    args = build_parser().parse_args(["classify-posts", "--post-id", "12", "--limit", "3"])

    assert args.post_id == 12
    assert args.limit == 3


def test_approval_parser_accepts_local_port_and_candidate_limit() -> None:
    args = build_parser().parse_args(["approval-dashboard", "--port", "9876", "--limit", "5"])

    assert args.port == 9876
    assert args.limit == 5


def test_approval_parser_rejects_privileged_port() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["approval-dashboard", "--port", "80"])


def test_classify_command_fails_closed_when_provider_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "disabled.sqlite3"))

    result = main(["classify-posts"])

    assert result == 2
    assert "AI provider is disabled" in capsys.readouterr().err


def test_classify_command_runs_offline_and_prints_only_candidate_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    database_path = tmp_path / "classifier.sqlite3"
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("AI_PROVIDER", "heuristic")
    database = Database(database_path)
    database.initialize()
    saved = database.save_post(
        FacebookPost(
            external_post_id="fixture-post",
            group_id="fixture-group",
            group_name="Synthetic Fixture Group",
            author_name="Sarah Example",
            post_text="Looking for someone in Louisville to repair our deck this week.",
        )
    ).post

    result = main(["classify-posts", "--post-id", str(saved.id)])
    output = capsys.readouterr().out

    assert result == 0
    assert "classified=1 candidates=1 ignored=0" in output
    assert "CANDIDATE" in output
    assert "DRAFT" in output
    assert "Looking for someone" not in output
    lead = database.get_lead_for_post(saved.id or 0)
    assert lead is not None
    assert lead.status is LeadStatus.CANDIDATE


def test_approval_dashboard_command_uses_local_service_without_facebook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    database_path = tmp_path / "approvals.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    calls: dict[str, int] = {}

    def fake_dashboard(
        service: object,
        *,
        port: int,
        candidate_limit: int,
    ) -> None:
        calls["port"] = port
        calls["candidate_limit"] = candidate_limit
        assert service.__class__.__name__ == "LocalApprovalService"

    monkeypatch.setattr(cli_module, "run_local_approval_dashboard", fake_dashboard)

    result = main(["approval-dashboard", "--port", "9876", "--limit", "4"])

    assert result == 0
    assert calls == {"port": 9876, "candidate_limit": 4}
    assert database_path.exists()


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

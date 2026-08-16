import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import lead_agent.cli as cli_module
from lead_agent.approvals import ApprovalAction, LocalApprovalService
from lead_agent.cli import build_parser, main
from lead_agent.config import Settings
from lead_agent.database import Database
from lead_agent.facebook_state import FacebookPageState, FacebookSafetyStop
from lead_agent.groups import FacebookGroup, GroupsConfigError
from lead_agent.models import FacebookPost, Lead, LeadIntent, LeadStatus, RejectionReason
from lead_agent.notifications import SmsDeliveryReceipt, SmsMessage
from lead_agent.operations import (
    CycleAlreadyRunningError,
    OperationPaths,
    OperationsState,
    QuietHoursActiveError,
    ScanCycleSummary,
)
from lead_agent.scanner import ScanSummary, TransientFacebookReadError


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
    assert payload["approval_expiration_minutes"] == 600
    assert payload["posting_approval_max_age_minutes"] == 20
    assert payload["daily_posting_limit"] == 5
    assert payload["per_group_daily_posting_limit"] == 2
    assert payload["business_timezone"] == "America/New_York"
    assert payload["operations_quiet_hours_enabled"] is True
    assert payload["operations_quiet_hours_start"] == "22:00"
    assert payload["operations_quiet_hours_end"] == "05:00"
    assert payload["operations_minimum_group_post_yield_rate"] == 0.5
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


def test_doctor_reports_remote_readiness_without_phone_or_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    monkeypatch.setenv("NOTIFICATIONS_ENABLED", "true")
    monkeypatch.setenv("SMS_PROVIDER", "telnyx")
    monkeypatch.setenv("REMOTE_APPROVAL_BASE_URL", "https://approve.example")
    monkeypatch.setenv("APPROVAL_SIGNING_KEY", "signing-secret-" * 4)
    monkeypatch.setenv("SMS_RECIPIENT_NUMBER", "+15025550101")
    monkeypatch.setenv("TELNYX_API_KEY", "telnyx-secret")
    monkeypatch.setenv("TELNYX_FROM_NUMBER", "+15025550100")

    result = main(["doctor"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert result == 0
    assert payload["remote_approval_ready"] is True
    assert payload["sms_provider"] == "telnyx"
    assert "telnyx-secret" not in output
    assert "signing-secret" not in output
    assert "+15025550101" not in output
    assert "+15025550100" not in output


def test_init_db_creates_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    database_path = tmp_path / "data" / "test.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))

    result = main(["init-db"])

    assert result == 0
    assert database_path.exists()
    assert Database(database_path).list_posts() == []


def test_replay_reclassify_and_feedback_export_commands_stay_local(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    database_path = tmp_path / "data" / "maintenance.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    monkeypatch.setenv("AI_PROVIDER", "heuristic")
    database = Database(database_path)
    database.initialize()
    saved = database.save_post(
        FacebookPost(
            external_post_id="maintenance-fixture",
            post_url=("https://www.facebook.com/groups/111/posts/maintenance-fixture"),
            group_id="fixture-group",
            group_name="Synthetic Group",
            author_name="Fixture Person",
            post_text=(
                "Need someone to repair a leaking roof at my Louisville home this week. "
                "Please provide an estimate."
            ),
        )
    ).post
    assert saved.id is not None

    assert main(["classify-posts", "--post-id", str(saved.id)]) == 0
    capsys.readouterr()
    lead = database.get_lead_for_post(saved.id)
    assert lead is not None and lead.id is not None

    assert main(["classification-replay", "--lead-id", str(lead.id), "--changed-only"]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay == {"changed": 0, "leads_considered": 1, "outcomes": []}

    assert main(["reclassify-leads", "--lead-id", str(lead.id)]) == 0
    reclassified = json.loads(capsys.readouterr().out)
    assert reclassified["leads_considered"] == 1
    assert reclassified["changes"][0]["lead_id"] == lead.id

    service = LocalApprovalService(database, expiration_minutes=20)
    request_id = service.prepare_candidates(limit=1)[0].request.id
    assert request_id is not None
    service.decide(
        request_id,
        ApprovalAction.REJECT,
        rejection_reason=RejectionReason.WRONG_GEOGRAPHY,
    )
    output_path = tmp_path / "private" / "feedback.json"

    assert (
        main(
            [
                "export-regression-fixtures",
                "--lead-id",
                str(lead.id),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    exported = json.loads(capsys.readouterr().out)
    assert exported["fixtures_exported"] == 1
    assert output_path.exists()


def test_database_backup_and_restore_test_commands_are_disposable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    database_path = tmp_path / "data" / "backup-source.sqlite3"
    backup_dir = tmp_path / "private-backups"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("DATABASE_BACKUP_DIR", str(backup_dir))
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    database = Database(database_path)
    database.initialize()
    database.save_post(
        FacebookPost(
            external_post_id="backup-command-fixture",
            group_id="fixture-group",
            group_name="Synthetic Group",
            post_text="Synthetic database backup command fixture.",
        )
    )

    assert main(["database-backup"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["created"] == created["verified"] == 1
    backup_path = Path(created["backup_path"])
    assert backup_path.exists()

    assert main(["database-restore-test", "--backup-path", str(backup_path)]) == 0
    restored = json.loads(capsys.readouterr().out)
    assert restored == {"backup_path": str(backup_path.resolve()), "verified": True}
    assert not list(backup_dir.glob(".restore-test.*"))


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


def test_feedback_replay_and_backup_parsers_accept_bounded_options(tmp_path: Path) -> None:
    replay = build_parser().parse_args(
        ["classification-replay", "--lead-id", "12", "--limit", "20", "--changed-only"]
    )
    reclassify = build_parser().parse_args(["reclassify-leads", "--lead-id", "12"])
    exported = build_parser().parse_args(
        ["export-regression-fixtures", "--limit", "25", "--output", str(tmp_path / "out.json")]
    )
    restored = build_parser().parse_args(
        ["database-restore-test", "--backup-path", str(tmp_path / "backup.sqlite3")]
    )

    assert replay.lead_id == reclassify.lead_id == 12
    assert replay.limit == 20
    assert replay.changed_only is True
    assert exported.limit == 25
    assert exported.output == tmp_path / "out.json"
    assert restored.backup_path == tmp_path / "backup.sqlite3"


def test_approval_parser_accepts_local_port_and_candidate_limit() -> None:
    args = build_parser().parse_args(["approval-dashboard", "--port", "9876", "--limit", "5"])

    assert args.port == 9876
    assert args.limit == 5


def test_approval_parser_rejects_privileged_port() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["approval-dashboard", "--port", "80"])


def test_remote_approval_parser_accepts_safe_runtime_options() -> None:
    args = build_parser().parse_args(
        ["remote-approval", "--port", "9877", "--limit", "4", "--retry-failed"]
    )

    assert args.port == 9877
    assert args.limit == 4
    assert args.retry_failed is True


def test_run_cycle_parser_accepts_bounded_manual_options() -> None:
    args = build_parser().parse_args(
        [
            "run-cycle",
            "--max-posts",
            "12",
            "--classification-limit",
            "80",
            "--skip-notifications",
            "--ignore-quiet-hours",
        ]
    )

    assert args.max_posts == 12
    assert args.classification_limit == 80
    assert args.skip_notifications is True
    assert args.ignore_quiet_hours is True


def test_operations_pause_status_and_resume_do_not_require_ai_or_facebook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))

    assert main(["operations-pause"]) == 0
    capsys.readouterr()
    assert main(["run-cycle"]) == 0
    assert "paused" in capsys.readouterr().out
    assert main(["operations-status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["paused"] is True
    assert status["status"] == "paused"
    assert status["quiet_hours_enabled"] is True
    assert status["quiet_hours_start"] == "22:00"
    assert status["quiet_hours_end"] == "05:00"
    assert status["quiet_hours_timezone"] == "America/New_York"
    assert main(["operations-resume"]) == 0
    assert "resumed" in capsys.readouterr().out


def test_operations_status_does_not_report_expected_quiet_time_as_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    state = OperationsState(
        OperationPaths(
            state_dir=tmp_path / "data" / "operations",
            log_dir=tmp_path / "data" / "logs",
            screenshot_dir=tmp_path / "screenshots",
        )
    )
    state.mark_running(started_at=datetime(2026, 8, 8, tzinfo=UTC))
    monkeypatch.setattr(cli_module, "_quiet_hours_active", lambda *args, **kwargs: True)

    assert main(["operations-status"]) == 0
    status = json.loads(capsys.readouterr().out)

    assert status["quiet_hours_active"] is True
    assert status["stale"] is False


def test_group_report_contains_counts_but_no_post_content_or_urls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    database_path = tmp_path / "data" / "quality.sqlite3"
    groups_path = tmp_path / "groups.yaml"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("GROUPS_CONFIG_PATH", str(groups_path))
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    groups_path.write_text(
        """
groups:
  - id: fixture-group
    name: Fixture Group
    url: https://www.facebook.com/groups/111
    enabled: true
    priority: 2
""",
        encoding="utf-8",
    )
    database = Database(database_path)
    database.initialize()
    post = database.save_post(
        FacebookPost(
            external_post_id="private-post",
            post_url="https://www.facebook.com/groups/111/posts/private-post",
            group_id="fixture-group",
            group_name="Fixture Group",
            post_text="Private customer needs a deck repaired.",
        )
    ).post
    database.create_lead(
        Lead(
            facebook_post_id=post.id or 0,
            status=LeadStatus.CANDIDATE,
            service_category="decks",
            intent=LeadIntent.HIRING,
            overall_score=90,
            drafted_response="Fixture draft",
        )
    )

    result = main(["group-report"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert result == 0
    assert payload[0]["group_id"] == "fixture-group"
    assert payload[0]["priority"] == 2
    assert payload[0]["posts_discovered"] == 1
    assert "Private customer" not in output
    assert "facebook.com" not in output


def test_post_approved_parser_defaults_to_validation_only() -> None:
    args = build_parser().parse_args(["post-approved", "--lead-id", "12"])

    assert args.lead_id == 12
    assert args.submit is False


def test_post_approved_submit_requires_both_live_safety_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))

    result = main(["post-approved", "--lead-id", "12", "--submit"])

    assert result == 2
    assert "POSTING_ENABLED=false" in capsys.readouterr().err


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
        business_timezone: str,
    ) -> None:
        calls["port"] = port
        calls["candidate_limit"] = candidate_limit
        assert business_timezone == "America/New_York"
        assert service.__class__.__name__ == "LocalApprovalService"

    monkeypatch.setattr(cli_module, "run_local_approval_dashboard", fake_dashboard)

    result = main(["approval-dashboard", "--port", "9876", "--limit", "4"])

    assert result == 0
    assert calls == {"port": 9876, "candidate_limit": 4}
    assert database_path.exists()


def test_remote_approval_command_fails_closed_until_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))

    result = main(["remote-approval"])

    assert result == 2
    assert "NOTIFICATIONS_ENABLED=true" in capsys.readouterr().err


def test_remote_approval_command_uses_provider_abstraction_and_loopback_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "remote.sqlite3"))
    monkeypatch.setenv("NOTIFICATIONS_ENABLED", "true")
    monkeypatch.setenv("SMS_PROVIDER", "telnyx")
    monkeypatch.setenv("REMOTE_APPROVAL_BASE_URL", "https://approve.example")
    monkeypatch.setenv("APPROVAL_SIGNING_KEY", "s" * 48)
    monkeypatch.setenv("SMS_RECIPIENT_NUMBER", "+15025550101")
    monkeypatch.setenv("TELNYX_API_KEY", "fixture-secret")
    monkeypatch.setenv("TELNYX_FROM_NUMBER", "+15025550100")
    calls: dict[str, object] = {}

    class FakeProvider:
        name = "fake"

        def send(self, message: SmsMessage) -> SmsDeliveryReceipt:
            raise AssertionError(
                f"No fixture lead should send a message: {message.idempotency_key}"
            )

    def fake_run_server(
        controller: object,
        *,
        port: int,
        public_base_url: str,
        periodic_callback: object,
        callback_interval_seconds: int,
    ) -> None:
        calls.update(
            controller=controller.__class__.__name__,
            port=port,
            public_base_url=public_base_url,
            periodic_callback=callable(periodic_callback),
            callback_interval_seconds=callback_interval_seconds,
        )

    monkeypatch.setattr(cli_module, "build_sms_provider", lambda settings: FakeProvider())
    monkeypatch.setattr(cli_module, "run_remote_approval_server", fake_run_server)

    result = main(["remote-approval", "--port", "9877", "--limit", "4"])

    assert result == 0
    assert calls == {
        "controller": "RemoteApprovalController",
        "port": 9877,
        "public_base_url": "https://approve.example",
        "periodic_callback": True,
        "callback_interval_seconds": 10,
    }


def test_posting_queue_worker_exits_without_opening_browser_when_queue_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    groups_path = tmp_path / "config" / "groups.yaml"
    groups_path.parent.mkdir()
    groups_path.write_text(
        """
groups:
  - id: fixture-group
    name: Fixture Group
    url: https://www.facebook.com/groups/111
    enabled: true
    posting_enabled: true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "queue.sqlite3"))
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    monkeypatch.setenv("POSTING_ENABLED", "true")
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("POSTING_QUEUE_ENABLED", "true")
    monkeypatch.setenv("NOTIFICATIONS_ENABLED", "true")
    monkeypatch.setenv("SMS_PROVIDER", "telnyx")
    monkeypatch.setenv("REMOTE_APPROVAL_BASE_URL", "https://approve.example")
    monkeypatch.setenv("APPROVAL_SIGNING_KEY", "s" * 48)
    monkeypatch.setenv("SMS_RECIPIENT_NUMBER", "+15025550101")
    monkeypatch.setenv("TELNYX_API_KEY", "fixture-secret")
    monkeypatch.setenv("TELNYX_FROM_NUMBER", "+15025550100")

    class FakeProvider:
        name = "fake"

        def send(self, message: SmsMessage) -> SmsDeliveryReceipt:
            raise AssertionError(f"No outcome should be sent: {message.idempotency_key}")

    monkeypatch.setattr(cli_module, "_quiet_hours_active", lambda settings: False)
    monkeypatch.setattr(cli_module, "build_sms_provider", lambda settings: FakeProvider())
    monkeypatch.setattr(
        cli_module,
        "FacebookCommentBrowser",
        lambda settings: pytest.fail("An empty queue must not open Facebook"),
    )

    result = main(["process-posting-queue"])

    assert result == 0
    assert "No queued Facebook submissions" in capsys.readouterr().out


def test_unattended_cycle_reports_bounded_retries_and_isolates_group_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    groups = [
        FacebookGroup(
            id="recovering",
            name="Recovering Group",
            url="https://www.facebook.com/groups/111",
            enabled=True,
        ),
        FacebookGroup(
            id="failing",
            name="Failing Group",
            url="https://www.facebook.com/groups/222",
            enabled=True,
        ),
    ]
    failed_after_retry = TransientFacebookReadError(stage="navigation", kind="timeout")
    failed_after_retry.retry_count = 1
    outcomes: dict[str, ScanSummary | Exception] = {
        "recovering": ScanSummary(
            groups[0],
            posts_seen=6,
            new_posts=(),
            posts_requested=10,
            retry_count=1,
            recovered=True,
        ),
        "failing": failed_after_retry,
    }
    sleeps: list[float] = []

    class FakeBrowser:
        def __init__(self, settings: Settings) -> None:
            del settings

        async def __aenter__(self) -> "FakeBrowser":
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

    class FakeScanner:
        def __init__(self, database: Database, browser: object) -> None:
            del database, browser

        async def scan_group(
            self,
            group: FacebookGroup,
            *,
            max_posts: int,
            **kwargs: object,
        ) -> ScanSummary:
            del kwargs
            assert max_posts == 10
            outcome = outcomes[group.id]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(cli_module, "FacebookReadOnlyBrowser", FakeBrowser)
    monkeypatch.setattr(cli_module, "ReadOnlyScanService", FakeScanner)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "cycle.sqlite3",
        data_dir=tmp_path / "data",
        facebook_profile_path=tmp_path.parent / "browser-profile",
        facebook_group_max_retries=1,
        facebook_group_retry_backoff_seconds=5,
        facebook_group_delay_seconds=2,
    )

    summary = asyncio.run(cli_module._scan_groups_for_cycle(settings, groups, max_posts=10))

    assert summary.groups_scanned == 1
    assert summary.groups_failed == 1
    assert summary.groups_retried == 2
    assert summary.groups_recovered == 1
    assert summary.groups_partial == 1
    assert summary.groups_severely_partial == 0
    assert summary.posts_seen == 6
    assert summary.posts_requested == 20
    assert sleeps == [2]


def test_unattended_cycle_never_retries_a_facebook_safety_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    group = FacebookGroup(
        id="checkpoint",
        name="Checkpoint Group",
        url="https://www.facebook.com/groups/111",
        enabled=True,
    )

    class FakeBrowser:
        def __init__(self, settings: Settings) -> None:
            del settings

        async def __aenter__(self) -> "FakeBrowser":
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

    class FakeScanner:
        def __init__(self, database: Database, browser: object) -> None:
            del database, browser

        async def scan_group(
            self,
            group: FacebookGroup,
            *,
            max_posts: int,
            **kwargs: object,
        ) -> ScanSummary:
            del group, max_posts, kwargs
            raise FacebookSafetyStop(FacebookPageState.CHECKPOINT, "human review required")

    monkeypatch.setattr(cli_module, "FacebookReadOnlyBrowser", FakeBrowser)
    monkeypatch.setattr(cli_module, "ReadOnlyScanService", FakeScanner)
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "safety-stop.sqlite3",
        data_dir=tmp_path / "data",
        facebook_profile_path=tmp_path.parent / "browser-profile",
    )

    with pytest.raises(FacebookSafetyStop):
        asyncio.run(cli_module._scan_groups_for_cycle(settings, [group], max_posts=10))


def configure_cycle_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    groups_path = tmp_path / "groups.yaml"
    groups_path.write_text(
        """
groups:
  - id: fixture-group
    name: Fixture Group
    url: https://www.facebook.com/groups/111
    enabled: true
    posting_enabled: true
    priority: 1
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "data" / "cycle.sqlite3"))
    monkeypatch.setenv("GROUPS_CONFIG_PATH", str(groups_path))
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    monkeypatch.setenv("AI_PROVIDER", "heuristic")
    monkeypatch.setenv("OPERATIONS_QUIET_HOURS_ENABLED", "false")
    return groups_path


def test_run_cycle_command_executes_content_free_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_cycle_fixture(monkeypatch, tmp_path)

    async def fake_scan(
        settings: Settings,
        groups: list[FacebookGroup],
        *,
        max_posts: int,
    ) -> ScanCycleSummary:
        del settings
        assert [group.id for group in groups] == ["fixture-group"]
        assert max_posts == 10
        return ScanCycleSummary(
            groups_scanned=1,
            groups_failed=0,
            posts_seen=4,
            posts_new=0,
            duplicates=4,
            groups_retried=1,
            groups_recovered=1,
        )

    monkeypatch.setattr(cli_module, "_scan_groups_for_cycle", fake_scan)

    result = main(
        ["run-cycle", "--max-posts", "10", "--classification-limit", "5", "--skip-notifications"]
    )
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert result == 0
    assert payload["status"] == "success"
    assert payload["scan"]["groups_retried"] == 1
    assert payload["scan"]["groups_recovered"] == 1
    database = Database(tmp_path / "data" / "cycle.sqlite3")
    events = database.list_audit_events()
    assert [event.action for event in events] == ["cycle.run"]
    assert events[0].details == {
        "candidates_created": 0,
        "duplicates": 4,
        "groups_failed": 0,
        "groups_shortfall": 0,
        "groups_partial": 0,
        "groups_severely_partial": 0,
        "groups_feed_responsive_partial": 0,
        "groups_recovered": 1,
        "groups_retried": 1,
        "groups_scanned": 1,
        "notifications_considered": 0,
        "notifications_failed": 0,
        "notifications_sent": 0,
        "posts_classified": 0,
        "posts_ignored": 0,
        "posts_new": 0,
        "posts_seen": 4,
        "posts_requested": 0,
        "circuit_breaker_tripped": False,
        "backups_created": 1,
        "backups_verified": 1,
        "backups_removed": 0,
    }


def test_run_cycle_records_only_safe_failure_type_for_dashboard_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_cycle_fixture(monkeypatch, tmp_path)

    async def failed_scan(*args: object, **kwargs: object) -> ScanCycleSummary:
        del args, kwargs
        raise RuntimeError("private Facebook content must not enter history")

    monkeypatch.setattr(cli_module, "_scan_groups_for_cycle", failed_scan)

    result = main(["run-cycle", "--skip-notifications"])

    assert result == 2
    assert "private Facebook content" not in capsys.readouterr().err
    database = Database(tmp_path / "data" / "cycle.sqlite3")
    events = database.list_audit_events(component="operations", action="cycle.run")
    assert len(events) == 1
    assert events[0].result == "failed"
    assert events[0].details == {
        "error_code": "RuntimeError",
        "circuit_breaker_tripped": False,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ["run-cycle", "--max-posts", "51"],
        ["run-cycle", "--classification-limit", "1001"],
    ],
)
def test_run_cycle_command_enforces_safety_caps_without_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    configure_cycle_fixture(monkeypatch, tmp_path)

    result = main(arguments)

    assert result == 2
    assert "Stopped safely:" in capsys.readouterr().err


def test_run_cycle_command_handles_active_lock_without_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_cycle_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_run_operations_cycle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CycleAlreadyRunningError("fixture cycle already running")
        ),
    )

    result = main(["run-cycle"])

    assert result == 0
    assert "already running" in capsys.readouterr().out


def test_run_cycle_skips_quiet_hours_before_database_or_facebook_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    database_path = tmp_path / "data" / "quiet.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("FACEBOOK_PROFILE_PATH", str(tmp_path.parent / "browser-profile"))
    monkeypatch.setattr(cli_module, "_quiet_hours_active", lambda *args, **kwargs: True)

    result = main(["run-cycle"])

    assert result == 0
    assert "quiet hours are active" in capsys.readouterr().out
    assert not database_path.exists()


def test_cycle_gate_uses_eastern_wall_clock_and_manual_override(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        database_path=tmp_path / "quiet.sqlite3",
        groups_config_path=tmp_path / "missing-groups.yaml",
        facebook_profile_path=tmp_path.parent / "browser-profile",
    )
    overnight = datetime(2026, 8, 9, 3, tzinfo=ZoneInfo("America/New_York"))

    with pytest.raises(QuietHoursActiveError, match="22:00-05:00"):
        cli_module._run_operations_cycle(
            settings,
            max_posts=10,
            classification_limit=10,
            skip_notifications=True,
            now=overnight,
        )
    assert not settings.database_path.exists()
    with pytest.raises(GroupsConfigError):
        cli_module._run_operations_cycle(
            settings,
            max_posts=10,
            classification_limit=10,
            skip_notifications=True,
            ignore_quiet_hours=True,
            now=overnight,
        )


def test_scan_command_uses_allowlist_and_prints_new_posts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_cycle_fixture(monkeypatch, tmp_path)
    discovered = FacebookPost(
        external_post_id="222",
        post_url="https://www.facebook.com/groups/111/posts/222",
        group_id="fixture-group",
        group_name="Fixture Group",
        post_text="Need someone for a deck repair estimate.",
    )

    async def fake_scan(
        settings: Settings,
        groups: list[FacebookGroup],
        *,
        max_posts: int,
        pause_after_scan: bool,
    ) -> list[ScanSummary]:
        del settings
        assert max_posts == 10
        assert pause_after_scan is False
        return [ScanSummary(groups[0], posts_seen=1, new_posts=(discovered,))]

    monkeypatch.setattr(cli_module, "_scan_groups", fake_scan)

    result = main(["scan-facebook", "--group-id", "fixture-group", "--max-posts", "10"])
    output = capsys.readouterr().out

    assert result == 0
    assert "seen=1 new=1 duplicates=0" in output
    assert discovered.post_url is not None
    assert discovered.post_url in output
    assert discovered.post_text in output


def test_manual_login_command_reports_success_and_safe_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_cycle_fixture(monkeypatch, tmp_path)

    async def successful_login(settings: Settings) -> None:
        del settings

    monkeypatch.setattr(cli_module, "_manual_login", successful_login)
    assert main(["facebook-login"]) == 0
    assert "verified" in capsys.readouterr().out

    async def failed_login(settings: Settings) -> None:
        del settings
        raise FacebookSafetyStop(FacebookPageState.LOGIN_REQUIRED, "manual login required")

    monkeypatch.setattr(cli_module, "_manual_login", failed_login)
    assert main(["facebook-login"]) == 2
    assert "manual login required" in capsys.readouterr().err


def test_post_approved_dry_run_reports_validation_without_browser_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_cycle_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_module, "ApprovedPostingService", lambda *args, **kwargs: object())

    async def fake_execute(
        settings: Settings,
        service: object,
        *,
        lead_id: int,
        dry_run: bool,
    ) -> object:
        del settings, service
        assert lead_id == 12
        assert dry_run is True
        return SimpleNamespace(
            created=True,
            work=SimpleNamespace(attempt=SimpleNamespace(status=SimpleNamespace(value="dry_run"))),
        )

    monkeypatch.setattr(cli_module, "_execute_approved_posting", fake_execute)

    result = main(["post-approved", "--lead-id", "12"])

    assert result == 0
    assert "DRY RUN lead=12 validated" in capsys.readouterr().out


def test_post_approved_reports_pending_group_moderation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_cycle_fixture(monkeypatch, tmp_path)
    monkeypatch.setenv("POSTING_ENABLED", "true")
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setattr(cli_module, "ApprovedPostingService", lambda *args, **kwargs: object())

    async def fake_execute(
        settings: Settings,
        service: object,
        *,
        lead_id: int,
        dry_run: bool,
    ) -> object:
        del settings, service
        assert lead_id == 12
        assert dry_run is False
        return SimpleNamespace(
            created=True,
            work=SimpleNamespace(
                attempt=SimpleNamespace(status=SimpleNamespace(value="pending_moderation"))
            ),
        )

    monkeypatch.setattr(cli_module, "_execute_approved_posting", fake_execute)

    result = main(["post-approved", "--lead-id", "12", "--submit"])

    assert result == 0
    assert "PENDING MODERATION lead=12" in capsys.readouterr().out


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

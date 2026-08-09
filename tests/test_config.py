from datetime import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from lead_agent.config import (
    NotificationConfigurationError,
    PostingDisabledError,
    Settings,
    UnsafeReadOnlyModeError,
)


def test_safe_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=None)

    assert settings.posting_enabled is False
    assert settings.dry_run is True
    assert settings.posting_allowed is False
    assert settings.scan_interval_seconds == 900
    assert settings.lead_threshold == 75
    assert settings.approval_expiration_minutes == 20
    assert settings.approval_local_port == 8765
    assert settings.remote_approval_port == 8766
    assert settings.posting_approval_max_age_minutes == 20
    assert settings.daily_posting_limit == 5
    assert settings.per_group_daily_posting_limit == 2
    assert settings.business_timezone == "America/New_York"
    assert settings.database_backup_retention_days == 14
    assert settings.database_backup_interval_hours == 24
    assert settings.database_backup_dir == Path("data/backups")
    assert settings.operations_quiet_hours_enabled is True
    assert settings.operations_quiet_hours_start == time(hour=22)
    assert settings.operations_quiet_hours_end == time(hour=5)
    assert settings.operations_minimum_group_post_yield_rate == 0.5
    assert settings.browser_channel is None
    assert settings.browser_headless is False
    assert settings.facebook_group_max_retries == 1
    assert settings.facebook_group_retry_backoff_seconds == 5
    assert settings.facebook_group_delay_seconds == 2
    assert settings.facebook_max_scrolls == 12
    assert settings.facebook_scroll_settle_seconds == 0.75
    assert settings.ai_provider == "disabled"
    assert settings.ai_model == "gemini-2.5-flash"
    assert settings.gemini_api_key is None
    assert settings.ai_max_posts_per_run == 20
    assert settings.sms_provider == "disabled"
    assert settings.remote_approval_ready is False


def test_backup_directory_expands_user_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=None, database_backup_dir=Path("~/private-lead-backups"))

    assert settings.database_backup_dir == Path.home() / "private-lead-backups"


def test_quiet_hours_must_use_distinct_minute_precision_times(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="start and end must differ"):
        Settings(
            _env_file=None,
            operations_quiet_hours_start=time(hour=5),
            operations_quiet_hours_end=time(hour=5),
        )
    with pytest.raises(ValidationError, match="hour-and-minute precision"):
        Settings(
            _env_file=None,
            operations_quiet_hours_start=time(hour=22, second=1),
        )


def test_business_timezone_must_be_valid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="IANA timezone"):
        Settings(_env_file=None, business_timezone="Louisville/Invalid")


@pytest.mark.parametrize(
    ("posting_enabled", "dry_run", "expected_message"),
    [
        (False, False, "POSTING_ENABLED=false"),
        (True, True, "DRY_RUN=true"),
    ],
)
def test_posting_interlock_fails_closed(
    posting_enabled: bool,
    dry_run: bool,
    expected_message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        _env_file=None,
        posting_enabled=posting_enabled,
        dry_run=dry_run,
    )

    with pytest.raises(PostingDisabledError, match=expected_message):
        settings.require_posting_allowed()


def test_posting_requires_two_explicit_configuration_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, posting_enabled=True, dry_run=False)

    settings.require_posting_allowed()

    assert settings.posting_allowed is True


@pytest.mark.parametrize(
    ("posting_enabled", "dry_run", "expected_message"),
    [
        (True, True, "POSTING_ENABLED=false"),
        (False, False, "DRY_RUN=true"),
    ],
)
def test_read_only_browser_requires_both_safe_flags(
    posting_enabled: bool,
    dry_run: bool,
    expected_message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        _env_file=None,
        posting_enabled=posting_enabled,
        dry_run=dry_run,
    )

    with pytest.raises(UnsafeReadOnlyModeError, match=expected_message):
        settings.require_read_only_mode()


def test_read_only_browser_accepts_safe_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, posting_enabled=False, dry_run=True)

    settings.require_read_only_mode()


def test_empty_browser_channel_uses_bundled_chromium(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=None, browser_channel="  ")

    assert settings.browser_channel is None


def test_remote_approval_requires_https_origin_and_e164_numbers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="HTTPS"):
        Settings(
            _env_file=None,
            facebook_profile_path=tmp_path.parent / "browser-profile",
            remote_approval_base_url="http://approve.example",
        )
    with pytest.raises(ValidationError, match=r"E\.164"):
        Settings(
            _env_file=None,
            facebook_profile_path=tmp_path.parent / "browser-profile",
            sms_recipient_number="502-555-1234",
        )


def test_remote_approval_origin_must_leave_room_for_compliant_sms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    Settings(
        _env_file=None,
        facebook_profile_path=tmp_path.parent / "browser-profile",
        remote_approval_base_url=f"https://{'a' * 41}.com",
    )
    with pytest.raises(ValidationError, match="too long for one SMS segment"):
        Settings(
            _env_file=None,
            facebook_profile_path=tmp_path.parent / "browser-profile",
            remote_approval_base_url=f"https://{'a' * 42}.com",
        )


def test_remote_approval_readiness_never_exposes_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        _env_file=None,
        facebook_profile_path=tmp_path.parent / "browser-profile",
        notifications_enabled=True,
        sms_provider=" TELNYX ",
        remote_approval_base_url="https://approve.example",
        approval_signing_key="s" * 48,
        sms_recipient_number="+15025550101",
        telnyx_api_key="telnyx-secret",
        telnyx_from_number="+15025550100",
    )

    settings.require_remote_approval_ready()

    assert settings.remote_approval_ready is True
    assert "telnyx-secret" not in repr(settings.telnyx_api_key)


def test_remote_approval_fails_closed_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        _env_file=None,
        facebook_profile_path=tmp_path.parent / "browser-profile",
    )

    with pytest.raises(NotificationConfigurationError, match="NOTIFICATIONS_ENABLED"):
        settings.require_remote_approval_ready()


def test_environment_overrides_and_comma_separated_services(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LEAD_THRESHOLD", "88")
    monkeypatch.setenv("ENABLED_SERVICES", "decks, drywall, flooring")
    monkeypatch.setenv("OPERATIONS_QUIET_HOURS_ENABLED", "false")
    monkeypatch.setenv("OPERATIONS_QUIET_HOURS_START", "21:30")
    monkeypatch.setenv("OPERATIONS_QUIET_HOURS_END", "04:30")

    settings = Settings(_env_file=None)

    assert settings.lead_threshold == 88
    assert settings.enabled_services == ["decks", "drywall", "flooring"]
    assert settings.operations_quiet_hours_enabled is False
    assert settings.operations_quiet_hours_start == time(hour=21, minute=30)
    assert settings.operations_quiet_hours_end == time(hour=4, minute=30)


def test_ai_provider_and_secret_configuration_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        _env_file=None,
        facebook_profile_path=tmp_path.parent / "browser-profile",
        ai_provider=" GEMINI ",
        ai_model=" gemini-fixture ",
        gemini_api_key="placeholder",
    )

    assert settings.ai_provider == "gemini"
    assert settings.ai_model == "gemini-fixture"
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "placeholder"
    assert "placeholder" not in repr(settings.gemini_api_key)


def test_invalid_ai_provider_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="AI_PROVIDER"):
        Settings(
            _env_file=None,
            facebook_profile_path=tmp_path.parent / "browser-profile",
            ai_provider="unknown",
        )


def test_browser_profile_must_be_outside_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="must be outside the repository"):
        Settings(_env_file=None, facebook_profile_path=tmp_path / "facebook-profile")


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="LOG_LEVEL"):
        Settings(_env_file=None, log_level="verbose")

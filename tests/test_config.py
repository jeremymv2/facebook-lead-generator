from pathlib import Path

import pytest
from pydantic import ValidationError

from lead_agent.config import PostingDisabledError, Settings


def test_safe_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=None)

    assert settings.posting_enabled is False
    assert settings.dry_run is True
    assert settings.posting_allowed is False
    assert settings.scan_interval_seconds == 300
    assert settings.lead_threshold == 75


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


def test_environment_overrides_and_comma_separated_services(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LEAD_THRESHOLD", "88")
    monkeypatch.setenv("ENABLED_SERVICES", "decks, drywall, flooring")

    settings = Settings(_env_file=None)

    assert settings.lead_threshold == 88
    assert settings.enabled_services == ["decks", "drywall", "flooring"]


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

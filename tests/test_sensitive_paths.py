import pytest

from scripts.check_sensitive_paths import main, sensitive_path_reason


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.production",
        "config/groups.yaml",
        "facebook-profile/Default/Cookies",
        "playwright/.auth/user.json",
        "screenshots/facebook-checkpoint.png",
        "data/leads.sqlite3",
        "logs/scanner.log",
        "private/id_ed25519",
        "secrets/client_secret_google.json",
        "network/session.har",
        "storage-state-jeremy.json",
        ".cloudflared/fixture-tunnel.json",
    ],
)
def test_sensitive_paths_are_rejected(path: str) -> None:
    assert sensitive_path_reason(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        ".gitleaks.toml",
        "config/groups.example.yaml",
        "data/.gitkeep",
        "screenshots/.gitkeep",
        "tests/fixtures/facebook_posts.json",
        "src/lead_agent/config.py",
    ],
)
def test_safe_paths_are_allowed(path: str) -> None:
    assert sensitive_path_reason(path) is None


def test_main_reports_rejected_paths(capsys: pytest.CaptureFixture[str]) -> None:
    result = main([".env", "src/lead_agent/config.py"])

    captured = capsys.readouterr()
    assert result == 1
    assert ".env: environment file" in captured.err
    assert "config.py" not in captured.err


def test_main_accepts_safe_paths() -> None:
    assert main([".env.example", "README.md"]) == 0


def test_all_tracked_mode_checks_git_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.check_sensitive_paths.tracked_paths",
        lambda: ["src/lead_agent/config.py", "secrets/id_rsa"],
    )

    assert main(["--all-tracked"]) == 1

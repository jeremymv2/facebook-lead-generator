"""Application configuration and posting safety interlocks."""

from pathlib import Path
from typing import Annotated

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DEFAULT_SERVICES = [
    "general_contracting",
    "handyman",
    "kitchen_remodeling",
    "bathroom_remodeling",
    "cabinet_installation",
    "drywall",
    "painting",
    "carpentry",
    "doors",
    "windows",
    "decks",
    "pressure_washing",
    "fencing",
    "flooring",
    "tile",
    "plumbing_fixtures",
    "landscaping",
    "porches",
    "patios",
    "framing",
    "structural_repairs",
    "general_home_repairs",
]


class PostingDisabledError(RuntimeError):
    """Raised when code requests posting while a safety control is active."""


class UnsafeReadOnlyModeError(RuntimeError):
    """Raised when a read-only command is run with unsafe feature flags."""


class Settings(BaseSettings):
    """Validated settings loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    posting_enabled: bool = False
    dry_run: bool = True

    scan_interval_seconds: int = Field(default=300, ge=60)
    lead_threshold: int = Field(default=75, ge=0, le=100)
    service_area: str = "Louisville, Kentucky"
    service_radius_miles: int = Field(default=50, ge=1, le=250)
    enabled_services: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_SERVICES)
    )

    approval_expiration_minutes: int = Field(default=20, ge=1, le=120)
    daily_posting_limit: int = Field(default=5, ge=1, le=100)
    per_group_daily_posting_limit: int = Field(default=2, ge=1, le=50)
    screenshot_retention_days: int = Field(default=14, ge=1, le=365)

    data_dir: Path = Path("data")
    database_path: Path = Path("data/lead_agent.sqlite3")
    screenshot_dir: Path = Path("screenshots")
    groups_config_path: Path = Path("config/groups.yaml")
    facebook_profile_path: Path = Path("~/.jjmiller-lead-agent/facebook-profile")
    browser_channel: str | None = None
    browser_headless: bool = False
    facebook_navigation_timeout_seconds: int = Field(default=30, ge=10, le=120)
    facebook_post_load_timeout_seconds: int = Field(default=15, ge=5, le=60)
    max_posts_per_group: int = Field(default=20, ge=1, le=50)
    min_post_text_length: int = Field(default=15, ge=1, le=500)

    ai_provider: str = "disabled"
    ai_model: str = ""
    approval_api_url: HttpUrl | None = None
    notifications_enabled: bool = False

    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("enabled_services", mode="before")
    @classmethod
    def parse_enabled_services(cls, value: object) -> object:
        """Allow either JSON-like lists or a convenient comma-separated env value."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                stripped = stripped[1:-1]
            return [
                item.strip().strip('"').strip("'") for item in stripped.split(",") if item.strip()
            ]
        return value

    @field_validator("data_dir", "database_path", "screenshot_dir", "groups_config_path")
    @classmethod
    def normalize_local_path(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("facebook_profile_path")
    @classmethod
    def protect_browser_profile(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        repository_root = Path.cwd().resolve()
        if resolved == repository_root or repository_root in resolved.parents:
            raise ValueError("FACEBOOK_PROFILE_PATH must be outside the repository")
        return resolved

    @field_validator("browser_channel", mode="before")
    @classmethod
    def normalize_browser_channel(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("approval_api_url", mode="before")
    @classmethod
    def normalize_optional_url(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("LOG_LEVEL must be a standard Python logging level")
        return normalized

    @property
    def posting_allowed(self) -> bool:
        """Return true only when both independent posting controls permit submission."""
        return self.posting_enabled and not self.dry_run

    def require_posting_allowed(self) -> None:
        """Fail closed before any future code can submit a Facebook comment."""
        if not self.posting_enabled:
            raise PostingDisabledError("Posting is disabled by POSTING_ENABLED=false")
        if self.dry_run:
            raise PostingDisabledError("Posting is disabled while DRY_RUN=true")

    def require_read_only_mode(self) -> None:
        """Require both safe defaults before browser-based discovery can run."""
        if self.posting_enabled:
            raise UnsafeReadOnlyModeError(
                "Read-only Facebook commands require POSTING_ENABLED=false"
            )
        if not self.dry_run:
            raise UnsafeReadOnlyModeError("Read-only Facebook commands require DRY_RUN=true")


def load_settings() -> Settings:
    """Create settings at call time so tests and commands can control their environment."""
    return Settings()

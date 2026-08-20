"""Application configuration and posting safety interlocks."""

import re
from datetime import time
from pathlib import Path
from typing import Annotated, Self
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
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
    "roof_repair",
    "masonry",
    "exterior_repairs",
    "gutters_and_drainage",
    "outdoor_structures",
    "demolition",
    "installations_and_mounting",
    "insulation_and_air_sealing",
    "minor_plumbing_repairs",
    "appliance_installation",
    "electrical_fixtures",
    "ventilation",
    "property_maintenance",
    "cleanup_and_hauling",
    "project_coordination",
]

# The production review token is 43 characters. This limit leaves enough room for the exact
# registered brand name, review path, and required opt-out language in one 160-character SMS.
MAX_REMOTE_APPROVAL_BASE_URL_LENGTH = 53


class PostingDisabledError(RuntimeError):
    """Raised when code requests posting while a safety control is active."""


class UnsafeReadOnlyModeError(RuntimeError):
    """Raised when a read-only command is run with unsafe feature flags."""


class NotificationConfigurationError(RuntimeError):
    """Raised when remote approval notifications are not safely configured."""


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
    posting_queue_enabled: bool = False
    posting_queue_poll_interval_seconds: int = Field(default=60, ge=30, le=900)

    scan_interval_seconds: int = Field(default=900, ge=300, le=3600)
    lead_threshold: int = Field(default=75, ge=0, le=100)
    service_area: str = "Louisville, Kentucky"
    service_radius_miles: int = Field(default=50, ge=1, le=250)
    enabled_services: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_SERVICES)
    )

    approval_expiration_minutes: int = Field(default=600, ge=1, le=720)
    candidate_duplicate_window_hours: int = Field(default=72, ge=1, le=720)
    approval_local_port: int = Field(default=8765, ge=1024, le=65535)
    remote_approval_port: int = Field(default=8766, ge=1024, le=65535)
    posting_approval_max_age_minutes: int = Field(default=20, ge=1, le=120)
    daily_posting_limit: int = Field(default=5, ge=1, le=100)
    per_group_daily_posting_limit: int = Field(default=2, ge=1, le=50)
    business_timezone: str = "America/New_York"
    screenshot_retention_days: int = Field(default=14, ge=1, le=365)
    operations_log_retention_days: int = Field(default=14, ge=1, le=365)
    operations_log_max_bytes: int = Field(default=5_000_000, ge=65_536, le=100_000_000)
    database_backup_retention_days: int = Field(default=14, ge=1, le=365)
    database_backup_interval_hours: int = Field(default=24, ge=1, le=168)
    operations_quiet_hours_enabled: bool = True
    operations_quiet_hours_start: time = time(hour=22)
    operations_quiet_hours_end: time = time(hour=5)
    operations_degraded_cycle_limit: int = Field(default=2, ge=1, le=10)
    operations_incomplete_group_rate_threshold: float = Field(default=0.25, ge=0.1, le=1)
    operations_minimum_group_post_yield_rate: float = Field(default=0.5, gt=0, le=1)
    operations_healthy_group_post_yield_rate: float = Field(default=0.8, gt=0, le=1)
    cycle_classification_limit: int = Field(default=100, ge=1, le=1_000)

    data_dir: Path = Path("data")
    database_path: Path = Path("data/lead_agent.sqlite3")
    database_backup_dir: Path = Path("data/backups")
    screenshot_dir: Path = Path("screenshots")
    groups_config_path: Path = Path("config/groups.yaml")
    facebook_profile_path: Path = Path("~/.jjmiller-lead-agent/facebook-profile")
    browser_channel: str | None = None
    browser_headless: bool = False
    facebook_navigation_timeout_seconds: int = Field(default=30, ge=10, le=120)
    facebook_post_load_timeout_seconds: int = Field(default=15, ge=5, le=60)
    facebook_scan_max_wait_seconds: int = Field(default=25, ge=10, le=90)
    facebook_scan_idle_seconds: int = Field(default=5, ge=2, le=15)
    facebook_retry_scan_max_wait_seconds: int = Field(default=45, ge=10, le=120)
    facebook_retry_scan_idle_seconds: int = Field(default=10, ge=2, le=30)
    facebook_group_max_retries: int = Field(default=1, ge=0, le=2)
    facebook_group_retry_backoff_seconds: float = Field(default=5, ge=1, le=30)
    facebook_group_delay_seconds: float = Field(default=2, ge=0.25, le=30)
    facebook_max_scrolls: int = Field(default=20, ge=0, le=30)
    facebook_scroll_settle_seconds: float = Field(default=0.75, ge=0.25, le=5)
    facebook_retry_scroll_settle_seconds: float = Field(default=1.25, ge=0.25, le=5)
    max_posts_per_group: int = Field(default=10, ge=1, le=50)
    min_post_text_length: int = Field(default=15, ge=1, le=500)

    ai_provider: str = "disabled"
    ai_model: str = "gemini-2.5-flash"
    gemini_api_key: SecretStr | None = None
    ai_request_timeout_seconds: int = Field(default=30, ge=5, le=120)
    ai_max_posts_per_run: int = Field(default=20, ge=1, le=100)
    ai_max_input_characters: int = Field(default=5000, ge=500, le=20000)
    notifications_enabled: bool = False
    notification_poll_interval_seconds: int = Field(default=10, ge=5, le=300)
    remote_approval_base_url: HttpUrl | None = None
    approval_signing_key: SecretStr | None = None
    sms_provider: str = "disabled"
    sms_recipient_number: str | None = None
    sms_request_timeout_seconds: int = Field(default=15, ge=5, le=60)
    telnyx_api_key: SecretStr | None = None
    telnyx_from_number: str | None = None

    log_level: str = "INFO"
    log_json: bool = True

    @property
    def operations_state_dir(self) -> Path:
        return self.data_dir / "operations"

    @property
    def operations_log_dir(self) -> Path:
        return self.data_dir / "logs"

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

    @field_validator(
        "data_dir",
        "database_path",
        "database_backup_dir",
        "screenshot_dir",
        "groups_config_path",
    )
    @classmethod
    def normalize_local_path(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("operations_quiet_hours_start", "operations_quiet_hours_end")
    @classmethod
    def validate_quiet_hours_time(cls, value: time) -> time:
        if value.tzinfo is not None:
            raise ValueError("quiet-hours times must be local wall-clock times")
        if value.second or value.microsecond:
            raise ValueError("quiet-hours times must use hour-and-minute precision")
        return value

    @model_validator(mode="after")
    def validate_quiet_hours_window(self) -> Self:
        if self.operations_quiet_hours_start == self.operations_quiet_hours_end:
            raise ValueError("quiet-hours start and end must differ")
        return self

    @model_validator(mode="after")
    def validate_group_yield_thresholds(self) -> Self:
        if (
            self.operations_minimum_group_post_yield_rate
            >= self.operations_healthy_group_post_yield_rate
        ):
            raise ValueError(
                "severe group yield threshold must be below the healthy yield threshold"
            )
        return self

    @field_validator("scan_interval_seconds")
    @classmethod
    def validate_fixed_scan_interval(cls, value: int) -> int:
        if 3600 % value:
            raise ValueError("SCAN_INTERVAL_SECONDS must evenly divide one hour")
        return value

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

    @field_validator("ai_provider")
    @classmethod
    def validate_ai_provider(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if normalized not in {"disabled", "heuristic", "gemini"}:
            raise ValueError("AI_PROVIDER must be disabled, heuristic, or gemini")
        return normalized

    @field_validator("ai_model")
    @classmethod
    def normalize_ai_model(cls, value: str) -> str:
        return value.strip()

    @field_validator("gemini_api_key", "approval_signing_key", "telnyx_api_key", mode="before")
    @classmethod
    def normalize_optional_secret(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("remote_approval_base_url", mode="before")
    @classmethod
    def normalize_optional_url(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("remote_approval_base_url")
    @classmethod
    def validate_remote_approval_base_url(cls, value: HttpUrl | None) -> HttpUrl | None:
        if value is None:
            return None
        parts = urlsplit(str(value))
        if parts.scheme != "https":
            raise ValueError("REMOTE_APPROVAL_BASE_URL must use HTTPS")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("REMOTE_APPROVAL_BASE_URL must be a plain HTTPS origin")
        if parts.path not in {"", "/"}:
            raise ValueError("REMOTE_APPROVAL_BASE_URL must not include a path")
        if len(str(value).rstrip("/")) > MAX_REMOTE_APPROVAL_BASE_URL_LENGTH:
            raise ValueError("REMOTE_APPROVAL_BASE_URL is too long for one SMS segment")
        hostname = (parts.hostname or "").casefold()
        if hostname in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("REMOTE_APPROVAL_BASE_URL must be reachable from the phone")
        return value

    @field_validator("sms_provider")
    @classmethod
    def validate_sms_provider(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if normalized not in {"disabled", "telnyx"}:
            raise ValueError("SMS_PROVIDER must be disabled or telnyx")
        return normalized

    @field_validator("sms_recipient_number", "telnyx_from_number", mode="before")
    @classmethod
    def normalize_phone_number(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            if re.fullmatch(r"\+[1-9]\d{7,14}", stripped) is None:
                raise ValueError("SMS phone numbers must use E.164 format, such as +15025551234")
            return stripped
        return value

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("LOG_LEVEL must be a standard Python logging level")
        return normalized

    @field_validator("business_timezone")
    @classmethod
    def validate_business_timezone(cls, value: str) -> str:
        normalized = value.strip()
        try:
            ZoneInfo(normalized)
        except ZoneInfoNotFoundError as error:
            raise ValueError("BUSINESS_TIMEZONE must be a valid IANA timezone") from error
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

    def require_posting_queue_enabled(self) -> None:
        if not self.posting_queue_enabled:
            raise PostingDisabledError("Queued posting is disabled by POSTING_QUEUE_ENABLED=false")

    def require_read_only_mode(self) -> None:
        """Require both safe defaults before browser-based discovery can run."""
        if self.posting_enabled:
            raise UnsafeReadOnlyModeError(
                "Read-only Facebook commands require POSTING_ENABLED=false"
            )
        if not self.dry_run:
            raise UnsafeReadOnlyModeError("Read-only Facebook commands require DRY_RUN=true")

    @property
    def remote_approval_ready(self) -> bool:
        """Return whether every secret and endpoint needed for tunneled SMS review exists."""
        signing_key = (
            self.approval_signing_key.get_secret_value()
            if self.approval_signing_key is not None
            else ""
        )
        return (
            self.notifications_enabled
            and self.sms_provider == "telnyx"
            and self.remote_approval_base_url is not None
            and len(signing_key) >= 32
            and self.sms_recipient_number is not None
            and self.telnyx_api_key is not None
            and self.telnyx_from_number is not None
        )

    def require_remote_approval_ready(self) -> None:
        """Fail closed unless tunneled remote approval and Telnyx are fully configured."""
        if not self.notifications_enabled:
            raise NotificationConfigurationError(
                "Remote approval requires NOTIFICATIONS_ENABLED=true"
            )
        if self.sms_provider != "telnyx":
            raise NotificationConfigurationError("Remote approval requires SMS_PROVIDER=telnyx")
        if self.remote_approval_base_url is None:
            raise NotificationConfigurationError("REMOTE_APPROVAL_BASE_URL is required")
        if (
            self.approval_signing_key is None
            or len(self.approval_signing_key.get_secret_value()) < 32
        ):
            raise NotificationConfigurationError(
                "APPROVAL_SIGNING_KEY must contain at least 32 characters"
            )
        if self.sms_recipient_number is None:
            raise NotificationConfigurationError("SMS_RECIPIENT_NUMBER is required")
        if self.telnyx_api_key is None:
            raise NotificationConfigurationError("TELNYX_API_KEY is required")
        if self.telnyx_from_number is None:
            raise NotificationConfigurationError("TELNYX_FROM_NUMBER is required")


def load_settings() -> Settings:
    """Create settings at call time so tests and commands can control their environment."""
    return Settings()

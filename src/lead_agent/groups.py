"""Validated configuration for explicitly approved Facebook groups."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class GroupsConfigError(ValueError):
    """Raised when the approved group catalog is missing or invalid."""


class FacebookGroup(BaseModel):
    """One group with independent read-only scanning and posting permissions."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$", min_length=2, max_length=80)
    name: str = Field(min_length=1, max_length=200)
    url: str
    enabled: bool = False
    posting_enabled: bool = False
    priority: int = Field(default=1, ge=1, le=100)
    geography: str | None = Field(default=None, max_length=200)

    @field_validator("name", "geography", mode="before")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value

    @field_validator("url")
    @classmethod
    def validate_group_url(cls, value: str) -> str:
        normalized = value.strip()
        parts = urlsplit(normalized)
        hostname = (parts.hostname or "").casefold()
        if parts.scheme != "https":
            raise ValueError("Facebook group URLs must use https")
        if hostname != "facebook.com" and not hostname.endswith(".facebook.com"):
            raise ValueError("Group URL must use the facebook.com domain")
        if not parts.path.casefold().startswith("/groups/"):
            raise ValueError("Group URL must point to a Facebook group")
        return normalized.rstrip("/")

    @model_validator(mode="after")
    def posting_requires_scanning(self) -> FacebookGroup:
        if self.posting_enabled and not self.enabled:
            raise ValueError("Posting cannot be enabled for a scan-disabled group")
        return self


class GroupCatalog(BaseModel):
    """The complete allowlist of Facebook groups."""

    groups: list[FacebookGroup] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_ids(self) -> GroupCatalog:
        group_ids = [group.id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("Group IDs must be unique")
        return self

    def enabled_groups(self) -> list[FacebookGroup]:
        return sorted(
            (group for group in self.groups if group.enabled),
            key=lambda group: (group.priority, group.id),
        )

    def enabled_group(self, group_id: str) -> FacebookGroup:
        for group in self.enabled_groups():
            if group.id == group_id:
                return group
        raise GroupsConfigError(f"Enabled Facebook group '{group_id}' was not found")

    def posting_enabled_groups(self) -> list[FacebookGroup]:
        """Return only groups explicitly approved for Facebook comments."""
        return sorted(
            (group for group in self.groups if group.enabled and group.posting_enabled),
            key=lambda group: (group.priority, group.id),
        )


def load_group_catalog(path: Path) -> GroupCatalog:
    """Load an allowlist with safe YAML parsing and Pydantic validation."""
    if not path.exists():
        raise GroupsConfigError(
            f"Group configuration does not exist: {path}. "
            "Copy config/groups.example.yaml to config/groups.yaml first."
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return GroupCatalog.model_validate(raw or {})
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise GroupsConfigError(f"Invalid group configuration at {path}: {error}") from error

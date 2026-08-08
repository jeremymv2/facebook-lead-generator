from pathlib import Path

import pytest

from lead_agent.groups import GroupsConfigError, load_group_catalog


def test_catalog_loads_only_enabled_groups_in_priority_order(tmp_path: Path) -> None:
    path = tmp_path / "groups.yaml"
    path.write_text(
        """
groups:
  - id: second
    name: Second group
    url: https://www.facebook.com/groups/222
    enabled: true
    priority: 2
  - id: first
    name: First group
    url: https://facebook.com/groups/111/
    enabled: true
    priority: 1
  - id: disabled
    name: Disabled group
    url: https://www.facebook.com/groups/333
    enabled: false
    priority: 1
""",
        encoding="utf-8",
    )

    catalog = load_group_catalog(path)

    assert [group.id for group in catalog.enabled_groups()] == ["first", "second"]
    assert catalog.enabled_group("first").url == "https://facebook.com/groups/111"


def test_catalog_rejects_non_facebook_and_non_group_urls(tmp_path: Path) -> None:
    path = tmp_path / "groups.yaml"
    path.write_text(
        """
groups:
  - id: unsafe
    name: Unsafe
    url: https://example.com/groups/111
    enabled: true
""",
        encoding="utf-8",
    )

    with pytest.raises(GroupsConfigError, match=r"facebook\.com"):
        load_group_catalog(path)


def test_catalog_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "groups.yaml"
    path.write_text(
        """
groups:
  - id: duplicate
    name: One
    url: https://facebook.com/groups/111
  - id: duplicate
    name: Two
    url: https://facebook.com/groups/222
""",
        encoding="utf-8",
    )

    with pytest.raises(GroupsConfigError, match="unique"):
        load_group_catalog(path)


def test_missing_catalog_and_disabled_lookup_are_clear(tmp_path: Path) -> None:
    with pytest.raises(GroupsConfigError, match="does not exist"):
        load_group_catalog(tmp_path / "missing.yaml")

    path = tmp_path / "groups.yaml"
    path.write_text(
        """
groups:
  - id: disabled
    name: Disabled
    url: https://facebook.com/groups/111
    enabled: false
""",
        encoding="utf-8",
    )

    with pytest.raises(GroupsConfigError, match="Enabled Facebook group"):
        load_group_catalog(path).enabled_group("disabled")

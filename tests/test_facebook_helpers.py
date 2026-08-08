import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lead_agent.facebook import (
    cleanup_old_screenshots,
    extract_post_id,
    facebook_group_key,
    select_facebook_permalink,
    select_message_text,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://facebook.com/groups/111/posts/222/", "222"),
        ("https://facebook.com/permalink/333", "333"),
        ("https://facebook.com/permalink.php?story_fbid=444&id=111", "444"),
        ("https://facebook.com/photo/?fbid=555", "555"),
        ("https://facebook.com/groups/111", None),
    ],
)
def test_extract_post_id(url: str, expected: str | None) -> None:
    assert extract_post_id(url) == expected


def test_group_key_is_scoped_to_the_group_path() -> None:
    assert facebook_group_key(
        "https://www.facebook.com/groups/LouisvilleOwners/?sorting=recent"
    ) == ("louisvilleowners")
    assert facebook_group_key("https://www.facebook.com/login") is None


def test_permalink_selection_ignores_external_and_tracking_urls() -> None:
    selected = select_facebook_permalink(
        [
            "https://example.com/posts/unsafe",
            "/groups/111/posts/222/?ref=share#comments",
        ],
        "https://www.facebook.com/groups/111",
    )

    assert selected == "https://www.facebook.com/groups/111/posts/222"


def test_permalink_selection_prefers_post_path_over_photo_identifier() -> None:
    selected = select_facebook_permalink(
        [
            "/photo/?fbid=999",
            "/groups/111/posts/222/?ref=share",
        ],
        "https://www.facebook.com/groups/111",
    )

    assert selected == "https://www.facebook.com/groups/111/posts/222"


def test_permalink_selection_rejects_photo_only_identifier() -> None:
    assert (
        select_facebook_permalink(
            ["/photo/?fbid=999"],
            "https://www.facebook.com/groups/111",
        )
        is None
    )


def test_message_selection_prefers_semantic_post_text() -> None:
    selected = select_message_text(
        "Author Need a deck repair Like Comment Share",
        ["Need a deck repair in Louisville next week."],
        ["Author", "Need a deck repair"],
        min_length=15,
    )

    assert selected == "Need a deck repair in Louisville next week."


def test_message_selection_falls_back_and_rejects_short_text() -> None:
    assert (
        select_message_text(
            "Author Looking for a painter this month.",
            [],
            ["Looking for a painter this month."],
            min_length=15,
        )
        == "Looking for a painter this month."
    )
    assert select_message_text("Like", [], [], min_length=15) is None


def test_message_selection_chooses_the_most_complete_owned_post_text() -> None:
    selected = select_message_text(
        "Top-level article text with comments removed.",
        [],
        [
            "Short top-level post preview.",
            "The complete top-level post text that should be retained by the extractor.",
        ],
        min_length=15,
    )

    assert selected == "The complete top-level post text that should be retained by the extractor."


def test_screenshot_cleanup_deletes_only_expired_png_files(tmp_path: Path) -> None:
    now = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    expired = tmp_path / "expired.png"
    current = tmp_path / "current.png"
    unrelated = tmp_path / "keep.txt"
    for path in (expired, current, unrelated):
        path.write_text("diagnostic", encoding="utf-8")
    old_time = (now - timedelta(days=15)).timestamp()
    os.utime(expired, (old_time, old_time))

    deleted = cleanup_old_screenshots(tmp_path, retention_days=14, now=now)

    assert deleted == 1
    assert not expired.exists()
    assert current.exists()
    assert unrelated.exists()


def test_browser_adapter_contains_no_facebook_write_actions() -> None:
    source = (Path(__file__).parents[1] / "src/lead_agent/facebook.py").read_text(encoding="utf-8")

    for forbidden_call in (".click(", ".fill(", ".type(", ".press(", ".check("):
        assert forbidden_call not in source

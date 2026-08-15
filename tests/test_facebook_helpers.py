import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from lead_agent.facebook import (
    STORY_MESSAGE_SELECTOR,
    FacebookPostCandidate,
    build_facebook_post,
    cleanup_old_screenshots,
    extract_post_id,
    facebook_group_key,
    is_facebook_comment_label,
    merge_facebook_post,
    select_facebook_permalink,
    select_message_text,
)
from lead_agent.groups import FacebookGroup
from lead_agent.models import FacebookPost, is_exact_facebook_post_url

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "facebook_post_candidates.json"


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


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Comment by Billy Herdt a day ago", True),
        ("Reply by Example Person 2 hours ago", True),
        ("Post by Example Person", False),
        (None, False),
    ],
)
def test_comment_article_labels_are_rejected(label: str | None, expected: bool) -> None:
    assert is_facebook_comment_label(label) is expected


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


def test_permalink_selection_removes_comment_identity_from_parent_post_url() -> None:
    selected = select_facebook_permalink(
        ["/groups/111/posts/222/?comment_id=333&ref=share"],
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("https://www.facebook.com/groups/111", False),
        ("https://www.facebook.com/photo/?fbid=222", False),
        ("https://example.com/groups/111/posts/222", False),
        ("http://www.facebook.com/groups/111/posts/222", False),
        ("https://www.facebook.com/groups/111/posts/222", True),
        ("https://www.facebook.com/groups/111/permalink/222", True),
        ("https://www.facebook.com/permalink.php?story_fbid=222&id=111", False),
        ("https://www.facebook.com/groups/111?multi_permalinks=222", True),
    ],
)
def test_exact_facebook_post_url_requires_one_https_post(
    value: str | None,
    expected: bool,
) -> None:
    assert is_exact_facebook_post_url(value) is expected


def test_later_hydrated_story_merges_permalink_into_content_only_render() -> None:
    group_id = "fixture-group"
    text = "Hello, I need a concrete contractor to replace an existing driveway slab."
    content_only = FacebookPost(
        group_id=group_id,
        group_name="Fixture Group",
        post_text=text,
    )
    hydrated = FacebookPost(
        external_post_id="222",
        post_url="https://www.facebook.com/groups/111/posts/222",
        group_id=group_id,
        group_name="Fixture Group",
        author_name="Fixture Customer",
        post_text=text,
    )
    collected = {content_only.identity_key: content_only}

    merge_facebook_post(collected, hydrated)

    assert list(collected.values()) == [content_only]
    assert content_only.identity_key.startswith("content:")
    assert content_only.external_post_id == "222"
    assert content_only.post_url == "https://www.facebook.com/groups/111/posts/222"
    assert content_only.author_name == "Fixture Customer"


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


def test_message_selection_removes_duplicated_see_more_preview() -> None:
    complete = (
        "Need someone to mow two yards today, trim the bushes, and remove weeds along the fence. "
        "Update: I have found someone."
    )
    duplicated_preview = (
        f"{complete} Need someone to mow two yards today, trim the bushes, and remove weeds along "
        "the fence.… See more"
    )

    selected = select_message_text(
        duplicated_preview,
        [duplicated_preview],
        [],
        min_length=15,
    )

    assert selected == complete


def test_sanitized_candidate_fixtures_cover_supported_selector_shapes() -> None:
    fixtures = cast(list[dict[str, object]], json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
    group = FacebookGroup(
        id="fixture-group",
        name="Synthetic Fixture Group",
        url="https://www.facebook.com/groups/111",
        enabled=True,
    )

    for fixture in fixtures:
        selector = cast(str | None, fixture["selector"])
        if selector is not None:
            assert selector in STORY_MESSAGE_SELECTOR, cast(str, fixture["name"])
        raw = cast(dict[str, object], fixture["candidate"])
        candidate = FacebookPostCandidate(
            full_text=cast(str, raw["full_text"]),
            semantic_messages=tuple(cast(list[str], raw["semantic_messages"])),
            automatic_texts=tuple(cast(list[str], raw["automatic_texts"])),
            hrefs=tuple(cast(list[str], raw["hrefs"])),
            article_label=cast(str | None, raw["article_label"]),
            author_name=cast(str | None, raw.get("author_name")),
            is_nested_article=cast(bool, raw["is_nested_article"]),
        )
        expected = cast(dict[str, object], fixture["expected"])
        post = build_facebook_post(candidate, group, min_length=15)

        if not cast(bool, expected["accepted"]):
            assert post is None, cast(str, fixture["name"])
            continue
        assert post is not None, cast(str, fixture["name"])
        assert post.post_text == expected["text"]
        assert post.post_url == expected["post_url"]
        assert post.external_post_id == expected["external_post_id"]


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

from datetime import UTC, datetime

import pytest

from lead_agent.models import FacebookPost, Lead, canonicalize_facebook_url, hash_post_text


def test_post_normalizes_text_and_builds_id_identity() -> None:
    post = FacebookPost(
        external_post_id="12345",
        group_id="louisville-homeowners",
        group_name="Louisville Homeowners",
        post_text="  Need   a deck repaired.\nThis week! ",
    )

    assert post.post_text == "Need a deck repaired. This week!"
    assert post.identity_key == "facebook-id:12345"
    assert post.text_hash == hash_post_text("Need a deck repaired. This week!")


def test_url_identity_ignores_tracking_query_and_fragment() -> None:
    post = FacebookPost(
        post_url="https://WWW.FACEBOOK.COM/groups/123/posts/456/?ref=share#comments",
        group_id="123",
        group_name="Test Group",
        post_text="Looking for flooring installation.",
    )

    assert post.identity_key == "facebook-url:https://www.facebook.com/groups/123/posts/456"


def test_content_identity_is_stable_when_facebook_url_is_unavailable() -> None:
    first = FacebookPost(
        group_id="group-one",
        group_name="Group One",
        author_name="Sarah Smith",
        post_text="Need drywall repair in Louisville.",
        posted_at=datetime(2026, 8, 7, 15, 30, tzinfo=UTC),
    )
    second = FacebookPost(
        group_id="GROUP-ONE",
        group_name="Group One",
        author_name="sarah smith",
        post_text="Need  drywall repair in Louisville.",
    )

    assert first.identity_key == second.identity_key


def test_empty_post_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="post_text"):
        FacebookPost(group_id="group", group_name="Group", post_text=" \n ")


@pytest.mark.parametrize("score", [-1, 101])
def test_lead_scores_are_bounded(score: int) -> None:
    with pytest.raises(ValueError, match="overall_score"):
        Lead(facebook_post_id=1, overall_score=score)


def test_confidence_is_a_fraction() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Lead(facebook_post_id=1, confidence=1.1)


def test_canonicalize_facebook_url_preserves_path() -> None:
    assert (
        canonicalize_facebook_url("https://facebook.com/groups/example/posts/1/?tracking=yes")
        == "https://facebook.com/groups/example/posts/1"
    )


def test_canonicalize_permalink_retains_post_identity_query() -> None:
    first = canonicalize_facebook_url(
        "https://facebook.com/permalink.php?story_fbid=111&id=222&ref=share"
    )
    second = canonicalize_facebook_url(
        "https://facebook.com/permalink.php?id=222&story_fbid=333&tracking=ignored"
    )

    assert first == "https://facebook.com/permalink.php?id=222&story_fbid=111"
    assert second == "https://facebook.com/permalink.php?id=222&story_fbid=333"
    assert first != second

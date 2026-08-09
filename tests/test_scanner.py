import asyncio
from pathlib import Path

import pytest

from lead_agent.database import Database
from lead_agent.facebook_state import FacebookPageState, FacebookSafetyStop
from lead_agent.groups import FacebookGroup
from lead_agent.models import FacebookPost
from lead_agent.scanner import (
    ReadOnlyScanService,
    TransientFacebookReadError,
    safe_scan_error_code,
)


class FakeReader:
    def __init__(
        self,
        posts: list[FacebookPost] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.posts = posts or []
        self.error = error

    async def read_group(
        self,
        group: FacebookGroup,
        *,
        max_posts: int,
    ) -> list[FacebookPost]:
        if self.error is not None:
            raise self.error
        return self.posts[:max_posts]


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "scanner.sqlite3")
    database.initialize()
    return database


@pytest.fixture
def group() -> FacebookGroup:
    return FacebookGroup(
        id="louisville-homeowners",
        name="Louisville Homeowners",
        url="https://www.facebook.com/groups/111",
        enabled=True,
    )


def post(group: FacebookGroup) -> FacebookPost:
    return FacebookPost(
        external_post_id="222",
        post_url="https://www.facebook.com/groups/111/posts/222",
        group_id=group.id,
        group_name=group.name,
        post_text="Looking for a deck contractor in Louisville.",
    )


def test_second_scan_reports_duplicate_without_reinserting(
    database: Database,
    group: FacebookGroup,
) -> None:
    scanner = ReadOnlyScanService(database, FakeReader([post(group)]))

    first = asyncio.run(scanner.scan_group(group, max_posts=20))
    second = asyncio.run(scanner.scan_group(group, max_posts=20))

    assert len(first.new_posts) == 1
    assert len(second.new_posts) == 0
    assert second.duplicates == 1
    assert len(database.list_posts()) == 1
    state = database.get_group_scan_state(group.id)
    assert state is not None
    assert state.posts_seen == 1
    assert state.posts_new == 0


def test_scan_state_keeps_persisted_identity_when_permalink_hydrates(
    database: Database,
    group: FacebookGroup,
) -> None:
    text = "Looking for a deck contractor in Louisville."
    content_only = FacebookPost(
        group_id=group.id,
        group_name=group.name,
        post_text=text,
    )
    first = asyncio.run(
        ReadOnlyScanService(database, FakeReader([content_only])).scan_group(group, max_posts=20)
    )
    hydrated = post(group)
    second = asyncio.run(
        ReadOnlyScanService(database, FakeReader([hydrated])).scan_group(group, max_posts=20)
    )

    state = database.get_group_scan_state(group.id)
    assert len(first.new_posts) == 1
    assert len(second.new_posts) == 0
    assert len(database.list_posts()) == 1
    assert state is not None
    assert state.last_known_post_identity == content_only.identity_key


def test_safety_stop_is_recorded_and_re_raised(
    database: Database,
    group: FacebookGroup,
) -> None:
    scanner = ReadOnlyScanService(
        database,
        FakeReader(
            error=FacebookSafetyStop(
                FacebookPageState.CHECKPOINT,
                "Facebook displayed an account checkpoint",
            )
        ),
    )

    with pytest.raises(FacebookSafetyStop):
        asyncio.run(scanner.scan_group(group, max_posts=20))

    state = database.get_group_scan_state(group.id)
    assert state is not None
    assert state.last_error == "FacebookSafetyStop:checkpoint"
    assert state.consecutive_failures == 1
    assert database.list_audit_events()[0].result == "stopped"


def test_unexpected_errors_store_only_the_error_type(
    database: Database,
    group: FacebookGroup,
) -> None:
    scanner = ReadOnlyScanService(
        database,
        FakeReader(error=RuntimeError("page content must not enter audit logs")),
    )

    with pytest.raises(RuntimeError):
        asyncio.run(scanner.scan_group(group, max_posts=20))

    state = database.get_group_scan_state(group.id)
    assert state is not None
    assert state.last_error == "RuntimeError"


def test_reader_cannot_cross_the_group_allowlist_boundary(
    database: Database,
    group: FacebookGroup,
) -> None:
    wrong_group_post = FacebookPost(
        external_post_id="333",
        group_id="unapproved-group",
        group_name="Unapproved",
        post_text="This post must not be persisted by the approved group scan.",
    )
    scanner = ReadOnlyScanService(database, FakeReader([wrong_group_post]))

    with pytest.raises(ValueError, match="unexpected group"):
        asyncio.run(scanner.scan_group(group, max_posts=20))

    assert database.list_posts() == []
    state = database.get_group_scan_state(group.id)
    assert state is not None
    assert state.last_error == "ValueError"


def test_transient_error_records_only_stage_and_kind(
    database: Database,
    group: FacebookGroup,
    tmp_path: Path,
) -> None:
    error = TransientFacebookReadError(
        stage="feed",
        kind="timeout",
        screenshot_path=tmp_path / "local-only.png",
    )
    scanner = ReadOnlyScanService(database, FakeReader(error=error))

    with pytest.raises(TransientFacebookReadError, match="feed:timeout"):
        asyncio.run(scanner.scan_group(group, max_posts=20))

    state = database.get_group_scan_state(group.id)
    assert state is not None
    assert state.last_error == "TransientFacebookReadError:feed:timeout"
    assert "local-only" not in state.last_error
    assert safe_scan_error_code(error) == "TransientFacebookReadError:feed:timeout"

import asyncio
from pathlib import Path

import pytest

from lead_agent.database import Database
from lead_agent.facebook_state import FacebookPageState, FacebookSafetyStop
from lead_agent.groups import FacebookGroup
from lead_agent.models import FacebookPost
from lead_agent.scanner import (
    FacebookReadDiagnostics,
    FacebookReadResult,
    ReadOnlyScanService,
    TransientFacebookReadError,
    safe_scan_error_code,
)


class FakeReader:
    def __init__(
        self,
        posts: list[FacebookPost] | None = None,
        error: Exception | None = None,
        outcomes: list[FacebookReadResult | list[FacebookPost] | Exception] | None = None,
    ) -> None:
        self.posts = posts or []
        self.error = error
        self.outcomes = outcomes
        self.calls = 0

    async def read_group(
        self,
        group: FacebookGroup,
        *,
        max_posts: int,
    ) -> FacebookReadResult | list[FacebookPost]:
        self.calls += 1
        if self.outcomes is not None:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
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


def post(group: FacebookGroup, post_id: str = "222") -> FacebookPost:
    return FacebookPost(
        external_post_id=post_id,
        post_url=f"https://www.facebook.com/groups/111/posts/{post_id}",
        group_id=group.id,
        group_name=group.name,
        post_text=f"Looking for a deck contractor in Louisville for project {post_id}.",
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


def test_scan_with_fewer_posts_than_requested_is_recorded_as_partial(
    database: Database,
    group: FacebookGroup,
) -> None:
    scanner = ReadOnlyScanService(database, FakeReader([post(group)]))

    summary = asyncio.run(scanner.scan_group(group, max_posts=10))

    assert summary.partial is True
    assert summary.posts_requested == 10
    state = database.get_group_scan_state(group.id)
    assert state is not None
    assert state.posts_seen == 1
    assert state.posts_requested == 10
    assert state.last_scan_partial is True
    assert state.last_success_at is None
    group_event = next(
        event for event in database.list_audit_events() if event.action == "group.scan"
    )
    assert group_event.result == "partial"
    assert group_event.details["posts_requested"] == 10


def test_eight_of_ten_posts_is_healthy_with_a_minor_shortfall(
    database: Database,
    group: FacebookGroup,
) -> None:
    posts = [post(group, str(post_id)) for post_id in range(8)]
    scanner = ReadOnlyScanService(database, FakeReader(posts))

    summary = asyncio.run(scanner.scan_group(group, max_posts=10))

    assert summary.shortfall is True
    assert summary.partial is False
    assert summary.severe_partial is False
    state = database.get_group_scan_state(group.id)
    assert state is not None
    assert state.last_scan_partial is False
    assert state.last_success_at is not None
    group_event = next(
        event for event in database.list_audit_events() if event.action == "group.scan"
    )
    assert group_event.result == "success"
    assert group_event.details["shortfall"] is True


def test_severe_partial_retry_merges_unique_posts_before_one_persisted_outcome(
    database: Database,
    group: FacebookGroup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = [post(group, str(post_id)) for post_id in range(3)]
    second = [post(group, str(post_id)) for post_id in range(2, 7)]
    reader = FakeReader(outcomes=[first, second])
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    summary = asyncio.run(
        ReadOnlyScanService(database, reader).scan_group(
            group,
            max_posts=10,
            max_retries=1,
            retry_backoff_seconds=5,
        )
    )

    assert summary.posts_seen == 7
    assert len(summary.new_posts) == 7
    assert summary.retry_count == 1
    assert summary.recovered is True
    assert summary.partial is True
    assert summary.severe_partial is False
    assert reader.calls == 2
    assert sleeps == [5]
    assert len(database.list_posts()) == 7
    group_events = [event for event in database.list_audit_events() if event.action == "group.scan"]
    assert len(group_events) == 1
    assert group_events[0].details["retry_count"] == 1
    assert group_events[0].details["recovered"] is True


def test_retry_merge_preserves_distinct_permalinks_with_identical_text(
    database: Database,
    group: FacebookGroup,
) -> None:
    shared_text = "Looking for a reliable Louisville contractor for this project."
    posts = [
        FacebookPost(
            external_post_id=post_id,
            post_url=f"https://www.facebook.com/groups/111/posts/{post_id}",
            group_id=group.id,
            group_name=group.name,
            post_text=shared_text,
        )
        for post_id in ("one", "two")
    ]
    reader = FakeReader(outcomes=[[posts[0]], [posts[1]]])

    summary = asyncio.run(
        ReadOnlyScanService(database, reader).scan_group(
            group,
            max_posts=10,
            max_retries=1,
        )
    )

    assert summary.posts_seen == 2
    assert {value.external_post_id for value in database.list_posts()} == {"one", "two"}


def test_failed_severe_retry_preserves_first_attempt_posts(
    database: Database,
    group: FacebookGroup,
) -> None:
    initial_posts = [post(group, str(post_id)) for post_id in range(3)]
    reader = FakeReader(
        outcomes=[
            initial_posts,
            TransientFacebookReadError(stage="navigation", kind="timeout"),
        ]
    )

    summary = asyncio.run(
        ReadOnlyScanService(database, reader).scan_group(
            group,
            max_posts=10,
            max_retries=1,
        )
    )

    assert summary.posts_seen == 3
    assert summary.retry_count == 1
    assert summary.recovered is False
    assert summary.severe_partial is True
    assert len(database.list_posts()) == 3
    event = next(event for event in database.list_audit_events() if event.action == "group.scan")
    assert event.result == "partial"
    assert event.details["retry_error"] == "TransientFacebookReadError:navigation:timeout"


def test_scan_audit_records_content_free_feed_diagnostics_and_screenshot_flag(
    database: Database,
    group: FacebookGroup,
    tmp_path: Path,
) -> None:
    result = FacebookReadResult(
        posts=(post(group),),
        diagnostics=FacebookReadDiagnostics(
            elapsed_ms=2500,
            iterations=4,
            scrolls=3,
            story_nodes_seen=6,
            visible_articles_seen=5,
            readable_posts=1,
            permalinked_posts=1,
            detached_nodes=2,
            duplicate_identities=1,
            progress_events=1,
            stop_reason="idle",
        ),
        severe_screenshot_path=tmp_path / "severe.png",
    )

    asyncio.run(
        ReadOnlyScanService(database, FakeReader(outcomes=[result])).scan_group(
            group,
            max_posts=10,
        )
    )

    event = next(event for event in database.list_audit_events() if event.action == "group.scan")
    assert event.details["elapsed_ms"] == 2500
    assert event.details["scrolls"] == 3
    assert event.details["detached_nodes"] == 2
    assert event.details["stop_reason"] == "idle"
    assert event.details["severe_screenshot_captured"] is True
    assert "severe.png" not in str(event.details)


def test_transient_failure_retries_once_but_safety_stop_never_retries(
    database: Database,
    group: FacebookGroup,
) -> None:
    recovering = FakeReader(
        outcomes=[
            TransientFacebookReadError(stage="feed", kind="timeout"),
            [post(group, str(post_id)) for post_id in range(8)],
        ]
    )
    recovered = asyncio.run(
        ReadOnlyScanService(database, recovering).scan_group(
            group,
            max_posts=10,
            max_retries=1,
        )
    )
    assert recovered.retry_count == 1
    assert recovered.recovered is True
    assert recovering.calls == 2

    checkpoint = FakeReader(
        outcomes=[
            FacebookSafetyStop(
                FacebookPageState.CHECKPOINT,
                "Facebook displayed an account checkpoint",
            ),
            [post(group)],
        ]
    )
    with pytest.raises(FacebookSafetyStop):
        asyncio.run(
            ReadOnlyScanService(database, checkpoint).scan_group(
                group,
                max_posts=10,
                max_retries=1,
            )
        )
    assert checkpoint.calls == 1


def test_scan_state_keeps_persisted_identity_when_permalink_hydrates(
    database: Database,
    group: FacebookGroup,
) -> None:
    text = "Looking for a deck contractor in Louisville for project 222."
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

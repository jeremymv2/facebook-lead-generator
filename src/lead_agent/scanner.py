"""Read-only scan orchestration independent of Playwright details."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from lead_agent.database import Database
from lead_agent.facebook_state import FacebookSafetyStop
from lead_agent.groups import FacebookGroup
from lead_agent.models import AuditEvent, FacebookPost


class FacebookReader(Protocol):
    async def read_group(
        self,
        group: FacebookGroup,
        *,
        max_posts: int,
        attempt: int = 0,
        known_identities: frozenset[str] = frozenset(),
    ) -> FacebookReadResult | list[FacebookPost]: ...


class TransientFacebookReadError(RuntimeError):
    """A content-free Playwright failure that may be retried once by an unattended cycle."""

    def __init__(
        self,
        *,
        stage: str,
        kind: str,
        screenshot_path: Path | None = None,
    ) -> None:
        self.stage = stage
        self.kind = kind
        self.screenshot_path = screenshot_path
        self.retry_count = 0
        super().__init__(self.safe_code)

    @property
    def safe_code(self) -> str:
        return f"{self.stage}:{self.kind}"


@dataclass(frozen=True, slots=True)
class FacebookReadDiagnostics:
    """Content-free measurements explaining how one Facebook feed rendered."""

    elapsed_ms: int = 0
    iterations: int = 0
    scrolls: int = 0
    story_nodes_seen: int = 0
    visible_articles_seen: int = 0
    readable_posts: int = 0
    permalinked_posts: int = 0
    missing_permalinks: int = 0
    detached_nodes: int = 0
    duplicate_identities: int = 0
    progress_events: int = 0
    top_level_story_nodes_seen: int = 0
    collapsed_unexpanded_observations: int = 0
    comment_observations: int = 0
    nested_article_observations: int = 0
    short_text_observations: int = 0
    feed_movement_events: int = 0
    feed_height_growth_events: int = 0
    loading_observations: int = 0
    stop_reason: str = "unknown"

    @property
    def feed_responsive(self) -> bool:
        """Whether Facebook visibly rendered enough top-level content to avoid a false alarm."""
        return self.top_level_story_nodes_seen >= 2 or self.collapsed_unexpanded_observations > 0

    def combine(self, other: FacebookReadDiagnostics) -> FacebookReadDiagnostics:
        """Combine bounded retry diagnostics without exposing Facebook content."""
        return FacebookReadDiagnostics(
            elapsed_ms=self.elapsed_ms + other.elapsed_ms,
            iterations=self.iterations + other.iterations,
            scrolls=self.scrolls + other.scrolls,
            story_nodes_seen=max(self.story_nodes_seen, other.story_nodes_seen),
            visible_articles_seen=max(
                self.visible_articles_seen,
                other.visible_articles_seen,
            ),
            readable_posts=max(self.readable_posts, other.readable_posts),
            permalinked_posts=max(self.permalinked_posts, other.permalinked_posts),
            missing_permalinks=max(self.missing_permalinks, other.missing_permalinks),
            detached_nodes=self.detached_nodes + other.detached_nodes,
            duplicate_identities=self.duplicate_identities + other.duplicate_identities,
            progress_events=self.progress_events + other.progress_events,
            top_level_story_nodes_seen=max(
                self.top_level_story_nodes_seen,
                other.top_level_story_nodes_seen,
            ),
            collapsed_unexpanded_observations=(
                self.collapsed_unexpanded_observations + other.collapsed_unexpanded_observations
            ),
            comment_observations=self.comment_observations + other.comment_observations,
            nested_article_observations=(
                self.nested_article_observations + other.nested_article_observations
            ),
            short_text_observations=(self.short_text_observations + other.short_text_observations),
            feed_movement_events=self.feed_movement_events + other.feed_movement_events,
            feed_height_growth_events=(
                self.feed_height_growth_events + other.feed_height_growth_events
            ),
            loading_observations=self.loading_observations + other.loading_observations,
            stop_reason=other.stop_reason,
        )

    def audit_details(self) -> dict[str, object]:
        return {
            "elapsed_ms": self.elapsed_ms,
            "iterations": self.iterations,
            "scrolls": self.scrolls,
            "story_nodes_seen": self.story_nodes_seen,
            "visible_articles_seen": self.visible_articles_seen,
            "readable_posts": self.readable_posts,
            "permalinked_posts": self.permalinked_posts,
            "missing_permalinks": self.missing_permalinks,
            "detached_nodes": self.detached_nodes,
            "duplicate_identities": self.duplicate_identities,
            "progress_events": self.progress_events,
            "top_level_story_nodes_seen": self.top_level_story_nodes_seen,
            "collapsed_unexpanded_observations": self.collapsed_unexpanded_observations,
            "comment_observations": self.comment_observations,
            "nested_article_observations": self.nested_article_observations,
            "short_text_observations": self.short_text_observations,
            "feed_movement_events": self.feed_movement_events,
            "feed_height_growth_events": self.feed_height_growth_events,
            "loading_observations": self.loading_observations,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True, slots=True)
class FacebookReadResult:
    posts: tuple[FacebookPost, ...]
    diagnostics: FacebookReadDiagnostics = FacebookReadDiagnostics()
    severe_screenshot_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ScanSummary:
    group: FacebookGroup
    posts_seen: int
    new_posts: tuple[FacebookPost, ...]
    posts_requested: int = 0
    retry_count: int = 0
    recovered: bool = False
    healthy_yield_rate: float = 0.8
    severe_yield_rate: float = 0.5
    diagnostics: FacebookReadDiagnostics = FacebookReadDiagnostics()

    @property
    def duplicates(self) -> int:
        return self.posts_seen - len(self.new_posts)

    @property
    def partial(self) -> bool:
        return (
            self.posts_requested > 0
            and self.posts_seen / self.posts_requested < self.healthy_yield_rate
        )

    @property
    def shortfall(self) -> bool:
        return self.posts_requested > 0 and self.posts_seen < self.posts_requested

    @property
    def severe_partial(self) -> bool:
        return (
            self.posts_requested > 0
            and self.posts_seen / self.posts_requested < self.severe_yield_rate
        )

    @property
    def materially_incomplete(self) -> bool:
        """A severe shortfall without evidence that Facebook rendered a usable feed."""
        return self.severe_partial and not self.diagnostics.feed_responsive


class ReadOnlyScanService:
    """Persist visible posts once and maintain per-group scan state."""

    def __init__(self, database: Database, reader: FacebookReader) -> None:
        self.database = database
        self.reader = reader

    async def scan_group(
        self,
        group: FacebookGroup,
        *,
        max_posts: int,
        max_retries: int = 0,
        retry_backoff_seconds: float = 0,
        healthy_yield_rate: float = 0.8,
        severe_yield_rate: float = 0.5,
    ) -> ScanSummary:
        if max_retries < 0 or retry_backoff_seconds < 0:
            raise ValueError("scan retry settings cannot be negative")
        if not 0 < severe_yield_rate < healthy_yield_rate <= 1:
            raise ValueError("scan yield thresholds are invalid")
        collected: dict[str, FacebookPost] = {}
        diagnostics = FacebookReadDiagnostics()
        retry_count = 0
        retry_error: str | None = None
        severe_screenshot_captured = False
        try:
            while True:
                try:
                    raw_result = await self.reader.read_group(
                        group,
                        max_posts=max_posts,
                        attempt=retry_count,
                        known_identities=frozenset(collected),
                    )
                except FacebookSafetyStop:
                    raise
                except TransientFacebookReadError as error:
                    if retry_count >= max_retries:
                        if collected:
                            retry_error = safe_scan_error_code(error)
                            break
                        error.retry_count = retry_count
                        raise
                    retry_count += 1
                    if retry_backoff_seconds:
                        await asyncio.sleep(retry_backoff_seconds)
                    continue

                result = (
                    raw_result
                    if isinstance(raw_result, FacebookReadResult)
                    else FacebookReadResult(posts=tuple(raw_result))
                )
                if any(post.group_id != group.id for post in result.posts):
                    raise ValueError("Facebook reader returned a post for an unexpected group")
                diagnostics = diagnostics.combine(result.diagnostics)
                severe_screenshot_captured = (
                    severe_screenshot_captured or result.severe_screenshot_path is not None
                )
                _merge_read_posts(collected, result.posts)
                severe = max_posts > 0 and len(collected) / max_posts < severe_yield_rate
                if not severe or retry_count >= max_retries:
                    break
                retry_count += 1
                if retry_backoff_seconds:
                    await asyncio.sleep(retry_backoff_seconds)
        except Exception as error:
            safe_error = safe_scan_error_code(error)
            self.database.record_group_scan_failure(
                group_id=group.id,
                group_name=group.name,
                group_url=group.url,
                error=safe_error,
                posts_requested=max_posts,
            )
            self.database.record_audit_event(
                AuditEvent(
                    component="facebook_scanner",
                    action="group.scan",
                    result="stopped" if isinstance(error, FacebookSafetyStop) else "failed",
                    group_id=group.id,
                    details={
                        "error": safe_error,
                        "posts_requested": max_posts,
                        "retry_count": retry_count,
                    },
                )
            )
            raise

        discovered = list(collected.values())[:max_posts]
        permalinked_posts = sum(post.post_url is not None for post in discovered)
        diagnostics = replace(
            diagnostics,
            readable_posts=len(discovered),
            permalinked_posts=permalinked_posts,
            missing_permalinks=max(0, len(discovered) - permalinked_posts),
        )
        new_posts: list[FacebookPost] = []
        persisted_posts: list[FacebookPost] = []
        for post in discovered:
            save_result = self.database.save_post(post)
            persisted_posts.append(save_result.post)
            if save_result.created:
                new_posts.append(save_result.post)
                self.database.record_audit_event(
                    AuditEvent(
                        component="facebook_scanner",
                        action="post.discovered",
                        result="new",
                        post_id=save_result.post.id,
                        group_id=group.id,
                        details={"identity_key": save_result.post.identity_key},
                    )
                )

        last_identity = persisted_posts[0].identity_key if persisted_posts else None
        summary = ScanSummary(
            group=group,
            posts_seen=len(discovered),
            new_posts=tuple(new_posts),
            posts_requested=max_posts,
            retry_count=retry_count,
            recovered=(
                retry_count > 0
                and (max_posts == 0 or len(discovered) / max_posts >= severe_yield_rate)
            ),
            healthy_yield_rate=healthy_yield_rate,
            severe_yield_rate=severe_yield_rate,
            diagnostics=diagnostics,
        )
        self.database.record_group_scan_success(
            group_id=group.id,
            group_name=group.name,
            group_url=group.url,
            posts_seen=len(discovered),
            posts_new=len(new_posts),
            posts_requested=max_posts,
            is_partial=summary.partial,
            last_known_post_identity=last_identity,
        )
        self.database.record_audit_event(
            AuditEvent(
                component="facebook_scanner",
                action="group.scan",
                result="partial" if summary.partial else "success",
                group_id=group.id,
                details={
                    "posts_seen": len(discovered),
                    "posts_new": len(new_posts),
                    "posts_requested": max_posts,
                    "shortfall": summary.shortfall,
                    "severe_partial": summary.severe_partial,
                    "retry_count": retry_count,
                    "recovered": summary.recovered,
                    "retry_error": retry_error,
                    "severe_screenshot_captured": severe_screenshot_captured,
                    **diagnostics.audit_details(),
                },
            )
        )
        return summary


def _merge_read_posts(
    collected: dict[str, FacebookPost],
    discovered: tuple[FacebookPost, ...],
) -> None:
    """Merge retry results by stable identity or content while preserving hydration."""
    for post in discovered:
        existing = next(
            (
                value
                for value in collected.values()
                if value.identity_key == post.identity_key
                or (
                    value.external_post_id is not None
                    and value.external_post_id == post.external_post_id
                )
                or (
                    value.text_hash == post.text_hash
                    and (value.post_url is None or post.post_url is None)
                )
            ),
            None,
        )
        if existing is None:
            collected[post.identity_key] = post
            continue
        if existing.external_post_id is None and post.external_post_id is not None:
            existing.external_post_id = post.external_post_id
        if existing.post_url is None and post.post_url is not None:
            existing.post_url = post.post_url
        if existing.author_name is None and post.author_name is not None:
            existing.author_name = post.author_name


def safe_scan_error_code(error: Exception) -> str:
    """Return a stable diagnostic code without exception messages or page content."""
    if isinstance(error, FacebookSafetyStop):
        return f"{type(error).__name__}:{error.state.value}"
    if isinstance(error, TransientFacebookReadError):
        return f"{type(error).__name__}:{error.safe_code}"
    return type(error).__name__

"""Content-free historical metrics for the local review dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from lead_agent.database import ApprovalFeedbackSummary, Database
from lead_agent.models import AuditEvent, GroupScanState, PostingAttemptStatus

DEFAULT_HISTORY_LIMIT = 48


@dataclass(frozen=True, slots=True)
class CycleTrend:
    """One completed unattended cycle reconstructed from its safe audit event."""

    occurred_at: datetime
    status: str
    groups_scanned: int
    groups_failed: int
    groups_shortfall: int
    groups_partial: int
    groups_severely_partial: int
    groups_retried: int
    groups_recovered: int
    posts_seen: int
    posts_new: int
    duplicates: int
    posts_classified: int
    candidates_created: int
    posts_ignored: int
    notifications_sent: int

    @property
    def groups_attempted(self) -> int:
        return self.groups_scanned + self.groups_failed

    @property
    def group_success_percent(self) -> float:
        if self.groups_attempted == 0:
            return 0.0
        return round(self.groups_complete * 100 / self.groups_attempted, 1)

    @property
    def groups_complete(self) -> int:
        return max(0, self.groups_scanned - self.groups_partial)

    @property
    def groups_near_complete(self) -> int:
        return max(0, self.groups_shortfall - self.groups_partial)


@dataclass(frozen=True, slots=True)
class PostingOutcomeSummary:
    """Content-free totals for live Facebook posting outcomes."""

    posted: int
    pending_moderation: int
    needs_attention: int
    failed: int


@dataclass(frozen=True, slots=True)
class RecentSuccessfulPost:
    """One verified public Facebook reply for the local dashboard."""

    lead_id: int
    group_name: str
    posted_at: datetime
    facebook_reply_url: str


@dataclass(frozen=True, slots=True)
class DashboardTrendSnapshot:
    """Recent cycle history plus the latest content-free state for each group."""

    cycles: tuple[CycleTrend, ...]
    groups: tuple[GroupScanState, ...]
    feedback: ApprovalFeedbackSummary
    posting: PostingOutcomeSummary
    recent_successful_posts: tuple[RecentSuccessfulPost, ...] = ()

    @property
    def groups_scanned(self) -> int:
        return sum(cycle.groups_scanned for cycle in self.cycles)

    @property
    def groups_failed(self) -> int:
        return sum(cycle.groups_failed for cycle in self.cycles)

    @property
    def groups_partial(self) -> int:
        return sum(cycle.groups_partial for cycle in self.cycles)

    @property
    def groups_shortfall(self) -> int:
        return sum(cycle.groups_shortfall for cycle in self.cycles)

    @property
    def groups_near_complete(self) -> int:
        return max(0, self.groups_shortfall - self.groups_partial)

    @property
    def groups_severely_partial(self) -> int:
        return sum(cycle.groups_severely_partial for cycle in self.cycles)

    @property
    def groups_complete(self) -> int:
        return max(0, self.groups_scanned - self.groups_partial)

    @property
    def group_success_percent(self) -> float:
        attempted = self.groups_scanned + self.groups_failed
        if attempted == 0:
            return 0.0
        return round(self.groups_complete * 100 / attempted, 1)

    @property
    def posts_seen(self) -> int:
        return sum(cycle.posts_seen for cycle in self.cycles)

    @property
    def posts_new(self) -> int:
        return sum(cycle.posts_new for cycle in self.cycles)

    @property
    def candidates_created(self) -> int:
        return sum(cycle.candidates_created for cycle in self.cycles)

    @property
    def retries(self) -> int:
        return sum(cycle.groups_retried for cycle in self.cycles)

    @property
    def recoveries(self) -> int:
        return sum(cycle.groups_recovered for cycle in self.cycles)

    @property
    def degraded_groups(self) -> int:
        return sum(group.last_error is not None or group.last_scan_partial for group in self.groups)


class DashboardMetricsService:
    """Build bounded trend snapshots from existing operational audit records."""

    def __init__(self, database: Database, *, history_limit: int = DEFAULT_HISTORY_LIMIT) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be positive")
        self.database = database
        self.history_limit = history_limit

    def snapshot(self) -> DashboardTrendSnapshot:
        newest_events = self.database.list_audit_events(
            component="operations",
            action="cycle.run",
            limit=self.history_limit,
            newest_first=True,
        )
        cycles = tuple(_cycle_from_event(event) for event in reversed(newest_events))
        posting_counts = self.database.posting_attempt_status_counts()
        recent_successful_posts = tuple(
            RecentSuccessfulPost(
                lead_id=work.lead.id or 0,
                group_name=work.post.group_name,
                posted_at=work.attempt.completed_at,
                facebook_reply_url=work.attempt.facebook_reply_url,
            )
            for work in self.database.list_recent_successful_posting_work(limit=10)
            if work.attempt.completed_at is not None
            and work.attempt.facebook_reply_url is not None
            and work.lead.id is not None
        )
        return DashboardTrendSnapshot(
            cycles=cycles,
            groups=tuple(self.database.list_group_scan_states()),
            feedback=self.database.approval_feedback_summary(),
            posting=PostingOutcomeSummary(
                posted=posting_counts.get(PostingAttemptStatus.POSTED, 0),
                pending_moderation=posting_counts.get(PostingAttemptStatus.PENDING_MODERATION, 0),
                needs_attention=posting_counts.get(PostingAttemptStatus.NEEDS_ATTENTION, 0),
                failed=posting_counts.get(PostingAttemptStatus.FAILED, 0),
            ),
            recent_successful_posts=recent_successful_posts,
        )


def _cycle_from_event(event: AuditEvent) -> CycleTrend:
    details = event.details
    posts_seen = _counter(details, "posts_seen")
    posts_new = _counter(details, "posts_new")
    posts_classified = _counter(details, "posts_classified")
    candidates_created = _counter(details, "candidates_created")
    groups_partial = _counter(details, "groups_partial")
    return CycleTrend(
        occurred_at=event.occurred_at,
        status=event.result if event.result in {"success", "degraded", "failed"} else "unknown",
        groups_scanned=_counter(details, "groups_scanned"),
        groups_failed=_counter(details, "groups_failed"),
        groups_shortfall=_counter(
            details,
            "groups_shortfall",
            fallback=groups_partial,
        ),
        groups_partial=groups_partial,
        groups_severely_partial=_counter(details, "groups_severely_partial"),
        groups_retried=_counter(details, "groups_retried"),
        groups_recovered=_counter(details, "groups_recovered"),
        posts_seen=posts_seen,
        posts_new=posts_new,
        duplicates=_counter(details, "duplicates", fallback=max(0, posts_seen - posts_new)),
        posts_classified=posts_classified,
        candidates_created=candidates_created,
        posts_ignored=_counter(
            details,
            "posts_ignored",
            fallback=max(0, posts_classified - candidates_created),
        ),
        notifications_sent=_counter(details, "notifications_sent"),
    )


def _counter(details: dict[str, object], name: str, *, fallback: int = 0) -> int:
    value = details.get(name)
    return value if type(value) is int and value >= 0 else fallback

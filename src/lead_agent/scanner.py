"""Read-only scan orchestration independent of Playwright details."""

from __future__ import annotations

from dataclasses import dataclass
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
    ) -> list[FacebookPost]: ...


@dataclass(frozen=True, slots=True)
class ScanSummary:
    group: FacebookGroup
    posts_seen: int
    new_posts: tuple[FacebookPost, ...]

    @property
    def duplicates(self) -> int:
        return self.posts_seen - len(self.new_posts)


class ReadOnlyScanService:
    """Persist visible posts once and maintain per-group scan state."""

    def __init__(self, database: Database, reader: FacebookReader) -> None:
        self.database = database
        self.reader = reader

    async def scan_group(self, group: FacebookGroup, *, max_posts: int) -> ScanSummary:
        try:
            discovered = await self.reader.read_group(group, max_posts=max_posts)
            if any(post.group_id != group.id for post in discovered):
                raise ValueError("Facebook reader returned a post for an unexpected group")
        except Exception as error:
            safe_error = _safe_error_name(error)
            self.database.record_group_scan_failure(
                group_id=group.id,
                group_name=group.name,
                group_url=group.url,
                error=safe_error,
            )
            self.database.record_audit_event(
                AuditEvent(
                    component="facebook_scanner",
                    action="group.scan",
                    result="stopped" if isinstance(error, FacebookSafetyStop) else "failed",
                    group_id=group.id,
                    details={"error": safe_error},
                )
            )
            raise

        new_posts: list[FacebookPost] = []
        for post in discovered:
            result = self.database.save_post(post)
            if result.created:
                new_posts.append(result.post)
                self.database.record_audit_event(
                    AuditEvent(
                        component="facebook_scanner",
                        action="post.discovered",
                        result="new",
                        post_id=result.post.id,
                        group_id=group.id,
                        details={"identity_key": result.post.identity_key},
                    )
                )

        last_identity = discovered[0].identity_key if discovered else None
        self.database.record_group_scan_success(
            group_id=group.id,
            group_name=group.name,
            group_url=group.url,
            posts_seen=len(discovered),
            posts_new=len(new_posts),
            last_known_post_identity=last_identity,
        )
        self.database.record_audit_event(
            AuditEvent(
                component="facebook_scanner",
                action="group.scan",
                result="success",
                group_id=group.id,
                details={"posts_seen": len(discovered), "posts_new": len(new_posts)},
            )
        )
        return ScanSummary(
            group=group,
            posts_seen=len(discovered),
            new_posts=tuple(new_posts),
        )


def _safe_error_name(error: Exception) -> str:
    if isinstance(error, FacebookSafetyStop):
        return f"{type(error).__name__}:{error.state.value}"
    return type(error).__name__

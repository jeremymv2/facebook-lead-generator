"""Sanitized review-feedback exports for classifier regression development."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from lead_agent.database import Database
from lead_agent.models import ApprovalReview, RejectionReason, normalize_post_text

MAX_FIXTURE_TEXT_CHARACTERS = 1_500


@dataclass(frozen=True, slots=True)
class RegressionExportSummary:
    output_path: Path
    feedback_considered: int
    fixtures_exported: int
    feedback_skipped: int


def export_regression_fixtures(
    database: Database,
    output_path: Path,
    *,
    limit: int,
    lead_id: int | None = None,
) -> RegressionExportSummary:
    """Write deterministic, sanitized candidate fixtures from structured rejection feedback."""
    reviews = database.list_rejected_approval_reviews(limit=limit, lead_id=lead_id)
    fixtures: list[dict[str, object]] = []
    for review in reviews:
        fixture = _fixture_from_review(review)
        if fixture is not None:
            fixtures.append(fixture)
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_path.parent.chmod(0o700)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(fixtures, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(output_path)
        output_path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return RegressionExportSummary(
        output_path=output_path,
        feedback_considered=len(reviews),
        fixtures_exported=len(fixtures),
        feedback_skipped=len(reviews) - len(fixtures),
    )


def sanitize_fixture_text(text: str, *, author_name: str | None = None) -> str:
    """Redact common direct identifiers while retaining classifier-relevant language."""
    sanitized = text
    if author_name and len(author_name.strip()) >= 3:
        sanitized = re.sub(re.escape(author_name.strip()), "[name]", sanitized, flags=re.IGNORECASE)
    substitutions = (
        (r"https?://\S+|www\.\S+", "[url]"),
        (r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[email]"),
        (r"(?<!\w)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\w)", "[phone]"),
        (
            r"\b\d{1,6}\s+(?:[A-Z][\w'.-]+\s+){0,4}"
            r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Court|Ct|Boulevard|Blvd)\b\.?,?",
            "[address]",
        ),
        (r"\b(?:[A-Z][\w&'.-]+\s+){1,4}(?:LLC|Inc\.?|Company|Co\.)\b", "[business]"),
        (r"#\w*(?:llc|company|service|services)\w*", "[business]"),
    )
    for pattern, replacement in substitutions:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return normalize_post_text(sanitized)[:MAX_FIXTURE_TEXT_CHARACTERS]


def _fixture_from_review(review: ApprovalReview) -> dict[str, object] | None:
    reason = review.request.rejection_reason
    if reason is None:
        return None
    expected = _expected_for_reason(reason)
    if expected is None:
        return None
    lead_id = review.lead.id or 0
    return {
        "name": f"review_feedback_{lead_id}_{reason.value}",
        "text": sanitize_fixture_text(
            review.post.post_text,
            author_name=review.post.author_name,
        ),
        "expected": expected,
        "review_before_commit": True,
    }


def _expected_for_reason(reason: RejectionReason) -> dict[str, object] | None:
    mapping: dict[RejectionReason, dict[str, object] | None] = {
        RejectionReason.PROVIDER_ADVERTISEMENT: {
            "intent": "competitor_advertisement",
            "maximum_score": 10,
        },
        RejectionReason.EMPLOYMENT_RECRUITING: {
            "intent": "unrelated",
            "maximum_score": 10,
        },
        RejectionReason.SALE_LISTING: {"intent": "selling", "maximum_score": 10},
        RejectionReason.ADVICE_ONLY: {"intent": "advice", "maximum_score": 40},
        RejectionReason.WRONG_GEOGRAPHY: {"maximum_geographic_score": 20},
        RejectionReason.IRRELEVANT_SERVICE: {"maximum_score": 40},
        RejectionReason.DUPLICATE_OR_REPOST: None,
        RejectionReason.RESOLVED: {"intent": "resolved", "maximum_score": 10},
        RejectionReason.OTHER: None,
    }
    return mapping[reason]

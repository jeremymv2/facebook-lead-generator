"""Provider-independent SMS delivery for expiring remote approval links."""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lead_agent.approvals import LocalApprovalService
from lead_agent.config import Settings
from lead_agent.database import Database
from lead_agent.models import (
    ApprovalReview,
    AuditEvent,
    NotificationStatus,
    utc_now,
)

TELNYX_MESSAGES_URL = "https://api.telnyx.com/v2/messages"
E164_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")
SMS_BRAND_NAME = "JJ Miller & Co LLC"
SMS_OPT_OUT_INSTRUCTION = "Reply STOP to opt out."


class SmsProviderError(RuntimeError):
    """Raised when an SMS provider cannot accept a message."""


@dataclass(frozen=True, slots=True)
class SmsMessage:
    to: str
    body: str
    idempotency_key: str

    def __post_init__(self) -> None:
        if E164_PATTERN.fullmatch(self.to) is None:
            raise ValueError("SMS destination must use E.164 format")
        if not self.body or len(self.body) > 160 or not self.body.isascii():
            raise ValueError("SMS body must be non-empty ASCII and at most 160 characters")
        if not self.idempotency_key.strip():
            raise ValueError("SMS idempotency key cannot be empty")


@dataclass(frozen=True, slots=True)
class SmsDeliveryReceipt:
    provider_message_id: str
    status: str


class SmsProvider(Protocol):
    """Minimal provider boundary used by the approval workflow."""

    @property
    def name(self) -> str: ...

    def send(self, message: SmsMessage) -> SmsDeliveryReceipt: ...


class JsonPostTransport(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: int,
    ) -> Mapping[str, object]: ...


class UrllibJsonPostTransport:
    """Small HTTPS JSON transport that relies on Python's verified default TLS context."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: int,
    ) -> Mapping[str, object]:
        request = Request(
            url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw_response = response.read(65_537)
        except HTTPError as error:
            error_code = _telnyx_error_code(error.read(16_384))
            raise SmsProviderError(f"Telnyx rejected the message ({error_code})") from None
        except (TimeoutError, URLError, OSError):
            raise SmsProviderError("Telnyx request failed before acceptance") from None
        if len(raw_response) > 65_536:
            raise SmsProviderError("Telnyx returned an oversized response")
        try:
            decoded = json.loads(raw_response)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SmsProviderError("Telnyx returned an invalid response") from None
        if not isinstance(decoded, dict):
            raise SmsProviderError("Telnyx returned an invalid response")
        return cast(dict[str, object], decoded)


class TelnyxSmsProvider:
    """Telnyx Messaging API adapter with no approval-workflow knowledge."""

    name = "telnyx"

    def __init__(
        self,
        *,
        api_key: str,
        from_number: str,
        timeout_seconds: int,
        transport: JsonPostTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Telnyx API key cannot be empty")
        if E164_PATTERN.fullmatch(from_number) is None:
            raise ValueError("Telnyx sender must use E.164 format")
        self._api_key = api_key
        self._from_number = from_number
        self._timeout_seconds = timeout_seconds
        self._transport = transport or UrllibJsonPostTransport()

    def send(self, message: SmsMessage) -> SmsDeliveryReceipt:
        response = self._transport.post(
            TELNYX_MESSAGES_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            payload={
                "from": self._from_number,
                "to": message.to,
                "text": message.body,
            },
            timeout_seconds=self._timeout_seconds,
        )
        data = response.get("data")
        if not isinstance(data, Mapping):
            raise SmsProviderError("Telnyx response did not contain message data")
        message_id = data.get("id")
        if not isinstance(message_id, str) or not message_id:
            raise SmsProviderError("Telnyx response did not contain a message ID")
        return SmsDeliveryReceipt(provider_message_id=message_id, status="accepted")


@dataclass(frozen=True, slots=True)
class NotificationSummary:
    considered: int
    sent: int
    failed: int


class ApprovalNotificationService:
    """Create opaque links and send at most one SMS for each approval request."""

    def __init__(
        self,
        database: Database,
        approvals: LocalApprovalService,
        provider: SmsProvider,
        *,
        public_base_url: str,
        recipient_number: str,
        token_factory: Callable[[], str] | None = None,
        relay_healthcheck: Callable[[], bool] | None = None,
    ) -> None:
        self.database = database
        self.approvals = approvals
        self.provider = provider
        self.public_base_url = public_base_url.rstrip("/")
        self.recipient_number = recipient_number
        self.token_factory = token_factory or (lambda: secrets.token_urlsafe(32))
        self.relay_healthcheck = relay_healthcheck or (lambda: True)

    def notify_candidates(
        self,
        *,
        limit: int,
        retry_failed: bool = False,
        now: datetime | None = None,
    ) -> NotificationSummary:
        if not self.relay_healthcheck():
            return NotificationSummary(considered=0, sent=0, failed=0)
        timestamp = now or utc_now()
        self.approvals.prepare_candidates(limit=limit, now=timestamp)
        reviews = self.database.list_notifiable_approval_reviews(include_failed=retry_failed)[
            :limit
        ]
        sent = 0
        failed = 0
        for review in reviews:
            if self._notify_review(review, retry_failed=retry_failed, now=timestamp):
                sent += 1
            else:
                failed += 1
        return NotificationSummary(considered=len(reviews), sent=sent, failed=failed)

    def _notify_review(
        self,
        review: ApprovalReview,
        *,
        retry_failed: bool,
        now: datetime,
    ) -> bool:
        request_id = review.request.id
        if request_id is None:  # pragma: no cover - persistence contract
            raise RuntimeError("Approval request is missing its ID")
        token = self.token_factory()
        token_hash = remote_token_hash(token)
        review_url = f"{self.public_base_url}/review/{token}"
        message = SmsMessage(
            to=self.recipient_number,
            body=_approval_sms_body(review, review_url),
            idempotency_key=f"approval:{request_id}",
        )
        claimed = self.database.claim_approval_notification(
            request_id,
            provider=self.provider.name,
            remote_token_hash=token_hash,
            attempted_at=now,
            retry_failed=retry_failed,
        )
        if not claimed:
            return False
        try:
            receipt = self.provider.send(message)
        except SmsProviderError as error:
            self.database.complete_approval_notification(
                request_id,
                status=NotificationStatus.FAILED,
                completed_at=now,
                error_code=type(error).__name__,
            )
            self._record_event(review, result="failed", details={"error": type(error).__name__})
            return False
        self.database.complete_approval_notification(
            request_id,
            status=NotificationStatus.SENT,
            completed_at=now,
            provider_message_id=receipt.provider_message_id,
        )
        self._record_event(review, result="sent")
        return True

    def _record_event(
        self,
        review: ApprovalReview,
        *,
        result: str,
        details: dict[str, object] | None = None,
    ) -> None:
        self.database.record_audit_event(
            AuditEvent(
                component="notification",
                action="approval.sms",
                result=result,
                lead_id=review.lead.id,
                post_id=review.post.id,
                group_id=review.post.group_id,
                details={
                    "approval_request_id": review.request.id or 0,
                    "provider": self.provider.name,
                    **(details or {}),
                },
            )
        )


def build_sms_provider(settings: Settings) -> SmsProvider:
    settings.require_remote_approval_ready()
    if settings.telnyx_api_key is None or settings.telnyx_from_number is None:
        raise RuntimeError("Validated Telnyx settings are unexpectedly missing")
    return TelnyxSmsProvider(
        api_key=settings.telnyx_api_key.get_secret_value(),
        from_number=settings.telnyx_from_number,
        timeout_seconds=settings.sms_request_timeout_seconds,
    )


def remote_token_hash(token: str) -> str:
    import hashlib

    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _approval_sms_body(review: ApprovalReview, review_url: str) -> str:
    score = review.lead.overall_score if review.lead.overall_score is not None else "new"
    lead_id = review.lead.id if review.lead.id is not None else "new"
    service = (review.lead.service_category or "project").replace("_", " ")
    service = re.sub(r"[^A-Za-z0-9 /&-]", "", service)[:30].strip() or "project"
    body = (
        f"{SMS_BRAND_NAME} lead {lead_id}: {service}, score {score}. "
        f"{review_url} {SMS_OPT_OUT_INSTRUCTION}"
    )
    if len(body) <= 160:
        return body
    compact = f"{SMS_BRAND_NAME} lead {lead_id}: {service}. {review_url} {SMS_OPT_OUT_INSTRUCTION}"
    if len(compact) <= 160:
        return compact
    fallback = f"{SMS_BRAND_NAME} lead {lead_id}. {review_url} {SMS_OPT_OUT_INSTRUCTION}"
    if len(fallback) > 160:
        raise ValueError("REMOTE_APPROVAL_BASE_URL is too long for a single SMS segment")
    return fallback


def _telnyx_error_code(raw_response: bytes) -> str:
    try:
        decoded = json.loads(raw_response)
        errors = decoded.get("errors") if isinstance(decoded, dict) else None
        first = errors[0] if isinstance(errors, list) and errors else None
        code = first.get("code") if isinstance(first, dict) else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        code = None
    return str(code) if code else "HTTP error"

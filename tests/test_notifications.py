from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from email.message import Message
from io import BytesIO
from pathlib import Path
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

import lead_agent.notifications as notifications_module
from lead_agent.approvals import LocalApprovalService
from lead_agent.config import Settings
from lead_agent.database import Database
from lead_agent.models import FacebookPost, Lead, LeadIntent, LeadStatus, NotificationStatus
from lead_agent.notifications import (
    ApprovalNotificationService,
    SmsDeliveryReceipt,
    SmsMessage,
    SmsProviderError,
    TelnyxSmsProvider,
    UrllibJsonPostTransport,
    build_sms_provider,
    remote_token_hash,
)

VALID_DRAFT = (
    "JJ Miller & Co. can help with your deck project. Licensed & Insured. Free estimates. "
    "Text me at 502-528-0858 or visit https://jjmillerco.com."
)


def create_candidate(database: Database) -> None:
    post = database.save_post(
        FacebookPost(
            external_post_id="notification-fixture",
            post_url=("https://www.facebook.com/groups/111/posts/notification-fixture"),
            group_id="fixture-group",
            group_name="Synthetic Fixture Group",
            post_text="Looking for someone in Louisville to repair our deck this week.",
        )
    ).post
    database.create_lead(
        Lead(
            facebook_post_id=post.id or 0,
            status=LeadStatus.CANDIDATE,
            service_category="decks",
            intent=LeadIntent.HIRING,
            overall_score=95,
            drafted_response=VALID_DRAFT,
        )
    )


class FakeSmsProvider:
    name = "fake"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.messages: list[SmsMessage] = []

    def send(self, message: SmsMessage) -> SmsDeliveryReceipt:
        self.messages.append(message)
        if self.fail:
            raise SmsProviderError("Synthetic provider failure")
        return SmsDeliveryReceipt(provider_message_id="message-fixture", status="accepted")


class FakeTransport:
    def __init__(self, response: Mapping[str, object]) -> None:
        self.response = response
        self.url = ""
        self.headers: Mapping[str, str] = {}
        self.payload: Mapping[str, object] = {}
        self.timeout_seconds = 0
        self.get_url = ""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: int,
    ) -> Mapping[str, object]:
        self.url = url
        self.headers = headers
        self.payload = payload
        self.timeout_seconds = timeout_seconds
        return self.response

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: int,
    ) -> Mapping[str, object]:
        self.get_url = url
        self.headers = headers
        self.timeout_seconds = timeout_seconds
        return self.response


class FakeUrlResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeUrlResponse":
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self, limit: int) -> bytes:
        assert limit == 65_537
        return self.body


def test_telnyx_adapter_isolated_behind_sms_contract() -> None:
    transport = FakeTransport({"data": {"id": "telnyx-message-id"}})
    provider = TelnyxSmsProvider(
        api_key="secret-api-key",
        from_number="+15025550100",
        timeout_seconds=12,
        transport=transport,
    )

    receipt = provider.send(
        SmsMessage(
            to="+15025550101",
            body=(
                "JJ Miller & Co LLC lead. Review: https://approve.example/review/token "
                "Reply STOP to opt out."
            ),
            idempotency_key="approval:1",
        )
    )

    assert provider.name == "telnyx"
    assert receipt.provider_message_id == "telnyx-message-id"
    assert transport.url == "https://api.telnyx.com/v2/messages"
    assert transport.headers["Authorization"] == "Bearer secret-api-key"
    assert transport.payload == {
        "from": "+15025550100",
        "to": "+15025550101",
        "text": (
            "JJ Miller & Co LLC lead. Review: https://approve.example/review/token "
            "Reply STOP to opt out."
        ),
    }
    assert transport.timeout_seconds == 12


def test_notification_sends_one_tokenized_single_segment_sms(tmp_path: Path) -> None:
    database = Database(tmp_path / "notifications.sqlite3")
    database.initialize()
    create_candidate(database)
    approvals = LocalApprovalService(database, expiration_minutes=20)
    provider = FakeSmsProvider()
    token = "A" * 43
    notifier = ApprovalNotificationService(
        database,
        approvals,
        provider,
        public_base_url="https://approve.example",
        recipient_number="+15025550101",
        token_factory=lambda: token,
    )
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)

    first = notifier.notify_candidates(limit=10, now=now)
    second = notifier.notify_candidates(limit=10, now=now)

    assert first.considered == 1
    assert first.sent == 1
    assert first.failed == 0
    assert second.considered == 0
    assert len(provider.messages) == 1
    message = provider.messages[0]
    assert message.to == "+15025550101"
    assert message.body == (
        "JJ Miller & Co LLC lead 1: decks, score 95. "
        f"https://approve.example/review/{token} Reply STOP to opt out."
    )
    assert len(message.body) <= 160
    assert message.body.isascii()
    assert "Looking for someone" not in message.body
    review = database.get_approval_review_by_remote_token_hash(remote_token_hash(token))
    assert review is not None
    notification = database.get_approval_notification(review.request.id or 0)
    assert notification is not None
    assert notification.status is NotificationStatus.SENT
    assert notification.provider_message_id == "message-fixture"
    assert [event.action for event in database.list_audit_events()] == [
        "approval.requested",
        "approval.sms",
    ]


def test_notification_records_actual_telnyx_delivery_status(tmp_path: Path) -> None:
    class DeliveryAwareFakeSmsProvider(FakeSmsProvider):
        name = "telnyx"

        def delivery_status(self, provider_message_id: str) -> str:
            assert provider_message_id == "message-fixture"
            return "delivered"

    database = Database(tmp_path / "delivery.sqlite3")
    database.initialize()
    create_candidate(database)
    approvals = LocalApprovalService(database, expiration_minutes=20)
    notifier = ApprovalNotificationService(
        database,
        approvals,
        DeliveryAwareFakeSmsProvider(),
        public_base_url="https://approve.example",
        recipient_number="+15025550101",
    )
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)

    notifier.notify_candidates(limit=1, now=now)
    summary = notifier.reconcile_delivery_statuses(
        minimum_interval_seconds=60,
        now=now + timedelta(seconds=60),
    )

    notification = database.get_approval_notification(1)
    assert notification is not None
    assert summary.checked == 1
    assert summary.delivered == 1
    assert notification.delivery_status == "delivered"
    assert notification.delivery_checked_at == now + timedelta(seconds=60)
    assert "approval.sms_delivery" in [event.action for event in database.list_audit_events()]


def test_telnyx_adapter_reads_recipient_delivery_status() -> None:
    transport = FakeTransport({"data": {"to": [{"status": "delivered"}]}})
    provider = TelnyxSmsProvider(
        api_key="secret-api-key",
        from_number="+15025550100",
        timeout_seconds=12,
        transport=transport,
    )

    assert provider.delivery_status("telnyx-message-id") == "delivered"
    assert transport.get_url == "https://api.telnyx.com/v2/messages/telnyx-message-id"


def test_notification_uses_compliant_fallback_at_maximum_origin_length(tmp_path: Path) -> None:
    database = Database(tmp_path / "fallback.sqlite3")
    database.initialize()
    create_candidate(database)
    approvals = LocalApprovalService(database, expiration_minutes=20)
    provider = FakeSmsProvider()
    token = "A" * 43
    public_base_url = f"https://{'a' * 41}.com"
    notifier = ApprovalNotificationService(
        database,
        approvals,
        provider,
        public_base_url=public_base_url,
        recipient_number="+15025550101",
        token_factory=lambda: token,
    )

    summary = notifier.notify_candidates(limit=1)

    assert summary.sent == 1
    assert provider.messages[0].body == (
        f"JJ Miller & Co LLC lead 1. {public_base_url}/review/{token} Reply STOP to opt out."
    )
    assert len(provider.messages[0].body) <= 160


def test_failed_send_clears_link_and_requires_explicit_retry(tmp_path: Path) -> None:
    database = Database(tmp_path / "retry.sqlite3")
    database.initialize()
    create_candidate(database)
    approvals = LocalApprovalService(database, expiration_minutes=20)
    provider = FakeSmsProvider(fail=True)
    tokens = iter(("A" * 43, "B" * 43))
    notifier = ApprovalNotificationService(
        database,
        approvals,
        provider,
        public_base_url="https://approve.example",
        recipient_number="+15025550101",
        token_factory=lambda: next(tokens),
    )
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)

    failed = notifier.notify_candidates(limit=10, now=now)
    skipped = notifier.notify_candidates(limit=10, now=now)
    provider.fail = False
    retried = notifier.notify_candidates(limit=10, retry_failed=True, now=now)

    assert failed.failed == 1
    assert skipped.considered == 0
    assert retried.sent == 1
    assert database.get_approval_review_by_remote_token_hash(remote_token_hash("A" * 43)) is None
    review = database.get_approval_review_by_remote_token_hash(remote_token_hash("B" * 43))
    assert review is not None
    notification = database.get_approval_notification(review.request.id or 0)
    assert notification is not None
    assert notification.status is NotificationStatus.SENT
    assert notification.attempt_count == 2


def test_unreachable_relay_does_not_start_expiration_or_send_sms(tmp_path: Path) -> None:
    database = Database(tmp_path / "offline-relay.sqlite3")
    database.initialize()
    create_candidate(database)
    approvals = LocalApprovalService(database, expiration_minutes=20)
    provider = FakeSmsProvider()
    notifier = ApprovalNotificationService(
        database,
        approvals,
        provider,
        public_base_url="https://approve.example",
        recipient_number="+15025550101",
        relay_healthcheck=lambda: False,
    )

    summary = notifier.notify_candidates(limit=10)

    assert summary.considered == 0
    assert provider.messages == []
    assert database.list_pending_approval_reviews() == []
    assert len(database.list_candidate_leads(limit=10)) == 1


@pytest.mark.parametrize(
    ("to", "body", "idempotency_key", "expected"),
    [
        ("502-555-0101", "Review", "approval:1", "E.164"),
        ("+15025550101", "", "approval:1", "non-empty ASCII"),
        ("+15025550101", "snowman ☃", "approval:1", "non-empty ASCII"),
        ("+15025550101", "Review", "   ", "cannot be empty"),
    ],
)
def test_sms_message_rejects_noncompliant_fields(
    to: str,
    body: str,
    idempotency_key: str,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        SmsMessage(to=to, body=body, idempotency_key=idempotency_key)


@pytest.mark.parametrize(
    ("api_key", "from_number", "expected"),
    [
        ("   ", "+15025550100", "API key"),
        ("fixture", "502-555-0100", "E.164"),
    ],
)
def test_telnyx_provider_rejects_invalid_credentials(
    api_key: str,
    from_number: str,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        TelnyxSmsProvider(
            api_key=api_key,
            from_number=from_number,
            timeout_seconds=5,
        )


def test_urllib_transport_posts_json_and_decodes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: int) -> FakeUrlResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeUrlResponse(b'{"data":{"id":"message-id"}}')

    monkeypatch.setattr(notifications_module, "urlopen", fake_urlopen)
    transport = UrllibJsonPostTransport()

    response = transport.post(
        "https://api.telnyx.com/v2/messages",
        headers={"Authorization": "Bearer redacted"},
        payload={"to": "+15025550101", "text": "Review"},
        timeout_seconds=7,
    )

    assert response == {"data": {"id": "message-id"}}
    request = cast(Request, captured["request"])
    assert request.full_url == "https://api.telnyx.com/v2/messages"
    assert request.method == "POST"
    assert captured["timeout"] == 7


def test_urllib_transport_reduces_http_failure_to_telnyx_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request: object, timeout: int) -> FakeUrlResponse:
        del request, timeout
        raise HTTPError(
            "https://api.telnyx.com/v2/messages",
            422,
            "rejected",
            hdrs=Message(),
            fp=BytesIO(b'{"errors":[{"code":"40300"}]}'),
        )

    monkeypatch.setattr(notifications_module, "urlopen", fail)

    with pytest.raises(SmsProviderError, match="40300"):
        UrllibJsonPostTransport().post(
            "https://api.telnyx.com/v2/messages",
            headers={},
            payload={},
            timeout_seconds=5,
        )


def test_urllib_transport_reduces_network_failure_without_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request: object, timeout: int) -> FakeUrlResponse:
        del request, timeout
        raise URLError("private network detail")

    monkeypatch.setattr(notifications_module, "urlopen", fail)

    with pytest.raises(SmsProviderError, match="failed before acceptance") as captured:
        UrllibJsonPostTransport().post(
            "https://api.telnyx.com/v2/messages",
            headers={},
            payload={},
            timeout_seconds=5,
        )
    assert "private network detail" not in str(captured.value)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b"x" * 65_537, "oversized"),
        (b"not-json", "invalid response"),
        (b"[]", "invalid response"),
    ],
)
def test_urllib_transport_rejects_invalid_provider_responses(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    expected: str,
) -> None:
    monkeypatch.setattr(
        notifications_module,
        "urlopen",
        lambda request, timeout: FakeUrlResponse(body),
    )

    with pytest.raises(SmsProviderError, match=expected):
        UrllibJsonPostTransport().post(
            "https://api.telnyx.com/v2/messages",
            headers={},
            payload={},
            timeout_seconds=5,
        )


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"data": {}},
        {"data": {"id": ""}},
        {"data": {"id": 123}},
    ],
)
def test_telnyx_provider_requires_message_id(response: Mapping[str, object]) -> None:
    provider = TelnyxSmsProvider(
        api_key="fixture",
        from_number="+15025550100",
        timeout_seconds=5,
        transport=FakeTransport(response),
    )

    with pytest.raises(SmsProviderError, match=r"message data|message ID"):
        provider.send(
            SmsMessage(
                to="+15025550101",
                body="Review requested",
                idempotency_key="approval:1",
            )
        )


def test_build_sms_provider_uses_validated_telnyx_settings(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        facebook_profile_path=tmp_path.parent / "browser-profile",
        notifications_enabled=True,
        sms_provider="telnyx",
        remote_approval_base_url="https://approve.example",
        approval_signing_key="s" * 48,
        sms_recipient_number="+15025550101",
        telnyx_api_key="fixture-secret",
        telnyx_from_number="+15025550100",
    )

    provider = build_sms_provider(settings)

    assert provider.name == "telnyx"

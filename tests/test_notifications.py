from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from lead_agent.approvals import LocalApprovalService
from lead_agent.database import Database
from lead_agent.models import FacebookPost, Lead, LeadIntent, LeadStatus, NotificationStatus
from lead_agent.notifications import (
    ApprovalNotificationService,
    SmsDeliveryReceipt,
    SmsMessage,
    SmsProviderError,
    TelnyxSmsProvider,
    remote_token_hash,
)

VALID_DRAFT = (
    "JJ Miller & Co. can help with your deck project. Free estimates. "
    "Text me at 502-528-0858 or visit https://jjmillerco.com."
)


def create_candidate(database: Database) -> None:
    post = database.save_post(
        FacebookPost(
            external_post_id="notification-fixture",
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
        "JJ Miller & Co LLC lead 95: decks. Review: "
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
        f"JJ Miller & Co LLC lead. Review: {public_base_url}/review/{token} Reply STOP to opt out."
    )
    assert len(provider.messages[0].body) == 160


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

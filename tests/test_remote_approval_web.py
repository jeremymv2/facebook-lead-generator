import re
import socket
import threading
import time
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from http.server import HTTPServer
from pathlib import Path
from typing import cast
from urllib.error import URLError
from urllib.parse import urlencode

import pytest

import lead_agent.remote_approval_web as remote_web_module
from lead_agent.approvals import (
    ApprovalAction,
    ApprovalExpiredError,
    ApprovalStateError,
    LocalApprovalService,
)
from lead_agent.database import Database
from lead_agent.models import FacebookPost, Lead, LeadIntent, LeadStatus, PostingJobStatus
from lead_agent.notifications import remote_token_hash
from lead_agent.remote_approval_web import (
    LOOPBACK_HOST,
    RemoteApprovalController,
    RemoteApprovalHTTPServer,
    RemoteApprovalTokenError,
    RemoteCSRFFailure,
    _handler_class,
    relay_is_healthy,
)

TOKEN = "A" * 43
PUBLIC_BASE_URL = "https://approve.example"
VALID_DRAFT = (
    "JJ Miller & Co. can help with your deck project. Licensed & Insured. Free estimates. "
    "Text me at 502-528-0858 or visit https://jjmillerco.com."
)


def prepared_controller(
    tmp_path: Path,
) -> tuple[RemoteApprovalController, Database, int, datetime]:
    database = Database(tmp_path / "remote.sqlite3")
    database.initialize()
    post = database.save_post(
        FacebookPost(
            external_post_id="remote-fixture",
            post_url="https://www.facebook.com/groups/111/posts/remote-fixture",
            group_id="fixture-group",
            group_name="<script>Fixture Group</script>",
            post_text="<img src=x onerror=alert(1)> Need a deck repair in Louisville.",
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
    now = datetime.now(UTC)
    approvals = LocalApprovalService(database, expiration_minutes=20)
    request_id = approvals.prepare_candidates(limit=10, now=now)[0].request.id or 0
    claimed = database.claim_approval_notification(
        request_id,
        provider="fake",
        remote_token_hash=remote_token_hash(TOKEN),
        attempted_at=now,
    )
    assert claimed is True
    controller = RemoteApprovalController(approvals, signing_key="s" * 48)
    return controller, database, request_id, now


def csrf_from_page(page: str) -> str:
    match = re.search(r'name="csrf_token" value="([a-f0-9]{64})"', page)
    assert match is not None
    return match.group(1)


def test_remote_page_resolves_only_opaque_token_and_escapes_content(tmp_path: Path) -> None:
    controller, _, _, now = prepared_controller(tmp_path)

    page = controller.render(TOKEN, now=now)

    assert "&lt;script&gt;Fixture Group&lt;/script&gt;" in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "<script>Fixture Group</script>" not in page
    assert f'action="/review/{TOKEN}/approve"' in page
    assert "cannot post to Facebook" in page


def test_remote_decision_requires_token_bound_csrf_and_is_one_time(tmp_path: Path) -> None:
    controller, database, request_id, now = prepared_controller(tmp_path)
    csrf = csrf_from_page(controller.render(TOKEN, now=now))

    with pytest.raises(RemoteCSRFFailure, match="CSRF"):
        controller.submit(
            TOKEN,
            ApprovalAction.APPROVE.value,
            {"csrf_token": ["wrong"]},
            now=now,
        )

    approved = controller.submit(
        TOKEN,
        ApprovalAction.APPROVE.value,
        {"csrf_token": [csrf]},
        now=now + timedelta(minutes=1),
    )

    assert approved.request.status.value == "approved"
    assert database.get_approval_request(request_id).status.value == "approved"  # type: ignore[union-attr]
    assert "Draft approved" in controller.render(TOKEN, now=now + timedelta(minutes=2))
    assert VALID_DRAFT not in controller.render(TOKEN, now=now + timedelta(minutes=2))
    with pytest.raises(ApprovalStateError, match="already been reviewed"):
        controller.submit(
            TOKEN,
            ApprovalAction.REJECT.value,
            {"csrf_token": [csrf]},
            now=now + timedelta(minutes=2),
        )


def test_remote_approve_and_post_atomically_queues_only_an_enabled_group(
    tmp_path: Path,
) -> None:
    controller, database, request_id, now = prepared_controller(tmp_path)
    enabled = RemoteApprovalController(
        controller.approvals,
        signing_key="s" * 48,
        posting_queue_enabled=True,
        posting_enabled_group_ids={"fixture-group"},
    )
    page = enabled.render(TOKEN, now=now)
    csrf = csrf_from_page(page)

    assert f'action="/review/{TOKEN}/approve-post"' in page
    approved = enabled.submit(
        TOKEN,
        "approve-post",
        {"csrf_token": [csrf]},
        now=now + timedelta(minutes=1),
    )

    job = database.get_posting_job_for_approval(request_id)
    assert approved.request.status.value == "approved"
    assert job is not None
    assert job.status is PostingJobStatus.QUEUED
    terminal = enabled.render(TOKEN, now=now + timedelta(minutes=2))
    assert "Facebook submission is queued on your Mac" in terminal


def test_remote_post_action_fails_closed_for_scan_only_group(tmp_path: Path) -> None:
    controller, database, request_id, now = prepared_controller(tmp_path)
    queue_enabled = RemoteApprovalController(
        controller.approvals,
        signing_key="s" * 48,
        posting_queue_enabled=True,
        posting_enabled_group_ids=set(),
    )
    page = queue_enabled.render(TOKEN, now=now)

    assert "approve-post" not in page
    with pytest.raises(ApprovalStateError, match="not enabled"):
        queue_enabled.submit(
            TOKEN,
            "approve-post",
            {"csrf_token": [csrf_from_page(page)]},
            now=now + timedelta(minutes=1),
        )
    assert database.get_approval_request(request_id).status.value == "pending"  # type: ignore[union-attr]
    assert database.get_posting_job_for_approval(request_id) is None


def test_remote_post_action_is_hidden_and_blocked_without_exact_permalink(
    tmp_path: Path,
) -> None:
    controller, database, request_id, now = prepared_controller(tmp_path)
    review = database.get_approval_request(request_id)
    assert review is not None
    lead = database.get_lead(review.lead_id)
    assert lead is not None
    post = database.get_post(lead.facebook_post_id)
    assert post is not None
    with database.connection() as connection:
        connection.execute("UPDATE facebook_posts SET post_url = NULL WHERE id = ?", (post.id,))
    enabled = RemoteApprovalController(
        controller.approvals,
        signing_key="s" * 48,
        posting_queue_enabled=True,
        posting_enabled_group_ids={"fixture-group"},
    )
    page = enabled.render(TOKEN, now=now)

    assert "approve-post" not in page
    assert "Review only: an exact Facebook post link was not captured" in page
    with pytest.raises(ApprovalStateError, match="review-only"):
        enabled.submit(
            TOKEN,
            "approve-post",
            {"csrf_token": [csrf_from_page(page)]},
            now=now + timedelta(minutes=1),
        )
    assert database.get_approval_request(request_id).status.value == "pending"  # type: ignore[union-attr]
    assert database.get_posting_job_for_approval(request_id) is None


def test_remote_controller_rejects_weak_or_unknown_tokens(tmp_path: Path) -> None:
    controller, _, _, now = prepared_controller(tmp_path)

    with pytest.raises(ValueError, match="at least 32"):
        RemoteApprovalController(controller.approvals, signing_key="short")
    with pytest.raises(RemoteApprovalTokenError, match="invalid"):
        controller.resolve("not-a-valid-token", now=now)
    with pytest.raises(RemoteApprovalTokenError, match="invalid"):
        controller.resolve("B" * 43, now=now)


def test_remote_controller_expires_before_accepting_a_decision(tmp_path: Path) -> None:
    controller, _, _, now = prepared_controller(tmp_path)
    csrf = csrf_from_page(controller.render(TOKEN, now=now))

    with pytest.raises(ApprovalExpiredError, match="expired"):
        controller.submit(
            TOKEN,
            ApprovalAction.APPROVE.value,
            {"csrf_token": [csrf]},
            now=now + timedelta(minutes=21),
        )


class FakeHTTPServer:
    server_name = LOOPBACK_HOST
    server_port = 8766


def handle_request(
    controller: RemoteApprovalController,
    method: str,
    path: str,
    *,
    body: str | bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], str]:
    client, server = socket.socketpair()
    try:
        payload = body if isinstance(body, bytes) else (body or "").encode()
        request_headers = {
            "Host": "approve.example",
            "Connection": "close",
            **(headers or {}),
        }
        if payload and "Content-Length" not in request_headers:
            request_headers["Content-Length"] = str(len(payload))
        header_lines = "\r\n".join(f"{name}: {value}" for name, value in request_headers.items())
        client.sendall(f"{method} {path} HTTP/1.1\r\n{header_lines}\r\n\r\n".encode() + payload)
        client.shutdown(socket.SHUT_WR)
        handler = _handler_class(
            controller,
            port=8766,
            public_base_url=PUBLIC_BASE_URL,
        )

        def serve() -> None:
            handler(server, (LOOPBACK_HOST, 12345), cast(HTTPServer, FakeHTTPServer()))
            server.shutdown(socket.SHUT_WR)

        worker = threading.Thread(
            target=serve,
        )
        worker.start()
        chunks: list[bytes] = []
        while chunk := client.recv(65_536):
            chunks.append(chunk)
        worker.join(timeout=2)
        assert not worker.is_alive()
        raw_headers, raw_body = b"".join(chunks).split(b"\r\n\r\n", maxsplit=1)
        response_lines = raw_headers.decode("iso-8859-1").split("\r\n")
        status = int(response_lines[0].split()[1])
        response_headers = dict(line.split(": ", maxsplit=1) for line in response_lines[1:])
        return status, response_headers, raw_body.decode()
    finally:
        client.close()
        server.close()


def test_tunneled_http_surface_has_no_listing_and_enforces_origin(tmp_path: Path) -> None:
    controller, _, _, now = prepared_controller(tmp_path)

    status, _, _ = handle_request(controller, "GET", "/")
    assert status == HTTPStatus.NOT_FOUND

    status, _, _ = handle_request(
        controller,
        "GET",
        f"/review/{TOKEN}",
        headers={"Host": "attacker.invalid"},
    )
    assert status == HTTPStatus.MISDIRECTED_REQUEST

    status, headers, page = handle_request(controller, "GET", f"/review/{TOKEN}")
    assert status == HTTPStatus.OK
    assert headers["Cache-Control"] == "no-store"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    csrf = csrf_from_page(page)
    form = urlencode({"csrf_token": csrf})
    form_headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://attacker.invalid",
    }

    status, _, _ = handle_request(
        controller,
        "POST",
        f"/review/{TOKEN}/approve",
        body=form,
        headers=form_headers,
    )
    assert status == HTTPStatus.FORBIDDEN

    form_headers["Origin"] = "https://approve.example"
    status, headers, _ = handle_request(
        controller,
        "POST",
        f"/review/{TOKEN}/approve",
        body=form,
        headers=form_headers,
    )
    assert status == HTTPStatus.SEE_OTHER
    assert headers["Location"] == f"/review/{TOKEN}?result=approved"

    terminal = controller.render(TOKEN, now=now + timedelta(minutes=1))
    assert "Draft approved" in terminal


def test_relay_healthcheck_requires_exact_success_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        status = HTTPStatus.OK

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, limit: int) -> bytes:
            assert limit == 16
            return b"ok"

    monkeypatch.setattr(remote_web_module, "urlopen", lambda request, timeout: FakeResponse())

    assert relay_is_healthy("https://approve.example") is True


def test_tunneled_http_surface_rejects_malformed_requests(tmp_path: Path) -> None:
    controller, _, _, _ = prepared_controller(tmp_path)
    valid_path = f"/review/{TOKEN}/approve"
    base_headers = {"Origin": PUBLIC_BASE_URL}

    status, _, body = handle_request(
        controller,
        "GET",
        "/health",
        headers={"Host": "127.0.0.1:8766"},
    )
    assert status == HTTPStatus.OK
    assert body == "ok"

    cases: list[tuple[str, str | bytes, dict[str, str], HTTPStatus]] = [
        (
            valid_path,
            "",
            {**base_headers, "Host": "attacker.invalid"},
            HTTPStatus.MISDIRECTED_REQUEST,
        ),
        ("/not-a-decision", "", base_headers, HTTPStatus.NOT_FOUND),
        (valid_path, "x", base_headers, HTTPStatus.UNSUPPORTED_MEDIA_TYPE),
        (
            valid_path,
            "x",
            {
                **base_headers,
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": "bad",
            },
            HTTPStatus.BAD_REQUEST,
        ),
        (
            valid_path,
            "",
            {**base_headers, "Content-Type": "application/x-www-form-urlencoded"},
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        ),
        (
            valid_path,
            "x",
            {
                **base_headers,
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": "4097",
            },
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        ),
        (
            valid_path,
            b"\xff",
            {**base_headers, "Content-Type": "application/x-www-form-urlencoded"},
            HTTPStatus.BAD_REQUEST,
        ),
    ]
    for path, body_value, headers, expected in cases:
        status, _, _ = handle_request(
            controller,
            "POST",
            path,
            body=body_value,
            headers=headers,
        )
        assert status == expected


def test_tunneled_http_surface_maps_review_errors_without_leaking_content(tmp_path: Path) -> None:
    controller, _, _, now = prepared_controller(tmp_path)
    csrf = csrf_from_page(controller.render(TOKEN, now=now))
    headers = {
        "Origin": PUBLIC_BASE_URL,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    status, _, _ = handle_request(
        controller,
        "POST",
        f"/review/{TOKEN}/approve",
        body=urlencode({"csrf_token": "wrong"}),
        headers=headers,
    )
    assert status == HTTPStatus.FORBIDDEN

    status, _, _ = handle_request(
        controller,
        "POST",
        f"/review/{'B' * 43}/approve",
        body=urlencode({"csrf_token": csrf}),
        headers=headers,
    )
    assert status == HTTPStatus.NOT_FOUND

    status, _, _ = handle_request(
        controller,
        "POST",
        f"/review/{TOKEN}/approve",
        body=urlencode({"csrf_token": csrf}),
        headers=headers,
    )
    assert status == HTTPStatus.SEE_OTHER

    status, _, _ = handle_request(
        controller,
        "POST",
        f"/review/{TOKEN}/reject",
        body=urlencode({"csrf_token": csrf}),
        headers=headers,
    )
    assert status == HTTPStatus.CONFLICT


def test_remote_server_runs_periodic_callback_only_when_due(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    times = iter((100.0, 100.0, 105.0, 111.0, 111.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    server = object.__new__(RemoteApprovalHTTPServer)
    server.periodic_callback = lambda: calls.append("called")
    server.callback_interval_seconds = 10
    server.next_callback_at = 0

    server.service_actions()
    server.service_actions()
    server.service_actions()

    assert calls == ["called", "called"]


@pytest.mark.parametrize("response_body", [b"no", b"ok"])
def test_relay_healthcheck_rejects_wrong_response(
    monkeypatch: pytest.MonkeyPatch,
    response_body: bytes,
) -> None:
    class FakeResponse:
        status = HTTPStatus.BAD_GATEWAY if response_body == b"ok" else HTTPStatus.OK

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, limit: int) -> bytes:
            del limit
            return response_body

    monkeypatch.setattr(remote_web_module, "urlopen", lambda request, timeout: FakeResponse())

    assert relay_is_healthy(PUBLIC_BASE_URL) is False


def test_relay_healthcheck_reduces_network_errors_to_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request: object, timeout: int) -> object:
        del request, timeout
        raise URLError("private network detail")

    monkeypatch.setattr(remote_web_module, "urlopen", fail)

    assert relay_is_healthy(PUBLIC_BASE_URL) is False

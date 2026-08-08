import re
import socket
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from http.server import HTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import urlencode

import pytest

import lead_agent.remote_approval_web as remote_web_module
from lead_agent.approvals import ApprovalAction, ApprovalStateError, LocalApprovalService
from lead_agent.database import Database
from lead_agent.models import FacebookPost, Lead, LeadIntent, LeadStatus
from lead_agent.notifications import remote_token_hash
from lead_agent.remote_approval_web import (
    LOOPBACK_HOST,
    RemoteApprovalController,
    RemoteCSRFFailure,
    _handler_class,
    relay_is_healthy,
)

TOKEN = "A" * 43
PUBLIC_BASE_URL = "https://approve.example"
VALID_DRAFT = (
    "JJ Miller & Co. can help with your deck project. Free estimates. "
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


class FakeHTTPServer:
    server_name = LOOPBACK_HOST
    server_port = 8766


def handle_request(
    controller: RemoteApprovalController,
    method: str,
    path: str,
    *,
    body: str | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], str]:
    client, server = socket.socketpair()
    try:
        payload = (body or "").encode()
        request_headers = {
            "Host": "approve.example",
            "Connection": "close",
            **(headers or {}),
        }
        if payload:
            request_headers["Content-Length"] = str(len(payload))
        header_lines = "\r\n".join(f"{name}: {value}" for name, value in request_headers.items())
        client.sendall(f"{method} {path} HTTP/1.1\r\n{header_lines}\r\n\r\n".encode() + payload)
        client.shutdown(socket.SHUT_WR)
        handler = _handler_class(
            controller,
            port=8766,
            public_base_url=PUBLIC_BASE_URL,
        )
        handler(server, (LOOPBACK_HOST, 12345), cast(HTTPServer, FakeHTTPServer()))
        server.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while chunk := client.recv(65_536):
            chunks.append(chunk)
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

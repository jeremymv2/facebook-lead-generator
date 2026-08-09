import socket
import threading
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import HTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import urlencode

import pytest

from lead_agent.approval_web import (
    LOOPBACK_HOST,
    CSRFFailure,
    LocalApprovalController,
    _handler_class,
    _safe_facebook_post_url,
)
from lead_agent.approvals import ApprovalAction, LocalApprovalService
from lead_agent.database import Database
from lead_agent.models import AuditEvent, FacebookPost, Lead, LeadIntent, LeadStatus

VALID_DRAFT = (
    "JJ Miller & Co. can help with your deck project. Free estimates. "
    "Text me at 502-528-0858 or visit https://jjmillerco.com."
)


def prepared_controller(tmp_path: Path) -> tuple[LocalApprovalController, int, datetime]:
    database = Database(tmp_path / "dashboard.sqlite3")
    database.initialize()
    post = database.save_post(
        FacebookPost(
            external_post_id="web-fixture",
            post_url="javascript:alert(1)",
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
    service = LocalApprovalService(database, expiration_minutes=20)
    request_id = service.prepare_candidates(limit=10, now=now)[0].request.id or 0
    return LocalApprovalController(service, csrf_token="fixture-csrf"), request_id, now


def test_dashboard_escapes_facebook_content_and_states_safety_boundary(tmp_path: Path) -> None:
    controller, _, now = prepared_controller(tmp_path)

    page = controller.render(now=now)

    assert "&lt;script&gt;Fixture Group&lt;/script&gt;" in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "<script>Fixture Group</script>" not in page
    assert "This dashboard cannot post to Facebook" in page
    assert 'name="csrf_token" value="fixture-csrf"' in page
    assert "javascript:alert" not in page


def test_dashboard_renders_cycle_trends_and_current_group_health(tmp_path: Path) -> None:
    controller, _, now = prepared_controller(tmp_path)
    database = controller.service.database
    database.record_audit_event(
        AuditEvent(
            component="operations",
            action="cycle.run",
            result="success",
            occurred_at=now,
            details={
                "groups_scanned": 8,
                "groups_failed": 0,
                "posts_seen": 80,
                "posts_new": 5,
                "posts_classified": 5,
                "candidates_created": 1,
                "notifications_sent": 0,
            },
        )
    )
    database.record_audit_event(
        AuditEvent(
            component="operations",
            action="cycle.run",
            result="degraded",
            occurred_at=now,
            details={
                "groups_scanned": 5,
                "groups_failed": 3,
                "groups_partial": 2,
                "groups_severely_partial": 1,
                "groups_retried": 4,
                "groups_recovered": 1,
                "posts_seen": 39,
                "posts_new": 5,
                "posts_classified": 5,
                "candidates_created": 0,
                "notifications_sent": 0,
            },
        )
    )
    database.record_group_scan_failure(
        group_id="group-degraded",
        group_name="<script>Degraded Group</script>",
        group_url="https://www.facebook.com/groups/999",
        error="TransientFacebookReadError:feed:timeout",
        occurred_at=now,
    )

    page = controller.render(now=now)

    assert "Historical trends" in page
    assert "Candidate review quality" in page
    assert "68.8%" in page
    assert "Full group coverage" in page
    assert "Recent cycle details" in page
    assert "Severe partial" in page
    assert "Current group health" in page
    assert "3/8" in page
    assert "4/1" in page
    assert "&lt;script&gt;Degraded Group&lt;/script&gt;" in page
    assert "<script>Degraded Group</script>" not in page


def test_controller_rejects_missing_csrf_and_accepts_once(tmp_path: Path) -> None:
    controller, request_id, now = prepared_controller(tmp_path)

    with pytest.raises(CSRFFailure, match="CSRF"):
        controller.submit(
            request_id,
            ApprovalAction.APPROVE.value,
            {},
            now=now,
        )

    result = controller.submit(
        request_id,
        ApprovalAction.APPROVE.value,
        {"csrf_token": ["fixture-csrf"]},
        now=now,
    )

    assert result.result == "approved"
    assert "No Facebook action" in result.message


class FakeHTTPServer:
    server_name = LOOPBACK_HOST
    server_port = 8765


def handle_request(
    controller: LocalApprovalController,
    method: str,
    path: str,
    *,
    body: str | bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], str]:
    client, server = socket.socketpair()
    try:
        payload = body if isinstance(body, bytes) else (body or "").encode("utf-8")
        request_headers = {
            "Host": f"{LOOPBACK_HOST}:8765",
            "Connection": "close",
            **(headers or {}),
        }
        if payload and "Content-Length" not in request_headers:
            request_headers["Content-Length"] = str(len(payload))
        header_lines = "\r\n".join(f"{name}: {value}" for name, value in request_headers.items())
        client.sendall(f"{method} {path} HTTP/1.1\r\n{header_lines}\r\n\r\n".encode() + payload)
        client.shutdown(socket.SHUT_WR)
        handler = _handler_class(controller, port=8765)

        def serve() -> None:
            handler(server, (LOOPBACK_HOST, 12345), cast(HTTPServer, FakeHTTPServer()))
            server.shutdown(socket.SHUT_WR)

        worker = threading.Thread(
            target=serve,
        )
        worker.start()
        chunks: list[bytes] = []
        while chunk := client.recv(65536):
            chunks.append(chunk)
        worker.join(timeout=2)
        assert not worker.is_alive()
        raw_headers, raw_body = b"".join(chunks).split(b"\r\n\r\n", maxsplit=1)
        response_header_lines = raw_headers.decode("iso-8859-1").split("\r\n")
        status = int(response_header_lines[0].split()[1])
        response_headers = dict(line.split(": ", maxsplit=1) for line in response_header_lines[1:])
        return status, response_headers, raw_body.decode("utf-8")
    finally:
        client.close()
        server.close()


def test_loopback_server_enforces_headers_csrf_and_one_time_decision(tmp_path: Path) -> None:
    controller, request_id, _ = prepared_controller(tmp_path)
    host = f"{LOOPBACK_HOST}:8765"
    origin = f"http://{host}"

    status, headers, page = handle_request(controller, "GET", "/", headers={"Host": host})
    assert status == HTTPStatus.OK
    assert headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert "Lead Review" in page

    status, _, _ = handle_request(controller, "GET", "/", headers={"Host": "attacker.invalid"})
    assert status == HTTPStatus.MISDIRECTED_REQUEST

    form = urlencode({"csrf_token": "fixture-csrf"})
    form_headers = {
        "Host": host,
        "Origin": "https://attacker.invalid",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    status, _, _ = handle_request(
        controller,
        "POST",
        f"/approvals/{request_id}/approve",
        body=form,
        headers=form_headers,
    )
    assert status == HTTPStatus.FORBIDDEN

    form_headers["Origin"] = origin
    wrong_csrf = urlencode({"csrf_token": "wrong"})
    status, _, _ = handle_request(
        controller,
        "POST",
        f"/approvals/{request_id}/approve",
        body=wrong_csrf,
        headers=form_headers,
    )
    assert status == HTTPStatus.FORBIDDEN

    status, headers, _ = handle_request(
        controller,
        "POST",
        f"/approvals/{request_id}/approve",
        body=form,
        headers=form_headers,
    )
    assert status == HTTPStatus.SEE_OTHER
    assert headers["Location"] == "/?result=approved"

    status, _, page = handle_request(
        controller,
        "POST",
        f"/approvals/{request_id}/reject",
        body=form,
        headers=form_headers,
    )
    assert status == HTTPStatus.CONFLICT
    assert "already been decided" in page


def test_controller_supports_explicit_edit_and_reject_decisions(tmp_path: Path) -> None:
    edit_directory = tmp_path / "edit"
    edit_directory.mkdir()
    edit_controller, edit_request_id, now = prepared_controller(edit_directory)
    edited = edit_controller.submit(
        edit_request_id,
        ApprovalAction.EDIT.value,
        {
            "csrf_token": ["fixture-csrf"],
            "response": [VALID_DRAFT],
        },
        now=now,
    )
    assert edited.result == "edited"
    assert "Edited response approved" in edited.message

    reject_directory = tmp_path / "reject"
    reject_directory.mkdir()
    reject_controller, reject_request_id, now = prepared_controller(reject_directory)
    rejected = reject_controller.submit(
        reject_request_id,
        ApprovalAction.REJECT.value,
        {
            "csrf_token": ["fixture-csrf"],
            "rejection_reason": ["provider_advertisement"],
        },
        now=now,
    )
    assert rejected.result == "rejected"
    assert "Lead rejected" in rejected.message


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("http://www.facebook.com/groups/111/posts/222", None),
        ("https://example.com/groups/111/posts/222", None),
        (
            "https://www.facebook.com/groups/111/posts/222",
            "https://www.facebook.com/groups/111/posts/222",
        ),
    ],
)
def test_dashboard_links_only_to_https_facebook_posts(
    value: str | None,
    expected: str | None,
) -> None:
    assert _safe_facebook_post_url(value) == expected


def test_loopback_server_rejects_malformed_requests(tmp_path: Path) -> None:
    controller, request_id, _ = prepared_controller(tmp_path)
    valid_path = f"/approvals/{request_id}/approve"
    host = f"{LOOPBACK_HOST}:8765"
    origin = f"http://{host}"

    status, _, body = handle_request(controller, "GET", "/health")
    assert status == HTTPStatus.OK
    assert body == "ok"
    status, _, _ = handle_request(controller, "GET", "/missing")
    assert status == HTTPStatus.NOT_FOUND
    status, _, page = handle_request(controller, "GET", "/?result=rejected")
    assert status == HTTPStatus.OK
    assert "Lead rejected" in page

    cases: list[tuple[str, str | bytes, dict[str, str], HTTPStatus]] = [
        (valid_path, "", {"Host": "attacker.invalid"}, HTTPStatus.MISDIRECTED_REQUEST),
        ("/not-an-approval", "", {"Origin": origin}, HTTPStatus.NOT_FOUND),
        (valid_path, "x", {"Origin": origin}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE),
        (
            valid_path,
            "x",
            {
                "Origin": origin,
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": "bad",
            },
            HTTPStatus.BAD_REQUEST,
        ),
        (
            valid_path,
            "",
            {"Origin": origin, "Content-Type": "application/x-www-form-urlencoded"},
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        ),
        (
            valid_path,
            "x",
            {
                "Origin": origin,
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": "4097",
            },
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        ),
        (
            valid_path,
            b"\xff",
            {"Origin": origin, "Content-Type": "application/x-www-form-urlencoded"},
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

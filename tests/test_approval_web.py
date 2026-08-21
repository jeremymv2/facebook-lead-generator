import socket
import threading
from datetime import UTC, datetime, timedelta
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
    _valid_local_origin,
)
from lead_agent.approvals import ApprovalAction, LocalApprovalService
from lead_agent.database import Database
from lead_agent.models import AuditEvent, FacebookPost, Lead, LeadIntent, LeadStatus

VALID_DRAFT = (
    "JJ Miller & Co. can help with your deck project. Licensed & Insured. Free estimates. "
    "Text me at 502-528-0858 or visit https://jjmillerco.com."
)


def prepared_controller(tmp_path: Path) -> tuple[LocalApprovalController, int, datetime]:
    database = Database(tmp_path / "dashboard.sqlite3")
    database.initialize()
    post = database.save_post(
        FacebookPost(
            external_post_id="web-fixture",
            post_url="https://www.facebook.com/groups/111/posts/web-fixture",
            group_id="fixture-group",
            group_name="<script>Fixture Group</script>",
            post_text="<img src=x onerror=alert(1)> Need a deck repair in Louisville.",
        )
    ).post
    lead = database.create_lead(
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
    return LocalApprovalController(service, csrf_token="fixture-csrf"), lead.id or 0, now


def test_dashboard_escapes_facebook_content_and_states_safety_boundary(tmp_path: Path) -> None:
    controller, _, now = prepared_controller(tmp_path)

    page = controller.render(now=now)

    assert "&lt;script&gt;Fixture Group&lt;/script&gt;" in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "<script>Fixture Group</script>" not in page
    assert "This dashboard cannot post to Facebook" in page
    assert "Remains in this local backlog until you approve or reject it" in page
    assert 'name="csrf_token" value="fixture-csrf"' in page
    assert controller.service.database.list_pending_approval_reviews() == []
    assert "View original Facebook post" in page
    assert 'href="https://www.facebook.com/groups/111/posts/web-fixture"' in page


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
                "groups_shortfall": 3,
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
    assert "Facebook posting outcomes" in page
    assert "Publicly posted" in page
    assert "Pending group moderation" in page
    assert "never\n        submitted again automatically" in page
    assert "Candidate review quality" in page
    assert "68.8%" in page
    assert "Healthy group coverage" in page
    assert "Recent cycle details" in page
    assert "Minor shortfall" in page
    assert "Severe partial" in page
    assert "Current group health" in page
    assert "3/8" in page
    assert "4/1" in page
    assert "&lt;script&gt;Degraded Group&lt;/script&gt;" in page
    assert "<script>Degraded Group</script>" not in page


def test_dashboard_refresh_adds_candidates_without_starting_expiration(tmp_path: Path) -> None:
    controller, _, now = prepared_controller(tmp_path)
    database = controller.service.database
    first_page = controller.render(now=now)
    post = database.save_post(
        FacebookPost(
            external_post_id="new-after-dashboard-start",
            post_url=("https://www.facebook.com/groups/111/posts/new-after-dashboard-start"),
            group_id="fixture-group",
            group_name="Synthetic Fixture Group",
            post_text="Looking for someone to replace a window in Louisville.",
        )
    ).post
    new_lead = database.create_lead(
        Lead(
            facebook_post_id=post.id or 0,
            status=LeadStatus.CANDIDATE,
            service_category="windows",
            intent=LeadIntent.HIRING,
            overall_score=90,
            drafted_response=VALID_DRAFT,
        )
    )

    refreshed_page = controller.render(now=now)

    assert "replace a window" not in first_page
    assert "replace a window" in refreshed_page
    persisted = database.get_lead(new_lead.id or 0)
    assert persisted is not None
    assert persisted.status is LeadStatus.CANDIDATE
    assert database.list_pending_approval_reviews() == []


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


def test_dashboard_can_approve_and_queue_posting(tmp_path: Path) -> None:
    controller, lead_id, now = prepared_controller(tmp_path)
    posting_controller = LocalApprovalController(
        controller.service,
        csrf_token="fixture-csrf",
        posting_queue_enabled=True,
        posting_enabled_group_ids={"fixture-group"},
        posting_approval_max_age_minutes=20,
    )

    page = posting_controller.render(now=now)
    assert "Approve draft" in page
    assert "Approve edited response &amp; queue post" in page

    result = posting_controller.submit(
        lead_id,
        "approve-post",
        {"csrf_token": ["fixture-csrf"]},
        now=now,
    )

    assert result.result == "approved"
    assert "queued for Facebook posting" in result.message
    approval = controller.service.database.list_pending_approval_reviews()
    assert approval == []
    job = controller.service.database.get_posting_job_for_approval(1)
    assert job is not None
    assert job.status.value == "queued"


def test_dashboard_can_queue_a_recent_existing_approval(tmp_path: Path) -> None:
    controller, lead_id, now = prepared_controller(tmp_path)
    controller.submit(
        lead_id,
        "approve",
        {"csrf_token": ["fixture-csrf"]},
        now=now,
    )
    posting_controller = LocalApprovalController(
        controller.service,
        csrf_token="fixture-csrf",
        posting_queue_enabled=True,
        posting_enabled_group_ids={"fixture-group"},
        posting_approval_max_age_minutes=20,
    )

    page = posting_controller.render(now=now)
    assert "Queue approved response for Facebook" in page

    result = posting_controller.submit(
        lead_id,
        "post",
        {"csrf_token": ["fixture-csrf"]},
        now=now + timedelta(minutes=1),
    )

    assert result.result == "queued"
    assert "queued for Facebook posting" in result.message


def test_dashboard_reopens_stale_approval_for_fresh_review(tmp_path: Path) -> None:
    controller, lead_id, now = prepared_controller(tmp_path)
    controller.submit(
        lead_id,
        "approve",
        {"csrf_token": ["fixture-csrf"]},
        now=now,
    )
    posting_controller = LocalApprovalController(
        controller.service,
        csrf_token="fixture-csrf",
        posting_queue_enabled=True,
        posting_enabled_group_ids={"fixture-group"},
        posting_approval_max_age_minutes=20,
    )

    page = posting_controller.render(now=now + timedelta(minutes=21))
    assert "Return to fresh review" in page

    result = posting_controller.submit(
        lead_id,
        "re-review",
        {"csrf_token": ["fixture-csrf"]},
        now=now + timedelta(minutes=21),
    )

    assert result.result == "reopened"
    reopened = controller.service.database.get_lead(lead_id)
    assert reopened is not None
    assert reopened.status.value == "candidate"


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
    controller, lead_id, _ = prepared_controller(tmp_path)
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
        f"/leads/{lead_id}/approve",
        body=form,
        headers=form_headers,
    )
    assert status == HTTPStatus.FORBIDDEN

    form_headers["Origin"] = origin
    wrong_csrf = urlencode({"csrf_token": "wrong"})
    status, _, _ = handle_request(
        controller,
        "POST",
        f"/leads/{lead_id}/approve",
        body=wrong_csrf,
        headers=form_headers,
    )
    assert status == HTTPStatus.FORBIDDEN

    status, headers, _ = handle_request(
        controller,
        "POST",
        f"/leads/{lead_id}/approve",
        body=form,
        headers=form_headers,
    )
    assert status == HTTPStatus.SEE_OTHER
    assert headers["Location"] == "/?result=approved"

    status, _, page = handle_request(
        controller,
        "POST",
        f"/leads/{lead_id}/reject",
        body=form,
        headers=form_headers,
    )
    assert status == HTTPStatus.CONFLICT
    assert "already been decided" in page


def test_loopback_server_accepts_verified_same_origin_browser_fallback(tmp_path: Path) -> None:
    controller, lead_id, _ = prepared_controller(tmp_path)
    host = f"{LOOPBACK_HOST}:8765"
    form = urlencode({"csrf_token": "fixture-csrf"})

    status, headers, _ = handle_request(
        controller,
        "POST",
        f"/leads/{lead_id}/approve",
        body=form,
        headers={
            "Host": host,
            "Origin": "null",
            "Sec-Fetch-Site": "same-origin",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    assert status == HTTPStatus.SEE_OTHER
    assert headers["Location"] == "/?result=approved"


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
        ("https://www.facebook.com/groups/111", None),
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


@pytest.mark.parametrize(
    ("origin", "referer", "fetch_site", "expected"),
    [
        (None, None, None, True),
        ("http://127.0.0.1:8765", None, None, True),
        ("http://localhost:8765/", None, None, True),
        ("http://LOCALHOST:8765", None, None, True),
        ("http://127.0.0.1:9999", None, None, False),
        ("https://127.0.0.1:8765", None, None, False),
        ("https://attacker.invalid", None, None, False),
        ("null", None, "same-origin", True),
        ("null", "http://127.0.0.1:8765/", "same-origin", True),
        ("codex://browser", "http://localhost:8765/?review=1", "same-origin", True),
        ("null", "http://127.0.0.1:8765/", "cross-site", False),
        ("null", "https://attacker.invalid/", "same-origin", False),
    ],
)
def test_local_origin_accepts_safe_browser_variants(
    origin: str | None,
    referer: str | None,
    fetch_site: str | None,
    expected: bool,
) -> None:
    assert (
        _valid_local_origin(
            origin,
            referer=referer,
            fetch_site=fetch_site,
            port=8765,
        )
        is expected
    )


def test_loopback_server_rejects_malformed_requests(tmp_path: Path) -> None:
    controller, request_id, _ = prepared_controller(tmp_path)
    valid_path = f"/leads/{request_id}/approve"
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

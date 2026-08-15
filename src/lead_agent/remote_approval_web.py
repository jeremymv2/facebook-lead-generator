"""Tokenized mobile approval pages served locally through an outbound-only relay."""

from __future__ import annotations

import hashlib
import hmac
import html
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

from lead_agent.approvals import (
    ApprovalAction,
    ApprovalError,
    ApprovalExpiredError,
    ApprovalStateError,
    LocalApprovalService,
)
from lead_agent.models import (
    ApprovalReview,
    ApprovalStatus,
    PostingJob,
    PostingJobStatus,
    RejectionReason,
    is_exact_facebook_post_url,
    utc_now,
)
from lead_agent.notifications import remote_token_hash

LOOPBACK_HOST = "127.0.0.1"
MAX_FORM_BYTES = 4096
REMOTE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}")
REVIEW_PATH = re.compile(r"^/review/([A-Za-z0-9_-]{43})$")
DECISION_PATH = re.compile(
    r"^/review/([A-Za-z0-9_-]{43})/(approve|approve-post|edit|edit-post|reject)$"
)


class RemoteApprovalTokenError(ApprovalError):
    """Raised when a public review token is invalid or unknown."""


class RemoteCSRFFailure(ApprovalError):
    """Raised when a tunneled form fails its token-bound CSRF check."""


def relay_is_healthy(public_base_url: str, *, timeout_seconds: int = 5) -> bool:
    """Confirm the configured HTTPS relay reaches this Mac before sending any link."""
    request = Request(
        f"{public_base_url.rstrip('/')}/health",
        headers={"Accept": "text/plain"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(16)
            return int(response.status) == HTTPStatus.OK and body == b"ok"
    except (TimeoutError, URLError, OSError):
        return False


class RemoteApprovalController:
    """Resolve one opaque review URL and apply one human decision."""

    def __init__(
        self,
        approvals: LocalApprovalService,
        *,
        signing_key: str,
        posting_queue_enabled: bool = False,
        posting_enabled_group_ids: set[str] | None = None,
    ) -> None:
        if len(signing_key) < 32:
            raise ValueError("Remote approval signing key must contain at least 32 characters")
        self.approvals = approvals
        self.database = approvals.database
        self._signing_key = signing_key.encode("utf-8")
        self.posting_queue_enabled = posting_queue_enabled
        self.posting_enabled_group_ids = posting_enabled_group_ids or set()

    def resolve(self, token: str, *, now: datetime | None = None) -> ApprovalReview:
        if REMOTE_TOKEN_PATTERN.fullmatch(token) is None:
            raise RemoteApprovalTokenError("This review link is invalid")
        timestamp = now or utc_now()
        self.approvals.expire_pending(now=timestamp)
        review = self.database.get_approval_review_by_remote_token_hash(remote_token_hash(token))
        if review is None:
            raise RemoteApprovalTokenError("This review link is invalid")
        return review

    def render(self, token: str, *, now: datetime | None = None) -> str:
        timestamp = now or utc_now()
        review = self.resolve(token, now=timestamp)
        if review.request.status is not ApprovalStatus.PENDING:
            request_id = review.request.id or 0
            return _render_terminal(
                review.request.status,
                posting_job=self.database.get_posting_job_for_approval(request_id),
            )
        posting_group_enabled = (
            self.posting_queue_enabled and review.post.group_id in self.posting_enabled_group_ids
        )
        exact_post_url_available = is_exact_facebook_post_url(review.post.post_url)
        return _render_pending(
            review,
            token=token,
            csrf_token=self._csrf_token(token),
            now=timestamp,
            posting_available=posting_group_enabled and exact_post_url_available,
            posting_group_enabled=posting_group_enabled,
        )

    def submit(
        self,
        token: str,
        action_value: str,
        form: dict[str, list[str]],
        *,
        now: datetime | None = None,
    ) -> ApprovalReview:
        review = self.resolve(token, now=now)
        if review.request.status is ApprovalStatus.EXPIRED:
            raise ApprovalExpiredError("This review link has expired")
        if review.request.status is not ApprovalStatus.PENDING:
            raise ApprovalStateError("This lead has already been reviewed")
        submitted_csrf = _one_form_value(form, "csrf_token")
        if not hmac.compare_digest(submitted_csrf, self._csrf_token(token)):
            raise RemoteCSRFFailure("Approval form failed CSRF validation")
        queue_posting = action_value in {"approve-post", "edit-post"}
        base_action = action_value.removesuffix("-post")
        if queue_posting and (
            not self.posting_queue_enabled
            or review.post.group_id not in self.posting_enabled_group_ids
        ):
            raise ApprovalStateError("This Facebook group is not enabled for queued posting")
        if queue_posting and not is_exact_facebook_post_url(review.post.post_url):
            raise ApprovalStateError(
                "This lead is review-only because its exact Facebook post link is unavailable"
            )
        try:
            action = ApprovalAction(base_action)
        except ValueError as error:  # pragma: no cover - route contract
            raise ApprovalStateError("Unknown approval action") from error
        return self.approvals.decide(
            review.request.id or 0,
            action,
            edited_response=_one_form_value(form, "response", required=False),
            rejection_reason=_one_form_value(form, "rejection_reason", required=False),
            queue_posting=queue_posting,
            now=now,
        )

    def _csrf_token(self, token: str) -> str:
        return hmac.new(
            self._signing_key,
            f"remote-approval-csrf:{token}".encode(),
            hashlib.sha256,
        ).hexdigest()


def _one_form_value(
    form: dict[str, list[str]],
    name: str,
    *,
    required: bool = True,
) -> str:
    values = form.get(name, [])
    if len(values) != 1:
        if not required and not values:
            return ""
        raise ApprovalStateError(f"Approval form field {name!r} is missing or repeated")
    return values[0]


def _render_pending(
    review: ApprovalReview,
    *,
    token: str,
    csrf_token: str,
    now: datetime,
    posting_available: bool,
    posting_group_enabled: bool,
) -> str:
    remaining_seconds = max(0, int((review.request.expires_at - now).total_seconds()))
    remaining_minutes = (remaining_seconds + 59) // 60
    score = review.lead.overall_score if review.lead.overall_score is not None else "—"
    service = (review.lead.service_category or "project").replace("_", " ").title()
    token_path = html.escape(token, quote=True)
    csrf = html.escape(csrf_token, quote=True)
    draft = html.escape(review.request.draft_response)
    rejection_options = "".join(
        f'<option value="{reason.value}">{html.escape(reason.value.replace("_", " ").title())}'
        "</option>"
        for reason in RejectionReason
    )
    edited_post_button = (
        f'<button class="post" type="submit" formaction="/review/{token_path}/edit-post">'
        "Approve edited response &amp; post</button>"
        if posting_available
        else ""
    )
    draft_post_form = (
        f"""
    <form method="post" action="/review/{token_path}/approve-post">
      <input type="hidden" name="csrf_token" value="{csrf}">
      <button class="post" type="submit">Approve &amp; post</button>
    </form>"""
        if posting_available
        else ""
    )
    safety = (
        "Approve &amp; post queues one guarded submission on your Mac. Facebook is never called "
        "inside this web request."
        if posting_available
        else (
            "Review only: an exact Facebook post link was not captured. Approval stores your "
            "decision, but posting remains unavailable until a later scan recovers the permalink."
            if posting_group_enabled
            else "This group is scan-only. Approval cannot post to Facebook; "
            "it only stores the decision."
        )
    )
    return _page(
        f"""
<section class="card">
  <h1>{html.escape(service)} lead</h1>
  <div class="meta">
    <span>{html.escape(review.post.group_name)}</span>
    <span>Score: {score}</span>
  </div>
  <p class="expiry">Review expires in approximately {remaining_minutes} minute(s).</p>
  <h2>Facebook post</h2>
  <p class="post">{html.escape(review.post.post_text)}</p>
  <h2>Proposed response</h2>
  <form method="post" action="/review/{token_path}/edit">
    <input type="hidden" name="csrf_token" value="{csrf}">
    <textarea name="response" maxlength="300" required>{draft}</textarea>
    <button class="edit" type="submit">Approve edited response</button>
    {edited_post_button}
  </form>
  <div class="actions">
    <form method="post" action="/review/{token_path}/approve">
      <input type="hidden" name="csrf_token" value="{csrf}">
      <button class="approve" type="submit">Approve draft</button>
    </form>
    {draft_post_form}
    <form method="post" action="/review/{token_path}/reject">
      <input type="hidden" name="csrf_token" value="{csrf}">
      <label>Reason for rejection
        <select name="rejection_reason" required>
          <option value="" selected disabled>Select a reason</option>
          {rejection_options}
        </select>
      </label>
      <button class="reject" type="submit">Reject</button>
    </form>
  </div>
</section>
<p class="safety">{safety}</p>
"""
    )


def _render_terminal(status: ApprovalStatus, *, posting_job: PostingJob | None = None) -> str:
    messages = {
        ApprovalStatus.APPROVED: "Draft approved",
        ApprovalStatus.EDITED: "Edited response approved",
        ApprovalStatus.REJECTED: "Lead rejected",
        ApprovalStatus.EXPIRED: "Review link expired",
    }
    message = messages.get(status, "Review unavailable")
    if posting_job is None:
        detail = "The decision is stored on your Mac. No Facebook action was requested."
    else:
        job_messages = {
            PostingJobStatus.QUEUED: "Facebook submission is queued on your Mac.",
            PostingJobStatus.PROCESSING: "Your Mac is validating the Facebook submission.",
            PostingJobStatus.POSTED: "The response was posted publicly to Facebook.",
            PostingJobStatus.PENDING_MODERATION: (
                "Facebook accepted the response for group admin review."
            ),
            PostingJobStatus.EXPIRED: "The posting approval expired before submission.",
            PostingJobStatus.FAILED: "Posting stopped safely before submission was confirmed.",
            PostingJobStatus.NEEDS_ATTENTION: (
                "Submission may have crossed Facebook's boundary and needs manual review."
            ),
        }
        detail = job_messages[posting_job.status]
    return _page(
        f"""
<section class="card terminal">
  <h1>{html.escape(message)}</h1>
  <p>{html.escape(detail)}</p>
</section>
"""
    )


def _page(content: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>JJ Miller &amp; Co. Lead Review</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #f4f5f7; color: #17202a; }}
    main {{ margin: 20px auto 48px; width: min(680px, calc(100% - 24px)); }}
    .card {{ background: white; border: 1px solid #d8dee4; border-radius: 14px;
      box-shadow: 0 2px 10px rgb(0 0 0 / 8%); padding: 20px; }}
    h1 {{ margin-top: 0; }}
    h2 {{ font-size: 1rem; margin: 22px 0 8px; }}
    .meta {{ color: #57606a; display: flex; flex-wrap: wrap; gap: 8px 16px; }}
    .expiry {{ color: #9a6700; font-weight: 700; }}
    .post {{ border-left: 4px solid #6e7781; padding: 10px 12px; white-space: pre-wrap; }}
    textarea, select {{ box-sizing: border-box; font: inherit; padding: 11px; width: 100%; }}
    textarea {{ min-height: 120px; }}
    .actions {{ display: flex; gap: 10px; margin-top: 12px; }}
    .actions form {{ flex: 1; }}
    button {{ border: 0; border-radius: 9px; color: white; cursor: pointer;
      font-size: 1rem; font-weight: 750; margin-top: 12px; padding: 13px; width: 100%; }}
    .approve {{ background: #1f883d; }} .edit {{ background: #0969da; }}
    .post {{ background: #8250df; }}
    .reject {{ background: #cf222e; }}
    .safety {{ color: #57606a; font-size: .9rem; text-align: center; }}
    .terminal {{ margin-top: 25vh; text-align: center; }}
  </style>
</head>
<body><main>{content}</main></body>
</html>"""


class RemoteApprovalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        periodic_callback: Callable[[], None],
        callback_interval_seconds: int,
    ) -> None:
        super().__init__(server_address, handler)
        self.periodic_callback = periodic_callback
        self.callback_interval_seconds = callback_interval_seconds
        self.next_callback_at = 0.0

    def service_actions(self) -> None:
        if time.monotonic() < self.next_callback_at:
            return
        self.next_callback_at = time.monotonic() + self.callback_interval_seconds
        try:
            self.periodic_callback()
        except Exception as error:  # pragma: no cover - defensive long-running boundary
            logging.getLogger("lead_agent.remote_approval").error(
                "Remote approval notification cycle failed",
                extra={
                    "action": "notification.cycle",
                    "result": "failed",
                    "error_type": type(error).__name__,
                },
            )


def run_remote_approval_server(
    controller: RemoteApprovalController,
    *,
    port: int,
    public_base_url: str,
    periodic_callback: Callable[[], None],
    callback_interval_seconds: int,
) -> None:  # pragma: no cover - interactive long-running server
    handler = _handler_class(controller, port=port, public_base_url=public_base_url)
    server = RemoteApprovalHTTPServer(
        (LOOPBACK_HOST, port),
        handler,
        periodic_callback=periodic_callback,
        callback_interval_seconds=callback_interval_seconds,
    )
    print(f"Remote approval origin: http://{LOOPBACK_HOST}:{port}")
    print(f"Expected secure relay URL: {public_base_url}")
    print("Authorized approvals may queue guarded Facebook posting. Press Ctrl-C to stop.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _handler_class(
    controller: RemoteApprovalController,
    *,
    port: int,
    public_base_url: str,
) -> type[BaseHTTPRequestHandler]:
    public_parts = urlsplit(public_base_url.rstrip("/"))
    public_host = public_parts.netloc
    public_origin = f"{public_parts.scheme}://{public_host}"
    local_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "JJMillerApproval/1"
        sys_version = ""

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/health" and self.headers.get("Host", "") in local_hosts | {public_host}:
                self._send_text(HTTPStatus.OK, "ok")
                return
            if self.headers.get("Host", "") != public_host:
                self._send_text(HTTPStatus.MISDIRECTED_REQUEST, "Invalid Host header")
                return
            match = REVIEW_PATH.fullmatch(path)
            if match is None:
                self._send_text(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                page = controller.render(match.group(1))
            except RemoteApprovalTokenError as error:
                self._send_text(HTTPStatus.NOT_FOUND, str(error))
                return
            self._send_html(HTTPStatus.OK, page)

        def do_POST(self) -> None:
            if self.headers.get("Host", "") != public_host:
                self._send_text(HTTPStatus.MISDIRECTED_REQUEST, "Invalid Host header")
                return
            if self.headers.get("Origin") != public_origin:
                self._send_text(HTTPStatus.FORBIDDEN, "Invalid Origin header")
                return
            match = DECISION_PATH.fullmatch(urlsplit(self.path).path)
            if match is None:
                self._send_text(HTTPStatus.NOT_FOUND, "Not found")
                return
            if self.headers.get_content_type() != "application/x-www-form-urlencoded":
                self._send_text(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Unsupported form encoding")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_text(HTTPStatus.BAD_REQUEST, "Invalid form length")
                return
            if not 0 < length <= MAX_FORM_BYTES:
                self._send_text(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Form is too large")
                return
            try:
                form = parse_qs(
                    self.rfile.read(length).decode("utf-8"),
                    keep_blank_values=True,
                    max_num_fields=10,
                )
            except (UnicodeDecodeError, ValueError):
                self._send_text(HTTPStatus.BAD_REQUEST, "Invalid form data")
                return
            try:
                review = controller.submit(match.group(1), match.group(2), form)
            except RemoteCSRFFailure as error:
                self._send_text(HTTPStatus.FORBIDDEN, str(error))
                return
            except RemoteApprovalTokenError as error:
                self._send_text(HTTPStatus.NOT_FOUND, str(error))
                return
            except ApprovalExpiredError as error:
                self._send_text(HTTPStatus.GONE, str(error))
                return
            except ApprovalError as error:
                self._send_text(HTTPStatus.CONFLICT, str(error))
                return
            status = review.request.status.value
            location = f"/review/{match.group(1)}?result={status}"
            self.send_response(HTTPStatus.SEE_OTHER)
            self._security_headers()
            self.send_header("Location", location)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _security_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
            self.send_header("Strict-Transport-Security", "max-age=31536000")

        def _send_html(self, status: HTTPStatus, content: str) -> None:
            payload = content.encode("utf-8")
            self.send_response(status)
            self._security_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_text(self, status: HTTPStatus, content: str) -> None:
            payload = content.encode("utf-8")
            self.send_response(status)
            self._security_headers()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler

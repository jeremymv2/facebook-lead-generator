"""Loopback-only HTML dashboard for local human lead review."""

from __future__ import annotations

import html
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit
from zoneinfo import ZoneInfo

from lead_agent.approvals import (
    ApprovalAction,
    ApprovalError,
    ApprovalExpiredError,
    ApprovalStateError,
    LocalApprovalService,
    LocalReviewItem,
)
from lead_agent.dashboard_metrics import CycleTrend, DashboardMetricsService, DashboardTrendSnapshot
from lead_agent.models import RejectionReason, is_exact_facebook_post_url, utc_now

LOOPBACK_HOST = "127.0.0.1"
MAX_FORM_BYTES = 4096
LOCAL_REVIEW_PATH = re.compile(r"^/leads/(\d+)/(approve|edit|reject)$")


class CSRFFailure(ApprovalError):
    """Raised when a local form does not carry the active server-session token."""


@dataclass(frozen=True, slots=True)
class DashboardResult:
    message: str
    result: str


class LocalApprovalController:
    """Framework-independent controller used by the local HTTP adapter and tests."""

    def __init__(
        self,
        service: LocalApprovalService,
        *,
        csrf_token: str | None = None,
        display_timezone: tzinfo = UTC,
        candidate_limit: int | None = None,
    ) -> None:
        self.service = service
        self.csrf_token = csrf_token or secrets.token_urlsafe(32)
        self.display_timezone = display_timezone
        self.candidate_limit = candidate_limit
        self.metrics = DashboardMetricsService(service.database)

    def render(self, *, message: str | None = None, now: datetime | None = None) -> str:
        reviews = self.service.list_local_backlog(limit=self.candidate_limit, now=now)
        return render_dashboard(
            reviews,
            trends=self.metrics.snapshot(),
            csrf_token=self.csrf_token,
            message=message,
            now=now or utc_now(),
            display_timezone=self.display_timezone,
        )

    def submit(
        self,
        lead_id: int,
        action_value: str,
        form: dict[str, list[str]],
        *,
        now: datetime | None = None,
    ) -> DashboardResult:
        try:
            submitted_token = _one_form_value(form, "csrf_token")
        except ApprovalStateError as error:
            raise CSRFFailure("Approval form expired or failed CSRF validation") from error
        if not compare_digest(submitted_token, self.csrf_token):
            raise CSRFFailure("Approval form expired or failed CSRF validation")
        try:
            action = ApprovalAction(action_value)
        except ValueError as error:  # pragma: no cover - route regex contract
            raise ApprovalStateError("Unknown approval action") from error
        edited_response = _one_form_value(form, "response", required=False)
        rejection_reason = _one_form_value(form, "rejection_reason", required=False)
        review = self.service.decide_local_lead(
            lead_id,
            action,
            edited_response=edited_response,
            rejection_reason=rejection_reason,
            now=now,
        )
        if action is ApprovalAction.REJECT:
            message = "Lead rejected. No Facebook action was taken."
        elif action is ApprovalAction.EDIT:
            message = "Edited response approved locally. No Facebook action was taken."
        else:
            message = "Draft approved locally. No Facebook action was taken."
        return DashboardResult(message=message, result=review.request.status.value)


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


def render_dashboard(
    reviews: list[LocalReviewItem],
    *,
    trends: DashboardTrendSnapshot,
    csrf_token: str,
    message: str | None,
    now: datetime,
    display_timezone: tzinfo = UTC,
) -> str:
    cards = "".join(_render_review(review, csrf_token=csrf_token) for review in reviews)
    if not cards:
        cards = (
            '<section class="empty"><h2>No leads awaiting review</h2>'
            "<p>New candidates will appear here when you refresh this page.</p></section>"
        )
    flash = f'<p class="flash">{html.escape(message)}</p>' if message else ""
    trend_content = _render_trends(trends, display_timezone=display_timezone)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>JJ Miller &amp; Co. Lead Review</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #f4f5f7; color: #17202a; }}
    main {{ width: min(1120px, calc(100% - 28px)); margin: 28px auto 60px; }}
    header {{ margin-bottom: 20px; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 0; }}
    .safety {{ color: #57606a; margin-top: 0; }}
    .card, .empty, .panel {{ background: white; border: 1px solid #d8dee4; border-radius: 12px;
      box-shadow: 0 2px 8px rgb(0 0 0 / 6%); margin: 18px 0; padding: 20px; }}
    .section-heading {{ align-items: baseline; display: flex; flex-wrap: wrap; gap: 8px 16px;
      justify-content: space-between; margin-top: 30px; }}
    .section-heading h2 {{ margin-bottom: 0; }}
    .section-heading p {{ color: #57606a; margin: 0; }}
    .kpis {{ display: grid; gap: 12px; grid-template-columns: repeat(5, minmax(0, 1fr)); }}
    .kpi {{ background: white; border: 1px solid #d8dee4; border-radius: 10px; padding: 14px; }}
    .kpi strong {{ display: block; font-size: 1.55rem; line-height: 1.2; }}
    .kpi span {{ color: #57606a; font-size: .88rem; }}
    .outcome-grid {{ display: grid; gap: 12px;
      grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .outcome {{ background: #f6f8fa; border-radius: 10px; padding: 14px; }}
    .outcome strong {{ display: block; font-size: 1.35rem; }}
    .outcome span {{ color: #57606a; font-size: .84rem; }}
    .trend-grid {{ display: grid; gap: 16px; grid-template-columns: minmax(0, 1.2fr)
      minmax(280px, .8fr); }}
    .panel {{ margin: 16px 0 0; min-width: 0; }}
    .trend-list {{ display: grid; gap: 9px; }}
    .trend-row {{ align-items: center; display: grid; gap: 10px;
      grid-template-columns: 76px minmax(120px, 1fr) 74px; }}
    .trend-row time, .trend-value {{ color: #57606a; font-size: .8rem; }}
    .trend-value {{ text-align: right; }}
    .stack {{ background: #f6f8fa; border-radius: 999px; display: flex; height: 12px;
      overflow: hidden; }}
    .stack .success {{ background: #1f883d; }}
    .stack .partial {{ background: #bf8700; }}
    .stack .failure {{ background: #cf222e; }}
    .throughput {{ align-items: end; display: flex; gap: 7px; height: 160px;
      justify-content: stretch; padding-top: 12px; }}
    .throughput-column {{ align-items: center; display: flex; flex: 1; flex-direction: column;
      height: 100%; justify-content: end; min-width: 0; }}
    .throughput-bar {{ background: #0969da; border-radius: 5px 5px 0 0; min-height: 2px;
      width: min(24px, 80%); }}
    .throughput-column time {{ color: #57606a; font-size: .7rem; margin-top: 6px;
      overflow: hidden; text-overflow: clip; white-space: nowrap; }}
    .legend {{ color: #57606a; font-size: .82rem; }}
    .dot {{ border-radius: 999px; display: inline-block; height: 9px; margin-right: 4px;
      width: 9px; }}
    .dot.success {{ background: #1f883d; }}
    .dot.partial {{ background: #bf8700; margin-left: 12px; }}
    .dot.failure {{ background: #cf222e; margin-left: 12px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; font-size: .86rem; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 9px 8px; text-align: right;
      white-space: nowrap; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    .status {{ border-radius: 999px; display: inline-block; font-size: .75rem;
      font-weight: 700; padding: 3px 8px; }}
    .status.success, .health.healthy {{ background: #dafbe1; color: #116329; }}
    .status.degraded, .health.degraded, .health.partial {{
      background: #fff8c5; color: #633c01; }}
    .status.failed, .status.unknown {{ background: #ffebe9; color: #82071e; }}
    .health {{ border-radius: 999px; display: inline-block; font-size: .75rem;
      font-weight: 700; padding: 3px 8px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px 16px; color: #57606a; }}
    .post {{ border-left: 4px solid #6e7781; padding: 10px 14px; white-space: pre-wrap; }}
    textarea, select {{ box-sizing: border-box; font: inherit; padding: 10px; width: 100%; }}
    textarea {{ min-height: 108px; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
    form {{ margin: 0; }}
    button {{ border: 0; border-radius: 8px; cursor: pointer; font-weight: 700;
      padding: 11px 16px; }}
    .approve {{ background: #1f883d; color: white; }}
    .edit {{ background: #0969da; color: white; }}
    .reject {{ background: #cf222e; color: white; }}
    .flash {{ background: #ddf4ff; border: 1px solid #54aeff; border-radius: 8px; padding: 12px; }}
    .expiry {{ color: #9a6700; font-weight: 650; }}
    .postability {{ background: #fff8c5; border-radius: 8px; color: #633c01;
      font-weight: 650; padding: 10px 12px; }}
    @media (max-width: 820px) {{
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .outcome-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .trend-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>JJ Miller &amp; Co. Lead Review</h1>
    <p class="safety">Local review only. This dashboard cannot post to Facebook.</p>
  </header>
  {flash}
  {trend_content}
  <div class="section-heading"><h2 id="lead-review">Lead review</h2></div>
  {cards}
</main>
</body>
</html>"""


def _render_trends(trends: DashboardTrendSnapshot, *, display_timezone: tzinfo) -> str:
    cycle_count = len(trends.cycles)
    period_label = f"Last {cycle_count} completed cycle(s)"
    kpis = f"""
    <div class="kpis">
      <div class="kpi"><strong>{trends.group_success_percent:g}%</strong>
        <span>Healthy group coverage</span></div>
      <div class="kpi"><strong>{trends.posts_seen}</strong><span>Posts seen</span></div>
      <div class="kpi"><strong>{trends.posts_new}</strong><span>New posts</span></div>
      <div class="kpi"><strong>{trends.candidates_created}</strong><span>Candidates</span></div>
      <div class="kpi"><strong>{trends.degraded_groups}</strong>
        <span>Groups degraded now</span></div>
    </div>"""
    if not trends.cycles:
        cycle_panels = (
            '<section class="panel"><h3>No cycle history yet</h3>'
            "<p>Completed unattended runs will appear here.</p></section>"
        )
        cycle_table = ""
    else:
        displayed_cycles = trends.cycles[-12:]
        maximum_posts = max(1, *(cycle.posts_seen for cycle in displayed_cycles))
        reliability_rows = "".join(
            _render_reliability_row(cycle, display_timezone=display_timezone)
            for cycle in displayed_cycles
        )
        throughput_bars = "".join(
            _render_throughput_bar(
                cycle,
                maximum_posts=maximum_posts,
                display_timezone=display_timezone,
            )
            for cycle in displayed_cycles
        )
        cycle_panels = f"""
        <div class="trend-grid">
          <section class="panel">
            <h3>Group scan reliability</h3>
            <p class="legend"><span class="dot success"></span>Complete
              <span class="dot partial"></span>Partial
              <span class="dot failure"></span>Failed</p>
            <div class="trend-list">{reliability_rows}</div>
          </section>
          <section class="panel">
            <h3>Posts seen per cycle</h3>
            <div class="throughput" role="img"
              aria-label="Posts seen across the most recent completed cycles">
              {throughput_bars}</div>
          </section>
        </div>"""
        cycle_table = _render_cycle_table(displayed_cycles, display_timezone=display_timezone)
    group_table = _render_group_health_table(trends, display_timezone=display_timezone)
    feedback_panel = _render_feedback_panel(trends)
    posting_panel = _render_posting_outcomes(trends)
    return f"""
    <section aria-labelledby="historical-trends">
      <div class="section-heading">
        <h2 id="historical-trends">Historical trends</h2>
        <p>{period_label}; times shown in {html.escape(str(display_timezone))}</p>
      </div>
      {kpis}
      {cycle_panels}
      {cycle_table}
      {group_table}
      {posting_panel}
      {feedback_panel}
    </section>"""


def _render_posting_outcomes(trends: DashboardTrendSnapshot) -> str:
    posting = trends.posting
    return f"""
    <section class="panel">
      <h3>Facebook posting outcomes</h3>
      <p class="legend">Live attempts only. Pending moderation is terminal and is never
        submitted again automatically.</p>
      <div class="outcome-grid">
        <div class="outcome"><strong>{posting.posted}</strong>
          <span>Publicly posted</span></div>
        <div class="outcome"><strong>{posting.pending_moderation}</strong>
          <span>Pending group moderation</span></div>
        <div class="outcome"><strong>{posting.needs_attention}</strong>
          <span>Needs manual review</span></div>
        <div class="outcome"><strong>{posting.failed}</strong>
          <span>Failed before confirmation</span></div>
      </div>
    </section>"""


def _render_feedback_panel(trends: DashboardTrendSnapshot) -> str:
    feedback = trends.feedback
    if feedback.reviewed == 0:
        detail = "<p>No completed lead reviews yet.</p>"
    elif feedback.rejection_reasons:
        rows = "".join(
            f"<tr><td>{html.escape(reason.replace('_', ' ').title())}</td><td>{count}</td></tr>"
            for reason, count in feedback.rejection_reasons
        )
        detail = f"""
        <div class="table-wrap"><table>
          <thead><tr><th>Rejection reason</th><th>Count</th></tr></thead>
          <tbody>{rows}</tbody>
        </table></div>"""
    else:
        detail = "<p>No rejected candidates yet.</p>"
    return f"""
    <section class="panel">
      <h3>Candidate review quality</h3>
      <div class="meta">
        <span>Reviewed: {feedback.reviewed}</span>
        <span>Accepted: {feedback.accepted}</span>
        <span>Rejected: {feedback.rejected}</span>
        <span>Acceptance rate: {feedback.acceptance_percent:g}%</span>
      </div>
      {detail}
    </section>"""


def _render_reliability_row(cycle: CycleTrend, *, display_timezone: tzinfo) -> str:
    attempted = cycle.groups_attempted
    success_width = cycle.groups_complete * 100 / attempted if attempted else 0
    partial_width = cycle.groups_partial * 100 / attempted if attempted else 0
    failure_width = cycle.groups_failed * 100 / attempted if attempted else 0
    time_label = _short_time(cycle.occurred_at, display_timezone)
    aria = (
        f"{cycle.groups_complete} groups healthy, {cycle.groups_partial} partial, and "
        f"{cycle.groups_failed} failed at {time_label}"
    )
    return f"""
    <div class="trend-row">
      <time datetime="{html.escape(cycle.occurred_at.isoformat(), quote=True)}">{time_label}</time>
      <div class="stack" role="img" aria-label="{html.escape(aria, quote=True)}">
        <span class="success" style="width:{success_width:.1f}%"></span>
        <span class="partial" style="width:{partial_width:.1f}%"></span>
        <span class="failure" style="width:{failure_width:.1f}%"></span>
      </div>
      <span class="trend-value">{cycle.groups_complete}/{attempted} healthy</span>
    </div>"""


def _render_throughput_bar(
    cycle: CycleTrend,
    *,
    maximum_posts: int,
    display_timezone: tzinfo,
) -> str:
    height = max(2.0, cycle.posts_seen * 100 / maximum_posts)
    time_label = _short_time(cycle.occurred_at, display_timezone)
    title = f"{time_label}: {cycle.posts_seen} posts seen, {cycle.posts_new} new"
    return f"""
    <div class="throughput-column" title="{html.escape(title, quote=True)}">
      <div class="throughput-bar" style="height:{height:.1f}%"></div>
      <time datetime="{html.escape(cycle.occurred_at.isoformat(), quote=True)}">{time_label}</time>
    </div>"""


def _render_cycle_table(cycles: tuple[CycleTrend, ...], *, display_timezone: tzinfo) -> str:
    rows: list[str] = []
    for value in reversed(cycles):
        status = html.escape(value.status)
        timestamp = _full_time(value.occurred_at, display_timezone)
        rows.append(
            f"""<tr>
              <td><time datetime="{html.escape(value.occurred_at.isoformat(), quote=True)}">
                {timestamp}</time></td>
              <td><span class="status {status}">{status.title()}</span></td>
              <td>{value.groups_complete}/{value.groups_attempted}</td>
              <td>{value.groups_near_complete}</td>
              <td>{value.groups_partial}</td>
              <td>{value.groups_severely_partial}</td>
              <td>{value.groups_retried}/{value.groups_recovered}</td>
              <td>{value.posts_seen}</td><td>{value.posts_new}</td>
              <td>{value.posts_classified}</td><td>{value.candidates_created}</td>
            </tr>"""
        )
    return f"""
    <section class="panel">
      <h3>Recent cycle details</h3>
      <div class="table-wrap"><table>
        <thead><tr><th>Completed</th><th>Status</th><th>Healthy groups</th>
          <th>Minor shortfall</th><th>Partial</th>
          <th>Severe partial</th>
          <th>Retries/recovered</th>
          <th>Seen</th><th>New</th><th>Classified</th><th>Candidates</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table></div>
    </section>"""


def _render_group_health_table(
    trends: DashboardTrendSnapshot,
    *,
    display_timezone: tzinfo,
) -> str:
    if not trends.groups:
        return ""
    rows: list[str] = []
    for group in sorted(trends.groups, key=lambda item: item.group_name.casefold()):
        health = (
            "degraded"
            if group.last_error is not None
            else "partial"
            if group.last_scan_partial
            else "healthy"
        )
        last_success = (
            _full_time(group.last_success_at, display_timezone)
            if group.last_success_at
            else "Never"
        )
        rows.append(
            f"""<tr>
              <td>{html.escape(group.group_name)}</td>
              <td><span class="health {health}">{health.title()}</span></td>
              <td>{group.consecutive_failures}</td>
              <td>{last_success}</td><td>{group.posts_seen}/{group.posts_requested}</td>
              <td>{group.posts_new}</td>
            </tr>"""
        )
    return f"""
    <section class="panel">
      <h3>Current group health</h3>
      <div class="table-wrap"><table>
        <thead><tr><th>Group</th><th>Health</th><th>Failure streak</th>
          <th>Last complete</th><th>Latest seen/requested</th><th>Latest new</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table></div>
    </section>"""


def _short_time(value: datetime, display_timezone: tzinfo) -> str:
    return value.astimezone(display_timezone).strftime("%I:%M %p").lstrip("0")


def _full_time(value: datetime, display_timezone: tzinfo) -> str:
    return value.astimezone(display_timezone).strftime("%b %-d, %-I:%M %p")


def _render_review(review: LocalReviewItem, *, csrf_token: str) -> str:
    lead_id = review.lead.id
    if lead_id is None:  # pragma: no cover - persisted lead contract
        raise RuntimeError("Approval review is missing its lead ID")
    lead = review.lead
    post = review.post
    score = lead.overall_score if lead.overall_score is not None else "—"
    service = (lead.service_category or "unknown").replace("_", " ")
    post_link = ""
    safe_post_url = _safe_facebook_post_url(post.post_url)
    if safe_post_url:
        safe_url = html.escape(safe_post_url, quote=True)
        post_link = (
            f'<p><a href="{safe_url}" target="_blank" rel="noreferrer">Open Facebook post</a></p>'
        )
    else:
        post_link = (
            '<p class="postability">Review only — an exact Facebook post link was not captured. '
            "This lead cannot be submitted until a later scan recovers its permalink.</p>"
        )
    csrf = html.escape(csrf_token, quote=True)
    draft = html.escape(review.draft_response)
    rejection_options = "".join(
        f'<option value="{reason.value}">{html.escape(reason.value.replace("_", " ").title())}'
        "</option>"
        for reason in RejectionReason
    )
    return f"""
<section class="card">
  <h2>{html.escape(service.title())} lead</h2>
  <div class="meta">
    <span>Group: {html.escape(post.group_name)}</span>
    <span>Score: {score}</span>
    <span>Intent: {html.escape(lead.intent.value if lead.intent else "unknown")}</span>
  </div>
  <p class="expiry">Remains in this local backlog until you approve or reject it.</p>
  <h3>Facebook post</h3>
  <p class="post">{html.escape(post.post_text)}</p>
  {post_link}
  <h3>Proposed response</h3>
  <form method="post" action="/leads/{lead_id}/edit">
    <input type="hidden" name="csrf_token" value="{csrf}">
    <textarea name="response" maxlength="300" required>{draft}</textarea>
    <div class="actions">
      <button class="edit" type="submit">Approve edited response</button>
    </div>
  </form>
  <div class="actions">
    <form method="post" action="/leads/{lead_id}/approve">
      <input type="hidden" name="csrf_token" value="{csrf}">
      <button class="approve" type="submit">Approve draft</button>
    </form>
    <form method="post" action="/leads/{lead_id}/reject">
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
</section>"""


def _safe_facebook_post_url(value: str | None) -> str | None:
    return value if is_exact_facebook_post_url(value) else None


def _is_local_dashboard_url(value: str | None, *, port: int, allow_path: bool) -> bool:
    if not value:
        return False
    parts = urlsplit(value)
    try:
        parsed_port = parts.port
    except ValueError:
        return False
    if (
        parts.scheme.casefold() != "http"
        or (parts.hostname or "").casefold() not in {"127.0.0.1", "localhost"}
        or parsed_port != port
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
    ):
        return False
    return allow_path or (parts.path in {"", "/"} and not parts.query)


def _valid_local_origin(
    origin: str | None,
    *,
    referer: str | None,
    fetch_site: str | None,
    port: int,
) -> bool:
    """Accept normalized loopback origins plus verified same-origin browser submissions."""
    if origin is None or _is_local_dashboard_url(origin, port=port, allow_path=False):
        return True
    if origin == "null" and fetch_site == "same-origin":
        return referer is None or _is_local_dashboard_url(
            referer,
            port=port,
            allow_path=True,
        )
    return fetch_site == "same-origin" and _is_local_dashboard_url(
        referer,
        port=port,
        allow_path=True,
    )


class LocalApprovalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_local_approval_dashboard(
    service: LocalApprovalService,
    *,
    port: int,
    candidate_limit: int | None,
    business_timezone: str,
) -> None:  # pragma: no cover - interactive local server
    controller = LocalApprovalController(
        service,
        display_timezone=ZoneInfo(business_timezone),
        candidate_limit=candidate_limit,
    )
    handler = _handler_class(controller, port=port)
    server = LocalApprovalHTTPServer((LOOPBACK_HOST, port), handler)
    print(f"Local approval dashboard: http://{LOOPBACK_HOST}:{port}")
    print("No Facebook actions can be taken. Press Ctrl-C to stop.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _handler_class(
    controller: LocalApprovalController,
    *,
    port: int,
) -> type[BaseHTTPRequestHandler]:
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "JJMillerLocalApproval/1"
        sys_version = ""

        def do_GET(self) -> None:
            if not self._valid_host():
                self._send_text(HTTPStatus.MISDIRECTED_REQUEST, "Invalid local Host header")
                return
            parts = urlsplit(self.path)
            if parts.path == "/health":
                self._send_text(HTTPStatus.OK, "ok")
                return
            if parts.path != "/":
                self._send_text(HTTPStatus.NOT_FOUND, "Not found")
                return
            query = parse_qs(parts.query, max_num_fields=3)
            result = query.get("result", [""])[0]
            messages = {
                "approved": "Draft approved locally. No Facebook action was taken.",
                "edited": "Edited response approved locally. No Facebook action was taken.",
                "rejected": "Lead rejected. No Facebook action was taken.",
            }
            self._send_html(HTTPStatus.OK, controller.render(message=messages.get(result)))

        def do_POST(self) -> None:
            if not self._valid_host():
                self._send_text(HTTPStatus.MISDIRECTED_REQUEST, "Invalid local Host header")
                return
            origin = self.headers.get("Origin")
            if not _valid_local_origin(
                origin,
                referer=self.headers.get("Referer"),
                fetch_site=self.headers.get("Sec-Fetch-Site"),
                port=port,
            ):
                self._send_text(HTTPStatus.FORBIDDEN, "Invalid local Origin header")
                return
            match = LOCAL_REVIEW_PATH.fullmatch(urlsplit(self.path).path)
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
                result = controller.submit(int(match.group(1)), match.group(2), form)
            except CSRFFailure as error:
                self._send_text(HTTPStatus.FORBIDDEN, str(error))
                return
            except ApprovalExpiredError as error:
                self._send_html(HTTPStatus.GONE, controller.render(message=str(error)))
                return
            except ApprovalError as error:
                self._send_html(HTTPStatus.CONFLICT, controller.render(message=str(error)))
                return
            location = "/?" + urlencode({"result": result.result})
            self.send_response(HTTPStatus.SEE_OTHER)
            self._security_headers()
            self.send_header("Location", location)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def _valid_host(self) -> bool:
            return self.headers.get("Host", "") in allowed_hosts

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

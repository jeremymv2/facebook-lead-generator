# JJ Miller & Co. Facebook Lead Agent

[![Coverage gate](https://github.com/jeremymv2/facebook-lead-generator/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/jeremymv2/facebook-lead-generator/actions/workflows/test.yml)
[![Lint](https://github.com/jeremymv2/facebook-lead-generator/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/jeremymv2/facebook-lead-generator/actions/workflows/lint.yml)
[![Secret scan](https://github.com/jeremymv2/facebook-lead-generator/actions/workflows/secret-scan.yml/badge.svg?branch=main)](https://github.com/jeremymv2/facebook-lead-generator/actions/workflows/secret-scan.yml)

A local-first, AI-assisted lead workflow for identifying high-value contracting requests in an
explicitly approved set of Louisville-area Facebook groups. The intended product flow is:

> Detect → classify → score → draft → notify → human approve/edit/reject → validate → post once

This repository is currently at **Phase 11: review feedback and recovery hardening**. It includes safe
configuration, a dedicated persistent Playwright profile, explicit group allowlisting, visible-post
extraction, alias-based SQLite duplicate prevention, durable per-group scan health, synthetic
selector fixtures, swappable structured lead providers, candidate-only response drafting,
an expiring loopback-only approve/edit/reject dashboard, a separately isolated tokenized mobile
review origin, provider-independent SMS delivery with a Telnyx adapter, structured logging, tests,
and CI. A macOS user launch agent can run locked scan/classify/notify cycles with pause controls,
content-free health, bounded scheduling, retention, duplicate-review suppression, and group-yield
reporting. Configurable overnight quiet hours prevent scheduled work when reviews are unlikely to
receive a prompt response. It includes a manually invoked, triple-gated Facebook commenting path, but it does
**not** include inbound SMS commands or autonomous posting.

## Safety status

The project fails closed:

- `POSTING_ENABLED=false` by default.
- `DRY_RUN=true` by default.
- Both controls are checked independently by `Settings.require_posting_allowed()`.
- Read-only Facebook code remains isolated from the separate posting adapter. A dry-run posting
  validation locates but never clicks or fills the comment composer.
- Facebook submission requires `POSTING_ENABLED=true`, `DRY_RUN=false`, and the explicit
  `post-approved --submit` command. Every layer rechecks the configuration before submission.
- Read-only commands refuse to start unless `POSTING_ENABLED=false` and `DRY_RUN=true`.
- AI access defaults to `AI_PROVIDER=disabled`; classification requires an explicit offline or
  Gemini provider choice.
- Classification and drafting operate only on SQLite records and have no Facebook browser access.
- Low-score, spam, competitor, sales, advice-only, and unrelated posts never receive drafts.
- The local approval dashboard binds only to `127.0.0.1`, validates CSRF and local Host/Origin
  headers, and remains incapable of Facebook submission.
- Approval decisions are atomic, expire quickly, and can transition only once.
- Rejections require a structured reason, making classifier quality measurable without placing
  source post text in operations history.
- Historical classifier replay is read-only. Reclassification is limited transactionally to
  candidate/ignored leads that have never entered review.
- The tunneled mobile surface has no lead-listing route. Each link uses a 256-bit random token,
  token-bound CSRF protection, strict Host/Origin validation, and no application request logging.
- SMS is not attempted until the configured HTTPS relay successfully reaches the Mac health check.
- Telnyx is isolated behind an `SmsProvider` protocol and is disabled unless every required local
  setting is explicitly enabled.
- Only groups explicitly marked `enabled: true` in the local allowlist can be scanned.
- Login pages, CAPTCHA, checkpoints, off-domain redirects, missing posts, and unreadable UI stop the
  scan and produce at most one local diagnostic screenshot.
- Posting additionally requires a fresh immutable approval, an exact post/group identifier,
  substantially unchanged source text, one recognizable composer, no visible duplicate response,
  available daily limits, and a durable one-live-attempt reservation.
- Once browser submission begins, an uncertain result is never retried automatically; the lead is
  moved to `needs_attention` for manual inspection.
- Browser profiles, cookies, databases, screenshots, `.env`, and local group configuration are
  excluded from Git.
- SQLite backups use the online backup API, private filesystem permissions, integrity checks, and
  disposable restore drills; they are never committed.
- The browser never enters Facebook credentials, handles MFA, solves CAPTCHA, bypasses checkpoints,
  or attempts to defeat platform security.

Browser automation may violate Facebook rules or place the account at risk. Account preservation
and human review take priority over lead volume.

## Architecture

The initial local runtime is intentionally small:

```text
Mac
├── validated configuration and safety interlocks
├── SQLite posts, leads, workflow state, and audit events
├── vendor-independent structured classifier/drafter
│   ├── deterministic offline heuristic provider
│   └── optional Gemini structured-output adapter
├── SQLite-backed approval state
├── loopback-only local review dashboard
├── tokenized loopback origin for one tunneled mobile review at a time
├── provider-independent SMS service
│   └── Telnyx adapter (disabled by default)
├── SQLite-backed posting-attempt ledger and daily limits
├── exact-post validation and separately gated Playwright comment adapter
├── JSON structured application logs
├── screenshots directory (gitignored)
└── dedicated Playwright browser profile outside the repository (cookies; never committed)

External, minimized
├── HTTPS relay transports requests to 127.0.0.1; it stores no application approval state
└── Telnyx delivers the SMS containing the expiring review link
```

The Playwright adapter produces plain `FacebookPost` records and passes them to a browser-independent
scan service and persistence layer. Classification is a separate bounded command behind an
`AIProvider` protocol. Pure helpers, synthetic fixtures, fake readers, and fake AI transports keep
the automated test suite independent of live Facebook and external model APIs.

## Requirements

- macOS (the intended production host) or Linux for development
- Python 3.12+
- Git
- A Facebook account that you log into manually

## Setup

On macOS with Python 3.12 installed:

```bash
./scripts/bootstrap_macos.sh
source .venv/bin/activate
cp .env.example .env
cp config/groups.example.yaml config/groups.yaml
```

Or perform the same setup manually:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/playwright install chromium
.venv/bin/pre-commit install --hook-type pre-commit --hook-type pre-push
cp .env.example .env
.venv/bin/lead-agent init-db
```

The database defaults to `data/lead_agent.sqlite3`. Runtime directories are created by `init-db`.

## Configuration

Settings are defined in `src/lead_agent/config.py`, loaded from environment variables and `.env`,
and validated by Pydantic. See `.env.example` for all current options.

Important settings include:

| Setting | Safe default | Purpose |
| --- | --- | --- |
| `POSTING_ENABLED` | `false` | Global posting kill switch |
| `DRY_RUN` | `true` | Prevents submission while exercising future workflows |
| `SCAN_INTERVAL_SECONDS` | `900` | Conservative interval between unattended cycles |
| `LEAD_THRESHOLD` | `75` | Minimum score for an approval candidate |
| `SERVICE_AREA` | `Louisville, Kentucky` | Primary geographic target |
| `APPROVAL_EXPIRATION_MINUTES` | `20` | Local review lifetime |
| `CANDIDATE_DUPLICATE_WINDOW_HOURS` | `72` | Suppress nearby exact-text reviews across groups |
| `APPROVAL_LOCAL_PORT` | `8765` | Loopback-only local review dashboard port |
| `REMOTE_APPROVAL_PORT` | `8766` | Separate loopback origin intended for a secure relay |
| `POSTING_APPROVAL_MAX_AGE_MINUTES` | `20` | Maximum terminal-approval age at posting time |
| `REMOTE_APPROVAL_BASE_URL` | empty | Stable HTTPS relay origin used in SMS links |
| `NOTIFICATIONS_ENABLED` | `false` | Explicit remote-notification switch |
| `SMS_PROVIDER` | `disabled` | `disabled` or `telnyx` |
| `SMS_RECIPIENT_NUMBER` | empty | Reviewer's phone in E.164 format |
| `TELNYX_FROM_NUMBER` | empty | Registered Telnyx sender in E.164 format |
| `TELNYX_API_KEY` | empty | Local Telnyx secret; never printed or stored in SQLite |
| `APPROVAL_SIGNING_KEY` | empty | Local secret for token-bound form protection |
| `DAILY_POSTING_LIMIT` | `5` | Global live-attempt cap per local calendar day |
| `PER_GROUP_DAILY_POSTING_LIMIT` | `2` | Per-group live-attempt cap per local calendar day |
| `BUSINESS_TIMEZONE` | `America/New_York` | Local calendar used by posting limits |
| `OPERATIONS_LOG_RETENTION_DAYS` | `14` | Rotated local operations-log retention |
| `OPERATIONS_LOG_MAX_BYTES` | `5000000` | Rotate an operations log after this size |
| `DATABASE_BACKUP_RETENTION_DAYS` | `14` | Private verified-backup retention |
| `DATABASE_BACKUP_INTERVAL_HOURS` | `24` | Minimum interval between automatic backups |
| `DATABASE_BACKUP_DIR` | `data/backups` | Gitignored directory for private SQLite backups |
| `OPERATIONS_QUIET_HOURS_ENABLED` | `true` | Skip scheduled work during the configured local window |
| `OPERATIONS_QUIET_HOURS_START` | `22:00` | Inclusive local quiet-hours start |
| `OPERATIONS_QUIET_HOURS_END` | `05:00` | Exclusive local quiet-hours end |
| `OPERATIONS_DEGRADED_CYCLE_LIMIT` | `2` | Consecutive materially incomplete cycles before automatic pause |
| `OPERATIONS_INCOMPLETE_GROUP_RATE_THRESHOLD` | `0.25` | Failed-plus-partial group rate that advances the circuit breaker |
| `CYCLE_CLASSIFICATION_LIMIT` | `100` | Maximum saved posts classified per cycle |
| `FACEBOOK_PROFILE_PATH` | `~/.jjmiller-lead-agent/facebook-profile` | Dedicated persistent profile |
| `BROWSER_HEADLESS` | `false` | Keeps manual login and scan behavior visible |
| `FACEBOOK_GROUP_MAX_RETRIES` | `1` | Bounded transient retries per group and cycle |
| `FACEBOOK_GROUP_RETRY_BACKOFF_SECONDS` | `5` | Delay before the one transient retry |
| `FACEBOOK_GROUP_DELAY_SECONDS` | `2` | Delay between unattended group navigations |
| `FACEBOOK_MAX_SCROLLS` | `12` | Maximum bounded lazy-load scrolls per group |
| `FACEBOOK_SCROLL_SETTLE_SECONDS` | `0.75` | Wait after each bounded scroll |
| `MAX_POSTS_PER_GROUP` | `10` | Conservative visible-post target per run |
| `MIN_POST_TEXT_LENGTH` | `15` | Ignores very short UI fragments |
| `AI_PROVIDER` | `disabled` | `disabled`, offline `heuristic`, or opt-in `gemini` |
| `AI_MODEL` | `gemini-2.5-flash` | Model used only by the Gemini provider |
| `GEMINI_API_KEY` | empty | Local secret required only for Gemini; never printed |
| `AI_MAX_POSTS_PER_RUN` | `20` | Bounded classification batch size |
| `AI_MAX_INPUT_CHARACTERS` | `5000` | Maximum post text sent per model request |
| `AI_REQUEST_TIMEOUT_SECONDS` | `30` | Per-request Gemini timeout |

The browser profile validator rejects paths inside the repository because a persistent profile
contains authentication cookies and other sensitive session data.

Check effective non-secret settings and safety state:

```bash
lead-agent doctor
```

This output intentionally excludes secrets and tokens.

## Database and duplicate prevention

SQLite currently stores:

- Discovered Facebook posts and durable processing state.
- Leads, classification fields, response fields, approval/posting timestamps, errors, and retries.
- Append-only audit events with structured JSON details.
- Per-group last attempt, last success, last-known post, safe error type, and scan counts.
- A schema version for future migrations.

Post identity prefers a Facebook post ID, then a canonical post URL, then a deterministic hash of
group, author, and normalized text. Database constraints and identity aliases make repeated scans
idempotent. A second uniqueness constraint permits only one lead per post. Schema version 4 adds
intent, residential/spam flags, provider/model metadata, and a classification-contract version.
Schema version 5 adds immutable approval draft snapshots, expiration, and one terminal local
decision per request. Schema version 6 adds hashed remote tokens and durable provider-delivery
metadata without storing SMS bodies, destination numbers, or plaintext review tokens. Schema
version 7 adds immutable posting inputs, dry-run outcomes, the no-retry submission boundary,
evidence paths, safe error codes, and a partial uniqueness constraint that permits only one live
attempt per lead. Schema version 8 adds bounded historical operations reporting. Schema version 9
adds structured rejection reasons and safely backfills earlier rejected reviews as `other`.

## Development commands

Run every local check:

```bash
./scripts/run_checks.sh
```

Or run checks individually:

```bash
ruff format --check .
ruff check .
mypy scripts src tests
pytest --cov=lead_agent --cov=scripts --cov-report=term-missing
pre-commit run --all-files --hook-stage pre-commit
pre-commit run --all-files --hook-stage pre-push
```

Pull requests expose separate required-ready status checks for lint/type safety, unit-test coverage,
and independent full-history secret scanning. Coverage fails below 90% and publishes its full
terminal report in the GitHub Actions job summary. Live Facebook tests must never run in GitHub
Actions.

## Secrets and local data

Never commit:

- Facebook credentials, cookies, browser storage, or profile contents
- AI, relay, notification, or approval signing keys
- `.env`
- SQLite runtime databases
- Screenshots or logs
- Private group configuration when it contains sensitive details

Use placeholders in `.env.example`. A later production hardening milestone can move long-lived
secrets to macOS Keychain.

### Credential leak prevention

This repository uses layered controls because no single scanner catches every leak:

- The pre-commit hook runs Gitleaks against staged content before a commit is created.
- The pre-push hook runs Gitleaks with redaction against complete local Git history.
- Private-key and large-file hooks catch common credential artifacts and accidental exports.
- A repository-specific path guard rejects `.env`, cookies, browser profiles, authentication state,
  screenshots, databases, logs, key files, and other private runtime artifacts even if force-added.
- GitHub Actions performs an independent full-history scan on pushes, pull requests, a weekly
  schedule, and manual requests. It does not upload a Gitleaks report artifact.
- Third-party GitHub Actions are pinned to immutable commit SHAs.

Install both versioned hooks after cloning:

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
```

Local hooks can be bypassed, so repository administrators should also enable GitHub Secret
Protection and push protection when the account plan permits it. Protect `main`, disallow direct
and force pushes, and require both `checks` and `Gitleaks full history` before merging.

If a scanner reports a real credential, do not merely delete the file: revoke or rotate the
credential first. Follow [SECURITY.md](SECURITY.md) for containment and Git-history cleanup.

## Test the read-only Facebook scanner

Keep your normal browser out of this workflow. Playwright uses the separate directory in
`FACEBOOK_PROFILE_PATH`; only one process can use that directory at a time.

1. Confirm `.env` retains the safety settings:

   ```dotenv
   POSTING_ENABLED=false
   DRY_RUN=true
   BROWSER_HEADLESS=false
   ```

2. Copy and edit the private group allowlist. Use the real Facebook group URL and set only the group
   you approve to `enabled: true`:

   ```bash
   cp config/groups.example.yaml config/groups.yaml
   ```

3. Check the non-secret configuration, then open the dedicated profile and log in yourself. Enter
   credentials and MFA only in the visible browser window; return to the terminal and press Enter
   after Facebook has fully loaded:

   ```bash
   lead-agent doctor
   lead-agent facebook-login
   ```

4. Scan one allowlisted group conservatively:

   ```bash
   lead-agent scan-facebook --group-id louisville-homeowners-example --max-posts 10
   ```

   Replace the example ID with the `id` in your `config/groups.yaml`. The command prints the counts
   and text only for newly stored posts.

5. Run the same command again. A successful deduplication check reports `new=0` for unchanged posts
   and does not add duplicate rows.

If the browser closes before you can inspect what Facebook rendered, pause it explicitly:

```bash
lead-agent scan-facebook --group-id louisville-homeowners-example --max-posts 10 --pause-after-scan
```

The browser remains open until you return to the terminal and press Enter. The pause does not enable
clicking, typing, commenting, or any other automated Facebook write action.

The scanner waits through Facebook's initial placeholder rendering and requires at least one
text-bearing post. If containers appear but readable post text never loads, the run stops safely and
captures a local screenshot instead of recording a misleading successful zero-post scan.
If Facebook yields fewer readable posts than the configured target, the group is recorded as
`partial`, not fully successful. The dashboard reports full, partial, and failed group coverage
separately. Two consecutive cycles with at least 25% failed-or-partial groups automatically pause
future unattended cycles; two consecutive fatal cycle errors do the same. An operator must review
the content-free health details and explicitly resume the worker.
Top-level post text is isolated from nested comment articles so a long reply is never paired with
the parent post's permalink. Facebook comment and reply articles are also rejected by their
semantic `aria-label`, including when Facebook exposes them outside the parent article subtree.
Current Facebook group feeds are read from semantic `story_message` nodes and paired with the
nearest owning group-post permalink; role-based article extraction remains a guarded fallback.
When Facebook exposes no post permalink, the scanner leaves the URL empty and uses the normalized
top-level text hash for deduplication; it never substitutes a comment, photo, or unrelated URL. If
Facebook exposes the permalink or post ID on a later scan, the database attaches those identifiers
to the original content discovery instead of inserting another row. Distinct Facebook post IDs are
never merged solely because their text matches.
`--max-posts` is a target and hard cap: the scanner accumulates unique posts across bounded
sub-viewport scrolls until it reaches the target, its scroll limit, or its load timeout. It rechecks
login, CAPTCHA, checkpoint, redirect, and page state after every scroll.

To scan every enabled group sequentially, omit `--group-id`. For a single content-free operational
test that scans all enabled groups, classifies locally, performs retention, and cannot send SMS,
use:

```bash
AI_PROVIDER=heuristic lead-agent run-cycle --max-posts 10 --skip-notifications
```

Unlike `scan-facebook`, `run-cycle` prints only aggregate counts and never prints discovered post
text. Unattended cycles wait between groups and retry a typed transient navigation/feed failure at
most once after a bounded backoff. Aggregate health reports retry and recovery counts. An exhausted
transient failure is recorded and the remaining groups continue; a Facebook login requirement,
checkpoint, CAPTCHA, unexpected domain, or other safety stop is never retried and still aborts the
whole cycle. Content-free logs identify only the group alias, attempt, stage, and safe error code.

If the scan stops safely, do not automate around the challenge. Review the terminal reason and the
single PNG in `screenshots/` if one was captured, then resolve login, MFA, CAPTCHA, or checkpoint
manually. Screenshots are local, ignored by Git, and cleaned up according to
`SCREENSHOT_RETENTION_DAYS`.

Inspect persisted group health without opening Facebook or printing post text:

```bash
lead-agent scan-status
lead-agent scan-status --group-id louisville-homeowners-example
```

Inspect group quality without exposing post text or URLs:

```bash
lead-agent group-report
```

The report shows discoveries, classifications, candidate yield, provider advertisements, and exact
text duplicates. Priority remains a manual allowlist decision; the report never edits
`config/groups.yaml`.

The JSON output reports the last attempt and success, latest safe error type, post counts, and
consecutive failure count. A successful recovery resets the failure streak while retaining the
historical last-failure timestamp. `lead-agent init-db`, scanning, and `scan-status` all apply the
idempotent schema upgrade from earlier databases automatically.

Synthetic candidate fixtures under `tests/fixtures/` cover each supported semantic message selector,
comment and nested-reply rejection, cross-group permalink rejection, short placeholders, and
no-permalink rendering. They contain no captured Facebook content or account data.

## Test lead classification and drafting

Classification does not run automatically after a Facebook scan. Start with the deterministic
offline provider, which makes no network requests and exists for smoke tests and regression
fixtures—not as a substitute for human judgment:

```dotenv
AI_PROVIDER=heuristic
```

Use a copy of the database while testing schema migration and classification:

```bash
cp -p data/lead_agent.sqlite3 data/lead_agent.phase4-test.sqlite3
DATABASE_PATH=data/lead_agent.phase4-test.sqlite3 lead-agent init-db
DATABASE_PATH=data/lead_agent.phase4-test.sqlite3 lead-agent classify-posts --limit 10
```

Only posts without an existing lead row are processed. Run the same command again; it should report
`considered=0 classified=0`. Strong candidates print a score, service, and locally stored `DRAFT`.
Ignored posts are counted but their source text is not printed. Classification cannot approve or
submit a draft to Facebook.

Drafts are intentionally brief and direct. Every locally validated draft identifies JJ Miller &
Co., states that estimates are free, links to `https://jjmillerco.com`, and asks the customer to text
`502-528-0858`. Generic greetings, filler, and requests to message through Facebook are rejected.
Posts that say the author already found or hired someone, is all set, or is no longer looking are
classified as resolved and never receive drafts.

To classify one saved post during focused testing:

```bash
DATABASE_PATH=data/lead_agent.phase4-test.sqlite3 lead-agent classify-posts --post-id 123
```

The optional Gemini provider uses Google's current structured-output interface and validates the
returned JSON again inside the application. Put the key only in the ignored `.env` file:

```dotenv
AI_PROVIDER=gemini
AI_MODEL=gemini-2.5-flash
GEMINI_API_KEY=REPLACE_ME
```

Enabling Gemini sends bounded post text and non-secret service settings to Google. It does not send
the Facebook author name, group ID, group name, post URL, browser cookies, or database contents.
Review Google's
[structured-output documentation](https://ai.google.dev/gemini-api/docs/structured-output) and
[current pricing/data-use terms](https://ai.google.dev/gemini-api/docs/pricing) before processing
real customer content. Keep `AI_PROVIDER=disabled` if external processing is not acceptable.

### Replay and safely reclassify saved leads

Preview how current classifier rules would treat historical leads without drafting responses or
changing SQLite:

```bash
lead-agent classification-replay --limit 100 --changed-only
lead-agent classification-replay --lead-id 168
```

Bulk reclassification selects only stale `candidate` or `ignored` leads that have never had an
approval request. A targeted ID may be on the current classifier version, but the same review and
status protections still apply. The database enforces these restrictions inside the update
transaction:

```bash
lead-agent reclassify-leads --limit 100
lead-agent reclassify-leads --lead-id 168
```

Neither command opens Facebook, sends SMS, approves a response, or posts a comment. Start with
`classification-replay`; use `reclassify-leads` only after reviewing the reported differences.

## Test local human approval

Use the same copied database after it contains classified candidate leads:

```bash
DATABASE_PATH=data/lead_agent.phase4-test.sqlite3 lead-agent approval-dashboard
```

Open `http://127.0.0.1:8765` on the same Mac. The page displays the source post, score, proposed
response, and remaining review time. Each candidate can be approved exactly as drafted, edited and
approved, or rejected. Edited responses must still satisfy the local company identity, free
estimate, full website URL, text-number, and length rules.

Every rejection requires one structured reason such as provider advertisement, employment
recruiting, wrong geography, irrelevant service, resolved, or duplicate/repost. The dashboard shows
the reviewed, accepted, and rejected totals, acceptance rate, and rejection-reason counts. These
metrics contain no source post text.

To turn eligible rejection feedback into a private, review-before-commit classifier fixture file:

```bash
lead-agent export-regression-fixtures --limit 100
```

The exporter redacts names, businesses, email addresses, phone numbers, street addresses, and URLs,
writes with private permissions under `data/`, and marks every fixture for manual review. Duplicate
and `other` reasons are intentionally skipped because they do not imply a classifier expectation.

The same page includes a content-free historical operations dashboard. It summarizes the newest 48
completed cycles, charts the latest 12 for group reliability and posts seen, lists recent cycle
counts, and shows the current failure streak and last success for every configured group. Existing
`cycle.run` audit events appear immediately; future events also retain retry, recovery, duplicate,
classification, candidate, and notification counters. Times use `BUSINESS_TIMEZONE`. No post text,
draft response, Facebook URL, phone number, or credential is copied into trend records.

Starting review changes a candidate to `pending_approval` and snapshots the draft so later data
changes cannot silently alter what was reviewed. The default review window is 20 minutes. A terminal
decision cannot be changed or replayed; expired requests require a new review workflow. Every state
change is written to the audit trail without logging source text or approved response content.

The dashboard is intentionally local-only: it binds to `127.0.0.1`, has no configurable public host,
uses an in-memory CSRF token, validates local Host/Origin headers, and imports no Facebook browser
submission capability. An approval only updates SQLite; the separate posting command must still
pass every posting interlock and validation.

## Test tunneled mobile approval and Telnyx

The remote design remains local-first. SQLite, drafts, decisions, token validation, and the mobile
HTML are all served by the Mac. A relay carries HTTPS requests to the dedicated loopback port, and
Telnyx delivers the SMS. The application has no Cloud Run service or cloud database.

[Smee](https://smee.io/) is useful for one-way webhook POST delivery, but it cannot proxy an
interactive browser request and return the local approval page. Use a browser-capable relay such as
a named [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/),
or a private [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve) URL when the Mac
and phone are in the same tailnet. Do not use a temporary URL that changes after restart.

Configure the selected relay to forward its stable HTTPS origin to
`http://127.0.0.1:8766`. Keep relay credentials outside this repository. Then add these values only
to the ignored `.env` file:

```dotenv
POSTING_ENABLED=false
DRY_RUN=true

NOTIFICATIONS_ENABLED=true
SMS_PROVIDER=telnyx
REMOTE_APPROVAL_BASE_URL=https://YOUR-STABLE-RELAY-HOST
APPROVAL_SIGNING_KEY=
SMS_RECIPIENT_NUMBER=+1XXXXXXXXXX
TELNYX_API_KEY=
TELNYX_FROM_NUMBER=+1XXXXXXXXXX
```

Generate the signing key locally without putting it in terminal history as a literal:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

Paste that output after `APPROVAL_SIGNING_KEY=` in `.env`, and paste the key created in the Telnyx
portal after `TELNYX_API_KEY=`. Never put either value in `.env.example` or this README.

The Telnyx sender and recipient must use E.164 notation. The recipient is the reviewer's phone; it
does not have to be the business's published customer text number. Complete the applicable Telnyx
sender registration and assign the number to a messaging profile before testing.

### Telnyx consent and 10DLC compliance

Register this internal notification program before sending through a US 10-digit long-code number.
Use the `LOW_VOLUME` campaign with the `ACCOUNT_NOTIFICATION` sub-use case; these are transactional
lead-review alerts to an authorized company administrator, not marketing or customer-care messages.
The exact pre-submission checklist and campaign copy live in
[`docs/telnyx-low-volume-campaign.md`](docs/telnyx-low-volume-campaign.md).

Every application-generated alert identifies `JJ Miller & Co LLC`, includes `Reply STOP to opt
out.`, uses only ASCII, and fits in one 160-character segment. The source post and draft never leave
the Mac in the SMS. Obtain the recipient's express consent before enabling notifications and retain
that consent outside the repository.

Telnyx handles its recognized STOP/START keywords and the carrier block list at the messaging
profile. Do not add a second local opt-out implementation that could disagree with the provider.
Configure the campaign's HELP reply through Telnyx Keyword Management before the first production
message. A custom inbound webhook is unnecessary unless the application later adds inbound SMS
features; if one is added, verify Telnyx's signed webhook before processing its payload.

Start the local service, then start or verify the relay in another terminal:

```bash
lead-agent doctor
lead-agent remote-approval
```

The service binds only to `127.0.0.1`. It checks the external `/health` route before creating an
approval request or spending an SMS. Once the relay is healthy, it prepares new candidates and
sends a branded, single-segment alert containing a random review link and STOP instruction. It never
puts the Facebook post text or proposed response in the SMS; those remain on the Mac and are
retrieved through the tokenized page. A second notification cycle does not resend the same approval.

Provider failures are recorded without message contents, tokens, API keys, or phone numbers. They
are not retried continuously. After fixing the provider configuration, permit one explicit retry
with a newly rotated review token:

```bash
lead-agent remote-approval --retry-failed
```

The tokenized service deliberately has no `/` dashboard or lead-list endpoint. A token expires with
the 20-minute approval window and can produce only one approve, edit, or reject transition. Approval
never directly posts anything to Facebook.

## Test approved posting safely

Keep the safe defaults in `.env`:

```dotenv
POSTING_ENABLED=false
DRY_RUN=true
```

Within 20 minutes after approving or editing a lead, run the validation-only command with its lead
ID:

```bash
lead-agent post-approved --lead-id 123
```

The command opens the exact saved permalink and requires all of the following before it succeeds:

- the group is still enabled in `config/groups.yaml`;
- the immutable approval snapshot is still fresh;
- the Facebook post ID and group match the saved target;
- the visible source text is unchanged except for tightly bounded rendering drift;
- no newly added resolution language says the customer already found help;
- the exact approved response is not already visible in the comments; and
- exactly one recognizable comment composer is present.

On success it prints `DRY RUN`, captures a local pre-posting screenshot, and records a schema-v7
dry-run attempt. It does not click, focus, fill, or submit the composer. Dry runs can be repeated and
do not change the approved lead status or consume a live posting slot.

Live submission is intentionally harder and should be used only after reviewing a successful dry
run. It requires all three explicit controls:

```dotenv
POSTING_ENABLED=true
DRY_RUN=false
```

```bash
lead-agent post-approved --lead-id 123 --submit
```

SQLite reserves the lead before the browser opens, enforces global and per-group daily limits, and
allows only one live attempt for that lead. Immediately before Playwright presses Enter, the
service durably records the no-retry submission boundary. It then succeeds only if the exact
approved response becomes visible in a Facebook comment article. Any uncertainty after that
boundary becomes `needs_attention` and is never retried automatically. Restore the safe defaults
after any controlled live test.

## Local unattended operation on macOS

The cycle launch agent invokes one process at `SCAN_INTERVAL_SECONDS`; launchd does not overlap an
already-running interval, and the application also takes a non-blocking file lock. The worker never
posts to Facebook. Keep `POSTING_ENABLED=false` and `DRY_RUN=true`.

By default, invocations from 10:00 PM through 4:59 AM in `BUSINESS_TIMEZONE` exit before loading the
group allowlist, opening SQLite or Facebook, classifying posts, creating backups, or sending SMS.
At exactly 5:00 AM, cycles are eligible again. launchd remains loaded during this window; the gate
is enforced inside `run-cycle`, so reinstalling the launch agent is unnecessary. The next normal
15-minute launchd interval performs the first eligible cycle.

`lead-agent doctor` and `lead-agent operations-status` show the configured window and whether it is
currently active. Expected inactivity during quiet hours is not reported as stale health. For an
intentional attended run during the window, use the explicit override:

```bash
lead-agent run-cycle --ignore-quiet-hours --skip-notifications
```

The override affects only that invocation and does not change the saved schedule. To disable the
gate persistently, set `OPERATIONS_QUIET_HOURS_ENABLED=false` in the ignored `.env` file.

For local-only classification, set this in the ignored `.env`:

```dotenv
AI_PROVIDER=heuristic
AI_MODEL=heuristic-v1
```

Exercise the exact unattended command manually before installing anything:

```bash
lead-agent operations-pause
lead-agent operations-status
lead-agent operations-resume
lead-agent run-cycle --max-posts 10 --skip-notifications
lead-agent group-report
```

Install and load only the cycle agent while Telnyx is pending:

```bash
.venv/bin/python scripts/manage_launchd.py install
.venv/bin/python scripts/manage_launchd.py status
```

The generated plist contains only executable paths and non-secret scheduling metadata. It reads the
ignored `.env` from the repository at runtime; credentials and phone numbers are never copied into
the plist. Logs live under `data/logs/`, and content-free health lives at
`data/operations/health.json`.

After the Telnyx campaign is active and `lead-agent doctor` reports
`remote_approval_ready: true`, install both agents:

```bash
.venv/bin/python scripts/manage_launchd.py install --include-remote-approval
```

Remove both JJ Miller & Co. launch agents with:

```bash
.venv/bin/python scripts/manage_launchd.py uninstall
```

The Mac must remain powered on, awake, connected to the internet, signed into its user session, and
logged into Facebook in the dedicated profile. Keep `BROWSER_HEADLESS=false` until the visible
scheduled cycle has been proven reliable; never automate around a Facebook checkpoint or MFA flow.

Pause creates a private local marker and prevents future cycles from opening Facebook. It does not
stop the separate approval server, so an already-sent review link can still be decided:

```bash
lead-agent operations-pause
lead-agent operations-resume
```

`lead-agent operations-status` reports `success`, `degraded`, `failed`, `paused`, or `never_run`,
plus aggregate counts and a stale-health flag. Failure state stores only the exception class name.
Expired PNG/JPEG/WebP screenshots and rotated `.log`/`.jsonl` files are removed within their
configured directories; databases, group configuration, browser profiles, and credentials are
never retention targets.

Each successful cycle also creates at most one private SQLite backup per configured interval,
verifies its integrity and schema by restoring it into a disposable database, and removes only
expired files matching the application's backup naming convention. Create and test a backup
manually without opening Facebook or sending SMS:

```bash
lead-agent database-backup
lead-agent database-restore-test
```

`database-restore-test --backup-path PATH` accepts only a named backup inside
`DATABASE_BACKUP_DIR`. It never replaces the production database. Backups are useful only if they
can be restored, so investigate any backup or restore-test failure before resuming unattended runs.

## Runtime and troubleshooting

For current configuration errors, run `lead-agent doctor`. For database initialization errors,
confirm the configured parent directory is writable, then rerun `lead-agent init-db`; schema setup
is idempotent.

## Roadmap

1. **Repository bootstrap:** config, models, SQLite, logging, tests, CI.
2. **Read-only Playwright proof of concept:** manually log in, scan configured
   groups, extract visible posts and URLs, save only new posts, and never comment.
3. **Selector fixtures and scanner hardening:** regression-test sanitized DOM
   candidates, deduplicate changing Facebook identities, and persist recoverable group health.
4. **Swappable AI classification/scoring and drafting:** validate structured
   classifications, draft only strong candidates, and remain incapable of Facebook submission.
5. **Loopback-only local human approval:** snapshot drafts and support expiring,
   one-time approve/edit/reject decisions with no Facebook posting capability.
6. **Tunneled remote/mobile approval:** keep state on the Mac, send Telnyx alerts,
   and expose only per-request cryptographic review URLs through an outbound HTTPS relay.
7. **Idempotent approved posting:** final validation, limits, screenshots,
   one live attempt, and stop-on-uncertainty behavior.
8. **Local unattended operations:** locked scan/classify/notify cycles, macOS
   launch agents, pause controls, content-free health, retention, duplicate review suppression,
   and group-quality reporting.
9. **Reliability and coverage hardening:** conservative scheduling, bounded
   transient group retries, safe retry diagnostics, security-boundary tests, and an enforced 90%
   branch-coverage floor.
10. **Historical operations dashboard:** local-only cycle charts, bounded audit
    queries, current group health, mobile-responsive tables, and richer content-free run counters.
11. **Review feedback and recovery hardening (this milestone):** structured rejection reasons,
    review-quality metrics, read-only historical replay, transactionally safe reclassification,
    sanitized regression exports, private automatic backups, and disposable restore drills.

The first browser milestone is successful only when a second read-only run stores no duplicate
posts and the software remains incapable of commenting.

## Source-control workflow

Meaningful work is developed on feature branches, checked locally, pushed, and reviewed through a
private GitHub pull request before merging to `main`. Production should pull reviewed `main`; code
should not be copied manually between chat and the runtime machine.

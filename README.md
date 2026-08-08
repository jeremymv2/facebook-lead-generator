# JJ Miller & Co. Facebook Lead Agent

A local-first, AI-assisted lead workflow for identifying high-value contracting requests in an
explicitly approved set of Louisville-area Facebook groups. The intended product flow is:

> Detect → classify → score → draft → notify → human approve/edit/reject → validate → post once

This repository is currently at **Phase 4: lead classification and drafting**. It includes safe
configuration, a dedicated persistent Playwright profile, explicit group allowlisting, visible-post
extraction, alias-based SQLite duplicate prevention, durable per-group scan health, synthetic
selector fixtures, swappable structured lead providers, candidate-only response drafting,
structured logging, tests, and CI. It does **not** include approvals, notifications, scheduling, or
Facebook comment posting.

## Safety status

The project fails closed:

- `POSTING_ENABLED=false` by default.
- `DRY_RUN=true` by default.
- Both controls are checked independently by `Settings.require_posting_allowed()`.
- The browser adapter exposes navigation and reading only; no comment or other submission
  implementation exists.
- Read-only commands refuse to start unless `POSTING_ENABLED=false` and `DRY_RUN=true`.
- AI access defaults to `AI_PROVIDER=disabled`; classification requires an explicit offline or
  Gemini provider choice.
- Classification and drafting operate only on SQLite records and have no Facebook browser access.
- Low-score, spam, competitor, sales, advice-only, and unrelated posts never receive drafts.
- Only groups explicitly marked `enabled: true` in the local allowlist can be scanned.
- Login pages, CAPTCHA, checkpoints, off-domain redirects, missing posts, and unreadable UI stop the
  scan and produce at most one local diagnostic screenshot.
- Browser profiles, cookies, databases, screenshots, `.env`, and local group configuration are
  excluded from Git.
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
├── JSON structured application logs
├── screenshots directory (gitignored)
└── dedicated Playwright browser profile outside the repository (cookies; never committed)
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
| `SCAN_INTERVAL_SECONDS` | `300` | Target interval between group scans |
| `LEAD_THRESHOLD` | `75` | Minimum score for an approval candidate |
| `SERVICE_AREA` | `Louisville, Kentucky` | Primary geographic target |
| `APPROVAL_EXPIRATION_MINUTES` | `20` | Planned approval lifetime |
| `DAILY_POSTING_LIMIT` | `5` | Planned global daily cap |
| `PER_GROUP_DAILY_POSTING_LIMIT` | `2` | Planned per-group cap |
| `FACEBOOK_PROFILE_PATH` | `~/.jjmiller-lead-agent/facebook-profile` | Dedicated persistent profile |
| `BROWSER_HEADLESS` | `false` | Keeps manual login and scan behavior visible |
| `FACEBOOK_MAX_SCROLLS` | `12` | Maximum bounded lazy-load scrolls per group |
| `FACEBOOK_SCROLL_SETTLE_SECONDS` | `0.75` | Wait after each bounded scroll |
| `MAX_POSTS_PER_GROUP` | `20` | Conservative visible-post cap per run |
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

The eventual posting workflow requires additional one-time approval and posting-attempt records;
those will be implemented with the approval/posting milestones before any submission code exists.

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

Pull requests run formatting, lint, type, unit-test, and independent full-history secret checks.
Live Facebook tests must never run in GitHub Actions.

## Secrets and local data

Never commit:

- Facebook credentials, cookies, browser storage, or profile contents
- AI, cloud, notification, or approval signing keys
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

To scan every enabled group sequentially, omit `--group-id`. The current proof of concept is manual;
it does not yet run every five minutes.

If the scan stops safely, do not automate around the challenge. Review the terminal reason and the
single PNG in `screenshots/` if one was captured, then resolve login, MFA, CAPTCHA, or checkpoint
manually. Screenshots are local, ignored by Git, and cleaned up according to
`SCREENSHOT_RETENTION_DAYS`.

Inspect persisted group health without opening Facebook or printing post text:

```bash
lead-agent scan-status
lead-agent scan-status --group-id louisville-homeowners-example
```

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
Ignored posts are counted but their source text is not printed. No command in this phase can approve
or submit the draft to Facebook.

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

Enabling Gemini sends bounded post text, a sanitized first name, and non-secret service settings to
Google. It does not send the Facebook group ID, group name, post URL, browser cookies, or database
contents. Review Google's
[structured-output documentation](https://ai.google.dev/gemini-api/docs/structured-output) and
[current pricing/data-use terms](https://ai.google.dev/gemini-api/docs/pricing) before processing
real customer content. Keep `AI_PROVIDER=disabled` if external processing is not acceptable.

## Runtime and troubleshooting

The scanner runs manually. A later reliability milestone will supply a `launchd` service with
bounded restart behavior, pause controls, health monitoring, log retention, and screenshot cleanup.

For current configuration errors, run `lead-agent doctor`. For database initialization errors,
confirm the configured parent directory is writable, then rerun `lead-agent init-db`; schema setup
is idempotent.

## Roadmap

1. **Repository bootstrap:** config, models, SQLite, logging, tests, CI.
2. **Read-only Playwright proof of concept:** manually log in, scan configured
   groups, extract visible posts and URLs, save only new posts, and never comment.
3. **Selector fixtures and scanner hardening:** regression-test sanitized DOM
   candidates, deduplicate changing Facebook identities, and persist recoverable group health.
4. **Swappable AI classification/scoring and drafting (this milestone):** validate structured
   classifications, draft only strong candidates, and remain incapable of Facebook submission.
5. Local then remote human approval with expiring one-time tokens.
6. Idempotent approved posting with final validation, limits, screenshots, and stop-on-uncertainty.
7. Notifications, dashboard, `launchd`, retention, and operational health.

The first browser milestone is successful only when a second read-only run stores no duplicate
posts and the software remains incapable of commenting.

## Source-control workflow

Meaningful work is developed on feature branches, checked locally, pushed, and reviewed through a
private GitHub pull request before merging to `main`. Production should pull reviewed `main`; code
should not be copied manually between chat and the runtime machine.

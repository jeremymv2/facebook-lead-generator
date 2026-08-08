# JJ Miller & Co. Facebook Lead Agent

A local-first, AI-assisted lead workflow for identifying high-value contracting requests in an
explicitly approved set of Louisville-area Facebook groups. The intended product flow is:

> Detect → classify → score → draft → notify → human approve/edit/reject → validate → post once

This repository is currently at **Phase 1: repository bootstrap**. It includes safe configuration,
domain models, SQLite persistence, duplicate prevention, structured logging, tests, and CI. It does
**not** include Facebook scraping, browser automation, AI calls, remote approvals, notifications, or
comment posting yet.

## Safety status

The project fails closed:

- `POSTING_ENABLED=false` by default.
- `DRY_RUN=true` by default.
- Both controls are checked independently by `Settings.require_posting_allowed()`.
- No Facebook submission implementation exists in this milestone.
- Browser profiles, cookies, databases, screenshots, `.env`, and local group configuration are
  excluded from Git.
- Future browser automation must stop on login challenges, CAPTCHA, checkpoints, missing controls,
  or unexpected Facebook pages. It must never attempt to bypass platform security.

Browser automation may violate Facebook rules or place the account at risk. Account preservation
and human review take priority over lead volume.

## Architecture

The initial local runtime is intentionally small:

```text
Mac
├── validated configuration and safety interlocks
├── SQLite posts, leads, workflow state, and audit events
├── JSON structured application logs
├── screenshots directory (gitignored)
└── dedicated Facebook browser profile outside the repository (future phase)
```

Core business models have no browser dependency. A later Playwright adapter will produce plain
`FacebookPost` records and pass them to the persistence layer. This keeps most tests independent of
Facebook and makes UI selector changes replaceable.

## Requirements

- macOS (the intended production host) or Linux for development
- Python 3.12+
- Git

Playwright is deliberately not installed in this phase.

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
| `FACEBOOK_PROFILE_PATH` | `~/.jjmiller-lead-agent/facebook-profile` | Future persistent profile |

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
- A schema version for future migrations.

Post identity prefers a Facebook post ID, then a canonical post URL, then a deterministic hash of
group, author, and normalized text. A database uniqueness constraint makes repeated scans
idempotent. A second uniqueness constraint permits only one lead per post.

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

## Facebook login and browser profile

Playwright setup and manual Facebook login arrive in the read-only proof-of-concept milestone. The
profile will live at `FACEBOOK_PROFILE_PATH` outside Git and persist the normal login session. The
software will never solve CAPTCHA, bypass MFA, or defeat account checkpoints; it will pause and
alert Jeremy instead.

## Runtime and troubleshooting

The bootstrap runs manually. A later reliability milestone will supply a `launchd` service with
bounded restart behavior, pause controls, health monitoring, log retention, and screenshot cleanup.

For current configuration errors, run `lead-agent doctor`. For database initialization errors,
confirm the configured parent directory is writable, then rerun `lead-agent init-db`; schema setup
is idempotent.

## Roadmap

1. **Repository bootstrap (this milestone):** config, models, SQLite, logging, tests, CI.
2. **Read-only Playwright proof of concept:** manually log in, scan one configured group, extract
   visible posts and URLs, save only new posts, and never comment.
3. Deduplication and group scan state hardening.
4. Swappable AI classification/scoring and drafting providers.
5. Local then remote human approval with expiring one-time tokens.
6. Idempotent approved posting with final validation, limits, screenshots, and stop-on-uncertainty.
7. Notifications, dashboard, `launchd`, retention, and operational health.

The first browser milestone is successful only when a second read-only run stores no duplicate
posts and the software remains incapable of commenting.

## Source-control workflow

Meaningful work is developed on feature branches, checked locally, pushed, and reviewed through a
private GitHub pull request before merging to `main`. Production should pull reviewed `main`; code
should not be copied manually between chat and the runtime machine.

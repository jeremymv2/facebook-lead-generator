#!/bin/sh
set -eu

.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/mypy scripts src tests
.venv/bin/pytest --cov=lead_agent --cov=scripts --cov-report=term-missing
.venv/bin/pre-commit run --all-files --hook-stage pre-commit
.venv/bin/pre-commit run --all-files --hook-stage pre-push

#!/bin/sh
set -eu

.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/mypy src tests
.venv/bin/pytest --cov=lead_agent --cov-report=term-missing

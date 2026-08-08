#!/bin/sh
set -eu

python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pre-commit install --hook-type pre-commit --hook-type pre-push
.venv/bin/lead-agent init-db

printf '%s\n' 'Bootstrap complete. Activate with: source .venv/bin/activate'

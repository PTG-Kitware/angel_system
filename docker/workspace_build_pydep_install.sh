#!/bin/bash
# Workspace build component -- Python dependency installation
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This script will be installed into the directory that houses the
# `poetry.lock` and `pyproject.toml` files.
cd "$SCRIPT_DIR"

poetry install --no-root

#!/bin/bash
# Workspace build component -- Python dependency installation
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This script will be installed into the directory that houses the
# `poetry.lock` and `pyproject.toml` files.
cd "$SCRIPT_DIR"

# Install requirements ONLY as translated into frozen requirements form and
# then installed entirely into user space (`~/.local/`).
poetry export --dev -f requirements.txt |
  pip3 install --no-deps --user -r /dev/stdin

# Poetry extraneously finds matplotlib requiring setuptools-scm.
# theoretically being "fixed" in matplotlib 3.6
pip uninstall -y setuptools-scm

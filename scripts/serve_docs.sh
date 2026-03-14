#!/usr/bin/env bash
# Serve the MkDocs documentation site locally with live reload.
# Usage: ./scripts/serve_docs.sh [--build]
#   (no args)  → start live-reload server at http://localhost:8000
#   --build    → build the static site into website/site/ and exit
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEBSITE_DIR="$REPO_ROOT/website"

cd "$WEBSITE_DIR"

# Install deps if the venv is missing
if [ ! -d .venv ]; then
    echo "No .venv found — running uv sync..."
    uv sync --group docs
fi

if [ "${1:-}" = "--build" ]; then
    uv run mkdocs build --strict
    echo "Static site written to: $WEBSITE_DIR/site/"
else
    uv run mkdocs serve
fi

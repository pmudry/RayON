#!/usr/bin/env bash
# Serve the MkDocs documentation site locally with live reload.
# Usage: ./scripts/serve_docs.sh [--build] [--port PORT]
#   (no args)  → start live-reload server at http://localhost:8000
#   --build    → build the static site into website/site/ and exit
#   --port N   → use port N instead of 8000
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
    # Resolve the desired port (default 8000, or --port N)
    PORT=8000
    if [ "${1:-}" = "--port" ] && [ -n "${2:-}" ]; then
        PORT="$2"
    fi

    # Kill any mkdocs serve we own on that port; if held by another user, try +1
    OWNER_PID=$(fuser "${PORT}/tcp" 2>/dev/null | tr -d ' ') || true
    if [ -n "$OWNER_PID" ]; then
        if kill "$OWNER_PID" 2>/dev/null; then
            echo "Stopped previous mkdocs serve (PID $OWNER_PID) on port $PORT."
            sleep 1
        else
            # Not our process — pick the next free port
            FALLBACK=$((PORT + 1))
            echo "Port $PORT is in use by another process. Falling back to port $FALLBACK."
            PORT="$FALLBACK"
        fi
    fi

    echo "Starting docs server at http://localhost:${PORT}"
    uv run mkdocs serve --dev-addr "127.0.0.1:${PORT}" --livereload --watch docs --watch mkdocs.yml
fi

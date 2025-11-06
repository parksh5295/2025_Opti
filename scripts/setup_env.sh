#!/usr/bin/env bash
# POSIX-friendly virtual environment setup script for Unix-like shells.

set -euo pipefail

ENV_DIR=".venv"
RECREATE=0
PYTHON_BIN="python3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-dir)
            ENV_DIR="$2"
            shift 2
            ;;
        --recreate)
            RECREATE=1
            shift
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        *)
            echo "[setup-env] Unknown option: $1" >&2
            echo "Usage: $0 [--env-dir PATH] [--recreate] [--python PATH]" >&2
            exit 1
            ;;
    esac
done

echo "[setup-env] Project root: $(pwd)"
echo "[setup-env] Target virtual environment directory: $ENV_DIR"

if [[ $RECREATE -eq 1 && -d "$ENV_DIR" ]]; then
    echo "[setup-env] Removing existing environment..."
    rm -rf "$ENV_DIR"
fi

if [[ ! -d "$ENV_DIR" ]]; then
    echo "[setup-env] Creating virtual environment..."
    "$PYTHON_BIN" -m venv "$ENV_DIR"
else
    echo "[setup-env] Reusing existing environment. Use --recreate to rebuild."
fi

ACTIVATE_SCRIPT="$ENV_DIR/bin/activate"
if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
    echo "[setup-env] Activation script not found: $ACTIVATE_SCRIPT" >&2
    exit 1
fi

echo "[setup-env] Activating environment..."
# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"

echo "[setup-env] Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

if [[ -f requirements.txt ]]; then
    echo "[setup-env] Installing dependencies from requirements.txt..."
    python -m pip install -r requirements.txt
else
    echo "[setup-env] requirements.txt not found; skipping dependency installation."
fi

echo "[setup-env] Done. Activate later with:"
echo "    source $ACTIVATE_SCRIPT"


#!/usr/bin/env bash
# Pipenv environment bootstrap script for Unix-like shells.

set -euo pipefail

INSTALL_PIPENV=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-pipenv)
            INSTALL_PIPENV=1
            shift
            ;;
        *)
            echo "[pipenv] Unknown option: $1" >&2
            echo "Usage: $0 [--install-pipenv]" >&2
            exit 1
            ;;
    esac
done

echo "[pipenv] Project root: $(pwd)"

if [[ $INSTALL_PIPENV -eq 1 ]]; then
    echo "[pipenv] Installing pipenv (user site)..."
    python3 -m pip install --upgrade --user pip
    python3 -m pip install --upgrade --user pipenv
fi

if ! command -v pipenv >/dev/null 2>&1; then
    echo "[pipenv] pipenv is not installed. Re-run with --install-pipenv or install manually." >&2
    exit 1
fi

echo "[pipenv] Installing dependencies from Pipfile..."
pipenv install --dev

echo "[pipenv] Done. Activate with:"
echo "    pipenv shell"
echo "[pipenv] Run the pipeline with:"
echo "    pipenv run python main.py --data-path ../Data/creditcard/creditcard.csv"


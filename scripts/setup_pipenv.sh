#!/usr/bin/env bash
# Pipenv environment bootstrap script for Unix-like shells.

set -euo pipefail

INSTALL_PIPENV=0
INSTALL_PYENV=0
RESET_PYENV=0
PYTHON_VERSION="3.11.1"
FORCE_SHELL_INIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-pipenv)
            INSTALL_PIPENV=1
            shift
            ;;
        --install-pyenv)
            INSTALL_PYENV=1
            shift
            ;;
        --reset-pyenv)
            RESET_PYENV=1
            shift
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --force-shell-init)
            FORCE_SHELL_INIT=1
            shift
            ;;
        *)
            echo "[pipenv] Unknown option: $1" >&2
            echo "Usage: $0 [--install-pipenv] [--install-pyenv] [--reset-pyenv] [--python-version X.Y.Z] [--force-shell-init]" >&2
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

if [[ $INSTALL_PYENV -eq 1 ]]; then
    if [[ $RESET_PYENV -eq 1 && -d "$HOME/.pyenv" ]]; then
        echo "[pipenv] Removing existing pyenv at $HOME/.pyenv..."
        rm -rf "$HOME/.pyenv"
    fi

    if ! command -v pyenv >/dev/null 2>&1; then
        echo "[pipenv] Installing pyenv..."
        curl https://pyenv.run | bash
        export PATH="$HOME/.pyenv/bin:$PATH"
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"
    else
        echo "[pipenv] pyenv already installed."
        export PATH="$HOME/.pyenv/bin:$PATH"
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"
    fi

    if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
        echo "[pipenv] Installing Python ${PYTHON_VERSION} via pyenv..."
        pyenv install "$PYTHON_VERSION"
    else
        echo "[pipenv] Python ${PYTHON_VERSION} already installed in pyenv."
    fi

    export PIPENV_PYTHON="$HOME/.pyenv/versions/${PYTHON_VERSION}/bin/python"
    echo "[pipenv] Using Python at $PIPENV_PYTHON"
fi

if [[ $FORCE_SHELL_INIT -eq 1 ]]; then
    echo "[pipenv] Initialising pyenv for current shell..."
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    if [[ -z "${PIPENV_PYTHON:-}" ]]; then
        if pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
            export PIPENV_PYTHON="$HOME/.pyenv/versions/${PYTHON_VERSION}/bin/python"
        fi
    fi
fi

if ! command -v pipenv >/dev/null 2>&1; then
    echo "[pipenv] pipenv is not installed. Re-run with --install-pipenv or install manually." >&2
    exit 1
fi

if [[ -n "${PIPENV_PYTHON:-}" ]]; then
    echo "[pipenv] Installing dependencies from Pipfile using Python ${PYTHON_VERSION}..."
    pipenv install --dev --python "$PIPENV_PYTHON"
else
    echo "[pipenv] Installing dependencies from Pipfile..."
    pipenv install --dev --python "${PYTHON_VERSION}"
fi

if [[ $FORCE_SHELL_INIT -eq 1 ]]; then
    echo "[pipenv] Reminder: add the following to your shell profile (.bashrc or similar):"
    cat <<'EOF'
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
EOF
fi

echo "[pipenv] Done. Activate with:"
echo "    pipenv shell"
echo "[pipenv] Run the pipeline with:"
echo "    pipenv run python main.py --data-path ../Data/creditcard/creditcard.csv"


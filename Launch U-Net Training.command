#!/bin/bash
APP_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd -- "$APP_DIR" || exit 1

VENV_DIR="$APP_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"

pause_on_error() {
    printf '\n%s\n' "$1"
    read -r -p 'Press Return to close… ' _
    exit 1
}

find_python() {
    for candidate in python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && \
           "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

install_project() {
    printf '\nInstalling the U-Net classifier and its Python dependencies…\n'
    "$PYTHON" -m pip install -e "$APP_DIR" || pause_on_error 'Installation failed. See the error above and README.md.'
}

if [ ! -x "$PYTHON" ]; then
    BOOTSTRAP_PYTHON="$(find_python)" || pause_on_error 'Python 3.12 or newer was not found. Install Python 3.12, then run this launcher again.'
    printf 'A local Python environment is needed in:\n  %s\n\n' "$VENV_DIR"
    read -r -p 'Create it and install the application now? [Y/n] ' ANSWER
    case "$ANSWER" in n|N|no|NO) pause_on_error 'Installation cancelled.' ;; esac
    "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR" || pause_on_error 'Could not create the Python environment.'
    install_project
fi

if ! "$PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' >/dev/null 2>&1; then
    pause_on_error 'The existing .venv uses Python older than 3.12. Remove .venv and run this launcher again.'
fi

if ! "$PYTHON" "$APP_DIR/train.py" --check >/dev/null 2>&1; then
    printf 'The environment is incomplete or out of date.\n'
    read -r -p 'Install or repair the application now? [Y/n] ' ANSWER
    case "$ANSWER" in n|N|no|NO) pause_on_error 'Installation cancelled.' ;; esac
    install_project
fi

"$PYTHON" "$APP_DIR/train.py" "$@"
RESULT=$?
if [ "$RESULT" -ne 0 ]; then
    printf '\nU-Net classifier exited with code %s. See the error above and README.md.\n' "$RESULT"
    read -r -p 'Press Return to close… ' _
fi
exit "$RESULT"

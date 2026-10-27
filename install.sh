#!/bin/sh
# regesh one-line installer: creates an isolated virtual environment, installs regesh and runs the
# harness installer, which detects Claude Code, Codex CLI, Gemini CLI, Hermes Agent, OpenCode and
# Cursor and hooks the affect layer into the ones you select.
#
#   curl -fsSL https://raw.githubusercontent.com/cheesejaguar/regesh/main/install.sh | sh
#
# Environment: REGESH_HOME (default ~/.regesh), REGESH_REF (git ref, default main),
# REGESH_REPO (default https://github.com/cheesejaguar/regesh), REGESH_PYTHON (interpreter),
# REGESH_INSTALL_ARGS (extra arguments for regesh-install, e.g. "--yes" or "--harness claude").
set -eu

REGESH_HOME="${REGESH_HOME:-$HOME/.regesh}"
REGESH_REF="${REGESH_REF:-main}"
REGESH_REPO="${REGESH_REPO:-https://github.com/cheesejaguar/regesh}"
VENV="$REGESH_HOME/venv"

say() { printf '%s\n' "$*" >&2; }
fail() { say "regesh install: $*"; exit 1; }

find_python() {
  if [ -n "${REGESH_PYTHON:-}" ]; then
    echo "$REGESH_PYTHON"; return
  fi
  for candidate in python3.14 python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        echo "$candidate"; return
      fi
    fi
  done
  return 1
}

PY="$(find_python)" || fail "Python 3.11 or newer is required (set REGESH_PYTHON to an interpreter)."
command -v git >/dev/null 2>&1 || fail "git is required to fetch regesh."

say "→ Python: $($PY --version 2>&1)"
mkdir -p "$REGESH_HOME"
if [ ! -x "$VENV/bin/python" ]; then
  say "→ creating virtual environment at $VENV"
  "$PY" -m venv "$VENV" || fail "could not create a virtual environment (on Debian/Ubuntu: apt install python3-venv)."
fi
say "→ installing regesh ($REGESH_REF) into $VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
"$VENV/bin/python" -m pip install --quiet --upgrade "git+$REGESH_REPO@$REGESH_REF" || fail "pip install failed."

# Make the console scripts reachable from a shell too (optional, never fatal).
if [ -d "$HOME/.local/bin" ]; then
  for tool in regesh regesh-install regesh-mcp regesh-hook; do
    ln -sf "$VENV/bin/$tool" "$HOME/.local/bin/$tool" 2>/dev/null || true
  done
fi

say "→ launching the harness installer"
# When piped through `curl | sh`, stdin is the script; reattach the terminal for the interface.
if [ -t 1 ] && [ -r /dev/tty ] && [ -z "${REGESH_INSTALL_ARGS:-}" ]; then
  exec "$VENV/bin/regesh-install" </dev/tty
else
  # shellcheck disable=SC2086
  exec "$VENV/bin/regesh-install" ${REGESH_INSTALL_ARGS:---yes}
fi

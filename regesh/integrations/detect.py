"""Detect installed agent harnesses from binaries on PATH and their configuration directories."""
from __future__ import annotations

from pathlib import Path
import platform
import shutil

HARNESSES = {
    "claude": {"title": "Claude Code", "binaries": ["claude"], "paths": [".claude", ".claude.json"],
               "surfaces": ["MCP server (~/.claude.json)", "hooks (~/.claude/settings.json)", "instructions (~/.claude/CLAUDE.md)"]},
    "codex": {"title": "Codex CLI", "binaries": ["codex"], "paths": [".codex"],
              "surfaces": ["MCP server (~/.codex/config.toml)", "hooks (~/.codex/hooks.json)", "instructions (~/.codex/AGENTS.md)"]},
    "gemini": {"title": "Gemini CLI", "binaries": ["gemini"], "paths": [".gemini"],
               "surfaces": ["MCP server + hooks (~/.gemini/settings.json)", "instructions (~/.gemini/GEMINI.md)"]},
    "hermes": {"title": "Hermes Agent", "binaries": ["hermes"], "paths": [".hermes"],
               "surfaces": ["MCP server (~/.hermes/config.yaml)", "plugin (~/.hermes/plugins/regesh)"]},
    "opencode": {"title": "OpenCode", "binaries": ["opencode"], "paths": [".config/opencode", ".opencode"],
                 "surfaces": ["MCP server (~/.config/opencode/opencode.json)", "plugin (~/.config/opencode/plugins/regesh.js)", "instructions (~/.config/opencode/AGENTS.md)"]},
    "cursor": {"title": "Cursor", "binaries": ["cursor"], "paths": [".cursor"], "apps": ["/Applications/Cursor.app"],
               "surfaces": ["MCP server (~/.cursor/mcp.json)"]},
}
ORDER = ["claude", "codex", "gemini", "hermes", "opencode", "cursor"]


def detect(home=None, path=None):
    """Return a list of dicts: {id, title, detected, evidence, surfaces} in display order."""
    home = Path(home) if home else Path.home()
    found = []
    for hid in ORDER:
        spec = HARNESSES[hid]; evidence = []
        for binary in spec["binaries"]:
            where = shutil.which(binary, path=path) if path is not None else shutil.which(binary)
            if where:
                evidence.append(f"binary {where}")
        for rel in spec["paths"]:
            p = home / rel
            if p.exists():
                evidence.append(f"config {p}")
        if platform.system() == "Darwin":
            for app in spec.get("apps", []):
                if Path(app).exists():
                    evidence.append(f"app {app}")
        found.append({"id": hid, "title": spec["title"], "detected": bool(evidence), "evidence": evidence, "surfaces": spec["surfaces"]})
    return found

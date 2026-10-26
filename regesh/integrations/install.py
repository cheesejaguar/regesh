"""Idempotent, reversible writers that hook the affect layer into installed agent harnesses.

Every file touched is backed up under ~/.regesh/backups/<timestamp>/ and listed in
~/.regesh/install_manifest.json. Our additions are marked (JSON keys we own, marker comments
in TOML/Markdown, a trailing ``# regesh-install`` tag on YAML lines) so ``uninstall`` removes
exactly what we added and leaves the user's own later edits alone.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

from .detect import HARNESSES, ORDER

TEMPLATES = Path(__file__).parent / "templates"
MD_START, MD_END = "<!-- regesh:start -->", "<!-- regesh:end -->"
TOML_START, TOML_END = "# regesh:start", "# regesh:end"
YAML_TAG = "# regesh-install"
CLAUDE_EVENTS = ["UserPromptSubmit", "PostToolUse", "PostToolUseFailure", "Stop", "SessionStart"]
CODEX_EVENTS = ["UserPromptSubmit", "PostToolUse", "Stop", "SessionStart"]
GEMINI_EVENTS = ["BeforeAgent", "AfterTool", "AfterAgent", "SessionStart"]


def now_tag():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def find_command(name, bin_dir=None):
    """Absolute path of a console script from this environment, falling back to PATH."""
    candidates = []
    if bin_dir:
        candidates.append(Path(bin_dir) / name)
    candidates.append(Path(sys.executable).parent / name)
    for c in candidates:
        if c.exists():
            return str(c)
    found = shutil.which(name)
    if found:
        return found
    return None


class Installer:
    def __init__(self, home=None, user_home=None, bin_dir=None, dry_run=False):
        self.home = Path(home) if home else Path(os.environ.get("REGESH_HOME", str(Path.home() / ".regesh")))
        self.user_home = Path(user_home) if user_home else Path.home()
        self.dry_run = dry_run
        self.manifest_path = self.home / "install_manifest.json"
        self.manifest = self._load_manifest()
        self.mcp_cmd = find_command("regesh-mcp", bin_dir)
        self.hook_cmd = find_command("regesh-hook", bin_dir)
        self.python = sys.executable
        self._stamp = now_tag()

    # ----- manifest / backups -----
    def _load_manifest(self):
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text())
            except ValueError:
                pass
        return {"version": 1, "harnesses": {}}

    def _save_manifest(self):
        if self.dry_run:
            return
        self.home.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest, indent=1, sort_keys=True) + "\n")

    def _backup(self, path):
        if not path.exists():
            return None
        dest = self.home / "backups" / self._stamp / path.name
        if not self.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            n = 1
            while dest.exists():
                dest = dest.with_name(f"{path.name}.{n}"); n += 1
            shutil.copy2(path, dest)
        return str(dest)

    def _record(self, report, path, kind, existed, backup, action):
        report["changes"].append({"file": str(path), "action": action})
        files = report["_files"]
        if not any(f["path"] == str(path) for f in files):
            files.append({"path": str(path), "kind": kind, "existed": existed, "backup": backup})

    def _write(self, path, text):
        if self.dry_run:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".regesh-tmp")
        tmp.write_text(text)
        os.replace(tmp, path)

    # ----- command strings -----
    def mcp_server(self, hid):
        if self.mcp_cmd:
            return self.mcp_cmd, ["--harness", hid]
        return self.python, ["-m", "regesh.integrations.mcp_server", "--harness", hid]

    def hook_command(self, hid, event):
        if self.hook_cmd:
            return f'"{self.hook_cmd}" {hid} {event}'
        return f'"{self.python}" -m regesh.integrations.hooks {hid} {event}'

    def _is_ours_hook(self, hook, hid):
        cmd = hook.get("command", "") if isinstance(hook, dict) else ""
        return "regesh" in cmd and f" {hid} " in cmd + " "

    # ----- generic writers -----
    def _json_update(self, report, path, kind, mutate):
        """mutate(data) -> True if changed. Skips with a note when the file is not valid JSON."""
        existed = path.exists()
        data = {}
        if existed:
            try:
                data = json.loads(path.read_text() or "{}")
            except ValueError:
                report["notes"].append(f"{path} is not valid JSON (comments?); left untouched"); return False
            if not isinstance(data, dict):
                report["notes"].append(f"{path} is not a JSON object; left untouched"); return False
        if not mutate(data):
            report["changes"].append({"file": str(path), "action": "already configured"}); return False
        backup = self._backup(path)
        self._write(path, json.dumps(data, indent=2) + "\n")
        self._record(report, path, kind, existed, backup, "updated" if existed else "created")
        return True

    def _json_remove(self, path, mutate):
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text() or "{}")
        except ValueError:
            return False
        if not isinstance(data, dict) or not mutate(data):
            return False
        entry = self._file_entry(path)
        if not data and entry and not entry.get("existed"):
            if not self.dry_run:
                path.unlink()
            return True
        self._write(path, json.dumps(data, indent=2) + "\n")
        return True

    @staticmethod
    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    def _restore_untouched(self, hid):
        """Files whose content is exactly what we wrote are restored from their backup byte for byte
        (or deleted when we created them). Returns the set of paths handled this way."""
        handled = set()
        for f in self.manifest.get("harnesses", {}).get(hid, {}).get("files", []):
            path = Path(f["path"])
            if not path.exists() or f.get("written_sha256") != self._sha(path):
                continue
            if self.dry_run:
                handled.add(str(path)); continue
            if f.get("existed") and f.get("backup") and Path(f["backup"]).exists():
                shutil.copy2(f["backup"], path); handled.add(str(path))
            elif not f.get("existed"):
                path.unlink(); handled.add(str(path))
        return handled

    def _rmdir_if_empty(self, *dirs):
        for d in dirs:
            try:
                if d.exists() and d.is_dir() and not any(d.iterdir()) and not self.dry_run:
                    d.rmdir()
            except OSError:
                pass

    def _markers_add(self, report, path, kind, body, start, end):
        existed = path.exists()
        text = path.read_text() if existed else ""
        block = f"{start}\n{body.rstrip()}\n{end}\n"
        if start in text and end in text:
            pattern = re.compile(re.escape(start) + r".*?" + re.escape(end) + r"\n?", re.S)
            new = pattern.sub(lambda m: block, text, count=1)
            if new == text:
                report["changes"].append({"file": str(path), "action": "already configured"}); return False
        else:
            new = text + ("\n" if text and not text.endswith("\n") else "") + ("\n" if text else "") + block
        backup = self._backup(path)
        self._write(path, new)
        self._record(report, path, kind, existed, backup, "updated" if existed else "created")
        return True

    @staticmethod
    def _markers_remove_text(text, start, end):
        if start not in text:
            return text
        pattern = re.compile(r"\n*" + re.escape(start) + r".*?" + re.escape(end) + r"[ \t]*\n?", re.S)
        new = pattern.sub("", text, count=1)
        return (new.rstrip("\n") + "\n") if new.strip() else ""

    # ----- per-harness surfaces -----
    def _mcp_json(self, report, path, hid, key="mcpServers", extra=None, list_command=False):
        cmd, args = self.mcp_server(hid)
        def mutate(data):
            servers = data.setdefault(key, {})
            if not isinstance(servers, dict):
                return False
            entry = {"command": [cmd, *args]} if list_command else {"command": cmd, "args": args}
            entry.update(extra or {})
            if servers.get("regesh") == entry:
                return False
            servers["regesh"] = entry
            return True
        return self._json_update(report, path, "json-mcp", mutate)

    def _hooks_json(self, report, path, hid, events, entry_factory, root_key="hooks"):
        def mutate(data):
            hooks = data.setdefault(root_key, {})
            if not isinstance(hooks, dict):
                return False
            changed = False
            for event in events:
                groups = hooks.setdefault(event, [])
                if not isinstance(groups, list):
                    continue
                present = any(self._is_ours_hook(h, hid) for g in groups if isinstance(g, dict) for h in g.get("hooks", []))
                if not present:
                    groups.append(entry_factory(event)); changed = True
            return changed
        return self._json_update(report, path, "json-hooks", mutate)

    def _remove_hooks_json(self, path, hid, root_key="hooks"):
        def mutate(data):
            hooks = data.get(root_key)
            if not isinstance(hooks, dict):
                return False
            changed = False
            for event, groups in list(hooks.items()):
                if not isinstance(groups, list):
                    continue
                kept = []
                for g in groups:
                    if isinstance(g, dict) and any(self._is_ours_hook(h, hid) for h in g.get("hooks", [])):
                        g = dict(g); g["hooks"] = [h for h in g["hooks"] if not self._is_ours_hook(h, hid)]
                        changed = True
                        if not g["hooks"]:
                            continue
                    kept.append(g)
                if kept:
                    hooks[event] = kept
                else:
                    if event in hooks and groups != kept:
                        del hooks[event]
            if not hooks and root_key in data:
                del data[root_key]
            return changed
        return self._json_remove(path, mutate)

    def _remove_mcp_json(self, path, key="mcpServers"):
        def mutate(data):
            servers = data.get(key)
            if isinstance(servers, dict) and "regesh" in servers:
                del servers["regesh"]
                if not servers:
                    del data[key]
                return True
            return False
        return self._json_remove(path, mutate)

    def instructions(self, report, path):
        body = (TEMPLATES / "instructions.md").read_text()
        return self._markers_add(report, path, "markdown", body, MD_START, MD_END)

    # Claude Code
    def install_claude(self, report):
        self._mcp_json(report, self.user_home / ".claude.json", "claude")
        self._hooks_json(report, self.user_home / ".claude" / "settings.json", "claude", CLAUDE_EVENTS,
                         lambda ev: {"hooks": [{"type": "command", "command": self.hook_command("claude", ev), "timeout": 15}]})
        self.instructions(report, self.user_home / ".claude" / "CLAUDE.md")
        report["notes"].append("Restart Claude Code (or run /mcp) to pick up the regesh MCP server and hooks.")

    def uninstall_claude(self, hid):
        self._remove_mcp_json(self.user_home / ".claude.json")
        self._remove_hooks_json(self.user_home / ".claude" / "settings.json", "claude")
        self._remove_markers(self.user_home / ".claude" / "CLAUDE.md", MD_START, MD_END)

    # Codex CLI
    def install_codex(self, report):
        self._codex_toml(report, self.user_home / ".codex" / "config.toml")
        self._hooks_json(report, self.user_home / ".codex" / "hooks.json", "codex", CODEX_EVENTS,
                         lambda ev: {"hooks": [{"type": "command", "command": self.hook_command("codex", ev), "timeout": 15}]})
        self.instructions(report, self.user_home / ".codex" / "AGENTS.md")
        report["notes"].append("Codex requires hooks to be trusted once: run `codex`, then `/hooks` and trust the regesh hooks.")

    def _codex_toml(self, report, path):
        existed = path.exists()
        text = path.read_text() if existed else ""
        try:
            parsed = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            report["notes"].append(f"{path} does not parse as TOML ({exc}); MCP server not added"); return False
        if isinstance(parsed.get("mcp_servers"), dict) and "regesh" in parsed["mcp_servers"]:
            report["changes"].append({"file": str(path), "action": "already configured"}); return False
        cmd, args = self.mcp_server("codex")
        block = (f"{TOML_START}\n[mcp_servers.regesh]\ncommand = {json.dumps(cmd)}\nargs = {json.dumps(args)}\n"
                 f"startup_timeout_sec = 20\n{TOML_END}\n")
        new = text + ("\n" if text and not text.endswith("\n") else "") + ("\n" if text else "") + block
        try:
            tomllib.loads(new)
        except tomllib.TOMLDecodeError as exc:
            report["notes"].append(f"adding the MCP table to {path} would break it ({exc}); left untouched"); return False
        backup = self._backup(path)
        self._write(path, new)
        self._record(report, path, "toml", existed, backup, "updated" if existed else "created")
        return True

    def uninstall_codex(self, hid):
        self._remove_markers(self.user_home / ".codex" / "config.toml", TOML_START, TOML_END)
        self._remove_hooks_json(self.user_home / ".codex" / "hooks.json", "codex")
        self._remove_markers(self.user_home / ".codex" / "AGENTS.md", MD_START, MD_END)

    # Gemini CLI
    def install_gemini(self, report):
        path = self.user_home / ".gemini" / "settings.json"
        self._mcp_json(report, path, "gemini", extra={"timeout": 10000})
        self._hooks_json(report, path, "gemini", GEMINI_EVENTS,
                         lambda ev: {"hooks": [{"name": f"regesh-{ev}", "type": "command", "command": self.hook_command("gemini", ev), "timeout": 15000}]})
        self.instructions(report, self.user_home / ".gemini" / "GEMINI.md")
        report["notes"].append("Restart Gemini CLI to load the regesh MCP server and hooks.")

    def uninstall_gemini(self, hid):
        path = self.user_home / ".gemini" / "settings.json"
        self._remove_mcp_json(path)
        self._remove_hooks_json(path, "gemini")
        self._remove_markers(self.user_home / ".gemini" / "GEMINI.md", MD_START, MD_END)

    # Hermes Agent
    def install_hermes(self, report):
        self._hermes_yaml(report, self.user_home / ".hermes" / "config.yaml")
        plugin_dir = self.user_home / ".hermes" / "plugins" / "regesh"
        hook = self.hook_cmd or f"{self.python} -m regesh.integrations.hooks"
        init = (TEMPLATES / "hermes_plugin_init.py").read_text().replace("__REGESH_HOOK__", hook)
        if not self.hook_cmd:  # module invocation needs the interpreter plus -m
            init = init.replace('[REGESH_HOOK, "hermes", event]', f'[{json.dumps(self.python)}, "-m", "regesh.integrations.hooks", "hermes", event]')
        self._plugin_file(report, plugin_dir / "__init__.py", init)
        self._plugin_file(report, plugin_dir / "plugin.yaml", (TEMPLATES / "hermes_plugin.yaml").read_text())
        enabled = self._hermes_enable(report)
        if not enabled:
            report["notes"].append("Enable the plugin with `hermes plugins enable regesh` (the installer could not run it).")
        report["notes"].append("Restart Hermes (or the gateway) so the regesh plugin and MCP server load.")

    def _hermes_enable(self, report):
        if self.dry_run:
            return True
        hermes = shutil.which("hermes")
        if not hermes:
            return False
        try:
            proc = subprocess.run([hermes, "plugins", "enable", "regesh"], capture_output=True, text=True, timeout=60,
                                  env={**os.environ, "HOME": str(self.user_home)})
        except (OSError, subprocess.SubprocessError):
            return False
        if proc.returncode == 0:
            report["changes"].append({"file": "hermes plugins enable regesh", "action": "ran"}); return True
        return False

    def _plugin_file(self, report, path, content):
        existed = path.exists()
        if existed and path.read_text() == content:
            report["changes"].append({"file": str(path), "action": "already configured"}); return False
        backup = self._backup(path)
        self._write(path, content)
        self._record(report, path, "plugin", existed, backup, "updated" if existed else "created")
        return True

    def _hermes_yaml(self, report, path):
        existed = path.exists()
        text = path.read_text() if existed else ""
        if re.search(r"^\s+regesh:.*$", text, re.M) and re.search(r"^mcp_servers:", text, re.M):
            report["changes"].append({"file": str(path), "action": "already configured"}); return False
        cmd, args = self.mcp_server("hermes")
        new = yaml_insert_mcp_server(text, "regesh", cmd, args)
        if new is None:
            report["notes"].append(f"{path} has an mcp_servers entry this installer cannot edit safely; add the regesh server by hand (see docs/install.md)"); return False
        try:
            import yaml  # optional validation only
            yaml.safe_load(new)
        except ImportError:
            pass
        except Exception as exc:  # pragma: no cover - depends on optional dependency
            report["notes"].append(f"the edited {path} did not validate as YAML ({exc}); left untouched"); return False
        backup = self._backup(path)
        self._write(path, new)
        self._record(report, path, "yaml", existed, backup, "updated" if existed else "created")
        return True

    def uninstall_hermes(self, hid):
        path = self.user_home / ".hermes" / "config.yaml"
        if path.exists():
            new = yaml_remove_tagged(path.read_text())
            if new is not None:
                self._write(path, new)
        plugin_dir = self.user_home / ".hermes" / "plugins" / "regesh"
        for name in ("__init__.py", "plugin.yaml"):
            p = plugin_dir / name
            if p.exists() and not self.dry_run:
                p.unlink()
        self._rmdir_if_empty(plugin_dir, plugin_dir.parent)
        hermes = shutil.which("hermes")
        if hermes and not self.dry_run:
            try:
                subprocess.run([hermes, "plugins", "disable", "regesh"], capture_output=True, text=True, timeout=60, env={**os.environ, "HOME": str(self.user_home)})
            except (OSError, subprocess.SubprocessError):
                pass

    # OpenCode
    def install_opencode(self, report):
        base = self.user_home / ".config" / "opencode"
        cfg = base / "opencode.json"
        if not cfg.exists() and (base / "opencode.jsonc").exists():
            report["notes"].append(f"{base / 'opencode.jsonc'} uses JSONC; add the regesh MCP server by hand (see docs/install.md)")
        else:
            self._mcp_json(report, cfg, "opencode", key="mcp", extra={"type": "local", "enabled": True}, list_command=True)
        hook = self.hook_cmd
        js = (TEMPLATES / "opencode_plugin.js").read_text()
        if hook:
            js = js.replace("__REGESH_HOOK__", hook)
        else:
            js = js.replace('spawn(REGESH_HOOK, ["opencode", event]', f'spawn({json.dumps(self.python)}, ["-m", "regesh.integrations.hooks", "opencode", event]').replace("__REGESH_HOOK__", self.python)
        self._plugin_file(report, base / "plugins" / "regesh.js", js)
        self.instructions(report, base / "AGENTS.md")
        report["notes"].append("Restart OpenCode to load the regesh plugin and MCP server.")

    def uninstall_opencode(self, hid):
        base = self.user_home / ".config" / "opencode"
        self._remove_mcp_json(base / "opencode.json", key="mcp")
        p = base / "plugins" / "regesh.js"
        if p.exists() and not self.dry_run:
            p.unlink()
        self._rmdir_if_empty(p.parent)
        self._remove_markers(base / "AGENTS.md", MD_START, MD_END)

    # Cursor
    def install_cursor(self, report):
        self._mcp_json(report, self.user_home / ".cursor" / "mcp.json", "cursor")
        report["notes"].append("Cursor exposes regesh as MCP tools only (no hook surface); reload MCP servers in Cursor settings.")

    def uninstall_cursor(self, hid):
        self._remove_mcp_json(self.user_home / ".cursor" / "mcp.json")

    def _remove_markers(self, path, start, end):
        if not path.exists():
            return
        text = path.read_text()
        new = self._markers_remove_text(text, start, end)
        if new != text:
            if new.strip():
                self._write(path, new)
            elif not self.dry_run:
                entry = self._file_entry(path)
                if entry and not entry.get("existed"):
                    path.unlink()
                else:
                    self._write(path, new)

    def _file_entry(self, path):
        for h in self.manifest.get("harnesses", {}).values():
            for f in h.get("files", []):
                if f["path"] == str(path):
                    return f
        return None

    # ----- public API -----
    def install(self, hid):
        if hid not in HARNESSES:
            raise ValueError(f"unknown harness {hid}")
        report = {"harness": hid, "title": HARNESSES[hid]["title"], "changes": [], "notes": [], "_files": [], "dry_run": self.dry_run}
        previous = self.manifest["harnesses"].get(hid, {}).get("files", [])
        getattr(self, f"install_{hid}")(report)
        files = {f["path"]: f for f in previous}
        for f in report["_files"]:
            files.setdefault(f["path"], f)
        for f in files.values():
            f["written_sha256"] = self._sha(Path(f["path"]))
        self.manifest["harnesses"][hid] = {"installed_at": datetime.now(timezone.utc).isoformat(), "mcp_command": list(self.mcp_server(hid)[0:1]) + self.mcp_server(hid)[1],
                                           "hook_command": self.hook_cmd or f"{self.python} -m regesh.integrations.hooks", "files": list(files.values())}
        self._save_manifest()
        report.pop("_files")
        return report

    def uninstall(self, hid):
        if hid not in HARNESSES:
            raise ValueError(f"unknown harness {hid}")
        self._restore_untouched(hid)
        getattr(self, f"uninstall_{hid}")(hid)
        entry = self.manifest["harnesses"].pop(hid, None)
        self._save_manifest()
        return {"harness": hid, "title": HARNESSES[hid]["title"], "removed": True, "had_manifest": entry is not None,
                "backups": [f["backup"] for f in (entry or {}).get("files", []) if f.get("backup")]}

    def installed(self):
        return sorted(self.manifest.get("harnesses", {}), key=ORDER.index)


# ----- YAML helpers (no dependency; Hermes config.yaml is block-style YAML) -----
def yaml_insert_mcp_server(text, name, command, args):
    """Insert ``name`` under the top-level ``mcp_servers`` mapping, tagging every added line.

    Returns the new text, or None when the existing mcp_servers value is a form we will not edit.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^mcp_servers:(.*)$", line)
        if not m:
            continue
        rest = m.group(1).split("#", 1)[0].strip()
        if rest in ("{}", "~", "null"):
            indent = 2
            lines[i] = f"mcp_servers:  {YAML_TAG}"
            return "\n".join(lines[: i + 1] + _server_lines(name, command, args, indent) + lines[i + 1:]) + "\n"
        if rest:
            return None   # flow mapping or anchor we do not understand
        indent = 2
        for nxt in lines[i + 1:]:
            if nxt.strip() and not nxt.lstrip().startswith("#"):
                lead = len(nxt) - len(nxt.lstrip())
                if lead > 0:
                    indent = lead
                break
        return "\n".join(lines[: i + 1] + _server_lines(name, command, args, indent) + lines[i + 1:]) + "\n"
    head = text if not text or text.endswith("\n") else text + "\n"
    if head.strip():
        head += "\n"
    return head + "\n".join([f"mcp_servers:  {YAML_TAG}"] + _server_lines(name, command, args, 2)) + "\n"


def _server_lines(name, command, args, indent):
    pad = " " * indent; inner = " " * (indent * 2)
    return [f"{pad}{name}:  {YAML_TAG}", f"{inner}command: {json.dumps(command)}  {YAML_TAG}", f"{inner}args: {json.dumps(args)}  {YAML_TAG}",
            f"{inner}enabled: true  {YAML_TAG}"]


def yaml_remove_tagged(text):
    if YAML_TAG not in text:
        return None
    lines = text.splitlines()
    kept = []
    for i, line in enumerate(lines):
        if not line.rstrip().endswith(YAML_TAG):
            kept.append(line); continue
        if line.startswith("mcp_servers:"):
            # keep the parent key if other children (not ours) follow it
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].startswith(" ")):
                if lines[j].strip() and not lines[j].rstrip().endswith(YAML_TAG):
                    kept.append("mcp_servers:"); break
                j += 1
    out = "\n".join(kept).rstrip("\n")
    return out + "\n" if out else ""

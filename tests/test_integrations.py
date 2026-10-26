"""Harness integrations: offline text affect, bridge, hooks CLI, MCP server, detection, installer."""
import io
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from regesh.integrations import hooks
from regesh.integrations.bridge import Bridge, judge_tool_result, context_line
from regesh.integrations.detect import detect, ORDER
from regesh.integrations.install import Installer, yaml_insert_mcp_server, yaml_remove_tagged, MD_START
from regesh.integrations.text_affect import OfflineTextAffect, OFFLINE_GATE
from regesh.integrations import cli


# ----- offline text affect -----
def test_offline_estimator_signs_and_gate():
    est = OfflineTextAffect()
    pos = est.estimate("Thank you so much, this really helped me today!")
    neutral = est.estimate("list the files in the directory")
    assert pos["partner_affect"] > .5 and not pos["gate_fired"] and pos["top_emotion"] == "gratitude"
    assert neutral["gate_fired"] and neutral["partner_affect"] == 0.
    assert set(pos) >= {"model", "partner_affect", "raw_valence", "gate_fired", "top_emotion", "top_probability", "support_proxy"}
    assert OFFLINE_GATE == {"label": "neutral", "tau": .2, "dead_band": .6}   # calibrated: results/bonding/text_affect_gate/gate.json


def test_offline_gate_matches_calibration_artifact():
    gate = json.loads((Path(__file__).parent.parent / "results/bonding/text_affect_gate/gate.json").read_text())["chosen_gate"]
    assert gate["tau"] == OFFLINE_GATE["tau"] and gate["dead_band"] == OFFLINE_GATE["dead_band"]


# ----- tool-result heuristic -----
@pytest.mark.parametrize("tool,text,error,exit_code,expect", [
    ("Bash", "3 passed in 0.1s", None, None, "positive"),
    ("Bash", "FAILED tests/x.py - AssertionError", None, None, "negative"),
    ("Bash", "", "boom", None, "negative"),
    ("Bash", "ok", None, 2, "negative"),
    ("Read", "def f(): pass", None, None, None),
    ("Bash", "1 failed, 2 passed", None, None, "negative"),
    ("Bash", "12 passed, 0 failed, 0 errors", None, None, "positive"),
])
def test_judge_tool_result(tool, text, error, exit_code, expect):
    v = judge_tool_result(tool, text, error=error, exit_code=exit_code)
    if expect is None:
        assert v is None
    else:
        assert (v["quality"] > 0) == (expect == "positive") and 0 <= v["control"] <= 1


# ----- bridge -----
def test_bridge_updates_affect_and_persists(tmp_path):
    b = Bridge("testharness", home=tmp_path)
    try:
        c0 = b.context()
        assert c0["bond"]["partner"] == "user" and "Affect layer (regesh)" in c0["line"]
        c1 = b.observe_tool_result("Bash", {"command": "pytest"}, {"stdout": "FAILED tests/x.py::t - AssertionError", "exit_code": 1})
        assert c1["mood"]["valence"] < c0["mood"]["valence"]
        c2 = b.observe_user_message("Thank you so much, this really helped me today!")
        assert c2["mood"]["valence"] > c1["mood"]["valence"]
        c3 = b.turn_end("Glad it helped.")
        assert c3["time"] == c0["time"] + 1
        assert len(c3["line"]) < 700
        state = b.agent.get_state()
        assert any(e["source"] == "testharness:prompt" for e in state["events"].values())
        digest = b.agent.snapshot()["digest"]
    finally:
        b.close()
    again = Bridge("testharness", home=tmp_path)   # reopen the same persisted agent
    try:
        assert again.agent.snapshot()["digest"] == digest
        assert again.context()["time"] == 1.
    finally:
        again.close()


def test_bridge_request_support_respects_controls(tmp_path):
    b = Bridge("h", home=tmp_path)
    try:
        r = b.request_support()
        assert r["requested"] is False and "context" in r
    finally:
        b.close()


def test_context_line_mentions_repair_and_request():
    ctx = {"mood": {"valence": -.2, "arousal": .3}, "controls": {"exploration": .4, "verification": .6, "planning_horizon": 4},
           "firewall": "affect state never changes facts, permissions or tool authority",
           "bond": {"partner": "user", "mode": "seek", "security": .2, "intimacy": .1, "expressed_valence": -.1, "coordination_cap": .5,
                    "repair": True, "coordination_band": "over", "request_support": True, "arousal": .8}}
    line = context_line(ctx)
    assert "repair threshold" in line and "above the human band" in line and "check-in" in line


# ----- hooks CLI -----
def run_hook(tmp_path, harness, event, payload):
    out = io.StringIO()
    os.environ["REGESH_HOME"] = str(tmp_path)
    try:
        code = hooks.run([harness, event], io.StringIO(payload if isinstance(payload, str) else json.dumps(payload)), out)
    finally:
        del os.environ["REGESH_HOME"]
    return code, out.getvalue()


def test_hooks_claude_codex_schema(tmp_path):
    for harness in ("claude", "codex"):
        code, out = run_hook(tmp_path, harness, "UserPromptSubmit", {"prompt": "thanks, that worked perfectly!", "session_id": "s"})
        assert code == 0
        data = json.loads(out)
        assert data["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert data["hookSpecificOutput"]["additionalContext"].startswith("Affect layer (regesh)")
        code, out = run_hook(tmp_path, harness, "PostToolUse", {"tool_name": "Bash", "tool_input": {"command": "pytest"}, "tool_response": {"stdout": "FAILED x"}})
        assert json.loads(out)["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
        code, out = run_hook(tmp_path, harness, "Stop", {"last_assistant_message": "done"})
        assert out == ""   # Stop must stay silent (anything else is a block decision)
        code, out = run_hook(tmp_path, harness, "SessionStart", {"source": "startup"})
        assert "additionalContext" in out


def test_hooks_gemini_hermes_opencode_generic(tmp_path):
    code, out = run_hook(tmp_path, "gemini", "BeforeAgent", {"prompt": "hi"})
    data = json.loads(out); assert "hookEventName" not in data["hookSpecificOutput"] and "additionalContext" in data["hookSpecificOutput"]
    code, out = run_hook(tmp_path, "gemini", "AfterTool", {"tool_name": "run_shell_command", "tool_response": {"llmContent": "3 passed", "error": None}})
    assert "additionalContext" in out
    code, out = run_hook(tmp_path, "gemini", "AfterAgent", {"prompt": "hi", "prompt_response": "hello"}); assert out == ""
    code, out = run_hook(tmp_path, "hermes", "pre_llm_call", {"user_message": [{"type": "text", "text": "thanks!"}], "session_id": "s"})
    assert json.loads(out)["context"].startswith("Affect layer")
    code, out = run_hook(tmp_path, "hermes", "post_tool_call", {"tool_name": "terminal", "args": {}, "result": json.dumps({"output": "Traceback (most recent call last)"})}); assert out == ""
    code, out = run_hook(tmp_path, "hermes", "post_llm_call", {"assistant_response": "ok"}); assert out == ""
    code, out = run_hook(tmp_path, "opencode", "tool.execute.after", {"tool": "bash", "args": {}, "output": {"title": "t", "output": "error: not found"}}); assert out == ""
    code, out = run_hook(tmp_path, "opencode", "session.idle", {}); assert out == ""
    code, out = run_hook(tmp_path, "generic", "prompt", {"text": "thank you!"}); assert "context" in json.loads(out)
    code, out = run_hook(tmp_path, "generic", "tool", {"tool_name": "bash", "tool_response": "exit code 1", "exit_code": 1}); assert out == ""


def test_hooks_fail_open(tmp_path):
    code, out = run_hook(tmp_path, "claude", "NoSuchEvent", {"x": 1})
    assert code == 0 and out == ""
    assert "unknown event" in (tmp_path / "hook.log").read_text()
    code, out = run_hook(tmp_path, "claude", "UserPromptSubmit", "this is not json")
    assert code == 0 and "additionalContext" in out   # plain text is treated as the prompt
    assert hooks.run([], io.StringIO(""), io.StringIO()) == 0


def test_hook_console_script_subprocess(tmp_path):
    env = {**os.environ, "REGESH_HOME": str(tmp_path)}
    proc = subprocess.run([sys.executable, "-m", "regesh.integrations.hooks", "claude", "UserPromptSubmit"], input=json.dumps({"prompt": "hello there"}),
                          capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0 and "additionalContext" in proc.stdout


# ----- MCP server -----
def test_mcp_server_round_trip(tmp_path):
    proc = subprocess.Popen([sys.executable, "-m", "regesh.integrations.mcp_server", "--harness", "mcp-test", "--home", str(tmp_path)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    counter = iter(range(1, 100))

    def rpc(method, params=None):
        i = next(counter)
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}}) + "\n"); proc.stdin.flush()
        msg = json.loads(proc.stdout.readline()); assert msg["id"] == i; return msg
    try:
        init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})
        assert init["result"]["serverInfo"]["name"] == "regesh" and "tools" in init["result"]["capabilities"]
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"); proc.stdin.flush()
        assert rpc("ping")["result"] == {}
        tools = [t["name"] for t in rpc("tools/list")["result"]["tools"]]
        assert tools == ["affect_context", "affect_observe_message", "affect_observe_outcome", "affect_express", "affect_request_support", "affect_snapshot"]
        r = rpc("tools/call", {"name": "affect_observe_message", "arguments": {"text": "Thank you so much, this really helped me today!"}})
        ctx = json.loads(r["result"]["content"][0]["text"]); assert r["result"]["isError"] is False and ctx["bond"]["partner"] == "user"
        r = rpc("tools/call", {"name": "affect_observe_outcome", "arguments": {"description": "tests failed", "quality": -.7}})
        assert json.loads(r["result"]["content"][0]["text"])["mood"]["valence"] < ctx["mood"]["valence"]
        r = rpc("tools/call", {"name": "affect_express", "arguments": {"valence": .2, "text": "Nice work."}}); assert r["result"]["isError"] is False
        r = rpc("tools/call", {"name": "affect_request_support", "arguments": {}}); assert "requested" in json.loads(r["result"]["content"][0]["text"])
        r = rpc("tools/call", {"name": "affect_observe_outcome", "arguments": {"description": "bad", "quality": 7}}); assert r["result"]["isError"] is True
        assert rpc("tools/call", {"name": "nope", "arguments": {}})["error"]["code"] == -32602
        res = rpc("resources/list")["result"]["resources"]; assert res[0]["uri"] == "regesh://affect/context"
        assert rpc("resources/read", {"uri": "regesh://affect/context"})["result"]["contents"][0]["text"].startswith("Affect layer")
        snap = json.loads(rpc("tools/call", {"name": "affect_snapshot", "arguments": {}})["result"]["content"][0]["text"]); assert snap["events"] == 2
        assert rpc("unknown/method")["error"]["code"] == -32601
    finally:
        proc.stdin.close(); proc.wait(timeout=30)
    assert proc.returncode == 0


# ----- detection -----
def test_detect_uses_binaries_and_config_dirs(tmp_path):
    bindir = tmp_path / "bin"; bindir.mkdir()
    for name in ("claude", "hermes"):
        p = bindir / name; p.write_text("#!/bin/sh\n"); p.chmod(0o755)
    (tmp_path / "home" / ".codex").mkdir(parents=True)
    found = {d["id"]: d for d in detect(home=tmp_path / "home", path=str(bindir))}
    assert [d for d in found] == ORDER
    assert found["claude"]["detected"] and found["hermes"]["detected"] and found["codex"]["detected"]
    assert not found["gemini"]["detected"] and not found["opencode"]["detected"] and not found["cursor"]["detected"]
    assert found["claude"]["evidence"][0].startswith("binary ")


# ----- installer -----
def seed_home(root):
    for d in (".claude", ".codex", ".gemini", ".hermes", ".config/opencode", ".cursor"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / ".claude.json").write_text('{"numStartups": 3, "mcpServers": {"other": {"command": "x"}}}')
    (root / ".claude/settings.json").write_text('{"permissions": {"allow": ["Bash"]}, "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo hi"}]}]}}')
    (root / ".codex/config.toml").write_text('model = "gpt-5"\n\n[mcp_servers.other]\ncommand = "x"\n')
    (root / ".codex/AGENTS.md").write_text("# my notes\n")
    (root / ".gemini/settings.json").write_text('{"theme": "dark"}')
    (root / ".hermes/config.yaml").write_text("model:\n  provider: openrouter\nmcp_servers:\n  other:\n    command: x\n")
    (root / ".config/opencode/opencode.json").write_text('{"$schema": "https://opencode.ai/config.json", "mcp": {"other": {"type": "remote", "url": "u"}}}')


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file() and ".regesh" not in p.parts}


def test_installer_install_idempotent_uninstall_exact(tmp_path):
    seed_home(tmp_path); before = snapshot(tmp_path)
    inst = Installer(home=tmp_path / ".regesh", user_home=tmp_path, bin_dir=tmp_path / "nowhere")
    reports = [inst.install(h) for h in ORDER]
    assert all(r["changes"] for r in reports)
    # Claude Code
    cj = json.loads((tmp_path / ".claude.json").read_text()); assert cj["numStartups"] == 3 and cj["mcpServers"]["regesh"]["args"] == ["--harness", "claude"] and "other" in cj["mcpServers"]
    st = json.loads((tmp_path / ".claude/settings.json").read_text())
    assert st["permissions"] == {"allow": ["Bash"]} and len(st["hooks"]["Stop"]) == 2 and set(st["hooks"]) == {"Stop", "UserPromptSubmit", "PostToolUse", "PostToolUseFailure", "SessionStart"}
    assert all(h["timeout"] == 15 and " claude " in h["command"] for g in st["hooks"]["UserPromptSubmit"] for h in g["hooks"])
    assert MD_START in (tmp_path / ".claude/CLAUDE.md").read_text()
    # Codex
    toml = tomllib.loads((tmp_path / ".codex/config.toml").read_text()); assert toml["model"] == "gpt-5" and toml["mcp_servers"]["regesh"]["args"] == ["--harness", "codex"] and "other" in toml["mcp_servers"]
    hk = json.loads((tmp_path / ".codex/hooks.json").read_text()); assert set(hk["hooks"]) == {"UserPromptSubmit", "PostToolUse", "Stop", "SessionStart"}
    assert (tmp_path / ".codex/AGENTS.md").read_text().startswith("# my notes\n\n" + MD_START)
    # Gemini: ms timeouts, named hooks, no hookEventName requirement
    gs = json.loads((tmp_path / ".gemini/settings.json").read_text()); assert gs["theme"] == "dark" and gs["mcpServers"]["regesh"]["timeout"] == 10000
    assert gs["hooks"]["BeforeAgent"][0]["hooks"][0]["timeout"] == 15000 and gs["hooks"]["BeforeAgent"][0]["hooks"][0]["name"] == "regesh-BeforeAgent"
    # Hermes
    yml = (tmp_path / ".hermes/config.yaml").read_text(); assert "  regesh:  # regesh-install" in yml and "  other:\n    command: x" in yml
    init = (tmp_path / ".hermes/plugins/regesh/__init__.py").read_text(); assert "__REGESH_HOOK__" not in init and "def register(ctx)" in init
    assert "provides_hooks" in (tmp_path / ".hermes/plugins/regesh/plugin.yaml").read_text()
    # OpenCode
    oc = json.loads((tmp_path / ".config/opencode/opencode.json").read_text()); assert oc["mcp"]["regesh"]["type"] == "local" and oc["mcp"]["regesh"]["command"][-2:] == ["--harness", "opencode"] and "other" in oc["mcp"]
    js = (tmp_path / ".config/opencode/plugins/regesh.js").read_text(); assert "__REGESH_HOOK__" not in js and "tool.execute.after" in js
    # Cursor
    assert json.loads((tmp_path / ".cursor/mcp.json").read_text())["mcpServers"]["regesh"]["args"] == ["--harness", "cursor"]
    manifest = json.loads((tmp_path / ".regesh/install_manifest.json").read_text()); assert set(manifest["harnesses"]) == set(ORDER)
    assert all(f["written_sha256"] for h in manifest["harnesses"].values() for f in h["files"])
    # idempotent
    after_first = snapshot(tmp_path)
    reports2 = [inst.install(h) for h in ORDER]
    assert snapshot(tmp_path) == after_first
    assert all(c["action"] == "already configured" for r in reports2 for c in r["changes"] if c["action"] != "ran")
    # uninstall restores byte for byte when nothing else changed
    for h in ORDER:
        inst.uninstall(h)
    assert snapshot(tmp_path) == before
    assert not (tmp_path / ".cursor/mcp.json").exists() and not (tmp_path / ".hermes/plugins").exists() and not (tmp_path / ".config/opencode/plugins").exists()
    assert json.loads((tmp_path / ".regesh/install_manifest.json").read_text())["harnesses"] == {}


def test_installer_uninstall_preserves_user_edits(tmp_path):
    seed_home(tmp_path)
    inst = Installer(home=tmp_path / ".regesh", user_home=tmp_path, bin_dir=tmp_path / "nowhere")
    for h in ORDER:
        inst.install(h)
    cj = json.loads((tmp_path / ".claude.json").read_text()); cj["numStartups"] = 9; (tmp_path / ".claude.json").write_text(json.dumps(cj))
    (tmp_path / ".codex/AGENTS.md").write_text((tmp_path / ".codex/AGENTS.md").read_text() + "\nuser addition\n")
    (tmp_path / ".hermes/config.yaml").write_text((tmp_path / ".hermes/config.yaml").read_text() + "extra: 1\n")
    st = json.loads((tmp_path / ".claude/settings.json").read_text()); st["model"] = "opus"; (tmp_path / ".claude/settings.json").write_text(json.dumps(st))
    for h in ORDER:
        inst.uninstall(h)
    cj = json.loads((tmp_path / ".claude.json").read_text()); assert cj["numStartups"] == 9 and "regesh" not in cj["mcpServers"] and "other" in cj["mcpServers"]
    assert (tmp_path / ".codex/AGENTS.md").read_text() == "# my notes\nuser addition\n"
    assert (tmp_path / ".hermes/config.yaml").read_text() == "model:\n  provider: openrouter\nmcp_servers:\n  other:\n    command: x\nextra: 1\n"
    st = json.loads((tmp_path / ".claude/settings.json").read_text()); assert st["model"] == "opus" and st["hooks"] == {"Stop": [{"hooks": [{"type": "command", "command": "echo hi"}]}]}
    assert "regesh" not in (tmp_path / ".codex/config.toml").read_text()


def test_installer_refuses_unsafe_files(tmp_path):
    (tmp_path / ".codex").mkdir(); (tmp_path / ".codex/config.toml").write_text('model = "x\n')   # broken TOML
    (tmp_path / ".config/opencode").mkdir(parents=True); (tmp_path / ".config/opencode/opencode.jsonc").write_text("{ // comment\n}")
    (tmp_path / ".hermes").mkdir(); (tmp_path / ".hermes/config.yaml").write_text("mcp_servers: {a: b}\n")
    (tmp_path / ".claude").mkdir(); (tmp_path / ".claude/settings.json").write_text("{not json")
    inst = Installer(home=tmp_path / ".regesh", user_home=tmp_path, bin_dir=tmp_path / "nowhere")
    codex = inst.install("codex"); assert any("does not parse as TOML" in n for n in codex["notes"]) and (tmp_path / ".codex/config.toml").read_text() == 'model = "x\n'
    oc = inst.install("opencode"); assert any("JSONC" in n for n in oc["notes"]) and not (tmp_path / ".config/opencode/opencode.json").exists()
    hermes = inst.install("hermes"); assert any("cannot edit safely" in n for n in hermes["notes"]) and (tmp_path / ".hermes/config.yaml").read_text() == "mcp_servers: {a: b}\n"
    claude = inst.install("claude"); assert any("not valid JSON" in n for n in claude["notes"]) and (tmp_path / ".claude/settings.json").read_text() == "{not json"


def test_installer_dry_run_writes_nothing(tmp_path):
    seed_home(tmp_path); before = snapshot(tmp_path)
    inst = Installer(home=tmp_path / ".regesh", user_home=tmp_path, bin_dir=tmp_path / "nowhere", dry_run=True)
    reports = [inst.install(h) for h in ORDER]
    assert snapshot(tmp_path) == before and all(r["dry_run"] for r in reports) and not (tmp_path / ".regesh/install_manifest.json").exists()


def test_installer_falls_back_to_module_invocation(tmp_path):
    inst = Installer(home=tmp_path / ".regesh", user_home=tmp_path, bin_dir=tmp_path / "nowhere")
    inst.mcp_cmd = None; inst.hook_cmd = None
    cmd, args = inst.mcp_server("claude"); assert cmd == sys.executable and args[:2] == ["-m", "regesh.integrations.mcp_server"]
    assert "-m regesh.integrations.hooks claude Stop" in inst.hook_command("claude", "Stop")
    inst.install("hermes"); init = (tmp_path / ".hermes/plugins/regesh/__init__.py").read_text()
    assert '"-m", "regesh.integrations.hooks", "hermes", event' in init
    inst.install("opencode"); js = (tmp_path / ".config/opencode/plugins/regesh.js").read_text()
    assert '"-m", "regesh.integrations.hooks", "opencode", event' in js


@pytest.mark.parametrize("text", ["", "model: x\n", "model: x\nmcp_servers:\n  other:\n    command: a\n", "mcp_servers: {}\n", "mcp_servers:\n    deep:\n        command: a\n"])
def test_yaml_insert_remove_round_trip(text):
    new = yaml_insert_mcp_server(text, "regesh", "/x/regesh-mcp", ["--harness", "hermes"])
    assert "regesh:  # regesh-install" in new and 'command: "/x/regesh-mcp"' in new
    back = yaml_remove_tagged(new)
    assert back == text or (text == "mcp_servers: {}\n" and back == "")
    try:
        import yaml
    except ImportError:
        return
    doc = yaml.safe_load(new); assert doc["mcp_servers"]["regesh"]["args"] == ["--harness", "hermes"]


def test_cli_yes_list_uninstall(tmp_path, capsys):
    seed_home(tmp_path)
    args = ["--user-home", str(tmp_path), "--home", str(tmp_path / ".regesh"), "--bin-dir", str(tmp_path / "nowhere")]
    assert cli.main(args + ["--yes", "--json"]) == 0
    reports = json.loads(capsys.readouterr().out); assert {r["harness"] for r in reports} == set(ORDER)   # every seeded harness detected
    assert cli.main(args + ["--list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out); assert all(h["detected"] and h["installed"] for h in listed)
    assert cli.main(args + ["--uninstall", "--harness", "claude,cursor"]) == 0
    out = capsys.readouterr().out; assert "Claude Code" in out and "Cursor" in out
    assert cli.main(args + ["--uninstall"]) == 0
    assert json.loads((tmp_path / ".regesh/install_manifest.json").read_text())["harnesses"] == {}
    with pytest.raises(SystemExit):
        cli.main(args + ["--harness", "bogus"])


def test_cli_no_detection_without_selection(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "emptybin"))
    assert cli.main(["--user-home", str(tmp_path / "empty"), "--home", str(tmp_path / ".regesh"), "--yes"]) == 1
    assert "no supported agent harness detected" in capsys.readouterr().out

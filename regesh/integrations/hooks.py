"""Harness hook adapter: ``regesh-hook <harness> <event>`` reads the harness's JSON payload on
stdin, updates the persistent affect loop and prints whatever that harness accepts as
additional context. Any failure is logged to ~/.regesh/hook.log and exits 0 with no output,
so a broken affect layer can never break the agent.

Harness/event schemas (verified against each harness's published hook documentation):

claude    UserPromptSubmit {prompt}                      -> hookSpecificOutput.additionalContext
          PostToolUse {tool_name, tool_input, tool_response} -> hookSpecificOutput.additionalContext
          PostToolUseFailure {tool_name, tool_input, error}  -> (no output)
          Stop {last_assistant_message?}                 -> (no output)
          SessionStart {source}                          -> hookSpecificOutput.additionalContext
codex     UserPromptSubmit / PostToolUse / Stop / SessionStart  (same payloads as above)
gemini    BeforeAgent {prompt} -> hookSpecificOutput.additionalContext
          AfterTool {tool_name, tool_input, tool_response{llmContent,error}} -> hookSpecificOutput.additionalContext
          AfterAgent {prompt, prompt_response} -> (no output)
          SessionStart -> hookSpecificOutput.additionalContext
hermes    pre_llm_call {user_message} -> {"context": ...}      (called by the regesh Hermes plugin)
          post_tool_call {tool_name, args, result} -> (no output)
          post_llm_call {assistant_response} -> (no output)
          on_session_start -> (no output)
opencode  tool.execute.after {tool, args, output{title,output,metadata}} -> (no output)
          session.idle -> (no output)                     (called by the regesh OpenCode plugin)
generic   prompt {text} -> {"context": ...}; tool {tool_name, tool_input, tool_response, error, exit_code}; turn_end {text}
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

from .bridge import HOME, Bridge

CONTEXT_EVENTS = {
    "claude": {"UserPromptSubmit", "PostToolUse", "SessionStart"},
    "codex": {"UserPromptSubmit", "PostToolUse", "SessionStart"},
    "gemini": {"BeforeAgent", "AfterTool", "SessionStart"},
}
HOOK_EVENT_NAMES = {"claude": True, "codex": True, "gemini": False}  # whether hookSpecificOutput needs hookEventName


def read_payload(stream):
    raw = stream.read()
    if not raw or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return {"text": raw}
    return value if isinstance(value, dict) else {"text": raw}


def prompt_text(payload):
    for key in ("prompt", "user_message", "text", "message", "input"):
        value = payload.get(key)
        if isinstance(value, list):   # Hermes may pass multimodal content blocks
            value = " ".join(str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in value)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def dispatch(bridge, harness, event, payload):
    """Apply one harness event. Returns the context dict when the event should inject context, else None."""
    if harness in ("claude", "codex"):
        if event == "UserPromptSubmit":
            return bridge.observe_user_message(prompt_text(payload), session=payload.get("session_id"))
        if event == "PostToolUse":
            return bridge.observe_tool_result(payload.get("tool_name"), payload.get("tool_input"), payload.get("tool_response"))
        if event == "PostToolUseFailure":
            bridge.observe_tool_result(payload.get("tool_name"), payload.get("tool_input"), None, error=payload.get("error") or "tool failed"); return None
        if event == "Stop":
            bridge.turn_end(payload.get("last_assistant_message")); return None
        if event == "SessionStart":
            return bridge.session_start()
    elif harness == "gemini":
        if event == "BeforeAgent":
            return bridge.observe_user_message(prompt_text(payload), session=payload.get("session_id"))
        if event == "AfterTool":
            resp = payload.get("tool_response") or {}
            error = resp.get("error") if isinstance(resp, dict) else None
            return bridge.observe_tool_result(payload.get("tool_name"), payload.get("tool_input"), resp, error=error)
        if event == "AfterAgent":
            bridge.turn_end(payload.get("prompt_response")); return None
        if event == "SessionStart":
            return bridge.session_start()
    elif harness == "hermes":
        if event == "pre_llm_call":
            return bridge.observe_user_message(prompt_text(payload), session=payload.get("session_id"))
        if event == "post_tool_call":
            result = payload.get("result")
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except ValueError:
                    pass
            bridge.observe_tool_result(payload.get("tool_name"), payload.get("args"), result); return None
        if event == "post_llm_call":
            bridge.turn_end(payload.get("assistant_response")); return None
        if event == "on_session_start":
            bridge.session_start(); return None
    elif harness == "opencode":
        if event == "tool.execute.after":
            output = payload.get("output") or {}
            bridge.observe_tool_result(payload.get("tool"), payload.get("args"), output); return None
        if event == "session.idle":
            bridge.turn_end(None); return None
        if event == "message":
            return bridge.observe_user_message(prompt_text(payload))
    if event in ("prompt", "message", "user"):
        return bridge.observe_user_message(prompt_text(payload))
    if event in ("tool", "tool_result"):
        bridge.observe_tool_result(payload.get("tool_name") or payload.get("tool"), payload.get("tool_input") or payload.get("args"),
                                   payload.get("tool_response") or payload.get("output"), error=payload.get("error"), exit_code=payload.get("exit_code")); return None
    if event in ("turn_end", "stop"):
        bridge.turn_end(payload.get("text") or payload.get("last_assistant_message")); return None
    if event in ("session_start", "context"):
        return bridge.context()
    raise ValueError(f"unknown event {event!r} for harness {harness!r}")


def render(harness, event, ctx):
    """Format the context for the harness's accepted output schema; None means print nothing."""
    if ctx is None:
        return None
    line = ctx["line"]
    if harness in CONTEXT_EVENTS:
        if event not in CONTEXT_EVENTS[harness]:
            return None
        inner = {"additionalContext": line}
        if HOOK_EVENT_NAMES[harness]:
            inner["hookEventName"] = event
        return json.dumps({"hookSpecificOutput": inner})
    return json.dumps({"context": line})


def log_failure(home, harness, event, exc):
    try:
        home.mkdir(parents=True, exist_ok=True)
        with open(home / "hook.log", "a") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat()} {harness} {event} {type(exc).__name__}: {exc}\n")
            fh.write("".join(traceback.format_exception(exc))[-2000:] + "\n")
    except OSError:
        pass


def run(argv, stdin=None, stdout=None):
    stdin = stdin or sys.stdin; stdout = stdout or sys.stdout
    home = Path(os.environ.get("REGESH_HOME", str(HOME)))
    if len(argv) < 2:
        return 0
    harness, event = argv[0], argv[1]
    try:
        payload = read_payload(stdin)
        bridge = Bridge(harness, home=home)
        try:
            ctx = dispatch(bridge, harness, event, payload)
        finally:
            bridge.close()
        text = render(harness, event, ctx)
        if text:
            stdout.write(text + "\n"); stdout.flush()
    except Exception as exc:  # fail open: the harness must never be blocked by the affect layer
        log_failure(home, harness, event, exc)
    return 0


def main():
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()

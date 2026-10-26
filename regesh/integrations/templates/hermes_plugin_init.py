"""regesh affect layer for Hermes Agent. Each hook shells out to `regesh-hook hermes <event>`
with the hook payload as JSON on stdin; the hook fails open, so a missing or broken regesh
install never blocks a turn. Installed by `regesh-install`; remove with `regesh-install --uninstall`."""
import json
import subprocess

REGESH_HOOK = "__REGESH_HOOK__"
TIMEOUT = 10


def _call(event, payload):
    try:
        proc = subprocess.run([REGESH_HOOK, "hermes", event], input=json.dumps(payload, default=str),
                              capture_output=True, text=True, timeout=TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


def pre_llm_call(session_id, user_message, conversation_history=None, is_first_turn=False, model=None, platform=None, **kwargs):
    result = _call("pre_llm_call", {"session_id": session_id, "user_message": user_message, "is_first_turn": is_first_turn,
                                    "model": model, "platform": platform})
    if result and result.get("context"):
        return {"context": result["context"]}
    return None


def post_tool_call(tool_name, args, result, task_id=None, duration_ms=None, **kwargs):
    _call("post_tool_call", {"tool_name": tool_name, "args": args, "result": result, "task_id": task_id, "duration_ms": duration_ms})


def post_llm_call(session_id, user_message, assistant_response, conversation_history=None, model=None, platform=None, **kwargs):
    _call("post_llm_call", {"session_id": session_id, "assistant_response": assistant_response})


def on_session_start(session_id, model=None, platform=None, **kwargs):
    _call("on_session_start", {"session_id": session_id, "model": model, "platform": platform})


def register(ctx):
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("post_llm_call", post_llm_call)
    ctx.register_hook("on_session_start", on_session_start)

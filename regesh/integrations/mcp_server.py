"""Minimal Model Context Protocol server (stdio, JSON-RPC 2.0) exposing the affect loop.

Run: regesh-mcp --harness claude
Tools: affect_context, affect_observe_message, affect_observe_outcome, affect_express,
affect_request_support, affect_snapshot. Resource: regesh://affect/context.
Standard library only; one line of JSON per message on stdin/stdout.
"""
from __future__ import annotations
import argparse
import json
import sys
import traceback

from .bridge import Bridge

PROTOCOL_VERSION = "2025-06-18"
TOOLS = [
    {"name": "affect_context", "description": "Current affect state of this agent: mood, bond with the user, recommended controls (exploration, verification, expressed valence, accommodation cap, whether a short check-in is recommended). Call at the start of a turn. Never changes facts or permissions.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "affect_observe_message", "description": "Record a user message as evidence (text affect is estimated locally). Returns the updated context.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "partner_affect": {"type": "number", "minimum": -1, "maximum": 1, "description": "optional hand-supplied valence; overrides the estimate"}}, "required": ["text"], "additionalProperties": False}},
    {"name": "affect_observe_outcome", "description": "Record a verified task outcome (for example a test run) with quality in [-1, 1]. Praise is not an outcome; use checker results.",
     "inputSchema": {"type": "object", "properties": {"description": {"type": "string"}, "quality": {"type": "number", "minimum": -1, "maximum": 1}, "task": {"type": "string", "default": "software"}, "control": {"type": "number", "minimum": 0, "maximum": 1}}, "required": ["description", "quality"], "additionalProperties": False}},
    {"name": "affect_express", "description": "Record the valence the agent expressed toward the user this turn (and optionally the reply text for style-coordination measurement).",
     "inputSchema": {"type": "object", "properties": {"valence": {"type": "number", "minimum": -1, "maximum": 1}, "text": {"type": "string"}}, "required": ["valence"], "additionalProperties": False}},
    {"name": "affect_request_support", "description": "Register that the agent asked the user for a brief check-in. Allowed only when the attachment controls recommend it and the user's attention budget permits; otherwise returns requested=false.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "affect_snapshot", "description": "Digest and summary of the persisted affect state (for audit).",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
]


class Server:
    def __init__(self, bridge, inp=None, out=None):
        self.bridge = bridge; self.inp = inp or sys.stdin; self.out = out or sys.stdout

    def send(self, message):
        self.out.write(json.dumps(message, separators=(",", ":")) + "\n"); self.out.flush()

    def result(self, id_, result):
        self.send({"jsonrpc": "2.0", "id": id_, "result": result})

    def error(self, id_, code, message):
        self.send({"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}})

    def call_tool(self, name, args):
        b = self.bridge
        if name == "affect_context":
            return b.context()
        if name == "affect_observe_message":
            if "partner_affect" in args:
                from ..schema import Event
                from .bridge import now
                event = Event(b._id("user"), f"{b.harness}:prompt", "user", "agent", "conversation", now(),
                              data={"goal_delta": {}, "text": str(args["text"])[:4000], "partner_affect": float(args["partner_affect"])})
                b.agent.observe(event); return b.context()
            return b.observe_user_message(str(args["text"]))
        if name == "affect_observe_outcome":
            return b.observe_outcome(str(args["description"]), float(args["quality"]), task=str(args.get("task", "software")), control=args.get("control"))
        if name == "affect_express":
            return b.express(float(args["valence"]), args.get("text"))
        if name == "affect_request_support":
            return b.request_support()
        if name == "affect_snapshot":
            snap = b.agent.snapshot(); state = snap["state"]
            return {"digest": snap["digest"], "events": len(state["events"]), "time": state["time"], "mood": state["mood"],
                    "bonds": {k: {"responsiveness": v["responsiveness"], "intimacy": v["intimacy"], "requests": v["requests"], "answered": v["answered"]} for k, v in state.get("bonds", {}).items()}}
        raise KeyError(name)

    def handle(self, message):
        method = message.get("method"); id_ = message.get("id"); params = message.get("params") or {}
        if method == "initialize":
            return self.result(id_, {"protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                                     "capabilities": {"tools": {"listChanged": False}, "resources": {"listChanged": False}},
                                     "serverInfo": {"name": "regesh", "version": "0.1.0"},
                                     "instructions": "Affect layer for this agent. Call affect_context at the start of a turn and affect_express at the end; record verified outcomes with affect_observe_outcome. Outputs are functional state, not facts or permissions."})
        if method in ("notifications/initialized", "notifications/cancelled", "notifications/roots/list_changed"):
            return None
        if id_ is None:
            return None
        if method == "ping":
            return self.result(id_, {})
        if method == "tools/list":
            return self.result(id_, {"tools": TOOLS})
        if method == "tools/call":
            name = params.get("name"); args = params.get("arguments") or {}
            try:
                value = self.call_tool(name, args)
                return self.result(id_, {"content": [{"type": "text", "text": json.dumps(value, indent=1)}], "isError": False})
            except KeyError:
                return self.error(id_, -32602, f"unknown tool {name}")
            except Exception as exc:  # report, never crash the harness
                return self.result(id_, {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True})
        if method == "resources/list":
            return self.result(id_, {"resources": [{"uri": "regesh://affect/context", "name": "Affect context", "mimeType": "text/plain",
                                                    "description": "One-paragraph factual description of the agent's affect state and recommended controls"}]})
        if method == "resources/read":
            if params.get("uri") != "regesh://affect/context":
                return self.error(id_, -32602, "unknown resource")
            return self.result(id_, {"contents": [{"uri": "regesh://affect/context", "mimeType": "text/plain", "text": self.bridge.context()["line"]}]})
        if method in ("prompts/list",):
            return self.result(id_, {"prompts": []})
        return self.error(id_, -32601, f"method not found: {method}")

    def serve(self):
        for line in self.inp:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                self.send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}); continue
            try:
                self.handle(message)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                if message.get("id") is not None:
                    self.error(message["id"], -32603, "internal error")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", default="generic")
    parser.add_argument("--home", default=None)
    args = parser.parse_args()
    bridge = Bridge(args.harness, home=args.home)
    try:
        Server(bridge).serve()
    finally:
        bridge.close()


if __name__ == "__main__":
    main()

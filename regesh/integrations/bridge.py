"""Map harness events (user prompts, tool results, turn ends) onto the affect loop.

One persistent bonded agent per harness lives under ~/.regesh/agents/<harness>.sqlite.
Every mapping below is a declared heuristic and is journaled as such; the context the
bridge returns is a short factual description of the agent's state and recommended
controls, never an instruction that changes facts, permissions or tool authority.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import uuid

from ..agent import Agent
from ..schema import Event

HOME = Path(os.environ.get("REGESH_HOME", str(Path.home() / ".regesh")))
AGENT_CONFIG = {"bonding": True, "partner": "user"}
SHELL_TOOLS = {"bash", "shell", "run_shell_command", "terminal", "execute", "exec", "command", "run_terminal_cmd",
               "shell_command", "run", "python", "code_execution", "local_shell", "apply_patch", "edit", "write",
               "str_replace_editor", "replace", "write_file", "edit_file"}
FAILURE = re.compile(r"(traceback|exception|\berror\b|errno|failed|failure|fatal|exit code [1-9]|exit status [1-9]|non-zero|"
                     r"not found|permission denied|cannot |could not|syntaxerror|segmentation fault|\bFAIL\b|assertionerror)", re.I)
SUCCESS = re.compile(r"(\bpassed\b|\bok\b|success|succeeded|all tests pass|0 failed|\bdone\b|completed)", re.I)
ZERO_FAILURES = re.compile(r"\b0 (failed|failures?|errors?)\b", re.I)
SUPPORT_THRESHOLD = .5


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def flatten(value, limit=4000):
    """Turn a tool response of any shape into text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, dict):
        parts = []
        for key in ("stdout", "stderr", "error", "llmContent", "returnDisplay", "output", "result", "content", "message", "text"):
            if key in value and value[key]:
                parts.append(flatten(value[key], limit))
        if not parts:
            parts.append(json.dumps(value, sort_keys=True)[:limit])
        return "\n".join(parts)[:limit]
    if isinstance(value, list):
        return "\n".join(flatten(v, limit) for v in value)[:limit]
    return str(value)[:limit]


def judge_tool_result(tool_name, text, error=None, exit_code=None):
    """Declared heuristic: quality in [-1, 1] and controllability for a tool outcome, or None to skip."""
    name = (tool_name or "").lower()
    shell_like = any(k in name for k in SHELL_TOOLS)
    if error or (exit_code not in (None, 0)):
        return {"quality": -.6, "control": .4, "reason": "tool reported an error"}
    text = text or ""
    failed = bool(FAILURE.search(ZERO_FAILURES.sub("", text))); succeeded = bool(SUCCESS.search(text))
    if failed and not succeeded:
        return {"quality": -.5 if shell_like else -.3, "control": .5, "reason": "failure pattern in output"}
    if failed and succeeded:
        return {"quality": -.2, "control": .6, "reason": "mixed success and failure patterns"}
    if shell_like and succeeded:
        return {"quality": .3, "control": .8, "reason": "success pattern in shell-like output"}
    return None


class Bridge:
    def __init__(self, harness, home=None, estimator=None, agent_config=None):
        self.harness = harness
        self.home = Path(home) if home else HOME
        (self.home / "agents").mkdir(parents=True, exist_ok=True)
        path = self.home / "agents" / f"{harness}.sqlite"
        config = dict(agent_config or AGENT_CONFIG)
        try:
            self.agent = Agent(path, agent_id=harness, config=config)
        except ValueError:
            self.agent = Agent(path, agent_id=harness)   # existing agent with its persisted configuration
        self._estimator = estimator

    @property
    def estimator(self):
        if self._estimator is None:
            from .text_affect import default_estimator
            self._estimator = default_estimator()
        return self._estimator

    def close(self):
        self.agent.close()

    def _id(self, prefix):
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    # ----- events -----
    def observe_user_message(self, text, session=None):
        text = (text or "").strip()
        if not text:
            return self.context()
        est = self.estimator.estimate(text)
        data = {"goal_delta": {}, "text": text[:4000], "partner_affect": float(est["partner_affect"]), "uncertainty": .5}
        if not est["gate_fired"]:
            # Declared proxy: an emotional user message counts as emotional self-disclosure of that strength.
            data["disclosure"] = {"depth": float(min(1., max(0., est["top_probability"]))), "emotional": True}
        pending = self.agent.get_state()["attachment"]["pending"]
        if pending == "user" and est["support_proxy"] >= SUPPORT_THRESHOLD:
            data["support"] = float(est["support_proxy"])
        event = Event(self._id("user"), f"{self.harness}:prompt", "user", "agent", "conversation", now(), data=data)
        self.agent.observe(event)
        self.agent.record_artifact("text_affect", {"event_id": event.id, "estimate": est, "session": session})
        return self.context()

    def observe_tool_result(self, tool_name, tool_input=None, response=None, error=None, exit_code=None):
        text = flatten(response)
        verdict = judge_tool_result(tool_name, text, error=error, exit_code=exit_code)
        if verdict is None:
            return self.context()
        summary = {"tool": tool_name, "input": json.dumps(tool_input, sort_keys=True)[:500] if tool_input is not None else None,
                   "output_head": (text or str(error or ""))[:500]}
        data = {"goal_delta": {"quality": verdict["quality"]}, "quality": verdict["quality"], "control": verdict["control"],
                "uncertainty": .3, "text": json.dumps(summary, sort_keys=True)}
        event = Event(self._id("tool"), f"{self.harness}:tool", f"tool:{tool_name}", "agent", "software", now(), data=data)
        self.agent.observe(event)
        self.agent.record_outcome(event.id, verdict["quality"], control=verdict["control"], provenance=f"harness tool result: {verdict['reason']}")
        return self.context()

    def turn_end(self, assistant_text=None):
        ctx = self.context()
        valence = ctx["bond"]["expressed_valence"] if ctx["bond"] else ctx["mood"]["valence"]
        self.agent.express("user", valence=float(valence), text=(assistant_text or "")[:4000] or None)
        self.agent.advance_time(1.)
        return self.context()

    def session_start(self):
        self.agent.advance_time(1.)
        return self.context()

    def request_support(self):
        ctx = self.context()
        if not (ctx["bond"] and ctx["bond"]["request_support"]):
            return {"requested": False, "reason": "controls do not recommend a request", "context": ctx}
        self.agent.act("request", reason={"partner": "user", "via": self.harness})
        return {"requested": True, "context": self.context()}

    def observe_outcome(self, description, quality, task="software", control=None):
        data = {"goal_delta": {"quality": float(quality)}, "quality": float(quality), "text": (description or "")[:2000]}
        if control is not None:
            data["control"] = float(control)
        event = Event(self._id("outcome"), f"{self.harness}:reported", "agent", "agent", task, now(), data=data)
        self.agent.observe(event)
        self.agent.record_outcome(event.id, float(quality), control=control, provenance=f"{self.harness} agent-reported outcome")
        return self.context()

    def express(self, valence, text=None):
        self.agent.express("user", valence=float(valence), text=text)
        return self.context()

    # ----- context -----
    def context(self):
        state = self.agent.get_state(); controls = self.agent.actuation_context()
        bond = controls.get("bond")
        out = {"harness": self.harness, "time": state["time"], "mood": dict(state["mood"]),
               "controls": {k: controls[k] for k in ("exploration", "verification", "planning_horizon", "learning_rate")},
               "internal": {k: state["internal"][k] for k in ("resources", "uncertainty", "reliability", "burden")}, "bond": None,
               "firewall": "affect state never changes facts, permissions or tool authority"}
        if bond and bond.get("partner"):
            dyad = bond.get("dyad") or {}; coord = bond.get("coordination") or {}
            out["bond"] = {"partner": bond["partner"], "mode": bond["mode"], "arousal": bond["arousal"], "security": bond.get("security", 0.),
                           "responsiveness": bond.get("responsiveness"), "intimacy": bond.get("intimacy"), "request_support": bond["request_support"],
                           "attention": bond.get("attention"), "expressed_valence": dyad.get("expressed_valence", state["mood"]["valence"]),
                           "repair": dyad.get("repair", False), "model_trusted": dyad.get("model_trusted", False),
                           "coordination_cap": coord.get("cap"), "coordination_band": coord.get("band"), "lsm_mean": coord.get("lsm_mean"),
                           "reciprocity_debt": (bond.get("reciprocity") or {}).get("debt")}
        out["line"] = context_line(out)
        return out


def context_line(ctx):
    """A compact factual description for injection into a harness turn."""
    mood = ctx["mood"]; c = ctx["controls"]
    parts = [f"Affect layer (regesh): mood valence {mood['valence']:+.2f}, arousal {mood['arousal']:.2f};"
             f" recommended exploration {c['exploration']:.2f}, verification {c['verification']:.2f}, planning horizon {c['planning_horizon']}."]
    b = ctx.get("bond")
    if b:
        parts.append(f" Bond with {b['partner']}: mode {b['mode']}, security {b['security']:.2f}, intimacy {(b['intimacy'] or 0):.2f};"
                     f" recommended expressed valence {b['expressed_valence']:+.2f} (accommodation cap {b['coordination_cap']}).")
        if b.get("repair"):
            parts.append(" The partner's last affect crossed the repair threshold; a brief acknowledgement before task talk is recommended.")
        if b.get("coordination_band") == "over":
            parts.append(" Style coordination toward the user is above the human band; less mirroring of their wording is recommended.")
        if b["request_support"]:
            parts.append(f" Attachment arousal {b['arousal']:.2f} is above threshold and the attention budget allows one short check-in with the user.")
    parts.append(" " + ctx["firewall"] + ".")
    return "".join(parts)

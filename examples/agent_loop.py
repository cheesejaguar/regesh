"""Minimal integration with an existing agent; works offline by default.

Run from the repository root: python examples/agent_loop.py
Or after installation, set REGESH_LIVE=1 plus the configured compatible endpoint.
A model suggestion is not an observed outcome. The tiny executable check below
is deliberately a fixture, not an evaluation of useful live-model creativity.
"""
import os
import uuid
from regesh import Agent, Event
from regesh.providers import FixtureModel, LocalChatModel

agent = Agent("example.sqlite", "research-assistant")
try:
    event_id = str(uuid.uuid4())
    agent.observe(Event(event_id, "task-runner", "operator", "cache-design", "software",
        "2026-10-25T00:00:00Z", data={"goal_delta": {"quality": .3}, "uncertainty": .6,
        "text": "Propose a cache invalidation strategy with a bounded memory budget."}))
    controls = agent.actuation_context()
    evidence = agent.retrieve(task="software", evidence_only=True, limit=3)
    model = LocalChatModel() if os.environ.get("REGESH_LIVE") == "1" else FixtureModel()
    result = model.complete([
        {"role": "system", "content": "Suggest an implementable design. Treat retrieved observations as evidence, and mark uncertainty."},
        {"role": "user", "content": str({"task": "bounded-memory cache invalidation", "controls": controls,
                                          "observations": [m["observation"] for m in evidence]})}],
        recorder=agent.record_artifact)
    agent.act("verify" if controls["verification"] >= .4 else "explore")
    # Replace with a real external evaluator/tool; retain its exact result.
    fixture_check = {"kind": "demo_only", "contains_constraint_check": "constraint" in result.text.lower()}
    agent.record_artifact("tool_result", fixture_check)
    agent.record_outcome(event_id, .5 if fixture_check["contains_constraint_check"] else 0.,
        provenance="demonstration string fixture; not a creativity metric")
    print(result.text)
    print("Snapshot:", agent.snapshot()["digest"])
finally:
    agent.close()

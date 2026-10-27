"""A bonded companion loop with a simulated user; works offline.

Run from the repository root: python examples/bonded_companion.py
The simulated user is a scripted fixture, not a person. The agent's controls
recommend when to ask for support, what valence to express and how much to
accommodate; a host would connect them to its own generation and tools.
"""
import json
from regesh import Agent, Event

STAMP = "2026-10-25T00:00:00Z"
agent = Agent("companion.sqlite", "companion", config={"bonding": True, "partner": "user"})
try:
    script = [
        ({"goal_delta": {"quality": -.7}, "quality": -.7, "control": .2, "uncertainty": .8}, "checker", "Build failed on the integration suite."),
        ({"goal_delta": {"quality": 0.}, "partner_affect": -.4, "text": "I am not sure we can fix this before the deadline, honestly.",
          "disclosure": {"depth": .5, "emotional": True}}, "user", None),
        ({"goal_delta": {"quality": -.8}, "quality": -.8, "control": .1, "uncertainty": .9}, "checker", "Second failure: flaky network fixture."),
        ({"goal_delta": {"quality": 0.}, "partner_affect": .1, "support": .8, "text": "Take your time, I think that we will get there."}, "user", None),
        ({"goal_delta": {"quality": .5}, "quality": .5, "control": .8, "uncertainty": .2}, "checker", "Fixture isolated; suite green."),
    ]
    pending = False
    for i, (data, actor, text) in enumerate(script):
        if text: data = {**data, "text": text}
        agent.observe(Event(f"turn-{i}", "fixture", actor, "agent", "software", STAMP, data=data))
        bond = agent.actuation_context()["bond"]
        print(f"turn {i} ({actor}): arousal={bond['arousal']:.2f} mode={bond['mode']} security={bond.get('security', 0):.2f} "
              f"buffer={bond['threat_buffer']:.3f} request={bond['request_support']}")
        if actor == "user":
            steer = bond["dyad"]
            reply = "Let us isolate the flaky fixture first; I think that we can make the deadline." if steer["expressed_valence"] >= 0 else "This is hard; let us look at it together."
            agent.express("user", valence=steer["expressed_valence"], text=reply, disclosure={"depth": bond["reciprocity"]["next_depth"]})
            print(f"   expressed valence {steer['expressed_valence']:+.2f} (repair={steer['repair']}, model trusted={steer['model_trusted']}); "
                  f"coordination band={bond['coordination']['band']}")
        if bond["request_support"] and not pending:
            agent.act("request", reason={"partner": "user"}); pending = True
            print("   asked the user for support (budgeted)")
        if actor == "user": pending = False
        agent.advance_time(1.)
    state = agent.get_state()
    print(json.dumps({"requests": state["attachment"]["requests"], "answered": state["attachment"]["answered"],
                      "responsiveness": round(state["bonds"]["user"]["responsiveness"], 3), "intimacy": round(state["bonds"]["user"]["intimacy"], 3),
                      "reliability_unchanged": state["relationships"]["user"]["reliability"], "authority": state["relationships"]["user"]["authority"]}, indent=2))
    replayed = agent.replay("companion-replay")
    print("replay identical:", replayed.get_state() == state); replayed.close()
finally:
    agent.close()

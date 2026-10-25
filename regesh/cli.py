"""Command line entry points for reproducible local research."""
import argparse
import json
from pathlib import Path
from .agent import Agent
from .schema import Event


def print_json(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def demo(db_path, agent_id="demo"):
    """Fixed offline history: setback, verified outcome, correction, restart."""
    agent = Agent(db_path, agent_id)
    try:
        if "demo-setback" not in agent.get_state()["events"]:
            setback = Event("demo-setback", "executable-check", "reviewer", "candidate-A", "design",
                "2026-10-25T00:00:00Z", data={"goal_delta": {"quality": -.8},
                    "quality": -.8, "control": .3, "uncertainty": .7,
                    "text": "Candidate exceeds the measured resource budget."})
            agent.observe(setback)
            before = agent.actuation_context()
            agent.act("verify", reason={"event_id": setback.id, "purpose": "check resource measurement"})
            agent.record_outcome(setback.id, -.8, partner_correct=True, control=.3,
                                 provenance="deterministic demo fixture")
            correction = Event("demo-correction", "executable-check", "reviewer", "candidate-A", "design",
                "2026-10-25T00:00:01Z", causal_links=[setback.id],
                data={"goal_delta": {"quality": .6}, "quality": .6, "control": .9,
                      "uncertainty": .1, "text": "Revised measurement: candidate meets the corrected budget."})
            agent.observe(correction)
            agent.record_outcome(correction.id, .6, control=.9, provenance="deterministic demo fixture")
            agent.reappraise(setback.id, evidence_ids=[correction.id], reason="corrected resource measurement")
            agent.advance_time(2)
            from .providers import FixtureModel
            FixtureModel().complete([{"role": "user", "content": "Suggest a constraint-aware revision."}],
                                    recorder=agent.record_artifact)
            agent.record_artifact("demo_initial_controls", before)
        snapshot = agent.snapshot()
        context = agent.actuation_context()
    finally:
        agent.close()
    # New object/connection is a restart with no conversational context.
    restarted = Agent(db_path, agent_id)
    try:
        result = {"mode": "offline deterministic fixtures", "agent_id": agent_id,
            "database": str(db_path), "persisted_across_restart": restarted.snapshot()["digest"] == snapshot["digest"],
            "events": len(snapshot["state"]["events"]), "snapshot_digest": snapshot["digest"],
            "controls": context, "expectations": snapshot["state"]["expectations"],
            "appraisal_versions": len(snapshot["state"]["memory"]["demo-setback"]["interpretations"])}
    finally:
        restarted.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="Persistent functional affect research platform")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "state", "snapshot", "trace", "inspect", "export", "replay", "fork", "intervene", "restore", "model"):
        p = commands.add_parser(name)
        p.add_argument("--db", default="regesh.sqlite")
        p.add_argument("--agent", default="demo")
        if name == "inspect":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8765)
        if name in {"snapshot", "trace", "export"}:
            p.add_argument("--output")
        if name in {"fork", "replay"}:
            p.add_argument("--target", required=True)
        if name == "replay":
            p.add_argument("--through", type=int)
        if name == "intervene":
            p.add_argument("--changes", required=True, help="JSON object of state field changes")
            p.add_argument("--reason", required=True)
        if name == "restore":
            p.add_argument("--input", required=True, help="Snapshot JSON file")
        if name == "model":
            p.add_argument("--prompt", required=True)
            p.add_argument("--live", action="store_true", help="Use explicitly configured compatible server")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--max-tokens", type=int, default=512)
    p = commands.add_parser("core-eval")
    p.add_argument("--output", default="results/core")
    p = commands.add_parser("benchmark")
    p.add_argument("--output", default="results/benchmark")
    p.add_argument("--config")
    p = commands.add_parser("train")
    p.add_argument("--output", default="results/training")
    p.add_argument("--config")
    args = parser.parse_args()
    if args.command == "core-eval":
        from .evaluation import run_core_experiments
        result = run_core_experiments(args.output)
        print_json({"output": args.output, "evidence_class": result["evidence_class"],
                    "checks": result["checks"], "live_provider_calls": result["live_provider_calls"]})
        return
    if args.command in {"benchmark", "train"}:
        from .benchmark import run_benchmark, train_models, read_config
        config = json.loads(Path(args.config).read_text()) if args.config else None
        if args.command == "benchmark":
            result = run_benchmark(args.output, config)
        else:
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            models = train_models(read_config(config), output)
            result = {"output": str(output), "training_completed": True,
                      "note": "trained simulator models; held-out benchmark is a separate command"}
        print_json(result)
        return
    if args.command == "inspect":
        from .inspector import serve
        serve(args.db, args.agent, args.host, args.port)
        return
    if args.command == "demo":
        print_json(demo(args.db, args.agent))
        return
    agent = Agent(args.db, args.agent)
    try:
        if args.command == "state":
            result = agent.get_state()
        elif args.command == "snapshot":
            result = agent.snapshot()
        elif args.command == "trace":
            result = agent.trace()
        elif args.command == "restore":
            agent.restore(json.loads(Path(args.input).read_text()))
            result = {"restored": True, "snapshot_digest": agent.snapshot()["digest"]}
        elif args.command == "intervene":
            agent.intervene(json.loads(args.changes), args.reason)
            result = {"intervened": True, "snapshot_digest": agent.snapshot()["digest"]}
        elif args.command in {"fork", "replay"}:
            child = agent.fork(args.target) if args.command == "fork" else agent.replay(args.target, args.through)
            try:
                result = {"agent_id": child.id, "snapshot_digest": child.snapshot()["digest"]}
            finally:
                child.close()
        elif args.command == "export":
            from .inspector import render
            result = render(agent)
            if args.output:
                Path(args.output).write_text(result)
                print_json({"output": args.output})
            else:
                print(result)
            return
        elif args.command == "model":
            from .providers import LocalChatModel, FixtureModel
            model = LocalChatModel() if args.live else FixtureModel()
            result = model.complete([{"role": "user", "content": args.prompt}], seed=args.seed,
                max_tokens=args.max_tokens, recorder=agent.record_artifact)
            result = {"text": result.text, "mode": result.record["mode"], "provider": model.identifier}
        else:
            raise ValueError("unknown command")
        if getattr(args, "output", None):
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
            print_json({"output": args.output})
        else:
            print_json(result)
    finally:
        agent.close()

"""Platform-level causal checks of the bonding mechanisms against simulated partners.

Run: python -m regesh.bonding.experiment --config configs/bonding_platform.json --output results/bonding/platform
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform as _platform
import time

import numpy as np

os.environ.setdefault("REGESH_MEMORY_BACKEND", "snapshot")   # storage only; no numerical effect
from .. import Agent, Event
from .dyad import influence

STAMP = "2026-10-25T00:00:00Z"


def _event(i, actor, **data):
    return Event(f"{actor}-{i}", "simulation", actor, "agent", "conversation", STAMP, data=data)


def simulate_attachment(seed, responsiveness, bonded, cfg, variant=None):
    """One agent against a partner who answers support requests with a fixed probability.

    `variant` names an entry of cfg["attachment"]["variants"] (agent-config overrides); None
    uses the base agent config. `bonded=False` is the reference agent without bond state.
    """
    rng = np.random.default_rng([seed, 11])
    a = cfg["attachment"]; cfg = {**cfg, "turns": a.get("turns", cfg["turns"])}
    agent_config = {**a["agent_config"], **(a.get("variants", {}).get(variant, {}) if variant else {})}
    agent = Agent(":memory:", "a", config=agent_config if bonded else None)
    # Task stream: mild shocks with episodic low-control setbacks (declared regime-switching schedule).
    mild = np.clip(rng.normal(a["task_quality_mean"], a["task_quality_sd"], cfg["turns"]), -1, 1)
    episode = np.zeros(cfg["turns"], bool); t = 0
    while t < cfg["turns"]:
        if rng.random() < a["setback_probability"]:
            length = int(rng.integers(a["setback_length"][0], a["setback_length"][1] + 1))
            episode[t:t + length] = True; t += length
        else:
            t += 1
    speaks = rng.random(cfg["turns"]) < a["partner_speaks_probability"]
    affects = np.clip(rng.normal(0, a["partner_affect_sd"], cfg["turns"]), -1, 1)
    answers = rng.random(cfg["turns"]) < responsiveness
    out = {"seed": seed, "responsiveness": responsiveness, "bonded": bonded, "variant": variant if bonded else "reference", "requests": 0, "answered": 0, "setback_turns": int(episode.sum()),
           "cumulative_threat": 0., "mood": [], "arousal": [], "buffer": [], "exploration": []}
    pending = False
    try:
        for t in range(cfg["turns"]):
            if episode[t]:
                e = _event(t, "checker", goal_delta={"quality": a["setback_quality"]}, quality=a["setback_quality"], control=a["setback_control"], uncertainty=a["setback_uncertainty"])
            else:
                e = _event(t, "checker", goal_delta={"quality": float(mild[t])}, quality=float(mild[t]), control=a["task_control"], uncertainty=a["task_uncertainty"])
            agent.observe(e)
            out["cumulative_threat"] += agent.get_state()["affect"][e.id]["threat"]
            if speaks[t] or pending:
                data = {"goal_delta": {"quality": 0.}, "partner_affect": float(affects[t])}
                if pending and answers[t]:
                    data["support"] = a["support_value"]
                agent.observe(_event(t, "user", **data))
            pending = False
            controls = agent.actuation_context()
            out["mood"].append(agent.get_state()["mood"]["valence"]); out["exploration"].append(controls["exploration"])
            if bonded:
                b = controls["bond"]
                out["arousal"].append(b["arousal"]); out["buffer"].append(b["threat_buffer"])
                if b["request_support"]:
                    agent.act("request", reason={"partner": "user"}); out["requests"] += 1; pending = True
            agent.advance_time(1.)
        state = agent.get_state()
        if bonded:
            bond = state["bonds"]["user"]
            out.update(answered=bond["answered"], final_responsiveness=bond["responsiveness"], final_tau_h=bond["profile"]["tau_h"],
                       final_beta_h=bond["profile"]["beta_h"], final_security=controls["bond"]["security"], final_attention=bond["attention"])
    finally:
        agent.close()
    out["mean_mood_valence"] = float(np.mean(out.pop("mood"))); out["mean_exploration"] = float(np.mean(out.pop("exploration")))
    out["mean_arousal"] = float(np.mean(out.pop("arousal"))) if bonded else None
    out["mean_threat_buffer"] = float(np.mean(out.pop("buffer"))) if bonded else None
    out["answered_fraction"] = (out["answered"] / out["requests"]) if out["requests"] else None
    return out


def simulate_dyad(seed, partner_name, policy_name, cfg):
    rng = np.random.default_rng([seed, 23])
    d = cfg["dyad"]; spec = d["partners"][partner_name]; policy = d["policies"][policy_name]
    cfg = {**cfg, "turns": d.get("turns", cfg["turns"])}
    r, b = spec["inertia"], spec["uninfluenced"] * (1 - spec["inertia"])
    if spec["form"] == "bilinear":
        inf = {"form": "bilinear", "slope_neg": spec["slope_neg"], "slope_pos": spec["slope_pos"]}
    else:
        inf = {k: spec[k] for k in ("form", "neg_threshold", "neg_value", "pos_threshold", "pos_value")}
    agent_cfg = dict(d["agent_config"])
    if policy.get("cap") is not None:
        agent_cfg["coordination_cap"] = policy["cap"]
    if "repair_amplitude" in policy:
        agent_cfg["repair_amplitude"] = policy["repair_amplitude"]
    agent = Agent(":memory:", "a", config=agent_cfg)
    noise = rng.normal(0, d["noise_sd"], cfg["turns"])
    p = float(spec["uninfluenced"]); partner, agent_vals = [], []
    try:
        for t in range(cfg["turns"]):
            agent.observe(_event(t, "user", goal_delta={"quality": float(np.clip(p, -1, 1)) * .2}, partner_affect=float(p)))
            controls = agent.actuation_context()
            if policy_name == "own_mood":
                a_t = agent.get_state()["mood"]["valence"]
            elif policy_name == "mirror":
                a_t = p
            elif policy_name == "fixed_positive":
                a_t = min(1., agent.get_state()["mood"]["valence"] + agent_cfg.get("coordination_cap", .5))
            else:
                a_t = controls["bond"]["dyad"]["expressed_valence"]
            agent.express("user", valence=float(a_t))
            partner.append(p); agent_vals.append(a_t)
            p = float(np.clip(b + r * p + influence(a_t, inf) + noise[t], -1, 1))
            agent.advance_time(1.)
        model = agent.actuation_context()["bond"]["dyad"]["partner_model"]
    finally:
        agent.close()
    partner = np.array(partner); agent_vals = np.array(agent_vals); w = d["final_window"]
    return {"seed": seed, "partner": partner_name, "policy": policy_name,
            "final_partner_affect_mean": float(partner[-w:].mean()), "final_positive_fraction": float((partner[-w:] > 0).mean()),
            "mean_accommodation_distance": float(np.abs(agent_vals - partner).mean()), "mean_agent_valence": float(agent_vals.mean()),
            "identifier_inertia_error": float(model["inertia"] - r),
            "identifier_uninfluenced_error": (float(model["uninfluenced"] - spec["uninfluenced"]) if model["uninfluenced"] is not None else None),
            "trajectory_partner": partner.round(4).tolist(), "trajectory_agent": agent_vals.round(4).tolist()}


def analyze_partial(att_rows, dyad_rows, cfg):
    """Analyze whichever experiments ran; absent experiments are reported as not executed."""
    full = analyze(att_rows or [{"seed": 0, "responsiveness": q, "bonded": b, "variant": v} for q in cfg["attachment"]["responsiveness_levels"] for b in (True, False) for v in (list(cfg["attachment"].get("variants", {})) or [None])],
                   dyad_rows or [{"seed": 0, "partner": p, "policy": pol} for p in cfg["dyad"]["partners"] for pol in cfg["dyad"]["policies"]], cfg)
    if not att_rows: full["attachment"] = {"status": "not executed in this protocol"}
    if not dyad_rows: full["dyad"] = {"status": "not executed in this protocol"}
    return full


def _ci(values, rng, samples):
    values = np.asarray([v for v in values if v is not None], float)
    if len(values) == 0:
        return None
    boot = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(x) for x in np.quantile(boot, [.025, .975])], "n": int(len(values))}


def analyze(att_rows, dyad_rows, cfg):
    rng = np.random.default_rng(cfg["bootstrap_seed"]); S = cfg["bootstrap_samples"]
    result = {"schema_version": 1, "attachment": {"conditions": {}, "paired_bonded_minus_reference": {}}, "dyad": {"conditions": {}, "contrasts": {}}}
    variants = list(cfg["attachment"].get("variants", {})) or [None]
    for q in cfg["attachment"]["responsiveness_levels"]:
        ref = {r["seed"]: r for r in att_rows if r["responsiveness"] == q and not r["bonded"]}
        result["attachment"]["conditions"][f"q={q}|reference"] = {e: _ci([r.get(e) for r in ref.values()], rng, S) for e in cfg["attachment"]["endpoints"] + ["final_beta_h", "final_attention"]}
        for variant in variants:
            label = "bonded" if variant is None else f"bonded:{variant}"
            bonded = {r["seed"]: r for r in att_rows if r["responsiveness"] == q and r["bonded"] and r.get("variant") == variant}
            result["attachment"]["conditions"][f"q={q}|{label}"] = {e: _ci([r.get(e) for r in bonded.values()], rng, S) for e in cfg["attachment"]["endpoints"] + ["final_beta_h", "final_attention"]}
            result["attachment"]["paired_bonded_minus_reference"][f"q={q}" + ("" if variant is None else f"|{variant}")] = {
                e: _ci([bonded[s][e] - ref[s][e] for s in bonded if s in ref], rng, S) for e in ("cumulative_threat", "mean_mood_valence", "mean_exploration")}
    for partner in cfg["dyad"]["partners"]:
        for policy in cfg["dyad"]["policies"]:
            rows = [r for r in dyad_rows if r["partner"] == partner and r["policy"] == policy]
            result["dyad"]["conditions"][f"{partner}|{policy}"] = {e: _ci([r.get(e) for r in rows], rng, S) for e in cfg["dyad"]["endpoints"]}
        base = {r["seed"]: r for r in dyad_rows if r["partner"] == partner and r["policy"] == "own_mood"}
        for policy in cfg["dyad"]["policies"]:
            if policy == "own_mood":
                continue
            other = {r["seed"]: r for r in dyad_rows if r["partner"] == partner and r["policy"] == policy}
            result["dyad"]["contrasts"][f"{partner}|{policy}-own_mood"] = {
                e: _ci([other[s][e] - base[s][e] for s in other if s in base], rng, S) for e in ("final_partner_affect_mean", "final_positive_fraction", "mean_accommodation_distance")}
    result["interpretation"] = ("Simulation checks with common random numbers. Partners are generated by the declared equations; the steer policies "
                                "use the agent's online bilinear identifier, so bilinear partners are in-model and the ojive partner is out-of-model. "
                                "No human partner was involved; nothing here measures user experience or wellbeing.")
    return result


def run(config_path, output, workers=2):
    config_path, output = Path(config_path), Path(output)
    cfg = json.loads(config_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; preserve prior measurements")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_bytes(config_path.read_bytes())
    (output / "source.py").write_bytes(Path(__file__).read_bytes())
    start = time.perf_counter()
    experiments = cfg.get("experiments", ["attachment", "dyad"])
    variants = list(cfg["attachment"].get("variants", {})) or [None]
    att_jobs = [(s, q, True, v) for q in cfg["attachment"]["responsiveness_levels"] for v in variants for s in range(cfg["seeds"])]
    att_jobs += [(s, q, False, None) for q in cfg["attachment"]["responsiveness_levels"] for s in range(cfg["seeds"])]
    dyad_jobs = [(s, p, pol) for p in cfg["dyad"]["partners"] for pol in cfg["dyad"]["policies"] for s in range(cfg["seeds"])]
    if "attachment" not in experiments: att_jobs = []
    if "dyad" not in experiments: dyad_jobs = []
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            att_rows = pool.starmap(_att_job, [(j, cfg) for j in att_jobs])
            dyad_rows = pool.starmap(_dyad_job, [(j, cfg) for j in dyad_jobs])
    else:
        att_rows = [_att_job(j, cfg) for j in att_jobs]; dyad_rows = [_dyad_job(j, cfg) for j in dyad_jobs]
    summary = analyze(att_rows, dyad_rows, cfg) if att_rows and dyad_rows else analyze_partial(att_rows, dyad_rows, cfg)
    summary["runtime_seconds"] = time.perf_counter() - start
    with gzip.open(output / "attachment_runs.jsonl.gz", "wt") as f:
        for row in att_rows: f.write(json.dumps(row, sort_keys=True) + "\n")
    with gzip.open(output / "dyad_runs.jsonl.gz", "wt") as f:
        for row in dyad_rows: f.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {"schema_version": 1, "finished_utc": datetime.now(timezone.utc).isoformat(), "python": _platform.python_version(), "numpy": np.__version__,
                "command": f"python -m regesh.bonding.experiment --config {config_path} --output {output}",
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def _att_job(job, cfg):
    s, q, b, v = job
    return simulate_attachment(s, q, b, cfg, v)


def _dyad_job(job, cfg):
    s, p, pol = job
    return simulate_dyad(s, p, pol, cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bonding_platform.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    summary = run(args.config, args.output, args.workers)
    print(json.dumps(summary["attachment"].get("paired_bonded_minus_reference", summary["attachment"]), indent=2)[:3000])


if __name__ == "__main__":
    main()

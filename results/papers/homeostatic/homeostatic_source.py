"""Independent Figure 6 reimplementation of Keramati & Gutkin (eLife, 2014).

The default run retains the published cohort and horizon, not the original RNG.
No agent state is clipped or reflected. A finite allocation guard raises rather
than silently manufacturing physiological stability. No LLM is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np


PAPER = "https://doi.org/10.7554/eLife.04811"


def drive(state, m=3.0, n=4.0):
    """Equation 1 in its one-dimensional, zero-setpoint specialization."""
    return np.abs(np.asarray(state, dtype=float)) ** (n / m)


def increase_probability(decrease_q, increase_q, beta, convention="inverse"):
    """Numerically stable two-choice softmax; convention is always explicit."""
    if beta <= 0 or convention not in ("inverse", "temperature"):
        raise ValueError("positive beta and declared softmax convention required")
    scale = beta if convention == "inverse" else 1.0 / beta
    difference = (np.asarray(increase_q) - np.asarray(decrease_q)) * scale
    return np.exp(-np.logaddexp(0.0, -difference))


def q_learning_update(old, reward, next_max, alpha, gamma):
    return old + alpha * (reward + gamma * next_max - old)


def simulate(agents, steps, seed, *, alpha=.4, beta=.05, gamma=.9,
             m=3., n=4., initial=30, convention="inverse", learn=True,
             radius=128, checkpoint_every=1, examples=4):
    """Independent tabular agents with raw trajectories for a declared subset.

    Arrays of Q values contain separate rows for every agent. Checkpoint moments
    summarize all agents; max deviations and terminal states remain individual.
    Visits are post-transition (one observation per executed action).
    """
    if agents < 1 or steps < 1 or checkpoint_every < 1:
        raise ValueError("positive agents, steps, checkpoint period required")
    if abs(initial) >= radius or not 0 <= alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError("invalid allocation or learning parameters")
    rng = np.random.default_rng(seed)
    state = np.full(agents, initial, dtype=np.int64)
    max_abs = np.abs(state).copy()
    rows = np.arange(agents)
    q = np.zeros((agents, 2 * radius + 1, 2)) if learn else None
    moments = [[0., float(state.mean()), float(np.mean(state.astype(float)**2)),
                float(np.mean(np.abs(state))), float(max_abs.max())]]
    visit_count = np.zeros(2 * radius + 1, dtype=np.int64)
    saved_steps = [0]
    example_states = [state[:examples].copy()]
    for step in range(1, steps + 1):
        idx = state + radius
        if learn:
            p = increase_probability(q[rows, idx, 0], q[rows, idx, 1], beta,
                                     convention)
        else:
            p = .5
        action = (rng.random(agents) < p).astype(np.int64)
        successor = state + 2 * action - 1
        if np.max(np.abs(successor)) >= radius:
            raise RuntimeError("allocation guard reached; no clipping performed")
        if learn:
            reward = drive(state, m, n) - drive(successor, m, n)
            maximum = np.maximum(q[rows, successor + radius, 0],
                                 q[rows, successor + radius, 1])
            q[rows, idx, action] = q_learning_update(
                q[rows, idx, action], reward, maximum, alpha, gamma)
        state = successor
        max_abs = np.maximum(max_abs, np.abs(state))
        np.add.at(visit_count, state + radius, 1)
        if step % checkpoint_every == 0 or step == steps:
            moments.append([float(step), float(state.mean()),
                            float(np.mean(state.astype(float)**2)),
                            float(np.mean(np.abs(state))), float(max_abs.max())])
            saved_steps.append(step)
            example_states.append(state[:examples].copy())
    return {"moments": np.array(moments), "final_states": state,
            "max_abs": max_abs, "visits": visit_count,
            "example_states": np.array(example_states, dtype=np.int16),
            "saved_steps": np.array(saved_steps),
            "final_q": q if q is not None else np.empty(0)}


def interval(values):
    """Normal interval across independent batches/seeds, not time samples."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.
    return {"n": len(values), "mean": mean, "standard_error": se,
            "normal_95_ci": [mean - 1.96 * se, mean + 1.96 * se]}


def run(config_path, output, sources_path=None):
    config_path, output = Path(config_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("preserve prior evidence: output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if sources_path is not None:
        sources_path = Path(sources_path)
        metadata = json.loads(sources_path.read_text())
        if metadata["config_sha256"] != hashlib.sha256(config_path.read_bytes()).hexdigest():
            raise ValueError("configuration differs from prospective source/protocol record")
        (output / "primary_sources.json").write_bytes(sources_path.read_bytes())
    source = Path(__file__).read_bytes()
    (output / "homeostatic_source.py").write_bytes(source)
    started = time.perf_counter()
    cohorts = []
    rows = []
    for condition in config["cohort_conditions"]:
        results = []
        for batch, seed in enumerate(config["cohort_seeds"]):
            kwargs = dict(config["parameters"])
            kwargs.update(condition["parameters"])
            result = simulate(config["agents_per_batch"], config["cohort_steps"],
                              seed, checkpoint_every=1, **kwargs)
            # Q is not needed to analyze the cohort and is intentionally omitted.
            result.pop("final_q")
            np.savez_compressed(output / f'cohort_{condition["name"]}_{batch:02d}.npz',
                                **result)
            results.append(result)
            rows.append({"condition": condition["name"], "batch": batch,
                         "seed": seed, "n": config["agents_per_batch"],
                         "final_mean": float(result["final_states"].mean()),
                         "final_sd": float(result["final_states"].std()),
                         "final_mean_absolute": float(np.abs(result["final_states"]).mean()),
                         "maximum_absolute_ever": int(result["max_abs"].max())})
        moments = np.mean([r["moments"] for r in results], axis=0)
        cohort = {"condition": condition["name"],
                  "agents": config["agents_per_batch"] * len(results),
                  "steps": config["cohort_steps"],
                  "final_mean": float(moments[-1, 1]),
                  "final_sd": float(np.sqrt(moments[-1, 2] - moments[-1, 1]**2)),
                  "final_mean_absolute": float(moments[-1, 3]),
                  "maximum_absolute_ever": int(max(r["max_abs"].max() for r in results)),
                  "final_mean_batch_uncertainty": interval([r["final_states"].mean() for r in results]),
                  "final_mean_absolute_batch_uncertainty": interval([np.abs(r["final_states"]).mean() for r in results])}
        cohorts.append(cohort)
        np.savez_compressed(output / f'cohort_{condition["name"]}_moments.npz',
                            moments=moments)
        print(json.dumps(cohort), flush=True)

    # A million-step horizon is the published Figure 6A horizon. Independent
    # replicas are vectorized across agents; the original A panel used one agent.
    long = simulate(config["long_replicates"], config["long_steps"],
                    config["long_joint_seed"], checkpoint_every=1000,
                    examples=config["long_replicates"], **config["parameters"])
    np.savez_compressed(output / "long_trajectories.npz", **long)
    long_summary = {"agents": config["long_replicates"],
                    "joint_generator_seed": config["long_joint_seed"],
                    "steps": config["long_steps"],
                    "maximum_absolute_per_agent": long["max_abs"].tolist(),
                    "fraction_replicates_maximum_le_40": float(np.mean(long["max_abs"] <= 40)),
                    "maximum_absolute_ever": int(long["max_abs"].max()),
                    "visits": int(long["visits"].sum())}
    # A paired contrast between primary learned and no-learning conditions uses
    # identical random variates at every batch/time, before policy divergence.
    contrasts = []
    primary = [r for r in rows if r["condition"] == "published_inverse_beta"]
    control = [r for r in rows if r["condition"] == "no_learning"]
    if primary and len(primary) == len(control):
        contrasts = interval([a["final_mean_absolute"] - b["final_mean_absolute"]
                              for a, b in zip(primary, control)])
    summary = {"schema_version": 1, "paper": PAPER,
               "reproduction_class": "independent_method_reimplementation",
               "cohorts": cohorts, "long": long_summary,
               "paired_learned_minus_no_learning_final_absolute_deviation": contrasts,
               "elapsed_seconds": time.perf_counter() - started,
               "published_numeric_targets": {
                   "single_agent_million_steps_maximum_deviation_le": 40,
                   "hundred_thousand_agents_1500_steps_maximum_deviation": 61,
                   "final_mean": "converges to zero; no tabulated numeric endpoint",
                   "final_sd": "converges; no tabulated numeric endpoint"},
               "scope": "simulated regulatory learning, not LLM affect or creativity",
               "deviations": [
                   "Independent implementation and PCG64 random streams; original source/RNG unavailable.",
                   "Primary interprets beta as inverse temperature; explicit opposite convention sensitivity retained.",
                   "100,000 independent agents generated in 32 independent batches; identical total/horizon.",
                   "32 long-run replicas replace the paper's single long-run realization.",
                   "Dense Q allocation has a fail-fast guard, never a clipping boundary."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "batches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in output.iterdir() if p.is_file()}
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__,
        "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source).hexdigest(), "files": files,
        "original_seed_available": False}, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/reproductions/homeostatic.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path,
                        default=Path("docs/reproductions/homeostatic_sources.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.sources), indent=2))


if __name__ == "__main__":
    main()

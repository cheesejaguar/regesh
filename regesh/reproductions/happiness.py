"""Independent implementation of Dubey et al. (2022) bandit experiments.

The primary outcome is *objective* reward; subjective reward r-rho is kept
separate. This module never mutates the Regesh controllers or their weights.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np


SETTINGS = ("stationary_spread", "stationary_close", "sudden_change", "constant_change")
PUBLISHED = {
    "stationary_spread": {
        "objective": [4384.07, 219.66], "fixed_compare": [4648.57, 182.93],
        "dynamic_compare": [4796.07, 326.45], "ucb": [4875.8066, 170.81]},
    "stationary_close": {
        "objective": [4792.05, 134.63], "fixed_compare": [4786.41, 133.16],
        "dynamic_compare": [4790.73, 132.74]},
    "sudden_change": {
        "objective": [4191.19, 180.47], "fixed_compare": [4561.3, 235.11]},
    "constant_change": {
        "objective": [32220.31, 9930.40], "fixed_compare": [30879.62, 9933.37],
        "dynamic_compare": [34609.12, 10717.57]},
}


def parameters(setting: str, condition: str) -> tuple[float, float | None]:
    """Published epsilon and learning rate; None means sample-average Eq. 11."""
    if setting not in SETTINGS:
        raise ValueError("unknown setting")
    if condition not in {"objective", "fixed_compare", "dynamic_compare", "ucb", "optimistic_objective"}:
        raise ValueError("unknown condition")
    if condition == "ucb":
        return 0.0, None
    if condition == "optimistic_objective":
        condition = "fixed_compare"
    epsilon = 0.1 if condition == "objective" or setting == "stationary_close" else 0.0
    if setting == "constant_change" and condition == "fixed_compare":
        epsilon = 0.1
    alpha = None if setting.startswith("stationary") else (0.8 if setting == "constant_change" and condition == "objective" else 0.1)
    return epsilon, alpha


def aspiration(setting: str, condition: str, step: int) -> float:
    if condition in {"objective", "ucb", "optimistic_objective"}:
        return 0.0
    if condition == "fixed_compare":
        return 5.0 if setting == "constant_change" else 3.5
    if setting == "stationary_spread":
        return (2.0, 2.5, 2.75, 3.0)[step // 50] if step < 200 else 3.5
    if setting == "stationary_close":
        return 2.5 if step < 50 else 3.5
    if setting == "constant_change":
        return 1.5 + 0.005 * step
    raise ValueError("No published dynamic-aspiration schedule for sudden change")


def random_argmax(values: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Uniform tie breaking without perturbing the ordering of unequal values."""
    mask = values == values.max(axis=1, keepdims=True)
    rank = np.floor(uniforms * mask.sum(axis=1)).astype(int)
    return np.argmax(mask & (np.cumsum(mask, axis=1) > rank[:, None]), axis=1)


def update(q: np.ndarray, counts: np.ndarray, actions: np.ndarray,
           reward: np.ndarray, rho: float, alpha: float | None) -> None:
    rows = np.arange(len(q))
    counts[rows, actions] += 1
    rate = 1.0 / counts[rows, actions] if alpha is None else alpha
    q[rows, actions] += rate * (reward - rho - q[rows, actions])


def simulate_block(config: dict, setting: str, seed: int) -> tuple[list[dict], list[dict]]:
    """Run independent rounds in parallel; share random innovations across policies."""
    n, k, steps = config["rounds_per_block"], config["arms"], config["steps"]
    environment_rng, policy_rng = [np.random.default_rng(s) for s in
        np.random.SeedSequence([seed, SETTINGS.index(setting)]).spawn(2)]
    if setting == "constant_change":
        means = environment_rng.normal(0, 1, (n, k))
    else:
        low, high = (0.87, 0.97) if setting == "stationary_close" else (-1, 0.9)
        means = np.column_stack((np.ones(n), environment_rng.uniform(low, high, (n, k - 1))))
    conditions = ["objective", "fixed_compare", "ucb", "optimistic_objective"]
    if setting != "sudden_change":
        conditions.insert(2, "dynamic_compare")
    q = {c: np.full((n, k), aspiration(setting, "fixed_compare", 0)
                  if c == "optimistic_objective" else config["initial_q"], dtype=float) for c in conditions}
    counts = {c: np.zeros((n, k), dtype=int) for c in conditions}
    sums = {c: {name: np.zeros(n) for name in (
        "objective_reward", "subjective_reward", "expected_regret", "optimal_choices",
        "post_change_optimal_choices", "late_optimal_choices", "action_disagreements_with_fixed")}
        for c in conditions}
    checkpoints = []
    rows = np.arange(n)
    for t in range(steps):
        if setting == "sudden_change" and t == steps // 2:
            # Independent uniform permutations in each round. Identity is permitted.
            means = np.take_along_axis(means, environment_rng.random((n, k)).argsort(axis=1), axis=1)
        if setting == "constant_change" and t:
            means += environment_rng.normal(0, 0.1, (n, k))
        reward_noise = environment_rng.normal(0, config["reward_noise_sd"], n)
        explore, tie, random_arm = policy_rng.random((3, n))
        best = means.argmax(axis=1)
        actions_by_condition = {}
        for c in conditions:
            epsilon, alpha = parameters(setting, c)
            if c == "ucb":
                values = q[c] + np.sqrt(2 * np.log(t + 1) / np.maximum(1, counts[c]))
                values[counts[c] == 0] = np.inf
            else:
                values = q[c]
            actions = random_argmax(values, tie)
            actions = np.where(explore < epsilon, (random_arm * k).astype(int), actions)
            actions_by_condition[c] = actions
            rewards = means[rows, actions] + reward_noise
            rho = aspiration(setting, c, t)
            update(q[c], counts[c], actions, rewards, rho, alpha)
            out = sums[c]
            out["objective_reward"] += rewards
            out["subjective_reward"] += rewards - rho
            out["expected_regret"] += means[rows, best] - means[rows, actions]
            out["optimal_choices"] += actions == best
            if t >= steps // 2:
                out["post_change_optimal_choices"] += actions == best
            if t >= steps - min(500, steps):
                out["late_optimal_choices"] += actions == best
        # Floating roundoff can differ after algebraic translation. It must not
        # alter coupled action choices in this experiment; record any violation.
        sums["optimistic_objective"]["action_disagreements_with_fixed"] += (
            actions_by_condition["fixed_compare"] != actions_by_condition["optimistic_objective"])
        if (t + 1) % config["checkpoint_every"] == 0 or t + 1 == steps:
            for c in conditions:
                checkpoints.append({"setting": setting, "seed_block": seed, "condition": c,
                    "step": t + 1, **{key: float(value.mean()) for key, value in sums[c].items()}})
    records = []
    for c in conditions:
        for i in range(n):
            records.append({"setting": setting, "seed_block": seed, "round": i, "condition": c,
                **{key: float(value[i]) for key, value in sums[c].items()}})
    return records, checkpoints


def interval(values: np.ndarray, rng: np.random.Generator, samples: int) -> list[float]:
    boot = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def analyze(records: list[dict], config: dict) -> dict:
    rng = np.random.default_rng(config["bootstrap_seed"])
    result = {"conditions": [], "contrasts": [], "interpretation": (
        "Independent implementation of named paper experiments; exact original-code/seed replay unavailable. "
        "Published SD is between-round dispersion, not confidence interval. Local confidence intervals resample "
        "20 independent seed-block means; paired controller differences preserve common random numbers.")}
    for setting in config["settings"]:
        subset = [r for r in records if r["setting"] == setting]
        names = list(dict.fromkeys(r["condition"] for r in subset))
        by_name = {}
        for c in names:
            group = [r for r in subset if r["condition"] == c]
            values = np.array([r["objective_reward"] for r in group])
            blockmeans = np.array([np.mean([r["objective_reward"] for r in group if r["seed_block"] == seed])
                                  for seed in config["seed_blocks"]])
            by_name[c] = blockmeans
            row = {"setting": setting, "condition": c, "rounds": len(group),
                   "objective_reward_mean": float(values.mean()), "objective_reward_sd": float(values.std(ddof=1)),
                   "objective_reward_ci95": interval(blockmeans, rng, config["bootstrap_samples"]),
                   "subjective_reward_mean": float(np.mean([r["subjective_reward"] for r in group])),
                   "expected_regret_mean": float(np.mean([r["expected_regret"] for r in group])),
                   "optimal_action_fraction": float(np.mean([r["optimal_choices"] for r in group])) / config["steps"],
                   "second_half_optimal_fraction": float(np.mean([r["post_change_optimal_choices"] for r in group])) / (config["steps"] - config["steps"] // 2),
                   "last_500_optimal_fraction": float(np.mean([r["late_optimal_choices"] for r in group])) / min(500, config["steps"]),
                   "action_disagreements_with_fixed": int(sum(r["action_disagreements_with_fixed"] for r in group))}
            published = PUBLISHED[setting].get(c)
            if published:
                row.update(published_mean=published[0], published_sd=published[1],
                           reproduced_minus_published=float(values.mean() - published[0]),
                           discrepancy_in_published_sd=float((values.mean() - published[0]) / published[1]))
            result["conditions"].append(row)
        for a, b in [(c, "objective") for c in names if c not in {"objective", "optimistic_objective"}] + [("optimistic_objective", "fixed_compare")]:
            difference = by_name[a] - by_name[b]
            result["contrasts"].append({"setting": setting, "condition_a": a, "condition_b": b,
                "paired_difference_mean": float(difference.mean()),
                "paired_seed_block_ci95": interval(difference, rng, config["bootstrap_samples"])})
    return result


def run(config_path: str | Path, output: str | Path) -> dict:
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; preserve prior measurements")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    source = Path(__file__).read_bytes()
    (output / "source.py").write_bytes(source)
    start = time.perf_counter()
    records, checkpoints = [], []
    for setting in config["settings"]:
        for seed in config["seed_blocks"]:
            rows, curve = simulate_block(config, setting, seed)
            records.extend(rows)
            checkpoints.extend(curve)
        print(f"completed {setting}: {len(config['seed_blocks']) * config['rounds_per_block']} rounds/condition", flush=True)
    summary = analyze(records, config)
    summary["runtime_seconds"] = time.perf_counter() - start
    for filename, rows in [("rounds.jsonl.gz", records), ("checkpoints.jsonl.gz", checkpoints)]:
        with gzip.open(output / filename, "wt") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {"schema_version": 1, "paper_url": "https://doi.org/10.1371/journal.pcbi.1010316",
        "primary_pdf_url": "https://cocosci.princeton.edu/papers/dubey2022pursuit.pdf",
        "primary_pdf_sha256": "f4068b6979811004dd27f26d5092dce618e6ec6b1e50d81224842857f294e6b3",
        "author_repo": "https://github.com/rach0012/happiness_RL",
        "inspected_author_repo_commit": "c0af9d85ad2b3724c939c8ea9ef197b3ce7d3123",
        "author_code_executed": False, "author_code_copied": False,
        "fidelity": "independent implementation of published bandit experiments",
        "python": platform.python_version(), "numpy": np.__version__,
        "command": f"python -m regesh.reproductions.happiness --config {config_path} --output {output}",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reproductions/happiness.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()

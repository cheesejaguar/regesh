"""Plot empirical learning curves and same-unit published/reproduced means."""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/papers/happiness")
    args = parser.parse_args()
    root = Path(args.run)
    summary = json.loads((root / "summary.json").read_text())
    with gzip.open(root / "checkpoints.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    conditions = ["objective", "fixed_compare", "dynamic_compare", "ucb"]
    colors = dict(zip(conditions, ["#666666", "#D55E00", "#0072B2", "#009E73"]))
    settings = ["stationary_spread", "stationary_close", "sudden_change", "constant_change"]
    plt.rcParams.update({"font.size": 9, "svg.hashsalt": "regesh-happiness-reproduction"})
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for axis, setting in zip(axes.flat, settings):
        for condition in conditions:
            points = defaultdict(list)
            for row in rows:
                if row["setting"] == setting and row["condition"] == condition:
                    points[row["step"]].append(row["objective_reward"] / row["step"])
            if points:
                x = sorted(points)
                axis.plot(x, [np.mean(points[t]) for t in x], label=condition.replace("_", " "), color=colors[condition])
        axis.set(title=setting.replace("_", " "), xlabel="Actions per round", ylabel="Cumulative objective reward / actions")
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Dubey et al. (2022), Experiments 2a/2b: independent implementation\n2,000 rounds per condition; 20 seed blocks; published selected hyperparameters", fontsize=11)
    figure.tight_layout()
    for ext in ["svg", "png"]:
        figure.savefig(root / f"learning_curves.{ext}", dpi=160, metadata={"Date": None} if ext == "svg" else {})
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for axis, setting in zip(axes.flat, settings):
        subset = [r for r in summary["conditions"] if r["setting"] == setting and r["condition"] in conditions]
        for i, row in enumerate(subset):
            lower, upper = row["objective_reward_ci95"]
            mean = row["objective_reward_mean"]
            axis.errorbar(i, mean, yerr=[[mean - lower], [upper - mean]], color=colors[row["condition"]], fmt="o", capsize=5)
            if "published_mean" in row:
                axis.scatter(i, row["published_mean"], marker="D", facecolors="none", edgecolors="black", s=50)
        axis.set_xticks(range(len(subset)), [r["condition"].replace("_", "\n") for r in subset], fontsize=8)
        axis.set(title=setting.replace("_", " "), ylabel="Objective reward over 5,000 actions")
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Filled circle: new mean and 95% seed-block bootstrap CI; hollow diamond: published mean\nPublished SD and missing author RNG details are reported separately; no equivalence test", fontsize=10)
    figure.tight_layout()
    for ext in ["svg", "png"]:
        figure.savefig(root / f"published_comparison.{ext}", dpi=160, metadata={"Date": None} if ext == "svg" else {})
    plt.close(figure)
    for path in root.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()

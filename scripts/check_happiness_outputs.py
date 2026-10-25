"""Independent stdlib arithmetic/integrity audit of the saved bandit run."""
import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def check(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"hash mismatch: {filename}")
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / "config.json").read_text())
    with gzip.open(root / "rounds.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    groups = defaultdict(list)
    ids = set()
    for row in rows:
        identity = (row["setting"], row["condition"], row["seed_block"], row["round"])
        if identity in ids:
            raise ValueError("duplicate round")
        ids.add(identity)
        groups[identity[:2]].append(row)
    checked = 0
    for entry in summary["conditions"]:
        group = groups[(entry["setting"], entry["condition"])]
        if len(group) != len(config["seed_blocks"]) * config["rounds_per_block"]:
            raise ValueError("incomplete condition")
        values = [r["objective_reward"] for r in group]
        for expected, actual in [(entry["objective_reward_mean"], statistics.mean(values)),
                                 (entry["objective_reward_sd"], statistics.stdev(values)),
                                 (entry["subjective_reward_mean"], statistics.mean(r["subjective_reward"] for r in group)),
                                 (entry["expected_regret_mean"], statistics.mean(r["expected_regret"] for r in group))]:
            if not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("independent metric mismatch")
            checked += 1
    equivalence = []
    for setting in config["settings"]:
        fixed = {(r["seed_block"], r["round"]): r for r in groups[(setting, "fixed_compare")]}
        ordinary = groups[(setting, "optimistic_objective")]
        for row in ordinary:
            match = fixed[(row["seed_block"], row["round"])]
            if match["objective_reward"] != row["objective_reward"] or row["action_disagreements_with_fixed"]:
                raise ValueError("optimistic-objective equivalence violated")
        equivalence.append({"setting": setting, "paired_rounds": len(ordinary),
                            "equal_objective_rewards": True, "action_disagreements": 0})
    return {"artifact_hashes_verified": len(manifest["files"]), "unique_round_condition_records": len(ids),
            "stdlib_metrics_matched": checked, "ordinary_equivalence": equivalence}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/papers/happiness")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = check(Path(args.run))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        target = Path(args.output)
        if target.exists():
            raise ValueError("refuse to replace an earlier audit")
        target.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

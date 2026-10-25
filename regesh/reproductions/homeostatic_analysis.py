"""Post hoc descriptive audit of the completed homeostatic reimplementation.

This audit was added after observing the primary cohort's extreme-value gap.
It does not modify the frozen experiment or select softmax parameters.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


def analyze(run_path, output):
    run_path, output = Path(run_path), Path(output)
    if output.exists():
        raise ValueError("preserve prior evidence: analysis output already exists")
    manifest = json.loads((run_path / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("manifest filename must be local")
        if hashlib.sha256((run_path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"measurement artifact hash mismatch: {name}")
    config = json.loads((run_path / "config.json").read_text())
    summary = json.loads((run_path / "summary.json").read_text())
    conditions = []
    for source in summary["cohorts"]:
        name = source["condition"]
        finals, maxima, temporal = [], [], []
        for batch in range(len(config["cohort_seeds"])):
            with np.load(run_path / f"cohort_{name}_{batch:02d}.npz", allow_pickle=False) as raw:
                finals.extend(raw["final_states"].tolist())
                maxima.extend(raw["max_abs"].tolist())
                temporal.append(raw["moments"])
                if int(raw["visits"].sum()) != config["agents_per_batch"] * config["cohort_steps"]:
                    raise ValueError("visit count does not equal executed actions")
        independent = {"n": len(finals), "final_mean": statistics.fmean(finals),
                       "final_sd": statistics.pstdev(finals),
                       "final_mean_absolute": statistics.fmean(abs(x) for x in finals),
                       "maximum_absolute_ever": max(maxima)}
        for key in ("final_mean", "final_sd", "final_mean_absolute", "maximum_absolute_ever"):
            if not math.isclose(independent[key], source[key], abs_tol=1e-10):
                raise ValueError(f"raw versus summary mismatch: {name}/{key}")
        moments = np.mean(temporal, axis=0)
        # The last 100 actions include both parity phases of the forced ±1 walk.
        last_window = moments[-min(100, len(moments)-1):]
        conditions.append({"condition": name, "independent_summary_recomputation": independent,
            "agents_exceeding_original_population_maximum_61": sum(x > 61 for x in maxima),
            "individual_maximum_quantiles": {
                str(q): float(np.quantile(maxima, q)) for q in (.5, .95, .99, .999, 1.)},
            "posthoc_last_100_actions_mean_absolute_deviation": float(last_window[:, 3].mean()),
            "posthoc_last_100_actions_rms_deviation": float(np.sqrt(last_window[:, 2].mean()))})
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "classification": "posthoc_descriptive_diagnostic_not_protocol_selection",
              "input_manifest_sha256": hashlib.sha256((run_path / "manifest.json").read_bytes()).hexdigest(),
              "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "conditions": conditions,
              "cautions": [
                  "The primary maximum differs from the paper; the number of agents above 61 is diagnostic, not a prospectively specified hypothesis test.",
                  "No source-code, seed, or verified beta convention is available for exact numerical reproduction.",
                  "Even/odd endpoint parity matters under forced unit actions, particularly in the temperature sensitivity.",
                  "Finite-horizon maxima do not prove an absolute long-run safety boundary."]}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()

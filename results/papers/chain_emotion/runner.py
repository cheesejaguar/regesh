"""Reanalyse author-released Chain-of-Emotion data without new participants or LLM calls.

The original experiment is Croissant et al., PLOS ONE 19(5), e0301033 (2024).
The three source CSVs are downloaded on request to an external cache and checked
against hashes published by OSF. No proprietary STEU items or LIWC dictionary
are required. This is a statistical reproduction, not a fresh model/human study.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import urllib.request

import numpy as np


PAPER_URL = "https://doi.org/10.1371/journal.pone.0301033"
OSF_URL = "https://doi.org/10.17605/OSF.IO/QPT6Z"
CONDITIONS = ("No Memory", "Memory", "Chain of Emotion")
SOURCES = {
    "experiment_1_STEU.csv": {
        "url": "https://osf.io/download/hqb8v/", "bytes": 16445,
        "sha256": "10cde1bbc7c562ac468a791fb609df6d63c2b0c491c6a5c536a144d807663e48",
    },
    "experiment_2_LIWC_Output.csv": {
        "url": "https://osf.io/download/bxy73/", "bytes": 56872,
        "sha256": "b9ff2c3b143142c798d904459ff32d7e23e9abb72b812f321a67e972267ac4ee",
    },
    "experiment_3_user_study.csv": {
        "url": "https://osf.io/download/d8szt/", "bytes": 4827,
        "sha256": "3af212b07a3ddd6a659e650ac8dc8e434cc9fdb434c7a7897ab4f63066d32a5f",
    },
}
USER_ITEMS = (
    "The agent's behavior was human-like.",
    "The agent's reactions were natural.",
    "The agent reacted to my input.",
    "The agent did not care about the scenario.",
    "The agent always knows their friends' emotions from their behavior.",
    "The agent is a good observer of others' emotions.",
    "The agent is sensitive to the feelings and emotions of others.",
    "The agent has a good understanding of the emotions of people around them.",
    "How capable was the agent?", "How competent was the agent?",
    "How warm was the agent?", "How friendly was the agent?",
)
LIWC_METRICS = ("WC", "Authentic", "Tone", "Affect", "emo_pos", "emo_neg")
# Facts transcribed from Tables 1--3 of the CC-BY published article, not fitted.
PUBLISHED_STEU = (24, 31, 35)
PUBLISHED_LIWC = {
    "WC": {"means": [18.00, 15.20, 17.00], "F": .20, "p": .65},
    "Authentic": {"means": [61.50, 61.90, 82.60], "F": 5.10, "p": .03},
    "Tone": {"means": [74.20, 62.00, 53.90], "F": 2.76, "p": .10},
    "Affect": {"means": [11.40, 13.50, 10.65], "F": .08, "p": .78},
    "emo_pos": {"means": [5.28, 4.13, 3.32], "F": 1.76, "p": .19},
    "emo_neg": {"means": [.59, 3.10, 1.57], "F": .37, "p": .55},
}
# Published table order is warmth/friendliness reversed relative to the CSV.
PUBLISHED_USER = {
    USER_ITEMS[0]: {"means": [4.82, 5.36, 5.75], "F": 1.95, "p": .15},
    USER_ITEMS[1]: {"means": [4.43, 4.89, 5.71], "F": 3.65, "p": .03},
    USER_ITEMS[2]: {"means": [5.43, 6.29, 6.54], "F": 3.62, "p": .04},
    USER_ITEMS[3]: {"means": [3.32, 3.14, 3.32], "F": .26, "p": .78},
    USER_ITEMS[4]: {"means": [4.79, 5.04, 5.71], "F": 1.32, "p": .27},
    USER_ITEMS[5]: {"means": [4.89, 5.29, 5.93], "F": 2.25, "p": .11},
    USER_ITEMS[6]: {"means": [5.14, 5.39, 6.25], "F": 3.31, "p": .04},
    USER_ITEMS[7]: {"means": [4.86, 5.11, 5.61], "F": .86, "p": .43},
    USER_ITEMS[8]: {"means": [4.86, 5.57, 5.14], "F": 1.68, "p": .19},
    USER_ITEMS[9]: {"means": [5.39, 5.39, 5.11], "F": 1.68, "p": .19},
    USER_ITEMS[10]: {"means": [6.07, 6.38, 5.55], "F": 2.48, "p": .09},
    USER_ITEMS[11]: {"means": [4.86, 5.18, 5.00], "F": .41, "p": .66},
}


def scipy_stats():
    try:
        from scipy import stats
    except ImportError as error:
        raise RuntimeError("Install regesh[papers] (SciPy) for this reproduction") from error
    return stats


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(cache: Path, fetch: bool = False) -> dict[str, Path]:
    """Only the three pinned author URLs can be fetched. Existing files checked."""
    paths = {}
    for name, spec in SOURCES.items():
        path = cache / name
        if not path.exists():
            if not fetch:
                raise FileNotFoundError(f"Missing {path}; rerun with --fetch")
            cache.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(spec["url"], timeout=60) as response:
                data = response.read(int(spec["bytes"]) + 1)
            if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise ValueError(f"Downloaded source does not match OSF hash: {name}")
            temporary = path.with_suffix(path.suffix + ".part")
            with temporary.open("xb") as stream:
                stream.write(data)
            temporary.replace(path)
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError(f"Source does not match pinned OSF hash: {name}")
        paths[name] = path
    return paths


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def holm(p_values: list[float]) -> list[float]:
    """Holm family-wise error correction, preserving original order."""
    out = [0.0] * len(p_values)
    largest = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        largest = max(largest, min(1.0, (len(p_values) - rank) * p_values[index]))
        out[index] = largest
    return out


def describe(values: np.ndarray) -> dict:
    return {"n": len(values), "mean": float(values.mean()), "sd": float(values.std(ddof=1))}


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [center-half, center+half]


def paired_bootstrap(differences: np.ndarray, rng: np.random.Generator, samples: int) -> list[float]:
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def repeated_measures(values: np.ndarray) -> dict:
    """One-factor within-participant ANOVA, no sphericity correction assumed."""
    stats = scipy_stats()
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Expected complete finite participant-by-condition matrix")
    n, k = values.shape
    grand = values.mean()
    total = float(((values-grand)**2).sum())
    condition = float(n*((values.mean(axis=0)-grand)**2).sum())
    subject = float(k*((values.mean(axis=1)-grand)**2).sum())
    error = max(0.0, total-condition-subject)
    numerator_df, denominator_df = k-1, (n-1)*(k-1)
    f_value = (condition/numerator_df)/(error/denominator_df)
    # Greenhouse-Geisser epsilon from the covariance of orthogonal contrasts.
    centering = np.eye(k)-np.ones((k, k))/k
    contrast_covariance = centering @ np.cov(values, rowvar=False, ddof=1) @ centering
    epsilon = float(np.trace(contrast_covariance)**2 /
                    ((k-1)*np.trace(contrast_covariance@contrast_covariance)))
    epsilon = min(1.0, max(1/(k-1), epsilon))
    return {"F": float(f_value), "df": [numerator_df, denominator_df],
            "p": float(stats.f.sf(f_value, numerator_df, denominator_df)),
            "greenhouse_geisser_epsilon": epsilon,
            "p_greenhouse_geisser": float(stats.f.sf(f_value, epsilon*numerator_df, epsilon*denominator_df)),
            "partial_eta_squared": condition/(condition+error)}


def analyse_steu(rows: list[dict], rng: np.random.Generator, bootstrap_samples: int) -> dict:
    stats = scipy_stats()
    if len(rows) != 42 or {int(row["STEU"]) for row in rows} != set(range(1, 43)):
        raise ValueError("Expected exactly the 42 distinct author-scored STEU items")
    scores = np.asarray([[int(row[k]) for k in ("No Memory Score", "Memory Score", "Appraisal Score")]
                         for row in sorted(rows, key=lambda row: int(row["STEU"]))])
    if not np.isin(scores, [0, 1]).all():
        raise ValueError("STEU scores must be author-supplied binary labels")
    conditions = []
    for index, name in enumerate(("No Memory", "Memory", "Appraisal Prompts")):
        correct = int(scores[:, index].sum())
        conditions.append({"condition": name, "correct": correct, "n": 42,
                           "accuracy": correct/42, "wilson_95_ci": wilson(correct, 42),
                           "published_correct": PUBLISHED_STEU[index],
                           "matches_published_count": correct == PUBLISHED_STEU[index],
                           "binomial_p_above_chance_0_2": float(stats.binomtest(correct, 42, .2, alternative="greater").pvalue)})
    comparisons = []
    for base, treatment in ((0, 1), (0, 2), (1, 2)):
        lost = int(((scores[:, base] == 1) & (scores[:, treatment] == 0)).sum())
        gained = int(((scores[:, base] == 0) & (scores[:, treatment] == 1)).sum())
        p_value = float(stats.binomtest(min(lost, gained), lost+gained, .5).pvalue) if lost+gained else 1.0
        delta = scores[:, treatment]-scores[:, base]
        comparisons.append({"base": conditions[base]["condition"], "treatment": conditions[treatment]["condition"],
                            "accuracy_delta": float(delta.mean()), "item_bootstrap_95_ci": paired_bootstrap(delta, rng, bootstrap_samples),
                            "gained_items": gained, "lost_items": lost, "mcnemar_exact_p": p_value})
    for row, adjusted in zip(comparisons, holm([row["mcnemar_exact_p"] for row in comparisons])):
        row["mcnemar_holm_p_3_contrasts"] = adjusted
    return {"conditions": conditions, "paired_comparisons": comparisons,
            "unit": "the same 42 fixed test items, not 126 independent samples",
            "fresh_model_calls": 0, "gold_verification": "author-released score columns; full STEU key not independently re-annotated"}


def analyse_liwc(rows: list[dict]) -> dict:
    stats = scipy_stats()
    # Punctuation-only segments have WC but no derived LIWC scores. Keep one
    # documented complete-case set for all six metrics, matching table N=73.
    complete = [row for row in rows if all(row[metric] != "" for metric in LIWC_METRICS)]
    x = np.asarray([float(row["Condition"]) for row in complete])
    if set(x) != {0., 1., 2.}:
        raise ValueError("Unexpected released LIWC condition codes")
    metrics = []
    for metric in LIWC_METRICS:
        y = np.asarray([float(row[metric]) for row in complete])
        groups = [y[x == code] for code in (0., 1., 2.)]
        test = stats.f_oneway(*groups)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope*x+intercept
        trend_f = float(((fitted-y.mean())**2).sum()/(((y-fitted)**2).sum()/(len(y)-2)))
        published = PUBLISHED_LIWC[metric]
        metrics.append({"metric": metric, "conditions": {name: describe(group) for name, group in zip(CONDITIONS, groups)},
                        "categorical_anova": {"F": float(test.statistic), "p": float(test.pvalue), "df": [2, len(y)-3]},
                        "numeric_condition_trend": {"F": trend_f, "p": float(stats.f.sf(trend_f, 1, len(y)-2)),
                                                    "df": [1, len(y)-2], "slope": float(slope)},
                        "published": published,
                        "numeric_trend_minus_published_F": trend_f-published["F"],
                        "mean_minus_published": [float(group.mean())-original for group, original in zip(groups, published["means"])]})
    for row, adjusted in zip(metrics, holm([row["numeric_condition_trend"]["p"] for row in metrics])):
        row["numeric_condition_trend"]["holm_p_6_metrics"] = adjusted
    for row, adjusted in zip(metrics, holm([row["categorical_anova"]["p"] for row in metrics])):
        row["categorical_anova"]["holm_p_6_metrics"] = adjusted
    return {"released_rows": len(rows), "complete_metric_rows": len(complete),
            "excluded_missing_metric_rows": [{"condition": row["Condition"], "segment": row["Segment"], "word_count": row["WC"]}
                                             for row in rows if row not in complete],
            "metrics": metrics,
            "interpretation": "Numeric condition 0/1/2 trend approximates published F and p; categorical ANOVA is a different estimand. Sentences from three conversations are not independent experimental replications.",
            "liwc_recomputed_from_text": False}


def analyse_users(rows: list[dict], rng: np.random.Generator, bootstrap_samples: int) -> dict:
    stats = scipy_stats()
    subjects: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row["group"] not in CONDITIONS or row["group"] in subjects.setdefault(row["id"], {}):
            raise ValueError("Unexpected or duplicated participant/condition")
        subjects[row["id"]][row["group"]] = row
    if len(subjects) != 30 or any(set(v) != set(CONDITIONS) for v in subjects.values()):
        raise ValueError("Expected complete 30-participant x 3-condition design")
    ordered = sorted(subjects, key=int)
    outputs = []
    pairwise = []
    for item_index, item in enumerate(USER_ITEMS):
        values = np.asarray([[float(subjects[subject][condition][item]) for condition in CONDITIONS] for subject in ordered])
        if not np.isfinite(values).all() or not ((values >= 1) & (values <= 7)).all():
            raise ValueError("Released user ratings must be complete and in [1,7]")
        groups = [values[:, i] for i in range(3)]
        ordinary = stats.f_oneway(*groups)
        rm = repeated_measures(values)
        published = PUBLISHED_USER[item]
        outputs.append({"item": item, "item_index": item_index, "conditions": {name: describe(group) for name, group in zip(CONDITIONS, groups)},
                        "published": published,
                        "mean_minus_published": [float(group.mean())-original for group, original in zip(groups, published["means"])],
                        "unpaired_diagnostic_anova": {"F": float(ordinary.statistic), "p": float(ordinary.pvalue), "df": [2, 87]},
                        "repeated_measures_anova": rm})
        for base, treatment in ((0, 1), (0, 2), (1, 2)):
            differences = values[:, treatment]-values[:, base]
            test = stats.ttest_rel(values[:, treatment], values[:, base])
            sem = float(stats.sem(differences))
            ci = stats.t.interval(.95, len(differences)-1, loc=float(differences.mean()), scale=sem)
            pairwise.append({"item": item, "base": CONDITIONS[base], "treatment": CONDITIONS[treatment],
                             "mean_delta": float(differences.mean()), "paired_t": float(test.statistic), "df": 29,
                             "p": float(test.pvalue), "paired_t_95_ci": [float(x) for x in ci],
                             "participant_bootstrap_95_ci": paired_bootstrap(differences, rng, bootstrap_samples),
                             "cohen_dz": float(differences.mean()/differences.std(ddof=1))})
    for row, adjusted in zip(outputs, holm([row["repeated_measures_anova"]["p"] for row in outputs])):
        row["repeated_measures_anova"]["holm_p_12_items"] = adjusted
    for row, adjusted in zip(outputs, holm([row["repeated_measures_anova"]["p_greenhouse_geisser"] for row in outputs])):
        row["repeated_measures_anova"]["holm_p_12_items_greenhouse_geisser"] = adjusted
    for row, adjusted in zip(pairwise, holm([row["p"] for row in pairwise])):
        row["holm_p_36_pairwise_contrasts"] = adjusted
    return {"participants": len(subjects), "rows": len(rows), "conditions": list(CONDITIONS),
            "items": outputs, "paired_contrasts": pairwise,
            "observed_rating_range": [min(float(row[item]) for row in rows for item in USER_ITEMS),
                                      max(float(row[item]) for row in rows for item in USER_ITEMS)],
            "missing_age_participants": len({row["id"] for row in rows if row["age"] in ("", "NA")}),
            "exclusions": "None: all 90 complete rating rows retained, including missing demographic age.",
            "published_vs_released": "Published Table3 means and F values are not exactly reproduced from all released ratings; no undocumented exclusions or item relabeling were selected to force agreement.",
            "new_participants": 0}


def write_report(summary: dict, output: Path) -> None:
    steu = summary["experiment_1"]
    users = summary["experiment_3"]
    lines = ["# Chain-of-Emotion: statistical reproduction", "",
             f"Source: Croissant et al. (2024), [PLOS ONE]({PAPER_URL}); [author data]({OSF_URL}).", "",
             "This re-executes analyses on the authors' original outputs and participant ratings. It does not rerun GPT-3.5, recruit participants, or recompute proprietary LIWC dictionaries.", "",
             "## Experiment 1: original 42-item STEU outcomes", "",
             "| Condition | Published | Reproduced | Accuracy |", "|---|---:|---:|---:|"]
    for row in steu["conditions"]:
        lines.append(f"| {row['condition']} | {row['published_correct']}/42 | {row['correct']}/42 | {row['accuracy']:.4f} |")
    lines += ["", "Paired exact McNemar tests use the same items and Holm-correct three contrasts:", ""]
    for row in steu["paired_comparisons"]:
        lines.append(f"- {row['treatment']} minus {row['base']}: {row['accuracy_delta']:+.4f}; paired bootstrap 95% CI {row['item_bootstrap_95_ci']}; exact p={row['mcnemar_exact_p']:.6g}, Holm p={row['mcnemar_holm_p_3_contrasts']:.6g}.")
    lines += ["", "## Experiment 2: released LIWC scores", "",
              "| Metric | Published F (p) | Reproduced numeric 0/1/2 trend F (p) | Categorical F (p) |", "|---|---:|---:|---:|"]
    for row in summary["experiment_2"]["metrics"]:
        p, trend, categorical = row["published"], row["numeric_condition_trend"], row["categorical_anova"]
        lines.append(f"| {row['metric']} | {p['F']:.2f} ({p['p']:.2f}) | {trend['F']:.5f} ({trend['p']:.5f}) | {categorical['F']:.5f} ({categorical['p']:.5f}) |")
    lines += ["", "Complete-metric filtering retains 73/75 sentence rows (22/24/27). The two excluded Chain-of-Emotion rows contain only punctuation. Numeric trend coding reproduces the reported F pattern, whereas a categorical condition test gives different statistics. All six trend tests fail Holm correction at .05. Sentences from only three generated conversations do not support an independent population-level replication.", "",
              "## Experiment 3: all released participant ratings", "",
              "The published description calls this a within-subject design; this analysis uses the 30 participant IDs and all 90 rows. Published Table 3 is not reproduced exactly. Raw ratings span 1–7, while the table caption says 0–6. Published residual df=84 differs from full-data independent df=87 and repeated-measures df=58; the source code/exclusion rule is unavailable.", "",
              "| Item | Published F (p) | Full-data unpaired diagnostic F (p) | Paired RM F (p; Holm p) |", "|---|---:|---:|---:|"]
    for row in users["items"]:
        p, ordinary, rm = row["published"], row["unpaired_diagnostic_anova"], row["repeated_measures_anova"]
        lines.append(f"| {row['item']} | {p['F']:.2f} ({p['p']:.2f}) | {ordinary['F']:.4f} ({ordinary['p']:.4f}) | {rm['F']:.4f} ({rm['p']:.4f}; {rm['holm_p_12_items']:.4f}) |")
    lines += ["", "Paired contrasts, participant bootstrap intervals, standardized differences, Greenhouse–Geisser sensitivity and Holm corrections are in summary.json. These are exploratory reanalyses of an existing small study, not preregistered confirmations or evidence of subjective experience. The strongest appraisal-versus-no-memory STEU result reproduces; subjective-rating claims require caution about analysis choices and multiple outcomes.", ""]
    (output/"report.md").write_text("\n".join(lines))


def run(output: Path, cache: Path, *, fetch: bool = False, seed: int = 20261025, bootstrap_samples: int = 10000) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be absent or empty; preserve previous evidence")
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    files = source_files(cache, fetch)
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    summary = {"schema_version": 1, "paper": PAPER_URL, "status": "statistical reproduction on author-released data",
               "experiment_1": analyse_steu(read_csv(files["experiment_1_STEU.csv"]), rng, bootstrap_samples),
               "experiment_2": analyse_liwc(read_csv(files["experiment_2_LIWC_Output.csv"])),
               "experiment_3": analyse_users(read_csv(files["experiment_3_user_study.csv"]), rng, bootstrap_samples)}
    output.mkdir(parents=True, exist_ok=True)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (output/"runner.py").write_bytes(Path(__file__).read_bytes())
    effective_config = {"schema_version": 1, "seed": seed, "bootstrap_samples": bootstrap_samples,
                        "population": "all 42 STEU items and all 30 complete participant triplets",
                        "liwc_filter": "complete six-metric cases",
                        "model_calls": 0, "new_participants": 0}
    (output/"config.json").write_text(json.dumps(effective_config, indent=2)+"\n")
    write_report(summary, output)
    import scipy
    manifest = {"schema_version": 1, "search_date": "2026-10-25", "paper": PAPER_URL,
                "osf": OSF_URL, "sources": SOURCES,
                "source_license": "Article CC-BY; OSF node_license=null on 2026-10-25. Raw source CSVs cached outside Git; no claim of unrestricted dataset redistribution.",
                "source_paths": {name: str(path.resolve()) for name, path in files.items()},
                "source_code_sha256": sha256(Path(__file__)), "seed": seed,
                "bootstrap_samples": bootstrap_samples, "elapsed_analysis_seconds": time.perf_counter()-start,
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "model_calls": 0, "new_participants": 0,
                "selection": "Existing paper selected for public raw data and relevance. Analyses are exploratory, not preregistered; early raw previews informed discrepancy checks.",
                "scope": ["Table1 STEU score reproduction", "Table2 released LIWC score analysis", "Table3 user rating reanalysis"],
                "unreproduced": ["Fresh GPT-3.5 outputs", "New human user study", "LIWC dictionary application", "Experiment3 LIWC/Table4: corresponding raw scores not released in the three-file node"],
                "summary_sha256": sha256(output/"summary.json"),
                "artifacts": {name: sha256(output/name) for name in ("summary.json", "runner.py", "config.json", "report.md")}}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/papers/chain_emotion"))
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    result = run(args.output, args.cache, fetch=args.fetch, seed=args.seed, bootstrap_samples=args.bootstrap_samples)
    print(json.dumps({"output": str(args.output), "STEU_counts": [row["correct"] for row in result["experiment_1"]["conditions"]],
                      "participants": result["experiment_3"]["participants"], "model_calls": 0}))


if __name__ == "__main__":
    main()

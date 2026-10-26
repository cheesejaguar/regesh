"""Post hoc neutral gate for the embedding sensor's zero-shot valence on conversation turns.

The frozen sensor study found that neutral small talk receives a mild positive
valence (mean |v| .20 on neutral DailyDialog turns). This declared post hoc
analysis selects a gate on the validation split only (set the emitted valence
to zero when the GoEmotions "neutral" probability is at least tau or |v| is
below a dead band) by three-way macro-F1, then reports the chosen gate once on
the test split. No model is retrained; no network is used.

Run: python scripts/sensor_neutral_gate.py --output results/bonding/sensor_gate
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from regesh.bonding.embeddings import EmbeddingCache, dailydialog_dialogues
from regesh.bonding.sensor import Sensor, DAILYDIALOG_VALENCE, valence_from_probabilities


def turns(cache_dir, split):
    path, dialogues = dailydialog_dialogues(cache_dir, split)
    texts, gold = [], []
    for d in dialogues:
        for t in d["dialogue"]:
            texts.append(t["text"]); gold.append(int(np.sign(DAILYDIALOG_VALENCE[t["emotion"]])))
    return path, texts, np.array(gold)


def three_way(values, gold):
    pred = np.sign(values).astype(int)
    rows = {}
    for c in (-1, 0, 1):
        tp = int(((pred == c) & (gold == c)).sum()); fp = int(((pred == c) & (gold != c)).sum()); fn = int(((pred != c) & (gold == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.; r = tp / (tp + fn) if tp + fn else 0.
        rows[str(c)] = {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if p + r else 0., "support": int((gold == c).sum())}
    emotional = gold != 0
    return {"accuracy": float((pred == gold).mean()), "macro_f1": float(np.mean([rows[c]["f1"] for c in rows])), "per_class": rows,
            "sign_agreement_emotional": float((pred[emotional] == gold[emotional]).mean()) if emotional.any() else None,
            "emotional_turns_zeroed": float((pred[emotional] == 0).mean()) if emotional.any() else None,
            "neutral_turns_nonzero": float((pred[~emotional] != 0).mean()) if (~emotional).any() else None}


def gated(v, p_neutral, tau, band):
    out = v.copy(); out[(p_neutral >= tau) | (np.abs(v) < band)] = 0.
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", default="results/bonding/sensor/sensor.json")
    parser.add_argument("--embeddings", default=".cache/embeddings/dailydialog-regesh-qwen-coder.npz")
    parser.add_argument("--dailydialog", default=".cache/dailydialog/parlai")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    sensor = Sensor.load(args.sensor); cache = EmbeddingCache(args.embeddings)
    neutral = sensor.labels.index("neutral")
    taus = [1.01, .9, .8, .7, .6, .5, .4, .3]; bands = [0., .1, .2, .3, .4]
    result = {"schema_version": 1, "status": "post hoc; gate selected on the validation split by three-way macro-F1, reported once on test",
              "sensor_digest": sensor.digest, "grid": {"tau": taus, "dead_band": bands}, "splits": {}}
    chosen = None
    for split in ("valid", "test"):
        path, texts, gold = turns(args.dailydialog, split)
        probs = sensor.probabilities(cache.get(texts)); v = np.array([valence_from_probabilities(p, sensor.labels, sensor.valence_signs) for p in probs])
        p_neutral = probs[:, neutral]
        grid = {f"tau={t}|band={b}": three_way(gated(v, p_neutral, t, b), gold) for t in taus for b in bands}
        ungated = three_way(v, gold); all_zero = three_way(np.zeros_like(v), gold)
        result["splits"][split] = {"source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "turns": len(texts),
                                   "ungated": ungated, "all_zero": all_zero, "grid": grid if split == "valid" else None,
                                   "mean_p_neutral_on_neutral_turns": float(p_neutral[gold == 0].mean()), "mean_p_neutral_on_emotional_turns": float(p_neutral[gold != 0].mean())}
        if split == "valid":
            chosen = max(grid, key=lambda k: grid[k]["macro_f1"])
            result["chosen_gate"] = {"key": chosen, "tau": float(chosen.split("|")[0][4:]), "dead_band": float(chosen.split("|")[1][5:]), "valid_metrics": grid[chosen]}
        else:
            result["splits"][split]["chosen_gate"] = grid[chosen]
    result["interpretation"] = ("A gate trades emotional-turn recall for fewer spurious valences on neutral turns. The chosen gate is a "
                                "validation-split choice; its test numbers are a single evaluation. The sensor itself is unchanged.")
    (out / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    (out / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.name != "manifest.json"}}, indent=2) + "\n")
    c = result["chosen_gate"]; t = result["splits"]["test"]
    print(json.dumps({"chosen": c["key"], "valid_macro_f1": c["valid_metrics"]["macro_f1"], "test_gated": {k: t["chosen_gate"][k] for k in ("accuracy", "macro_f1", "sign_agreement_emotional", "emotional_turns_zeroed", "neutral_turns_nonzero")},
                      "test_ungated": {k: t["ungated"][k] for k in ("accuracy", "macro_f1", "sign_agreement_emotional", "neutral_turns_nonzero")},
                      "test_all_zero": {k: t["all_zero"][k] for k in ("accuracy", "macro_f1")}}, indent=1))


if __name__ == "__main__":
    main()

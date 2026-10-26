"""Select the neutral gate for the offline hashed-token text-affect estimator.

Same procedure as scripts/sensor_neutral_gate.py but for the packaged GoEmotions
Bernoulli reference used by the hooks when no embedding server is configured:
choose (tau on the neutral probability, dead band on |valence|) on the DailyDialog
validation split by three-way macro-F1, report once on test. No training, no network.

Run: python scripts/text_affect_gate.py --output results/bonding/text_affect_gate
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from regesh.bonding.embeddings import dailydialog_dialogues
from regesh.bonding.sensor import DAILYDIALOG_VALENCE
from regesh.integrations.text_affect import OfflineTextAffect


def three_way(values, gold):
    pred = np.sign(values).astype(int); rows = {}
    for c in (-1, 0, 1):
        tp = int(((pred == c) & (gold == c)).sum()); fp = int(((pred == c) & (gold != c)).sum()); fn = int(((pred != c) & (gold == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.; r = tp / (tp + fn) if tp + fn else 0.
        rows[str(c)] = {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if p + r else 0., "support": int((gold == c).sum())}
    emotional = gold != 0
    return {"accuracy": float((pred == gold).mean()), "macro_f1": float(np.mean([rows[c]["f1"] for c in rows])), "per_class": rows,
            "sign_agreement_emotional": float((pred[emotional] == gold[emotional]).mean()),
            "emotional_turns_zeroed": float((pred[emotional] == 0).mean()), "neutral_turns_nonzero": float((pred[~emotional] != 0).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dailydialog", default=".cache/dailydialog/parlai"); parser.add_argument("--output", required=True)
    args = parser.parse_args(); out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    est = OfflineTextAffect(); neutral = est.labels.index("neutral")
    taus = [1.01, .9, .8, .7, .6, .5, .4, .3, .2]; bands = [0., .1, .2, .3, .4, .5, .6]
    result = {"schema_version": 1, "estimator": est.identifier, "grid": {"tau": taus, "dead_band": bands}, "splits": {}}
    chosen = None
    for split in ("valid", "test"):
        path, dialogues = dailydialog_dialogues(args.dailydialog, split)
        texts = [t["text"] for d in dialogues for t in d["dialogue"]]
        gold = np.array([int(np.sign(DAILYDIALOG_VALENCE[t["emotion"]])) for d in dialogues for t in d["dialogue"]])
        probs = np.vstack([est.probabilities(t) for t in texts])
        raw = np.array([est.estimate(t, gate=None)["raw_valence"] for t in texts]); p_neutral = probs[:, neutral]
        def gated(t, b):
            v = raw.copy(); v[(p_neutral >= t) | (np.abs(raw) < b)] = 0.; return v
        grid = {f"tau={t}|band={b}": three_way(gated(t, b), gold) for t in taus for b in bands}
        result["splits"][split] = {"source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "turns": len(texts),
                                   "ungated": three_way(raw, gold), "all_zero": three_way(np.zeros_like(raw), gold), "grid": grid if split == "valid" else None}
        if split == "valid":
            chosen = max(grid, key=lambda k: grid[k]["macro_f1"])
            result["chosen_gate"] = {"key": chosen, "tau": float(chosen.split("|")[0][4:]), "dead_band": float(chosen.split("|")[1][5:]), "valid_metrics": grid[chosen]}
        else:
            result["splits"][split]["chosen_gate"] = grid[chosen]
    (out / "gate.json").write_text(json.dumps(result, indent=2) + "\n"); (out / "source.py").write_bytes(Path(__file__).read_bytes())
    (out / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.name != "manifest.json"}}, indent=2) + "\n")
    c = result["chosen_gate"]; t = result["splits"]["test"]
    print(json.dumps({"chosen": c["key"], "valid_macro_f1": c["valid_metrics"]["macro_f1"],
                      "test_gated": {k: t["chosen_gate"][k] for k in ("accuracy", "macro_f1", "sign_agreement_emotional", "emotional_turns_zeroed", "neutral_turns_nonzero")},
                      "test_ungated": {k: t["ungated"][k] for k in ("accuracy", "macro_f1", "sign_agreement_emotional", "neutral_turns_nonzero")},
                      "test_all_zero": {k: t["all_zero"][k] for k in ("accuracy", "macro_f1")}}, indent=1))


if __name__ == "__main__":
    main()

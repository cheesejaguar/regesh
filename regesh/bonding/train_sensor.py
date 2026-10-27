"""Train and evaluate the partner-state sensor heads under the frozen protocol.

Run (after the embedding caches and their manifests exist; the trainer never queries the server):
  OPENBLAS_NUM_THREADS=2 python -m regesh.bonding.train_sensor --config configs/bonding_sensor.json \
      --embeddings .cache/embeddings --data .cache/regesh-data --dailydialog .cache/dailydialog/parlai \
      --output results/bonding/sensor

E1 trains linear and one-hidden-layer heads on standardized embeddings of the hash-selected 35% of
GoEmotions train rows (development rows excluded, used for early stopping and the global threshold)
and evaluates once on the official test set with the archived baseline's metrics, beside a constant
prior and the hashed-token Bernoulli reference refit on the same rows. E2 trains 7-class heads on
the hash-selected 20% of DailyDialog train dialogues, early-stops on valid, tests on the full test
split beside majority and Bernoulli baselines, and reports the GoEmotions sensor's zero-shot valence
sign agreement on the same turns. Outputs: trained heads (JSON), the deployable sensor bundle,
metrics, numeric predictions (no text), summary.json and manifest.json with SHA-256 digests.
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

from ..dataset_baseline import BernoulliReference, calibrate, evaluate, is_development, matrix, select_threshold, ranking_scores
from .embeddings import EmbeddingCache, goemotions_rows, dailydialog_dialogues, select_by_hash, DAILYDIALOG_EMOTIONS
from . import sensor as S


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _training_kwargs(cfg):
    t = cfg["training"]
    return {k: t[k] for k in ("epochs", "batch", "lr", "l2", "patience", "seed")}


def _per_item_counts(gold, predicted):
    """Per-item, per-class true positives, false positives and false negatives (n x k each)."""
    gold = np.asarray(gold, bool); predicted = np.asarray(predicted, bool)
    return (gold & predicted).astype(np.int64), (~gold & predicted).astype(np.int64), (gold & ~predicted).astype(np.int64)


def _f1_stats(gold, predicted):
    tp, fp, fn = _per_item_counts(gold, predicted)
    def micro(idx):
        a = tp[idx].sum(); return 2. * a / max(1, 2 * a + fp[idx].sum() + fn[idx].sum())
    def macro(idx):
        return S.macro_f1_from_counts(tp[idx].sum(axis=0), fp[idx].sum(axis=0), fn[idx].sum(axis=0))
    return {"micro_f1": micro, "macro_f1": macro}


def _one_hot(indices, k):
    out = np.zeros((len(indices), k), bool); out[np.arange(len(indices)), np.asarray(indices, int)] = True
    return out


def _bootstrap_block(stats, n, boot):
    return {name: S.bootstrap(fn, n, samples=boot["samples"], seed=boot["seed"]) for name, fn in stats.items()}


def _difference_block(stats_a, stats_b, n, boot):
    return {name: S.bootstrap(lambda idx, a=stats_a[name], b=stats_b[name]: a(idx) - b(idx), n, samples=boot["samples"], seed=boot["seed"]) for name in stats_a}


# ----------------------------------------------------------------------------- E1 GoEmotions

def run_goemotions(cfg, data_dir, cache_dir, output, log):
    g = cfg["goemotions"]; boot = cfg["bootstrap"]; data_dir = Path(data_dir); cache_dir = Path(cache_dir)
    labels = (data_dir / g["labels_file"]).read_text().splitlines()
    if tuple(labels) != S.GOEMOTIONS_LABELS:
        raise ValueError("labels file disagrees with the declared GoEmotions label order")
    train_path, train_rows = goemotions_rows(data_dir, "train"); test_path, test_rows = goemotions_rows(data_dir, "test")
    ids = [r[0] for r in train_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate official training IDs")
    keep = select_by_hash(ids, g["train_fraction"], g["train_salt"])
    selected = [r for r, k in zip(train_rows, keep) if k]
    fit = [r for r in selected if not is_development(r[0])]; dev = [r for r in selected if is_development(r[0])]
    test_ids = {r[0] for r in test_rows}
    if len(test_ids) != len(test_rows) or test_ids & set(ids):
        raise ValueError("duplicate or overlapping official test IDs")
    if not fit or not dev:
        raise ValueError("empty fit or development partition")
    cache = EmbeddingCache(cache_dir / cfg["embedding_cache_files"]["goemotions"])
    x_fit_raw = cache.get([r[1] for r in fit]); x_dev_raw = cache.get([r[1] for r in dev]); x_test_raw = cache.get([r[1] for r in test_rows])
    mean, std = S.fit_standardizer(x_fit_raw)
    x_fit, x_dev, x_test = (S.standardize(x, mean, std) for x in (x_fit_raw, x_dev_raw, x_test_raw))
    k = len(labels); y_fit = matrix(fit, k); y_dev = matrix(dev, k); gold = matrix(test_rows, k)
    log(f"E1 GoEmotions: fit {len(fit)} / development {len(dev)} / test {len(test_rows)} rows; dimension {x_fit.shape[1]}")
    out = {"counts": {"official_train": len(train_rows), "selected_35": len(selected), "fit": len(fit), "development": len(dev), "test": len(test_rows), "labels": k},
           "dimension": int(x_fit.shape[1]), "source_sha256": {"train": sha(train_path), "test": sha(test_path), "labels": sha(data_dir / g["labels_file"])},
           "standardization": {"mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(), "std_sha256": hashlib.sha256(std.tobytes()).hexdigest(),
                               "std_min": float(std.min()), "std_max": float(std.max()), "raw_norm_mean": float(np.linalg.norm(x_fit_raw, axis=1).mean())},
           "models": {}, "comparators": {}}
    heads, probabilities, predictions, stats = {}, {}, {}, {}
    for name, spec in g["models"].items():
        t0 = time.perf_counter()
        model, history = S.train_head(x_fit, y_fit.astype(float), x_dev, y_dev.astype(float), hidden=spec["hidden"], activation=spec.get("activation"), **_training_kwargs(cfg))
        seconds = time.perf_counter() - t0
        p_dev = model.predict(x_dev); threshold, dev_f1 = select_threshold(p_dev, y_dev)
        p_test = model.predict(x_test); metrics, predicted = evaluate(gold, p_test, threshold, labels)
        model.metadata.update(task="goemotions-28-multilabel", labels=labels, threshold=threshold, development_micro_f1=dev_f1, training_seconds=seconds,
                              standardization="per-dimension mean/std of the fit rows; stored in sensor.json", trained_utc=datetime.now(timezone.utc).isoformat())
        model.save(output / f"goemotions_{name}.json")
        heads[name] = model; probabilities[name] = p_test; predictions[name] = predicted; stats[name] = _f1_stats(gold, predicted)
        out["models"][name] = {"sizes": model.sizes, "parameters": model.metadata["parameters"], "epochs_run": model.metadata["epochs_run"], "best_dev_bce": model.metadata["best_dev_loss"],
                               "history": history, "threshold": threshold, "development_micro_f1": dev_f1, "training_seconds": seconds,
                               "head_file": f"goemotions_{name}.json", "head_digest": model.to_dict()["digest"], "test": metrics}
        log(f"  {name}: epochs {model.metadata['epochs_run']}, dev micro-F1 {dev_f1:.4f} at {threshold:.2f}, test micro-F1 {metrics['micro_f1']:.4f} macro-F1 {metrics['macro_f1']:.4f} ({seconds:.0f}s)")
    # Constant prior from the same fit rows, own development threshold.
    prior = (y_fit.sum(axis=0) + 1.) / (len(fit) + 2.)
    prior_threshold, prior_dev_f1 = select_threshold(np.tile(prior, (len(dev), 1)), y_dev)
    prior_metrics, prior_pred = evaluate(gold, np.tile(prior, (len(test_rows), 1)), prior_threshold, labels)
    stats["constant_prior"] = _f1_stats(gold, prior_pred)
    out["comparators"]["constant_prior"] = {"threshold": prior_threshold, "development_micro_f1": prior_dev_f1, "test": prior_metrics}
    # Hashed-token Bernoulli reference on exactly the same rows.
    t0 = time.perf_counter(); bern = BernoulliReference.fit(fit, labels); tuning = calibrate(bern, dev, y_dev); bern_seconds = time.perf_counter() - t0
    p_bern = bern.probabilities(test_rows); bern_metrics, bern_pred = evaluate(gold, p_bern, bern.threshold, labels)
    bern.save(output / "goemotions_bernoulli_35.npz"); probabilities["bernoulli_35"] = p_bern; stats["bernoulli_35"] = _f1_stats(gold, bern_pred)
    out["comparators"]["bernoulli_35"] = {"tuning": tuning, "training_seconds": bern_seconds, "model_file": "goemotions_bernoulli_35.npz", "test": bern_metrics}
    log(f"  bernoulli_35: test micro-F1 {bern_metrics['micro_f1']:.4f} macro-F1 {bern_metrics['macro_f1']:.4f}; constant prior micro-F1 {prior_metrics['micro_f1']:.4f}")
    archived = _read_json(Path(g["archived_reference_metrics"]))
    out["comparators"]["archived_reference"] = None if archived is None else {
        "file": g["archived_reference_metrics"], "micro_f1": archived["reference"]["micro_f1"], "macro_f1": archived["reference"]["macro_f1"],
        "support_weighted_f1": archived["reference"].get("support_weighted_f1"), "threshold": archived["reference"]["threshold"], "samples": archived["reference"]["samples"],
        "constant_prior_micro_f1": archived["constant_prior"]["micro_f1"], "training_rows": "39,013 fit rows (100% of official train minus development)"}
    n = len(test_rows)
    out["bootstrap"] = {name: _bootstrap_block(st, n, boot) for name, st in stats.items()}
    out["paired_differences"] = {}
    for a, b in [(m, c) for m in g["models"] for c in ("bernoulli_35", "constant_prior")] + [("mlp", "linear")]:
        if a in stats and b in stats:
            out["paired_differences"][f"{a}-{b}"] = _difference_block(stats[a], stats[b], n, boot)
    # Deployed sensor: higher development micro-F1, ties to the first declared model (linear).
    best = max(g["models"], key=lambda m: (out["models"][m]["development_micro_f1"], -list(g["models"]).index(m)))
    sensor = S.Sensor(heads[best], labels, mean, std, threshold=out["models"][best]["threshold"],
                      metadata={"selected_head": best, "selection_rule": g["deployed_head_selection"], "development_micro_f1": out["models"][best]["development_micro_f1"],
                                "embedding_model": cfg["embedding_model"], "protocol_frozen_utc": cfg["protocol_frozen_utc"], "fit_rows": len(fit)})
    sensor.save(output / "sensor.json")
    out["deployed_sensor"] = {"file": "sensor.json", "selected_head": best, "digest": sensor.digest, "dimension": sensor.dimension}
    with gzip.GzipFile(str(output / "goemotions_predictions.jsonl.gz"), mode="wb", mtime=0) as stream:
        for i, (identifier, _, human) in enumerate(test_rows):
            row = {"id": identifier, "gold_label_ids": list(human)}
            for name, p in probabilities.items():
                row[f"{name}_probabilities"] = [round(float(v), 4) for v in p[i]]
            stream.write((json.dumps(row, separators=(",", ":")) + "\n").encode())
    (output / "goemotions_metrics.json").write_text(json.dumps({"models": {m: out["models"][m]["test"] for m in out["models"]}, "comparators": {c: v.get("test") if isinstance(v, dict) else v for c, v in out["comparators"].items()}}, indent=2, allow_nan=False) + "\n")
    split = {"fit": sorted(r[0] for r in fit), "development": sorted(r[0] for r in dev), "test": sorted(test_ids)}
    return out, sensor, split


# ----------------------------------------------------------------------------- E2 DailyDialog

def dailydialog_turns(dialogues, keep=None, emotions=DAILYDIALOG_EMOTIONS):
    rows = []
    for i, d in enumerate(dialogues):
        if keep is not None and not keep[i]:
            continue
        for j, t in enumerate(d["dialogue"]):
            rows.append((i, j, t["text"], emotions.index(t["emotion"])))
    return rows


def run_dailydialog(cfg, dd_dir, cache_dir, output, log, ge_sensor):
    dcfg = cfg["dailydialog"]; boot = cfg["bootstrap"]; dd_dir = Path(dd_dir); cache_dir = Path(cache_dir)
    emotions = tuple(dcfg["emotions"])
    if emotions != DAILYDIALOG_EMOTIONS:
        raise ValueError("declared DailyDialog emotions disagree with regesh.bonding.embeddings")
    train_path, train_dlg = dailydialog_dialogues(dd_dir, "train"); valid_path, valid_dlg = dailydialog_dialogues(dd_dir, "valid"); test_path, test_dlg = dailydialog_dialogues(dd_dir, "test")
    ids = [f"train-{i}" for i in range(len(train_dlg))]; keep = select_by_hash(ids, dcfg["train_fraction"], dcfg["train_salt"])
    train = dailydialog_turns(train_dlg, keep, emotions); valid = dailydialog_turns(valid_dlg, None, emotions); test = dailydialog_turns(test_dlg, None, emotions)
    cache = EmbeddingCache(cache_dir / cfg["embedding_cache_files"]["dailydialog"])
    x_tr_raw = cache.get([t[2] for t in train]); x_va_raw = cache.get([t[2] for t in valid]); x_te_raw = cache.get([t[2] for t in test])
    mean, std = S.fit_standardizer(x_tr_raw)
    x_tr, x_va, x_te = (S.standardize(x, mean, std) for x in (x_tr_raw, x_va_raw, x_te_raw))
    k = len(emotions); y_tr = _one_hot([t[3] for t in train], k); y_va = _one_hot([t[3] for t in valid], k)
    gold = np.array([t[3] for t in test], int); scale = np.array([dcfg["valence_scale"][e] for e in emotions], float); valence_gold = scale[gold]
    train_counts = y_tr.sum(axis=0)
    log(f"E2 DailyDialog: train {len(train)} turns from {int(keep.sum())} dialogues / valid {len(valid)} / test {len(test)}")
    out = {"counts": {"train_dialogues_total": len(train_dlg), "train_dialogues_selected": int(keep.sum()), "train_turns": len(train), "valid_turns": len(valid), "test_turns": len(test), "classes": k},
           "dimension": int(x_tr.shape[1]), "source_sha256": {"train": sha(train_path), "valid": sha(valid_path), "test": sha(test_path)},
           "class_distribution": {"train": {e: int(c) for e, c in zip(emotions, train_counts)}, "test": {e: int((gold == i).sum()) for i, e in enumerate(emotions)}},
           "models": {}, "baselines": {}}
    predicted, stats, soft = {}, {}, {}
    def _stats(pred):
        g1, p1 = _one_hot(gold, k), _one_hot(pred, k); f1 = _f1_stats(g1, p1); pv = scale[pred]
        return {"accuracy": lambda idx: float(np.mean(gold[idx] == pred[idx])), "macro_f1": f1["macro_f1"], "valence_mae": lambda idx: S.valence_mae(valence_gold[idx], pv[idx])}
    def _metrics(pred):
        m = S.multiclass_metrics(gold, pred, list(emotions)); m["valence_mae"] = S.valence_mae(valence_gold, scale[pred]); return m
    for name, spec in dcfg["models"].items():
        t0 = time.perf_counter()
        model, history = S.train_head(x_tr, y_tr.astype(float), x_va, y_va.astype(float), hidden=spec["hidden"], activation=spec.get("activation"), **_training_kwargs(cfg))
        seconds = time.perf_counter() - t0
        p = model.predict(x_te); pred = p.argmax(axis=1); m = _metrics(pred)
        m["valence_mae_soft"] = S.valence_mae(valence_gold, (p @ scale) / np.maximum(p.sum(axis=1), 1e-12))
        m["valid_accuracy"] = float(np.mean(model.predict(x_va).argmax(axis=1) == y_va.argmax(axis=1)))
        model.metadata.update(task="dailydialog-7-class", labels=list(emotions), decoding="argmax of sigmoid outputs", training_seconds=seconds, trained_utc=datetime.now(timezone.utc).isoformat())
        model.save(output / f"dailydialog_{name}.json")
        predicted[name] = pred; stats[name] = _stats(pred); soft[name] = (p @ scale) / np.maximum(p.sum(axis=1), 1e-12)
        out["models"][name] = {"sizes": model.sizes, "parameters": model.metadata["parameters"], "epochs_run": model.metadata["epochs_run"], "best_valid_bce": model.metadata["best_dev_loss"],
                               "history": history, "training_seconds": seconds, "head_file": f"dailydialog_{name}.json", "head_digest": model.to_dict()["digest"], "test": m}
        log(f"  {name}: epochs {model.metadata['epochs_run']}, test accuracy {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f} valence MAE {m['valence_mae']:.4f} ({seconds:.0f}s)")
    majority = int(train_counts.argmax()); pred_major = np.full(len(test), majority, int)
    predicted["majority"] = pred_major; stats["majority"] = _stats(pred_major)
    out["baselines"]["majority"] = {"class": emotions[majority], "test": _metrics(pred_major)}
    t0 = time.perf_counter()
    bern = BernoulliReference.fit([(f"train-{i}-{j}", text, (label,)) for i, j, text, label in train], list(emotions)); bern_seconds = time.perf_counter() - t0
    pred_bern = bern.probabilities([(f"test-{i}-{j}", text, (label,)) for i, j, text, label in test]).argmax(axis=1)
    bern.save(output / "dailydialog_bernoulli_20.npz"); predicted["bernoulli_20"] = pred_bern; stats["bernoulli_20"] = _stats(pred_bern)
    out["baselines"]["bernoulli_20"] = {"model_file": "dailydialog_bernoulli_20.npz", "decoding": "argmax", "training_seconds": bern_seconds, "test": _metrics(pred_bern)}
    log(f"  majority accuracy {out['baselines']['majority']['test']['accuracy']:.4f} macro-F1 {out['baselines']['majority']['test']['macro_f1']:.4f}; bernoulli_20 accuracy {out['baselines']['bernoulli_20']['test']['accuracy']:.4f} macro-F1 {out['baselines']['bernoulli_20']['test']['macro_f1']:.4f}")
    n = len(test)
    out["bootstrap"] = {name: _bootstrap_block(st, n, boot) for name, st in stats.items()}
    out["paired_differences"] = {f"{a}-{b}": _difference_block(stats[a], stats[b], n, boot) for a in dcfg["models"] for b in ("bernoulli_20", "majority")}
    if "mlp" in stats and "linear" in stats:
        out["paired_differences"]["mlp-linear"] = _difference_block(stats["mlp"], stats["linear"], n, boot)
    # Zero-shot: the deployed GoEmotions sensor on raw DailyDialog test embeddings.
    z = dcfg["zero_shot"]; p_ge = ge_sensor.probabilities(x_te_raw)
    signs = np.array([ge_sensor.valence_signs[l] for l in ge_sensor.labels], float)
    v = np.clip((p_ge @ signs) / np.maximum(p_ge.sum(axis=1), 1e-12), -1., 1.)
    agreement = S.sign_agreement(valence_gold, v); mask = valence_gold != 0
    auc, ap = ranking_scores(v[mask], valence_gold[mask] > 0) if mask.any() else (None, None)
    three = S.three_way_sign(v, z["dead_band"])
    out["zero_shot"] = {
        "sensor_digest": ge_sensor.digest, "selected_head": ge_sensor.metadata.get("selected_head"),
        "sign_agreement": {**agreement, "bootstrap": S.bootstrap(lambda idx: float(np.mean(np.sign(v[idx][valence_gold[idx] != 0]) == valence_gold[idx][valence_gold[idx] != 0])) if (valence_gold[idx] != 0).any() else 0., n, samples=boot["samples"], seed=boot["seed"])},
        "baselines": {"chance": .5, "majority_sign": agreement["positive_fraction"]},
        "roc_auc_positive_vs_negative": auc, "average_precision_positive": ap,
        "three_way": {"dead_band": z["dead_band"], "accuracy": float(np.mean(three == valence_gold)), "predict_zero_everywhere": float(np.mean(valence_gold == 0)),
                      "confusion_gold_rows_neg_zero_pos": [[int(((valence_gold == gv) & (three == pv)).sum()) for pv in (-1., 0., 1.)] for gv in (-1., 0., 1.)]},
        "mean_valence_by_emotion": {e: {"n": int((gold == i).sum()), "mean": float(v[gold == i].mean()) if (gold == i).any() else None} for i, e in enumerate(emotions)},
        "mean_abs_valence_no_emotion": float(np.abs(v[gold == 0]).mean()) if (gold == 0).any() else None,
        "mean_abs_valence_emotional": float(np.abs(v[mask]).mean()) if mask.any() else None,
        "mean_support_proxy": float(np.mean([S.support_proxy_from_probabilities(row, ge_sensor.labels, ge_sensor.support_labels) for row in p_ge])),
    }
    log(f"  zero-shot sign agreement {agreement['agreement']:.4f} (n {agreement['n']}, majority sign {agreement['positive_fraction']:.4f}), AUC {auc if auc is None else round(auc, 4)}")
    with gzip.GzipFile(str(output / "dailydialog_predictions.jsonl.gz"), mode="wb", mtime=0) as stream:
        for idx, (i, j, _, label) in enumerate(test):
            row = {"dialogue": i, "turn": j, "gold": int(label), **{name: int(pred[idx]) for name, pred in predicted.items()},
                   **{f"{name}_soft_valence": round(float(sv[idx]), 4) for name, sv in soft.items()}, "zero_shot_valence": round(float(v[idx]), 4)}
            stream.write((json.dumps(row, separators=(",", ":")) + "\n").encode())
    (output / "dailydialog_metrics.json").write_text(json.dumps({"models": {m: out["models"][m]["test"] for m in out["models"]}, "baselines": {b: out["baselines"][b]["test"] for b in out["baselines"]}, "zero_shot": out["zero_shot"]}, indent=2, allow_nan=False) + "\n")
    return out, {"train_dialogues": [i for i, kept in zip(ids, keep) if kept]}


# ----------------------------------------------------------------------------- orchestration

def _headline(summary):
    g = summary["goemotions"]; d = summary["dailydialog"]
    def f1s(block, key):
        return {"micro_f1": block[key]["micro_f1"]["estimate"], "micro_f1_ci95": block[key]["micro_f1"]["ci95"], "macro_f1": block[key]["macro_f1"]["estimate"], "macro_f1_ci95": block[key]["macro_f1"]["ci95"]}
    head = {"goemotions": {name: f1s(g["bootstrap"], name) for name in g["bootstrap"]}, "goemotions_archived_reference": g["comparators"]["archived_reference"],
            "goemotions_differences": {k: {m: [v["estimate"], v["ci95"]] for m, v in blk.items()} for k, blk in g["paired_differences"].items()},
            "dailydialog": {name: {m: [v["estimate"], v["ci95"]] for m, v in blk.items()} for name, blk in d["bootstrap"].items()},
            "dailydialog_differences": {k: {m: [v["estimate"], v["ci95"]] for m, v in blk.items()} for k, blk in d["paired_differences"].items()},
            "zero_shot_sign_agreement": d["zero_shot"]["sign_agreement"], "zero_shot_baselines": d["zero_shot"]["baselines"], "zero_shot_roc_auc": d["zero_shot"]["roc_auc_positive_vs_negative"],
            "deployed_sensor": g["deployed_sensor"]}
    return head


def run(config_path, embeddings_dir, data_dir, dailydialog_dir, output, log=print):
    config_path, output, embeddings_dir = Path(config_path), Path(output), Path(embeddings_dir)
    cfg = json.loads(config_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; preserve prior measurements")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_bytes(config_path.read_bytes())
    (output / "source_train_sensor.py").write_bytes(Path(__file__).read_bytes())
    (output / "source_sensor.py").write_bytes((Path(__file__).parent / "sensor.py").read_bytes())
    started = datetime.now(timezone.utc).isoformat(); t0 = time.perf_counter()
    manifests = {k: _read_json(embeddings_dir / v) for k, v in cfg["embedding_manifests"].items()}
    for k, m in manifests.items():
        if m is None:
            log(f"warning: embedding manifest {k} missing under {embeddings_dir}")
    ge, sensor, ge_split = run_goemotions(cfg, data_dir, embeddings_dir, output, log)
    dd, dd_split = run_dailydialog(cfg, dailydialog_dir, embeddings_dir, output, log, sensor)
    (output / "split_ids.json").write_text(json.dumps({"goemotions": ge_split, "dailydialog": dd_split}, indent=2) + "\n")
    summary = {"schema_version": 1, "title": cfg["title"], "protocol_frozen_utc": cfg["protocol_frozen_utc"], "started_utc": started,
               "goemotions": ge, "dailydialog": dd, "embedding_manifests": manifests}
    summary["headline"] = _headline(summary)
    summary["interpretation"] = ("Recognition quality of a frozen-embedding head against hashed-token references on labelled corpora. The 35% / 20% training "
                                 "subsets, the fixed valence map and the proxy nature of support are declared in config.json. Nothing here measures benefit to users; "
                                 "sensor outputs are machine interpretations that fill affect evidence only and never establish facts, authority or permissions.")
    summary["runtime_seconds"] = time.perf_counter() - t0
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    _write_report(output, summary)
    manifest = {"schema_version": 1, "finished_utc": datetime.now(timezone.utc).isoformat(), "python": _platform.python_version(), "numpy": np.__version__,
                "command": f"python -m regesh.bonding.train_sensor --config {config_path} --embeddings {embeddings_dir} --data {data_dir} --dailydialog {dailydialog_dir} --output {output}",
                "config_sha256": sha(config_path), "source_sha256": {"train_sensor.py": sha(__file__), "sensor.py": sha(Path(__file__).parent / "sensor.py"), "mlp.py": sha(Path(__file__).parent / "mlp.py"),
                                                                     "embeddings.py": sha(Path(__file__).parent / "embeddings.py"), "dataset_baseline.py": sha(Path(__file__).parent.parent / "dataset_baseline.py")},
                "thread_variables": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}, "cpu_count": os.cpu_count(),
                "live_model_calls": 0, "embedding_cache_files": {k: (sha(embeddings_dir / v) if (embeddings_dir / v).exists() else None) for k, v in cfg["embedding_cache_files"].items()},
                "files": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def _write_report(output, summary):
    g = summary["goemotions"]; d = summary["dailydialog"]; h = summary["headline"]
    lines = [f"# {summary['title']}", "", f"Protocol frozen {summary['protocol_frozen_utc']}; run started {summary['started_utc']}. Numbers are single frozen-run test results with 95% item-bootstrap intervals.", "",
             f"## E1 GoEmotions (fit {g['counts']['fit']} / development {g['counts']['development']} / test {g['counts']['test']}; dimension {g['dimension']})", "",
             "| Model | Threshold | Micro-F1 [95%] | Macro-F1 [95%] | Support-weighted F1 | Hamming | Brier | Micro ECE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in list(g["models"]) + ["bernoulli_35", "constant_prior"]:
        t = g["models"][name]["test"] if name in g["models"] else g["comparators"][name]["test"]
        b = g["bootstrap"][name]
        lines.append(f"| {name} | {t['threshold']:.2f} | {t['micro_f1']:.4f} [{b['micro_f1']['ci95'][0]:.4f}, {b['micro_f1']['ci95'][1]:.4f}] | {t['macro_f1']:.4f} [{b['macro_f1']['ci95'][0]:.4f}, {b['macro_f1']['ci95'][1]:.4f}] | {t['support_weighted_f1']:.4f} | {t['hamming_loss']:.4f} | {t['brier']:.4f} | {t['micro_ece']:.4f} |")
    a = g["comparators"]["archived_reference"]
    if a:
        lines.append(f"| archived reference (100% train) | {a['threshold']:.2f} | {a['micro_f1']:.4f} | {a['macro_f1']:.4f} | {a['support_weighted_f1']:.4f} | | | |")
    lines += ["", "Paired differences (micro-F1, macro-F1):", ""]
    for k, blk in g["paired_differences"].items():
        lines.append(f"- {k}: micro {blk['micro_f1']['estimate']:+.4f} [{blk['micro_f1']['ci95'][0]:+.4f}, {blk['micro_f1']['ci95'][1]:+.4f}]; macro {blk['macro_f1']['estimate']:+.4f} [{blk['macro_f1']['ci95'][0]:+.4f}, {blk['macro_f1']['ci95'][1]:+.4f}]")
    lines += ["", f"Deployed sensor: `{g['deployed_sensor']['selected_head']}` head, digest `{g['deployed_sensor']['digest']}`.", "",
              f"## E2 DailyDialog (train {d['counts']['train_turns']} turns from {d['counts']['train_dialogues_selected']} dialogues / valid {d['counts']['valid_turns']} / test {d['counts']['test_turns']})", "",
              "| Model | Accuracy [95%] | Macro-F1 [95%] | Valence MAE [95%] |", "|---|---:|---:|---:|"]
    for name in list(d["models"]) + ["bernoulli_20", "majority"]:
        b = d["bootstrap"][name]
        lines.append(f"| {name} | {b['accuracy']['estimate']:.4f} [{b['accuracy']['ci95'][0]:.4f}, {b['accuracy']['ci95'][1]:.4f}] | {b['macro_f1']['estimate']:.4f} [{b['macro_f1']['ci95'][0]:.4f}, {b['macro_f1']['ci95'][1]:.4f}] | {b['valence_mae']['estimate']:.4f} [{b['valence_mae']['ci95'][0]:.4f}, {b['valence_mae']['ci95'][1]:.4f}] |")
    z = d["zero_shot"]; sa = z["sign_agreement"]
    lines += ["", f"Zero-shot GoEmotions sensor on DailyDialog test turns: sign agreement {sa['agreement']:.4f} [{sa['bootstrap']['ci95'][0]:.4f}, {sa['bootstrap']['ci95'][1]:.4f}] on {sa['n']} emotional turns "
              f"(majority sign {z['baselines']['majority_sign']:.4f}, chance .5); ROC-AUC positive vs negative {z['roc_auc_positive_vs_negative']}; three-way accuracy with dead band {z['three_way']['dead_band']}: {z['three_way']['accuracy']:.4f} vs {z['three_way']['predict_zero_everywhere']:.4f} for predicting 0 everywhere.",
              "", summary["interpretation"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/bonding_sensor.json")
    parser.add_argument("--embeddings", required=True, help="embedding cache directory")
    parser.add_argument("--data", required=True, help="GoEmotions dataset cache directory")
    parser.add_argument("--dailydialog", required=True, help="DailyDialog ParlAI JSON directory")
    parser.add_argument("--output", required=True, help="empty output directory")
    args = parser.parse_args()
    summary = run(args.config, args.embeddings, args.data, args.dailydialog, args.output)
    print(json.dumps(summary["headline"], indent=2))


if __name__ == "__main__":
    main()

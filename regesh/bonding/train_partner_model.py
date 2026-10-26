"""Train and evaluate the learned partner model against the online linear identifier and baselines.

Run: python -m regesh.bonding.train_partner_model --config configs/bonding_partner_model.json \
       --embeddings .cache/embeddings --dailydialog .cache/dailydialog/parlai --output results/bonding/partner_model

The output directory must be empty. The runner never calls an embedding endpoint: every
vector is read from the content-addressed cache written by `regesh.bonding.embeddings`.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform as _platform
import time

import numpy as np

from .embeddings import EmbeddingCache, dailydialog_dialogues, select_by_hash
from . import partner_model as pm

NH = len(pm.HISTORY_SCALARS)
A_VALENCE = pm.HISTORY_SCALARS.index("a_valence")


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def load_split(dailydialog_dir, split, spec):
    path, dialogues = dailydialog_dialogues(dailydialog_dir, split)
    ids = [f"{split}-{i}" for i in range(len(dialogues))]
    keep = select_by_hash(ids, spec["fraction"], spec["salt"])
    selected = [i for i, k in zip(ids, keep) if k]
    return {"path": path, "dialogues": dialogues, "keep": keep, "total": len(dialogues), "selected": int(keep.sum()),
            "selected_ids_sha256": sha(json.dumps(selected)), "source_sha256": sha(path.read_bytes())}


def manifest_check(embeddings_dir, split, info, spec):
    """Compare the embedding job's manifest for this split with the protocol's selection."""
    path = Path(embeddings_dir) / f"dailydialog-{split}-manifest.json"
    if not path.exists():
        return {"status": "manifest missing", "path": str(path)}
    m = json.loads(path.read_text())
    checks = {"fraction": m.get("fraction") == spec["fraction"], "selected_ids_sha256": m.get("selected_ids_sha256") == info["selected_ids_sha256"],
              "source_sha256": m.get("source_sha256") == info["source_sha256"]}
    return {"status": "ok" if all(checks.values()) else "mismatch", "checks": checks, "path": str(path),
            "manifest": {k: m.get(k) for k in ("fraction", "selected_units", "texts", "dimension", "model", "pooling", "finished_utc", "selected_ids_sha256", "source_sha256")}}


def run(config_path, embeddings_dir, dailydialog_dir, output, log=print):
    config_path, output = Path(config_path), Path(output)
    cfg = json.loads(config_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; preserve prior measurements")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_bytes(config_path.read_bytes())
    (output / "source_partner_model.py").write_bytes(Path(pm.__file__).read_bytes())
    (output / "source_train_partner_model.py").write_bytes(Path(__file__).read_bytes())
    started = datetime.now(timezone.utc).isoformat(); t0 = time.perf_counter()
    vmap = cfg["valence_map"]; emotions = pm.EMOTIONS
    summary = {"schema_version": 1, "title": cfg["title"], "protocol_frozen_utc": cfg["protocol_frozen_utc"], "started_utc": started,
               "data": {"splits": {}}, "models": {}, "training": {}, "timings": {}}

    # ---- data ------------------------------------------------------------------------------------
    splits, examples = {}, {}
    for split, spec in cfg["data"]["splits"].items():
        info = load_split(dailydialog_dir, split, spec); splits[split] = info
        examples[split] = pm.build_examples(info["dialogues"], info["keep"], split)
        tv, tc = pm.targets(examples[split], vmap)
        summary["data"]["splits"][split] = {"dialogues_total": info["total"], "dialogues_selected": info["selected"], "fraction": spec["fraction"],
                                            "selected_ids_sha256": info["selected_ids_sha256"], "source_sha256": info["source_sha256"],
                                            "examples": len(examples[split]), "target_distribution": pm.class_distribution(tc),
                                            "manifest_check": manifest_check(embeddings_dir, split, info, spec)}
        log(f"{split}: {info['selected']}/{info['total']} dialogues, {len(examples[split])} examples, majority {summary['data']['splits'][split]['target_distribution']['majority_fraction']:.3f}, manifest {summary['data']['splits'][split]['manifest_check']['status']}")
    cache = EmbeddingCache(Path(embeddings_dir) / cfg["embeddings"]["cache_file"])
    needed = {split: pm.example_texts(ex) for split, ex in examples.items()}
    missing = {split: len(cache.lookup(texts)[1]) for split, texts in needed.items()}
    summary["data"]["embeddings"] = {"cache_file": str(cache.path), "cache_texts": len(cache.keys), "dimension": int(cache.vectors.shape[1]) if cache.vectors is not None else None,
                                     "texts_needed": {s: len(t) for s, t in needed.items()}, "texts_missing": missing}
    if any(missing.values()):
        raise KeyError(f"embedding cache is missing utterances: {missing}; wait for the embedding job's manifests")
    sd_floor = cfg["embeddings"].get("sd_floor", 1e-3); components = cfg["embeddings"].get("pca_components")
    train_vectors = cache.get(needed["train"]); emb_mean, emb_sd = pm.fit_standardizer(train_vectors, sd_floor)
    projection, explained = (None, None)
    if components:
        projection, explained = pm.fit_projection((train_vectors - emb_mean) / emb_sd, components)
        summary["data"]["embeddings"]["projection"] = {"components": int(components), "fitted_on_utterances": int(len(train_vectors)),
                                                       "explained_variance_fraction": float(explained.sum()), "explained_by_component": [float(v) for v in explained]}
    parts = {s: pm.embedding_parts(ex, cache) for s, ex in examples.items()}
    scal = {s: pm.scalar_features(ex, vmap) for s, ex in examples.items()}
    tgt = {s: pm.targets(ex, vmap) for s, ex in examples.items()}
    keys = {s: [(e["split"], e["dialogue"]) for e in ex] for s, ex in examples.items()}
    tr_v, tr_c = tgt["train"]; dv_v, dv_c = tgt["valid"]; te_v, te_c = tgt["test"]
    summary["timings"]["data_seconds"] = time.perf_counter() - t0
    log(f"features ready in {summary['timings']['data_seconds']:.1f}s; dimension {emb_mean.shape[0]}")

    # ---- baselines -------------------------------------------------------------------------------
    preds = {}
    preds["persistence"] = pm.persistence_baseline(examples["test"], vmap)
    preds["constant"] = pm.constant_baseline(tr_v, tr_c, len(te_v))
    ident_default, thetas_default = pm.identifier_baseline(examples["test"], vmap)
    preds["identifier_default"] = (ident_default, None)
    theta0 = pm.pooled_bilinear_fit(scal["train"][:, A_VALENCE], scal["train"][:, NH], tr_v)
    ip = cfg["models"]["identifier_pooled_prior"]
    ident_pooled, thetas_pooled = pm.identifier_baseline(examples["test"], vmap, theta0=theta0, prior=ip["prior"], lam=ip["lam"])
    preds["identifier_pooled_prior"] = (ident_pooled, None)
    summary["models"]["persistence"] = {"description": cfg["models"]["persistence"]}
    summary["models"]["constant"] = {"description": cfg["models"]["constant"], "mean_train_valence": float(np.mean(tr_v)), "majority_class": emotions[int(preds["constant"][1][0])]}
    summary["models"]["identifier_default"] = {"description": cfg["models"]["identifier_default"], "final_theta_mean_over_test_dialogues": _final_thetas(thetas_default, keys["test"])}
    summary["models"]["identifier_pooled_prior"] = {**ip, "theta0": dict(zip(pm.RecursiveIdentifier.NAMES, map(float, theta0)))}
    grid = cfg["models"]["ridge"]["lambda_grid"]
    ridge_v, ridge_v_scores, lam_v = pm.RidgeModel.select(scal["train"], tr_v, scal["valid"], dv_v, grid)
    onehot = np.eye(len(emotions)); ridge_c, ridge_c_scores, lam_c = pm.RidgeModel.select(scal["train"], onehot[tr_c], scal["valid"], onehot[dv_c], grid)
    preds["ridge"] = (np.clip(ridge_v.predict(scal["test"])[:, 0], -1, 1), ridge_c.predict(scal["test"]).argmax(axis=1))
    summary["models"]["ridge"] = {"description": cfg["models"]["ridge"]["description"], "lambda_valence": lam_v, "lambda_class": lam_c,
                                  "valid_mse_by_lambda_valence": ridge_v_scores, "valid_mse_by_lambda_class": ridge_c_scores,
                                  "valence_weights": dict(zip(pm.SCALAR_FEATURES, map(float, ridge_v.W[:, 0])))}
    summary["timings"]["baselines_seconds"] = time.perf_counter() - t0 - summary["timings"]["data_seconds"]
    log("baselines done")

    # ---- learned models --------------------------------------------------------------------------
    mcfg = cfg["models"]["mlp"]; models = {}; per_seed = {}
    for fs in mcfg["variants"]:
        proto = pm.LearnedPartnerModel(fs, emb_mean, emb_sd, np.ones(len(emotions)) / len(emotions), vmap, embedding_projection=projection)
        x = {s: proto.assemble(*parts[s], scal[s][:, :NH], scal[s][:, NH:]) for s in examples}
        per_seed[fs] = {}
        for seed in mcfg["seeds"]:
            t1 = time.perf_counter()
            model, hist = pm.LearnedPartnerModel.train(fs, x["train"], tr_v, tr_c, x["valid"], dv_v, dv_c, embedding_mean=emb_mean, embedding_sd=emb_sd,
                                                       hidden=tuple(mcfg["hidden"]), activation=mcfg["activation"], lr=mcfg["lr"], l2=mcfg["l2"], batch=mcfg["batch"],
                                                       epochs=mcfg["epochs"], patience=mcfg["patience"], seed=seed, valence_map=vmap, embedding_projection=projection,
                                                       metadata={"embedding_sd_floor": sd_floor, "pca_components": components, "protocol_frozen_utc": cfg["protocol_frozen_utc"]})
            out = model.predict(x["test"]); name = f"mlp_{fs}"
            per_seed[fs][str(seed)] = {"argmax": pm.evaluate(out["valence"], te_v, out["class"], te_c), "balanced": pm.evaluate(out["valence"], te_v, out["class_balanced"], te_c),
                                       "seconds": time.perf_counter() - t1, "valence_head": model.valence_head.metadata, "class_head": model.class_head.metadata}
            summary["training"][f"{name}|seed={seed}"] = {"valence": hist["valence"], "class": hist["class"]}
            if seed == mcfg["headline_seed"]:
                models[fs] = model; preds[name] = (out["valence"], out["class"]); preds[f"{name}|balanced"] = (out["valence"], out["class_balanced"])
            log(f"{name} seed {seed}: {per_seed[fs][str(seed)]['seconds']:.0f}s, test MAE {per_seed[fs][str(seed)]['argmax']['mae']:.4f}, acc {per_seed[fs][str(seed)]['argmax']['accuracy']:.4f}, macroF1 {per_seed[fs][str(seed)]['argmax']['macro_f1']:.4f}")
        summary["models"][f"mlp_{fs}"] = {"feature_set": fs, "input_size": proto.input_size, **{k: mcfg[k] for k in ("hidden", "activation", "lr", "l2", "batch", "epochs", "patience")},
                                          "headline_seed": mcfg["headline_seed"], "per_seed_test": per_seed[fs],
                                          "seed_mean": {m: float(np.mean([per_seed[fs][str(s)]["argmax"][m] for s in mcfg["seeds"]])) for m in ("mae", "rmse", "sign_accuracy_nonneutral", "sign_accuracy_balanced", "accuracy", "macro_f1")}}
    summary["timings"]["training_seconds"] = time.perf_counter() - t0 - summary["timings"]["data_seconds"] - summary["timings"]["baselines_seconds"]

    # ---- test evaluation, once -----------------------------------------------------------------
    iv = cfg["metrics"]["intervals"]; headline = f"mlp_{mcfg['headline_variant']}"
    pairs = [(n, "persistence") for n in preds if n != "persistence"] + [(n, "identifier_default") for n in preds if n != "identifier_default"]
    pairs += [(headline, "mlp_scalars"), (headline, "mlp_embeddings"), (headline, "ridge")]
    pairs = [p for p in pairs if p[0] in preds and p[1] in preds]
    summary["test"] = pm.bootstrap_metrics(keys["test"], te_v, te_c, preds, seed=iv["seed"], samples=iv["samples"], pairs=pairs)
    summary["test"]["point_estimates"] = {n: pm.evaluate(v, te_v, c, None if c is None else te_c) for n, (v, c) in preds.items()}
    summary["timings"]["bootstrap_seconds"] = time.perf_counter() - t0 - sum(summary["timings"].values())
    log("bootstrap done")

    # ---- preference check (rank_candidates proxy) ---------------------------------------------
    pc = cfg["preference_check"]; e_prev, e_a, e_b = parts["test"]; h, c = scal["test"][:, :NH], scal["test"][:, NH:]
    checks = {}
    for fs, model in models.items():
        checks[f"mlp_{fs}"] = pm.preference_check(lambda ep, ea, eb, hh, cc, m=model: m.predict(m.assemble(ep, ea, eb, hh, cc))["valence"],
                                                  e_prev, e_a, e_b, h, c, te_v, keys["test"], seed=pc["seed"], samples=pc["bootstrap_samples"])
    checks["ridge"] = pm.preference_check(lambda ep, ea, eb, hh, cc: np.clip(ridge_v.predict(np.hstack([hh, cc]))[:, 0], -1, 1), e_prev, e_a, e_b, h, c, te_v, keys["test"], seed=pc["seed"], samples=pc["bootstrap_samples"])
    # The identifier ignores embeddings; its per-example parameters are passed in the E_prev slot so the same item/other pairing applies.
    checks["identifier_default"] = pm.preference_check(lambda th, ea, eb, hh, cc: pm.identifier_candidate_valence(th, hh[:, A_VALENCE], cc[:, 0]), thetas_default, e_a, e_b, h, c, te_v, keys["test"], seed=pc["seed"], samples=pc["bootstrap_samples"])
    checks["persistence"] = pm.preference_check(lambda ep, ea, eb, hh, cc: hh[:, A_VALENCE], e_prev, e_a, e_b, h, c, te_v, keys["test"], seed=pc["seed"], samples=pc["bootstrap_samples"])
    summary["preference_check"] = {"protocol": pc, **checks}
    log("preference check done")

    # ---- outputs -------------------------------------------------------------------------------
    digests = {}
    if mcfg["headline_variant"] in models:
        digests["partner_model.json.gz"] = models[mcfg["headline_variant"]].save(output / "partner_model.json.gz")
    if "scalars" in models:
        digests["partner_model_scalars.json.gz"] = models["scalars"].save(output / "partner_model_scalars.json.gz")
    summary["model_digests"] = digests
    with gzip.open(output / "predictions.jsonl.gz", "wt") as f:
        for i, e in enumerate(examples["test"]):
            row = {"dialogue": e["dialogue"], "turn": e["turn"], "target_emotion": e["target_emotion"], "target_valence": float(te_v[i]),
                   "a_emotion": e["a_emotion"], "b_emotion": e["b_emotion"]}
            for n, (v, cl) in preds.items():
                row[n] = {"valence": round(float(v[i]), 5), "class": None if cl is None else emotions[int(cl[i])]}
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary["runtime_seconds"] = time.perf_counter() - t0; summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    summary["interpretation"] = ("Observational next-turn prediction on scripted human-human dialogues. Differences between models describe predictive "
                                 "accuracy for the partner's next labelled affect given what speaker B actually said; they are not causal effects of an "
                                 "agent's expression, and the preference check is a weak proxy. Intervals are dialogue-level percentile bootstraps.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    manifest = {"schema_version": 1, "finished_utc": summary["finished_utc"], "python": _platform.python_version(), "numpy": np.__version__,
                "command": f"python -m regesh.bonding.train_partner_model --config {config_path} --embeddings {embeddings_dir} --dailydialog {dailydialog_dir} --output {output}",
                "files": {p.name: sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def _final_thetas(thetas, keys):
    """Mean of the last parameter vector used in each dialogue (the identifier's state after all but the last update)."""
    last = {}
    for i, k in enumerate(keys):
        last[k] = i
    rows = thetas[list(last.values())]
    return dict(zip(pm.RecursiveIdentifier.NAMES, map(float, rows.mean(axis=0))))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/bonding_partner_model.json")
    parser.add_argument("--embeddings", default=".cache/embeddings")
    parser.add_argument("--dailydialog", default=".cache/dailydialog/parlai")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run(args.config, args.embeddings, args.dailydialog, args.output)
    print(json.dumps({n: {m: v["point"] for m, v in est.items()} for n, est in summary["test"]["estimates"].items()}, indent=2))


if __name__ == "__main__":
    main()

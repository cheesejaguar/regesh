"""Offline checks for the learned partner model with synthetic dialogues and embeddings."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from regesh.bonding import partner_model as pm
from regesh.bonding.dyad import RecursiveIdentifier
from regesh.bonding.mlp import MLP


def turn(emotion, act, text):
    return {"emotion": emotion, "act": act, "text": text}


DIALOGUES = [
    {"topic": "t", "dialogue": [turn("no_emotion", "question", "a0"), turn("no_emotion", "inform", "b0")]},                          # 2 turns: 0 examples
    {"topic": "t", "dialogue": [turn("no_emotion", "question", "c0"), turn("happiness", "inform", "d0"), turn("happiness", "inform", "c1")]},   # 3 turns: 1
    {"topic": "t", "dialogue": [turn("anger", "directive", "e0"), turn("no_emotion", "commissive", "f0"), turn("sadness", "inform", "e1"), turn("no_emotion", "question", "f1")]},   # 4: 1
    {"topic": "t", "dialogue": [turn("no_emotion", "inform", "g0"), turn("no_emotion", "inform", "h0"), turn("surprise", "question", "g1"), turn("happiness", "inform", "h1"), turn("happiness", "inform", "g2")]},   # 5: 2
    {"topic": "t", "dialogue": [turn("no_emotion", "inform", f"{s}{k}") for k in range(4) for s in "ij"][:7]},   # 7: 3
]


class FakeCache:
    """Deterministic stand-in for EmbeddingCache: vectors from the SHA-256 of the text."""
    def __init__(self, dim=8):
        self.dim = dim

    def vector(self, text):
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=self.dim)

    def get(self, texts):
        return np.vstack([self.vector(t) for t in texts]).astype(np.float32)


def synthetic_dialogues(n, rng, length=7):
    """Dialogues whose A emotion at t+2 is a planted function of the B turn's embedding (first coordinate)."""
    cache = FakeCache(8); rows = []
    for d in range(n):
        turns = []
        for k in range(length):
            text = f"synthetic dialogue {d} turn {k}"
            emotion = "no_emotion"
            if k % 2 == 0 and k >= 2:
                prev_b = f"synthetic dialogue {d} turn {k - 1}"
                v = cache.vector(prev_b)[0]
                emotion = "happiness" if v > .6 else ("anger" if v < -.6 else "no_emotion")
            turns.append(turn(emotion, pm.ACTS[int(rng.integers(0, 4))], text))
        rows.append({"topic": "synthetic", "dialogue": turns})
    return rows, cache


class DatasetChecks(unittest.TestCase):
    def test_example_construction_is_deterministic_with_expected_counts(self):
        ex = pm.build_examples(DIALOGUES, split="toy")
        self.assertEqual([e["dialogue"] for e in ex], [1, 2, 3, 3, 4, 4, 4])
        self.assertEqual(pm.build_examples(DIALOGUES, split="toy"), ex)
        first = ex[0]
        self.assertIsNone(first["a_prev_text"]); self.assertEqual((first["a_text"], first["b_text"], first["target_emotion"]), ("c0", "d0", "happiness"))
        second_of_five = ex[3]
        self.assertEqual((second_of_five["turn"], second_of_five["exchange"], second_of_five["a_prev_text"], second_of_five["a_text"], second_of_five["b_text"]), (2, 1, "g0", "g1", "h1"))
        self.assertEqual(second_of_five["a_prev_emotion"], "no_emotion"); self.assertEqual(second_of_five["target_emotion"], "happiness")
        kept = pm.build_examples(DIALOGUES, keep=np.array([False, False, True, False, False]), split="toy")
        self.assertEqual(len(kept), 1); self.assertEqual(kept[0]["a_emotion"], "anger")
        self.assertEqual(pm.example_texts(ex)[:3], ["c0", "d0", "e0"])

    def test_feature_shapes_and_values(self):
        ex = pm.build_examples(DIALOGUES, split="toy"); cache = FakeCache(8)
        s = pm.scalar_features(ex)
        self.assertEqual(s.shape, (7, 13))
        np.testing.assert_allclose(s[3], [0., 1., .3, 0., 1., 0., 0., .1, .7, 1., 0., 0., 0.])   # g0 no_emotion prev, g1 surprise/question, exchange 1, h1 happiness/inform
        self.assertEqual(s[0, 1], 0.)   # no previous turn
        v, c = pm.targets(ex)
        self.assertEqual(v[1], -.6); self.assertEqual(c[1], pm.EMOTION_INDEX["sadness"])
        e_prev, e_a, e_b = pm.embedding_parts(ex, cache)
        self.assertEqual(e_a.shape, (7, 8)); self.assertTrue((e_prev[0] == 0).all()); self.assertFalse((e_prev[3] == 0).all())
        np.testing.assert_allclose(e_prev[3], cache.vector("g0"), rtol=1e-6)
        mean, sd = pm.fit_standardizer(cache.get(pm.example_texts(ex)))
        for fs, size in (("embeddings+scalars", 37), ("scalars", 13), ("embeddings", 26)):
            m = pm.LearnedPartnerModel(fs, mean, sd, np.full(7, 1 / 7), valence_head=MLP([size, 1]), class_head=MLP([size, 7], loss="bce"))
            self.assertEqual(m.input_size, size)
            x = m.features(ex, cache)
            self.assertEqual(x.shape, (7, size))
            if fs != "scalars":
                self.assertTrue((x[0, :8] == 0).all())   # standardized previous-turn block zeroed when absent
        dist = pm.class_distribution(c)
        self.assertEqual(dist["n"], 7); self.assertEqual(dist["majority"], "no_emotion")
        # Optional PCA projection: deterministic, orthonormal, shrinks each utterance block.
        z = (cache.get(pm.example_texts(ex)) - mean) / sd
        comps, explained = pm.fit_projection(z, 3)
        np.testing.assert_allclose(comps, pm.fit_projection(z, 3)[0]); self.assertEqual(comps.shape, (8, 3))
        np.testing.assert_allclose(comps.T @ comps, np.eye(3), atol=1e-10); self.assertTrue(0 < explained.sum() <= 1 + 1e-9)
        self.assertTrue((comps[np.abs(comps).argmax(axis=0), np.arange(3)] > 0).all())
        proj = pm.LearnedPartnerModel("embeddings+scalars", mean, sd, np.full(7, 1 / 7), valence_head=MLP([22, 1]), class_head=MLP([22, 7], loss="bce"), embedding_projection=comps)
        self.assertEqual((proj.latent, proj.input_size), (3, 22)); self.assertEqual(proj.features(ex, cache).shape, (7, 22))
        np.testing.assert_allclose(proj.features(ex, cache)[:, 3:6], z[[pm.example_texts(ex).index(e["a_text"]) for e in ex]] @ comps)


class BaselineChecks(unittest.TestCase):
    def test_persistence_constant_and_identifier_on_hand_built_sequences(self):
        ex = pm.build_examples(DIALOGUES, split="toy")
        pv, pc = pm.persistence_baseline(ex)
        self.assertEqual(pv[1], pm.VALENCE_MAP["anger"]); self.assertEqual(pc[1], pm.EMOTION_INDEX["anger"])
        cv, cc = pm.constant_baseline([.7, 0., -.7, 0.], [0, 4, 0, 0], 3)
        np.testing.assert_allclose(cv, 0.); self.assertEqual(cc.tolist(), [0, 0, 0])
        preds, thetas = pm.identifier_baseline(ex)
        # First prediction of each dialogue uses the default prior theta [0, .3, .2, .2]: .3 P + .2 min(A,0) + .2 max(A,0)
        self.assertAlmostEqual(preds[0], .3 * 0. + .2 * .7)                 # dialogue 1: A no_emotion, B happiness
        self.assertAlmostEqual(preds[1], .3 * -.7 + .2 * 0.)                # dialogue 2 restarts with the prior
        np.testing.assert_allclose(thetas[2], [0., .3, .2, .2])
        # Second example of the five-turn dialogue equals a hand-run identifier after one update.
        ident = RecursiveIdentifier(); ident.update(0., 0., .3)
        self.assertAlmostEqual(preds[3], np.clip(ident.predict(.3, .7), -1, 1)); np.testing.assert_allclose(thetas[3], ident.state["theta"])
        # Pooled least squares recovers a planted bilinear law exactly, and the warm-started identifier uses it.
        rng = np.random.default_rng(0); a = rng.uniform(-1, 1, 200); b = rng.uniform(-1, 1, 200)
        y = .1 + .4 * a - .5 * np.minimum(b, 0) + .2 * np.maximum(b, 0)
        np.testing.assert_allclose(pm.pooled_bilinear_fit(a, b, y), [.1, .4, -.5, .2], atol=1e-10)
        warm, th = pm.identifier_baseline(ex, theta0=[.1, .4, -.5, .2], prior=1.)
        np.testing.assert_allclose(th[0], [.1, .4, -.5, .2]); self.assertAlmostEqual(warm[0], .1 + .2 * .7)
        cand = pm.identifier_candidate_valence(np.array([[.1, .4, -.5, .2]]), np.array([0.]), np.array([-.6]))
        self.assertAlmostEqual(cand[0], .1 + .3)

    def test_ridge_selection_and_prediction_shape(self):
        rng = np.random.default_rng(1); x = rng.normal(size=(300, 5)); y = x @ np.array([1., -1., 0., .5, 0.]) + .01 * rng.normal(size=300)
        model, scores, best = pm.RidgeModel.select(x[:200], y[:200], x[200:], y[200:], [1e-2, 1., 1e3])
        self.assertEqual(best, 1e-2); self.assertEqual(model.predict(x[:3]).shape, (3, 1))
        self.assertLess(np.abs(model.predict(x[200:])[:, 0] - y[200:]).mean(), .05)
        self.assertEqual(set(scores), {"0.01", "1.0", "1000.0"})


class EvaluationChecks(unittest.TestCase):
    def test_metrics_and_bootstrap_intervals(self):
        tv = np.array([0., .7, -.7, 0., .3]); pv = np.array([0., .2, .1, .1, 0.])
        ev = pm.evaluate(pv, tv, [0, 4, 1, 0, 0], [0, 4, 1, 0, 6])
        self.assertAlmostEqual(ev["mae"], np.mean([0., .5, .8, .1, .3])); self.assertEqual(ev["n_nonneutral"], 3)
        self.assertAlmostEqual(ev["sign_accuracy_nonneutral"], 1 / 3)      # .2 vs .7 right, .1 vs -.7 wrong, 0 vs .3 wrong
        self.assertAlmostEqual(ev["sign_accuracy_balanced"], (.5 + 0.) / 2); self.assertEqual((ev["n_positive"], ev["n_negative"]), (2, 1))
        self.assertAlmostEqual(pm.evaluate(np.full(5, .1), tv)["sign_accuracy_balanced"], .5)   # constant positive guess scores .5
        self.assertAlmostEqual(ev["accuracy"], .8)
        self.assertAlmostEqual(ev["macro_f1"], np.mean([2 * (2 / 3) * 1 / (2 / 3 + 1), 1., 1., 0.]))   # classes present: no_emotion, anger, happiness, surprise
        keys = [("t", 0), ("t", 0), ("t", 1), ("t", 2), ("t", 2)]
        boot = pm.bootstrap_metrics(keys, tv, [0, 4, 1, 0, 6], {"m": (pv, np.array([0, 4, 1, 0, 0])), "p": (tv, np.array([0, 4, 1, 0, 6]))},
                                    seed=3, samples=100, pairs=[("m", "p"), ("m", "m")])
        self.assertEqual(boot["dialogues"], 3)
        est = boot["estimates"]["m"]["mae"]; self.assertLessEqual(est["ci95"][0], est["point"]); self.assertGreaterEqual(est["ci95"][1], est["point"])
        self.assertEqual(boot["paired_differences"]["m - m"]["mae"]["ci95"], [0., 0.])
        self.assertAlmostEqual(boot["paired_differences"]["m - p"]["mae"]["point"], ev["mae"])
        self.assertAlmostEqual(boot["estimates"]["p"]["macro_f1"]["point"], 1.)

    def test_preference_check_structure_and_sign_rule(self):
        rng = np.random.default_rng(5); n = 400; e_b = rng.normal(size=(n, 3)); e_a = rng.normal(size=(n, 3)); e_prev = np.zeros((n, 3))
        h = np.zeros((n, 8)); c = np.zeros((n, 5)); tv = np.where(e_b[:, 0] > 1, .7, np.where(e_b[:, 0] < -1, -.7, 0.))
        keys = [("t", i // 4) for i in range(n)]
        out = pm.preference_check(lambda ep, ea, eb, hh, cc: eb[:, 0], e_prev, e_a, e_b, h, c, tv, keys, seed=1, samples=50)
        self.assertEqual(out["n_items"], int((tv != 0).sum())); self.assertGreater(out["preference_accuracy"]["point"], .8)
        self.assertGreater(out["preference_accuracy_negative"]["point"], .8)
        ties = pm.preference_check(lambda ep, ea, eb, hh, cc: np.zeros(len(eb)), e_prev, e_a, e_b, h, c, tv, keys, seed=1, samples=10)
        self.assertEqual(ties["preference_accuracy"]["point"], 0.); self.assertEqual(ties["ties_fraction"], 1.)


class LearnedModelChecks(unittest.TestCase):
    def test_rank_candidates_orders_by_planted_linear_relation(self):
        d = 6; mean = np.zeros(d); sd = np.ones(d) * 2.
        v_head = MLP([3 * d + 2, 1]); v_head.W[0][:] = 0.; v_head.b[0][:] = 0.
        w = np.array([1., -2., .5, 0., 0., 3.]); v_head.W[0][2 * d:3 * d, 0] = w    # valence = w . z(B)
        model = pm.LearnedPartnerModel("embeddings", mean, sd, np.full(7, 1 / 7), valence_head=v_head, class_head=MLP([3 * d + 2, 7], loss="bce"))
        rng = np.random.default_rng(2); cands = rng.normal(size=(9, d))
        hist = pm.make_history(rng.normal(size=d), None, None, "no_emotion", "question", 3)
        out = model.rank_candidates(hist, cands)
        expected = (cands / 2.) @ w
        np.testing.assert_allclose(out["valence"], np.clip(expected, -1, 1), atol=1e-12)
        self.assertEqual(out["ranking"].tolist(), np.argsort(-np.clip(expected, -1, 1), kind="stable").tolist())
        self.assertEqual(out["class_probabilities"].shape, (9, 7)); np.testing.assert_allclose(out["class_probabilities"].sum(axis=1), 1.)
        # The scalar-only feature set ranks by the candidates' declared valence through the head, independent of embeddings.
        s_head = MLP([13, 1]); s_head.W[0][:] = 0.; s_head.W[0][8, 0] = 1.; s_head.b[0][:] = 0.
        scalar_model = pm.LearnedPartnerModel("scalars", mean, sd, np.full(7, 1 / 7), valence_head=s_head, class_head=MLP([13, 7], loss="bce"))
        ranked = scalar_model.rank_candidates(hist, cands[:3], candidate_scalars=np.vstack([pm.candidate_scalars(e, "inform") for e in ("sadness", "happiness", "no_emotion")]))
        self.assertEqual(ranked["ranking"].tolist(), [1, 2, 0])
        self.assertEqual(scalar_model.rank_candidates(hist, cands[:3])["valence"].tolist(), [0., 0., 0.])   # unknown candidates: neutral

    def test_training_recovers_planted_signal_and_round_trips_with_digest(self):
        rng = np.random.default_rng(7); rows, cache = synthetic_dialogues(260, rng)
        train, dev, test = pm.build_examples(rows[:180], split="train"), pm.build_examples(rows[180:220], split="valid"), pm.build_examples(rows[220:], split="test")
        tv, tc = pm.targets(train); dv, dc = pm.targets(dev); ev_, ec = pm.targets(test)
        mean, sd = pm.fit_standardizer(cache.get(pm.example_texts(train)))
        results = {}
        for fs in ("embeddings", "scalars"):
            proto = pm.LearnedPartnerModel(fs, mean, sd, np.full(7, 1 / 7), valence_head=MLP([1, 1]), class_head=MLP([1, 7], loss="bce"))
            x, xd, xt = proto.features(train, cache), proto.features(dev, cache), proto.features(test, cache)
            model, hist = pm.LearnedPartnerModel.train(fs, x, tv, tc, xd, dv, dc, embedding_mean=mean, embedding_sd=sd, hidden=(16,), lr=5e-3, epochs=60, patience=10, seed=0)
            pred = model.predict(xt); results[fs] = pm.evaluate(pred["valence"], ev_, pred["class"], ec)
            self.assertIn("valence", hist); self.assertLessEqual(hist["valence"][-1]["epoch"], 60)
        const_v, const_c = pm.constant_baseline(tv, tc, len(test)); const = pm.evaluate(const_v, ev_, const_c, ec)
        self.assertLess(results["embeddings"]["mae"], .6 * const["mae"])            # planted signal recovered from embeddings
        self.assertGreater(results["embeddings"]["accuracy"], const["accuracy"] + .1)
        self.assertGreater(results["scalars"]["mae"], .85 * const["mae"])            # nothing to learn without text
        comps, _ = pm.fit_projection((cache.get(pm.example_texts(train)) - mean) / sd, 4)
        x_proj = pm.LearnedPartnerModel("embeddings", mean, sd, np.full(7, 1 / 7), embedding_projection=comps).features(train, cache)
        self.assertEqual(x_proj.shape[1], 14)
        projected, _ = pm.LearnedPartnerModel.train("embeddings", x_proj, tv, tc, x_proj[:50], tv[:50], tc[:50], embedding_mean=mean, embedding_sd=sd, hidden=(4,), epochs=2, seed=0, embedding_projection=comps)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "projected.json.gz"; projected.save(path); copy = pm.LearnedPartnerModel.load(path)
            np.testing.assert_array_equal(copy.embedding_projection, comps); np.testing.assert_array_equal(copy.predict(x_proj[:3])["valence"], projected.predict(x_proj[:3])["valence"])
            self.assertEqual(copy.rank_candidates(pm.make_history(cache.vector("q"), None, None, "no_emotion", "inform", 0), cache.get(["r", "s"]))["valence"].shape, (2,))
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("model.json", "model.json.gz"):
                path = Path(tmp) / name; digest = model.save(path); copy = pm.LearnedPartnerModel.load(path)
                self.assertEqual(copy.to_dict()["digest"], digest); self.assertEqual(len(digest), 64)
                np.testing.assert_array_equal(copy.predict(xt[:5])["valence"], model.predict(xt[:5])["valence"])
                np.testing.assert_array_equal(copy.predict(xt[:5])["class_probabilities"], model.predict(xt[:5])["class_probabilities"])
                self.assertEqual(copy.feature_set, "scalars"); self.assertEqual(copy.metadata["hidden"], [16])
            tampered = Path(tmp) / "tampered.json"; d = json.loads((Path(tmp) / "model.json").read_text()); d["class_prior"][0] += .01
            tampered.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                pm.LearnedPartnerModel.load(tampered)


class RunnerChecks(unittest.TestCase):
    def test_run_writes_outputs_and_refuses_nonempty_output(self):
        from regesh.bonding import train_partner_model as tr
        from regesh.bonding.embeddings import EmbeddingCache
        rng = np.random.default_rng(11); rows, cache = synthetic_dialogues(90, rng)
        cfg = json.loads(Path("configs/bonding_partner_model.json").read_text())
        cfg["models"]["mlp"].update(epochs=3, seeds=[0, 1], hidden=[8]); cfg["metrics"]["intervals"]["samples"] = 20; cfg["preference_check"]["bootstrap_samples"] = 20
        cfg["data"]["splits"]["train"]["fraction"] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp / "dd").mkdir(); (tmp / "emb").mkdir()
            for split, chunk in (("train", rows[:60]), ("valid", rows[60:75]), ("test", rows[75:])):
                (tmp / "dd" / f"{split}.json").write_text("\n".join(json.dumps({"fold": split, **r}) for r in chunk) + "\n")
            texts = sorted({t["text"] for r in rows for t in r["dialogue"]})
            store = EmbeddingCache(tmp / "emb" / cfg["embeddings"]["cache_file"]); store.add(texts, cache.get(texts)); store.save()
            cfg_path = tmp / "cfg.json"; cfg_path.write_text(json.dumps(cfg)); out = tmp / "out"
            summary = tr.run(cfg_path, tmp / "emb", tmp / "dd", out, log=lambda m: None)
            n_test = len(pm.build_examples(rows[75:]))
            self.assertEqual(summary["data"]["splits"]["test"]["examples"], n_test)
            self.assertEqual(summary["data"]["splits"]["test"]["manifest_check"]["status"], "manifest missing")
            for name in ("persistence", "constant", "identifier_default", "identifier_pooled_prior", "ridge", "mlp_embeddings+scalars", "mlp_scalars", "mlp_embeddings", "mlp_embeddings+scalars|balanced"):
                self.assertIn(name, summary["test"]["estimates"])
            self.assertIn("mlp_embeddings+scalars - identifier_default", summary["test"]["paired_differences"])
            self.assertIn("mae", summary["test"]["paired_differences"]["mlp_embeddings+scalars - persistence"])
            self.assertNotIn("accuracy", summary["test"]["estimates"]["identifier_default"])
            self.assertEqual(set(summary["models"]["mlp_scalars"]["per_seed_test"]), {"0", "1"})
            self.assertIn("preference_accuracy", summary["preference_check"]["mlp_embeddings"])
            self.assertEqual(summary["preference_check"]["persistence"]["ties_fraction"], 1.)
            model = pm.LearnedPartnerModel.load(out / "partner_model.json.gz")
            self.assertEqual(model.to_dict()["digest"], summary["model_digests"]["partner_model.json.gz"]); self.assertEqual(model.feature_set, "embeddings+scalars")
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertTrue({"config.json", "source_partner_model.py", "source_train_partner_model.py", "summary.json", "predictions.jsonl.gz", "partner_model.json.gz", "partner_model_scalars.json.gz"} <= set(manifest["files"]))
            import gzip
            with gzip.open(out / "predictions.jsonl.gz", "rt") as f:
                lines = [json.loads(l) for l in f]
            self.assertEqual(len(lines), n_test); self.assertIn("ridge", lines[0])
            with self.assertRaises(ValueError):
                tr.run(cfg_path, tmp / "emb", tmp / "dd", out, log=lambda m: None)


if __name__ == "__main__":
    unittest.main()

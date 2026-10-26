"""Partner-state sensor checks: valence map arithmetic, event annotation, persistence, training and metrics (offline, synthetic vectors)."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from regesh import Agent, Event
from regesh.bonding import sensor as S
from regesh.bonding import train_sensor
from regesh.bonding.embeddings import EmbeddingCache
from regesh.bonding.mlp import MLP
from regesh.dataset_baseline import evaluate, matrix, micro_f1, select_threshold

LABELS = S.GOEMOTIONS_LABELS


def synthetic_sensor(dimension=8, seed=3):
    head = MLP([dimension, len(LABELS)], loss="bce", seed=seed)
    return S.Sensor(head, LABELS, np.zeros(dimension), np.ones(dimension), threshold=.3, metadata={"note": "synthetic"})


class ValenceMap(unittest.TestCase):
    def test_sentiment_groups_partition_the_28_labels(self):
        groups = S.SENTIMENT_GROUPS
        flat = [l for g in groups.values() for l in g]
        self.assertEqual(sorted(flat), sorted(LABELS)); self.assertEqual(len(flat), 28)
        self.assertEqual((len(groups["positive"]), len(groups["negative"]), len(groups["ambiguous"]), len(groups["neutral"])), (12, 11, 4, 1))
        self.assertEqual(S.VALENCE_MAP_NAME, "goemotions-sentiment-v1")

    def test_valence_is_the_mass_weighted_mean_sign(self):
        self.assertAlmostEqual(S.valence_from_probabilities({"joy": .5, "anger": .25, "neutral": .25}), (.5 - .25) / 1.)
        self.assertAlmostEqual(S.valence_from_probabilities({"joy": .2, "love": .2}), 1.)
        self.assertAlmostEqual(S.valence_from_probabilities({"sadness": .9, "curiosity": .1}), -.9)
        self.assertEqual(S.valence_from_probabilities({"curiosity": .7, "neutral": .3}), 0.)
        self.assertEqual(S.valence_from_probabilities(np.zeros(28)), 0.)
        vector = np.zeros(28); vector[LABELS.index("gratitude")] = .3; vector[LABELS.index("annoyance")] = .1
        self.assertAlmostEqual(S.valence_from_probabilities(vector), (.3 - .1) / .4)
        with self.assertRaises(ValueError):
            S.valence_from_probabilities(np.zeros(5))

    def test_support_proxy_is_noisy_or_over_the_declared_labels(self):
        self.assertAlmostEqual(S.support_proxy_from_probabilities({"caring": .5, "gratitude": .5}), .75)
        self.assertAlmostEqual(S.support_proxy_from_probabilities({"anger": .9}), 0.)
        self.assertAlmostEqual(S.support_proxy_from_probabilities({l: 1. for l in S.SUPPORT_PROXY_LABELS}), 1.)
        self.assertEqual(S.SUPPORT_PROXY_LABELS, ("caring", "gratitude", "approval", "love", "admiration"))

    def test_dailydialog_valence_scale(self):
        np.testing.assert_array_equal(S.dailydialog_valence(["happiness", "anger", "no_emotion", "surprise", "fear"]), [1., -1., 0., 0., -1.])


class SensorBundle(unittest.TestCase):
    def test_estimate_shape_ranges_and_dimension_check(self):
        sensor = synthetic_sensor()
        est = sensor.estimate(np.linspace(-1, 1, 8))
        self.assertEqual(set(est), {"emotion_probabilities", "partner_affect", "support_proxy"})
        self.assertEqual(set(est["emotion_probabilities"]), set(LABELS))
        self.assertTrue(all(0. <= p <= 1. for p in est["emotion_probabilities"].values()))
        self.assertTrue(-1. <= est["partner_affect"] <= 1.); self.assertTrue(0. <= est["support_proxy"] <= 1.)
        self.assertAlmostEqual(est["partner_affect"], S.valence_from_probabilities(est["emotion_probabilities"]))
        with self.assertRaises(ValueError):
            sensor.estimate(np.zeros(9))

    def test_save_load_round_trip_digest_and_tamper_detection(self):
        sensor = synthetic_sensor()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensor.json"; sensor.save(path); copy = S.Sensor.load(path)
            x = np.random.default_rng(0).normal(size=(4, 8))
            np.testing.assert_allclose(copy.probabilities(x), sensor.probabilities(x))
            self.assertEqual(copy.digest, sensor.digest); self.assertEqual(copy.threshold, .3); self.assertEqual(copy.metadata["note"], "synthetic")
            payload = json.loads(path.read_text()); payload["head"]["weights"]["W"][0][0][0] += 1.
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                S.Sensor.load(path)
        other = synthetic_sensor(); other.head.W[0][0, 0] += 1e-3
        self.assertNotEqual(other.digest, sensor.digest)
        shifted = S.Sensor(synthetic_sensor().head, LABELS, np.ones(8), np.ones(8))
        self.assertNotEqual(shifted.digest, sensor.digest)

    def test_shape_mismatches_are_rejected(self):
        head = MLP([8, 28], loss="bce", seed=0)
        with self.assertRaises(ValueError):
            S.Sensor(head, LABELS, np.zeros(7), np.ones(7))
        with self.assertRaises(ValueError):
            S.Sensor(head, LABELS[:27] + ("extra",), np.zeros(8), np.ones(8))
        with self.assertRaises(ValueError):
            S.Sensor(head, LABELS, np.zeros(8), np.ones(8), support_labels=("joyfulness",))


class AnnotateEvent(unittest.TestCase):
    def event(self, **data):
        return Event("u-1", "chat", "user", "agent", "conversation", "2026-10-25T00:00:00Z", data=data)

    def test_fills_only_missing_fields_and_records_provenance(self):
        sensor = synthetic_sensor(); x = np.linspace(-1, 1, 8); est = sensor.estimate(x)
        original = self.event(text="thanks, that helped")
        new = S.annotate_event(original, sensor, x)
        self.assertIsNot(new, original); self.assertNotIn("partner_affect", original.data); self.assertNotIn("sensor", original.data)
        self.assertEqual(new.data["partner_affect"], est["partner_affect"]); self.assertNotIn("support", new.data)
        prov = new.data["sensor"]
        self.assertEqual(prov["model_digest"], sensor.digest); self.assertEqual(prov["valence_map"], "goemotions-sentiment-v1")
        self.assertEqual(prov["emotion_probabilities"], est["emotion_probabilities"]); self.assertEqual(prov["filled"], ["partner_affect"])
        self.assertEqual(new.data["text"], "thanks, that helped"); self.assertEqual(new.id, original.id)
        # Hand-supplied evidence is never overwritten; support only with the opt-in flag.
        kept = S.annotate_event(self.event(partner_affect=.3, support=.9), sensor, x, fill_support=True)
        self.assertEqual(kept.data["partner_affect"], .3); self.assertEqual(kept.data["support"], .9); self.assertEqual(kept.data["sensor"]["filled"], [])
        with_support = S.annotate_event(self.event(), sensor, x, fill_support=True)
        self.assertEqual(with_support.data["support"], est["support_proxy"]); self.assertEqual(with_support.data["sensor"]["filled"], ["partner_affect", "support"])

    def test_annotated_event_is_observed_by_a_bonded_agent(self):
        sensor = synthetic_sensor(); x = np.linspace(-1, 1, 8)
        agent = Agent(":memory:", "companion", config={"bonding": True, "partner": "user"})
        try:
            for i in range(2):
                e = Event(f"u-{i}", "chat", "user", "agent", "conversation", "2026-10-25T00:00:00Z", data={"text": f"hello {i}"})
                agent.observe(S.annotate_event(e, sensor, x + i * .1))
            state = agent.get_state()
            last = state["bonds"]["user"]["dyad"]["last_partner_affect"]
            self.assertAlmostEqual(last, sensor.estimate(x + .1)["partner_affect"])
            self.assertEqual(state["events"]["u-1"]["data"]["sensor"]["model_digest"], sensor.digest)
            self.assertEqual(state["bonds"]["user"]["dyad"]["updates"], 1)
        finally:
            agent.close()


def separable_problem(seed=0, n=600, dimension=16, classes=3, noise=.6):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(classes, dimension)) * 2.
    y = np.zeros((n, classes), bool); x = np.empty((n, dimension))
    for i in range(n):
        active = rng.choice(classes, size=1 + (i % 2), replace=False); y[i, active] = True
        x[i] = centres[active].mean(axis=0) + rng.normal(scale=noise, size=dimension)
    return x, y


class Training(unittest.TestCase):
    def test_heads_reach_high_f1_on_a_separable_problem(self):
        x, y = separable_problem(); fit, dev, test = slice(0, 400), slice(400, 500), slice(500, 600)
        mean, std = S.fit_standardizer(x[fit]); xs = S.standardize(x, mean, std)
        self.assertAlmostEqual(float(xs[fit].mean()), 0., places=9); self.assertAlmostEqual(float(xs[fit].std(axis=0).mean()), 1., places=9)
        for hidden in (None, 16):
            model, history = S.train_head(xs[fit], y[fit], xs[dev], y[dev], hidden=hidden, epochs=80, batch=32, lr=5e-3, l2=1e-5, patience=8)
            self.assertEqual(model.sizes, [16] + ([16] if hidden else []) + [3])
            threshold, dev_f1 = select_threshold(model.predict(xs[dev]), y[dev])
            metrics, _ = evaluate(y[test], model.predict(xs[test]), threshold, ["a", "b", "c"])
            self.assertGreater(metrics["micro_f1"], .9, hidden); self.assertGreater(metrics["macro_f1"], .9, hidden)
            self.assertLessEqual(len(history), 80); self.assertEqual(model.metadata["hidden"], hidden)

    def test_standardizer_floors_constant_dimensions(self):
        x = np.array([[1., 5.], [1., 7.]]); mean, std = S.fit_standardizer(x)
        np.testing.assert_array_equal(mean, [1., 6.]); np.testing.assert_array_equal(std, [1., 1.])


class Metrics(unittest.TestCase):
    def test_multiclass_metrics_by_hand(self):
        m = S.multiclass_metrics([0, 0, 1, 1, 2], [0, 1, 1, 1, 2], ["a", "b", "c"])
        self.assertAlmostEqual(m["accuracy"], .8)
        self.assertAlmostEqual(m["per_class"][0]["f1"], 2 / 3); self.assertAlmostEqual(m["per_class"][1]["f1"], .8); self.assertAlmostEqual(m["per_class"][2]["f1"], 1.)
        self.assertAlmostEqual(m["macro_f1"], (2 / 3 + .8 + 1.) / 3)
        self.assertEqual(m["confusion"], [[1, 1, 0], [0, 2, 0], [0, 0, 1]])
        self.assertAlmostEqual(S.macro_f1_from_counts([1, 2, 1], [0, 1, 0], [1, 0, 0]), m["macro_f1"])
        self.assertEqual(S.macro_f1_from_counts([0, 0], [0, 0], [0, 0]), 0.)

    def test_valence_mae_sign_agreement_and_three_way(self):
        self.assertAlmostEqual(S.valence_mae([1, -1, 0, 0], [1, 0, 0, -1]), .5)
        agreement = S.sign_agreement([1, -1, 0, 1], [.2, .3, .5, 0.])
        self.assertEqual(agreement["n"], 3); self.assertAlmostEqual(agreement["agreement"], 1 / 3); self.assertAlmostEqual(agreement["positive_fraction"], 2 / 3)
        self.assertIsNone(S.sign_agreement([0, 0], [1, 1])["agreement"])
        np.testing.assert_array_equal(S.three_way_sign([.5, .05, -.05, -.3], .1), [1., 0., 0., -1.])

    def test_micro_f1_matches_dataset_baseline_and_bootstrap_is_deterministic(self):
        gold = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], bool); pred = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], bool)
        stats = train_sensor._f1_stats(gold, pred); idx = np.arange(4)
        self.assertAlmostEqual(stats["micro_f1"](idx), micro_f1(gold, pred)); self.assertAlmostEqual(stats["micro_f1"](idx), 2 * 3 / (2 * 3 + 1 + 1))
        self.assertAlmostEqual(stats["macro_f1"](idx), ((2 * 1 / (2 + 1 + 1)) + (2 * 2 / (4 + 0 + 0))) / 2)
        values = np.array([1., 2., 3., 4., 10.])
        a = S.bootstrap(lambda i: values[i].mean(), 5, samples=200, seed=7); b = S.bootstrap(lambda i: values[i].mean(), 5, samples=200, seed=7)
        self.assertEqual(a, b); self.assertEqual(a["estimate"], 4.); self.assertLessEqual(a["ci95"][0], 4.); self.assertGreaterEqual(a["ci95"][1], 4.)


class EndToEnd(unittest.TestCase):
    def test_runner_produces_summary_manifest_and_heads_on_synthetic_data(self):
        rng = np.random.default_rng(11); dimension = 12
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / "data"; dd = root / "dd"; emb = root / "emb"; out = root / "out"
            for p in (data, dd, emb): p.mkdir()
            (data / "goemotions-labels-emotions.txt").write_text("\n".join(LABELS))
            centres = rng.normal(size=(28, dimension)) * 2.; texts, vectors = [], []
            def ge_rows(prefix, n):
                lines = []
                for i in range(n):
                    labels = sorted(set(rng.choice(28, size=1 + (i % 2), replace=False).tolist())); text = f"{prefix} comment {i} " + " ".join(f"w{l}" for l in labels)
                    lines.append(f"{text}\t{','.join(map(str, labels))}\t{prefix}{i}"); texts.append(text); vectors.append(centres[labels].mean(axis=0) + rng.normal(scale=.5, size=dimension))
                return "\n".join(lines) + "\n"
            (data / "goemotions-train-train.tsv").write_text(ge_rows("tr", 900)); (data / "goemotions-test-test.tsv").write_text(ge_rows("te", 120))
            dd_centres = rng.normal(size=(7, dimension)) * 2.; emotions = list(S.DAILYDIALOG_VALENCE)
            def dd_dialogues(prefix, n):
                lines = []
                for i in range(n):
                    turns = []
                    for j in range(4):
                        e = int(rng.choice(7, p=[.6, .08, .06, .06, .1, .05, .05])); text = f"{prefix} dialogue {i} turn {j} e{e}"
                        turns.append({"emotion": emotions[e], "act": "inform", "text": text}); texts.append(text); vectors.append(dd_centres[e] + rng.normal(scale=.5, size=dimension))
                    lines.append(json.dumps({"fold": prefix, "topic": "t", "dialogue": turns}))
                return "\n".join(lines) + "\n"
            (dd / "train.json").write_text(dd_dialogues("train", 300)); (dd / "valid.json").write_text(dd_dialogues("valid", 20)); (dd / "test.json").write_text(dd_dialogues("test", 25))
            for name in ("ge.npz", "dd.npz"):
                cache = EmbeddingCache(emb / name); cache.add(texts, np.asarray(vectors, np.float32)); cache.save()
            (emb / "goemotions-test-manifest.json").write_text(json.dumps({"selected_units": 120}))
            cfg = json.loads((Path(__file__).resolve().parent.parent / "configs" / "bonding_sensor.json").read_text())
            cfg["training"].update(epochs=6, patience=3); cfg["bootstrap"].update(samples=40, seed=1)
            cfg["embedding_cache_files"] = {"goemotions": "ge.npz", "dailydialog": "dd.npz"}; cfg["goemotions"]["archived_reference_metrics"] = str(root / "missing.json")
            config = root / "config.json"; config.write_text(json.dumps(cfg))
            logged = []
            summary = train_sensor.run(config, emb, data, dd, out, log=logged.append)
            manifest = json.loads((out / "manifest.json").read_text())
            for name, digest in manifest["files"].items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest, name)
            self.assertIn("manifest.json", {p.name for p in out.iterdir()}); self.assertNotIn("manifest.json", manifest["files"])
            for name in ("summary.json", "config.json", "sensor.json", "goemotions_linear.json", "goemotions_mlp.json", "dailydialog_linear.json", "dailydialog_mlp.json",
                         "goemotions_bernoulli_35.npz", "dailydialog_bernoulli_20.npz", "goemotions_predictions.jsonl.gz", "dailydialog_predictions.jsonl.gz", "split_ids.json", "report.md", "source_sensor.py", "source_train_sensor.py"):
                self.assertIn(name, manifest["files"], name)
            self.assertEqual(manifest["live_model_calls"], 0)
            g = summary["goemotions"]; d = summary["dailydialog"]
            self.assertEqual(g["counts"]["fit"] + g["counts"]["development"], g["counts"]["selected_35"]); self.assertLess(g["counts"]["selected_35"], 900)
            self.assertEqual(g["counts"]["test"], 120); self.assertEqual(d["counts"]["test_turns"], 100); self.assertLess(d["counts"]["train_dialogues_selected"], 300)
            self.assertIsNone(g["comparators"]["archived_reference"]); self.assertTrue(any("missing" in m for m in logged))
            for name in ("linear", "mlp"):
                self.assertEqual(len(g["bootstrap"][name]["micro_f1"]["ci95"]), 2); self.assertIn("macro_f1", d["bootstrap"][name])
            self.assertEqual(set(summary["headline"]["goemotions"]), {"linear", "mlp", "bernoulli_35", "constant_prior"})
            sensor = S.Sensor.load(out / "sensor.json"); self.assertEqual(sensor.digest, g["deployed_sensor"]["digest"]); self.assertEqual(sensor.dimension, dimension)
            head = MLP.load(out / f"goemotions_{g['deployed_sensor']['selected_head']}.json"); self.assertEqual(head.to_dict()["digest"], sensor.head.to_dict()["digest"])
            rows = [json.loads(l) for l in gzip.open(out / "goemotions_predictions.jsonl.gz", "rt")]
            self.assertEqual(len(rows), 120); self.assertEqual(len(rows[0]["linear_probabilities"]), 28); self.assertNotIn("text", rows[0])
            self.assertIn("agreement", d["zero_shot"]["sign_agreement"]); self.assertEqual(d["zero_shot"]["sensor_digest"], sensor.digest)
            with self.assertRaises(ValueError):
                train_sensor.run(config, emb, data, dd, out, log=logged.append)


if __name__ == "__main__":
    unittest.main()

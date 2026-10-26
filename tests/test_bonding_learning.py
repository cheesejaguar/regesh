import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from regesh.bonding.mlp import MLP
from regesh.bonding import embeddings as emb


class MLPChecks(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        rng = np.random.default_rng(0)
        for loss in ("mse", "bce"):
            m = MLP([4, 6, 3], loss=loss, seed=1); x = rng.normal(size=(5, 4))
            y = rng.random((5, 3)) if loss == "bce" else rng.normal(size=(5, 3))
            gW, gb = m.gradients(x, y)
            for i in range(len(m.W)):
                for idx in [(0, 0), (1, 1)]:
                    old = m.W[i][idx]; eps = 1e-6
                    m.W[i][idx] = old + eps; lp = m.loss_value(x, y); m.W[i][idx] = old - eps; lm = m.loss_value(x, y); m.W[i][idx] = old
                    self.assertAlmostEqual((lp - lm) / (2 * eps), gW[i][idx], places=6)
                old = m.b[i][0]; m.b[i][0] = old + 1e-6; lp = m.loss_value(x, y); m.b[i][0] = old - 1e-6; lm = m.loss_value(x, y); m.b[i][0] = old
                self.assertAlmostEqual((lp - lm) / 2e-6, gb[i][0], places=6)

    def test_learns_nonlinear_target_and_round_trips(self):
        rng = np.random.default_rng(1); x = rng.normal(size=(1500, 2)); y = (np.sign(x[:, 0] * x[:, 1]) > 0).astype(float)[:, None]
        m = MLP([2, 16, 1], loss="bce", seed=0); history = m.fit(x, y, x_dev=x[:200], y_dev=y[:200], epochs=40, lr=5e-3, patience=6)
        self.assertGreater(((m.predict(x) > .5) == (y > .5)).mean(), .9)
        self.assertEqual(m.metadata["parameters"], 2 * 16 + 16 + 16 + 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "head.json"; m.save(path); copy = MLP.load(path)
            np.testing.assert_allclose(copy.predict(x[:5]), m.predict(x[:5]))
            self.assertEqual(copy.to_dict()["digest"], m.to_dict()["digest"])
        self.assertLessEqual(len(history), 40)


class EmbeddingCacheChecks(unittest.TestCase):
    def test_cache_round_trip_and_missing_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = emb.EmbeddingCache(Path(tmp) / "c.npz")
            texts = ["hello there", "goodbye", "hello there"]
            self.assertEqual(cache.add(texts, np.arange(9, dtype=np.float32).reshape(3, 3)), 2)   # duplicates stored once
            cache.save(); reopened = emb.EmbeddingCache(Path(tmp) / "c.npz")
            got = reopened.get(["goodbye", "hello there"])
            self.assertEqual(got.shape, (2, 3)); self.assertEqual(float(got[0, 0]), 3.)
            with self.assertRaises(KeyError): reopened.get(["unknown text"])

    def test_hash_selection_is_deterministic_and_proportional(self):
        ids = [f"train-{i}" for i in range(20000)]
        a = emb.select_by_hash(ids, .2, "salt"); b = emb.select_by_hash(ids, .2, "salt")
        self.assertTrue((a == b).all()); self.assertAlmostEqual(a.mean(), .2, delta=.01)
        self.assertNotEqual(emb.select_by_hash(ids[:10], .5, "salt").tolist(), emb.select_by_hash(ids[:10], .5, "other").tolist())

    def test_client_requires_configuration_and_text_key(self):
        with self.assertRaises(ValueError): emb.EmbeddingClient(base_url="", model="")
        self.assertEqual(emb.text_key("a"), emb.text_key("a")); self.assertNotEqual(emb.text_key("a"), emb.text_key("b"))


if __name__ == "__main__":
    unittest.main()

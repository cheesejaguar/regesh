"""Offline text-to-affect estimate for hooks, with the embedding sensor as an upgrade.

The default path uses the repository's archived GoEmotions reference (a hashed-token
Bernoulli classifier, 28 labels, micro-F1 .414 on the official test set) packaged as
`regesh/data/goemotions_reference.npz`; it needs no server. When the host configures
`REGESH_EMBEDDING_URL` (an llama.cpp server started with `--embedding --pooling mean`)
and a sensor bundle is available, the embedding head from `regesh.bonding.sensor` is
used instead (micro-F1 .486). Both map emotion probabilities to a valence in [-1, 1]
with the GoEmotions sentiment grouping and apply a neutral gate; see
results/bonding/text_affect_gate for the gate chosen for the offline path.
"""
from __future__ import annotations
from importlib import resources
import os
from pathlib import Path



from ..dataset_baseline import BernoulliReference
from ..bonding.sensor import valence_from_probabilities, support_proxy_from_probabilities, VALENCE_SIGNS

# Gate for the offline classifier, selected on the DailyDialog validation split and reported
# on test in results/bonding/text_affect_gate/gate.json (three-way macro-F1 criterion).
OFFLINE_GATE = {"label": "neutral", "tau": .2, "dead_band": .6}


class OfflineTextAffect:
    identifier = "goemotions-hashed-bernoulli-v1"

    def __init__(self, path=None):
        if path is None:
            with resources.as_file(resources.files("regesh").joinpath("data/goemotions_reference.npz")) as p:
                self.model = BernoulliReference.load(p)
        else:
            self.model = BernoulliReference.load(path)
        self.labels = tuple(self.model.labels)
        self.signs = {l: VALENCE_SIGNS[l] for l in self.labels}

    def probabilities(self, text):
        return self.model.probabilities([("", text, ())])[0]

    def estimate(self, text, gate=OFFLINE_GATE):
        p = self.probabilities(text)
        probabilities = {l: float(v) for l, v in zip(self.labels, p)}
        raw = valence_from_probabilities(p, self.labels, self.signs)
        gated = raw
        fired = False
        if gate is not None:
            fired = probabilities.get(gate["label"], 0.) >= gate["tau"] or abs(raw) < gate["dead_band"]
            gated = 0. if fired else raw
        top = max(probabilities, key=probabilities.get)
        return {"model": self.identifier, "partner_affect": float(gated), "raw_valence": float(raw), "gate_fired": bool(fired),
                "top_emotion": top, "top_probability": probabilities[top],
                "support_proxy": support_proxy_from_probabilities(p, self.labels)}


class EmbeddingTextAffect:
    """Embedding sensor path; only constructed when the host configured the server and bundle."""
    identifier = "embedding-sensor"

    def __init__(self, base_url, model, bundle):
        from ..bonding.embeddings import EmbeddingClient
        from ..bonding.sensor import Sensor, apply_gate, DEFAULT_NEUTRAL_GATE
        self.client = EmbeddingClient(base_url, model, timeout=20)
        self.sensor = Sensor.load(bundle); self._apply_gate = apply_gate; self.gate = DEFAULT_NEUTRAL_GATE

    def estimate(self, text, gate=None):
        est = self.sensor.estimate(self.client.embed([text])[0])
        affect, fired = self._apply_gate(est, self.gate if gate is None else gate)
        top = max(est["emotion_probabilities"], key=est["emotion_probabilities"].get)
        return {"model": f"{self.identifier}:{self.sensor.digest[:12]}", "partner_affect": float(affect), "raw_valence": est["partner_affect"],
                "gate_fired": bool(fired), "top_emotion": top, "top_probability": est["emotion_probabilities"][top], "support_proxy": est["support_proxy"]}


def default_estimator():
    """Embedding sensor if configured and reachable at construction, else the offline classifier."""
    url = os.environ.get("REGESH_EMBEDDING_URL"); model = os.environ.get("REGESH_EMBEDDING_MODEL", "regesh-qwen-coder")
    bundle = os.environ.get("REGESH_SENSOR_BUNDLE", str(Path.home() / ".regesh" / "sensor.json"))
    if url and Path(bundle).exists():
        try:
            return EmbeddingTextAffect(url, model, bundle)
        except Exception:
            pass
    return OfflineTextAffect()

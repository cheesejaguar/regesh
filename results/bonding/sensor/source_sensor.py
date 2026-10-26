"""Partner-state sensor: affect evidence estimated from the reply model's own embeddings.

The platform consumes `partner_affect` (dyadic identification and steering) and
`support` (comfort, request resolution) as evidence fields on each Event. Until
now a host had to supply them by hand. This module turns a mean-pooled embedding
of the user's utterance, produced by the same local language model that writes
the agent's replies, into those fields with a small trained head.

Boundary: the sensor is an optional evidence provider that the *host* runs before
`agent.observe`. The Agent never calls the network, and this module never does
either; the host obtains the embedding (for example with
`regesh.bonding.embeddings.EmbeddingClient` or a cache) and passes the vector in.
Everything the sensor adds to an Event is recorded under `data["sensor"]` so the
provenance of machine-estimated evidence is visible and replayable.

Valence uses a declared fixed map: the GoEmotions paper's sentiment grouping
(positive +1, negative -1, ambiguous 0, neutral 0), mass-weighted over the head's
28 label probabilities. The support value is a *proxy*: the probability that at
least one of five supportive labels is present. It detects supportive language,
not that the user answered the agent's request, and it is off by default.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .mlp import MLP
from .embeddings import DAILYDIALOG_EMOTIONS

GOEMOTIONS_LABELS = ("admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
                     "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
                     "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral")

# Demszky et al. (2020), GoEmotions, sentiment grouping (official repository data/sentiment_dict.json).
SENTIMENT_GROUPS = {
    "positive": ("admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"),
    "negative": ("anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"),
    "ambiguous": ("confusion", "curiosity", "realization", "surprise"),
    "neutral": ("neutral",),
}
VALENCE_MAP_NAME = "goemotions-sentiment-v1"
VALENCE_SIGNS = {**{l: 1. for l in SENTIMENT_GROUPS["positive"]}, **{l: -1. for l in SENTIMENT_GROUPS["negative"]},
                 **{l: 0. for l in SENTIMENT_GROUPS["ambiguous"]}, **{l: 0. for l in SENTIMENT_GROUPS["neutral"]}}
assert set(VALENCE_SIGNS) == set(GOEMOTIONS_LABELS) and len(GOEMOTIONS_LABELS) == 28

SUPPORT_PROXY_NAME = "goemotions-support-noisy-or-v1"
SUPPORT_PROXY_LABELS = ("caring", "gratitude", "approval", "love", "admiration")

# DailyDialog (Li et al. 2017) turn emotions mapped to the same three-valued scale.
DAILYDIALOG_VALENCE = {"no_emotion": 0., "anger": -1., "disgust": -1., "fear": -1., "happiness": 1., "sadness": -1., "surprise": 0.}
assert set(DAILYDIALOG_VALENCE) == set(DAILYDIALOG_EMOTIONS)


# ----------------------------------------------------------------------------- valence and support arithmetic

def _as_vector(probabilities, labels):
    if isinstance(probabilities, dict):
        return np.array([float(probabilities.get(l, 0.)) for l in labels], float)
    p = np.asarray(probabilities, float).ravel()
    if p.shape != (len(labels),):
        raise ValueError(f"expected {len(labels)} probabilities, got {p.shape}")
    return p


def valence_from_probabilities(probabilities, labels=GOEMOTIONS_LABELS, signs=VALENCE_SIGNS, eps=1e-12):
    """Mass-weighted mean sign: sum(sign_l p_l) / sum(p_l), clipped to [-1, 1]; 0 when there is no mass."""
    p = np.clip(_as_vector(probabilities, labels), 0., None)
    total = float(p.sum())
    if total <= eps:
        return 0.
    s = np.array([signs[l] for l in labels], float)
    return float(np.clip(float(p @ s) / total, -1., 1.))


def support_proxy_from_probabilities(probabilities, labels=GOEMOTIONS_LABELS, support_labels=SUPPORT_PROXY_LABELS):
    """Noisy-OR over the supportive labels: probability that at least one of them is present."""
    p = np.clip(_as_vector(probabilities, labels), 0., 1.)
    idx = [labels.index(l) for l in support_labels]
    return float(np.clip(1. - np.prod(1. - p[idx]), 0., 1.))


def dailydialog_valence(emotions, scale=DAILYDIALOG_VALENCE):
    return np.array([scale[e] for e in emotions], float)


# ----------------------------------------------------------------------------- standardization and training

def fit_standardizer(x, floor=1e-6):
    """Per-dimension mean and standard deviation of the training embeddings; tiny deviations become 1."""
    x = np.asarray(x, float)
    mean = x.mean(axis=0); std = x.std(axis=0)
    std = np.where(std < floor, 1., std)
    return mean, std


def standardize(x, mean, std):
    return (np.asarray(x, float) - mean) / std


def train_head(x, y, x_dev, y_dev, *, hidden=None, activation="tanh", epochs=60, batch=128, lr=1e-3, l2=1e-4, patience=6, seed=0, verbose=False):
    """Sigmoid-BCE head on standardized embeddings; linear when `hidden` is None. Returns (MLP, history)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    sizes = [x.shape[1]] + ([int(hidden)] if hidden else []) + [y.shape[1]]
    model = MLP(sizes, activation=activation or "tanh", loss="bce", seed=seed)
    history = model.fit(x, y, x_dev=np.asarray(x_dev, float), y_dev=np.asarray(y_dev, float), epochs=epochs, batch=batch, lr=lr, l2=l2, patience=patience, seed=seed, verbose=verbose)
    model.metadata.update(hidden=hidden, activation=activation if hidden else None, lr=lr, l2=l2, batch=batch, epochs_max=epochs, patience=patience, seed=seed,
                          train_rows=int(len(x)), dev_rows=int(len(x_dev)))
    return model, history


# ----------------------------------------------------------------------------- metrics (multiclass, valence, bootstrap)

def multiclass_metrics(gold, predicted, labels):
    """Accuracy, macro-F1 over all classes (F1 0 when undefined), per-class rows and the confusion matrix."""
    gold = np.asarray(gold, int); predicted = np.asarray(predicted, int); k = len(labels)
    confusion = np.zeros((k, k), int)
    np.add.at(confusion, (gold, predicted), 1)
    tp = np.diag(confusion).astype(float); fp = confusion.sum(axis=0) - tp; fn = confusion.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros(k), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(k), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=(precision + recall) > 0)
    return {"accuracy": float(np.mean(gold == predicted)), "macro_f1": float(f1.mean()), "samples": int(len(gold)),
            "per_class": [{"label": labels[i], "support": int(tp[i] + fn[i]), "predicted_count": int(tp[i] + fp[i]),
                           "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])} for i in range(k)],
            "confusion": confusion.tolist()}


def macro_f1_from_counts(tp, fp, fn):
    tp = np.asarray(tp, float); fp = np.asarray(fp, float); fn = np.asarray(fn, float)
    denominator = 2 * tp + fp + fn
    return float(np.mean(np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)))


def valence_mae(gold_valence, predicted_valence):
    return float(np.mean(np.abs(np.asarray(gold_valence, float) - np.asarray(predicted_valence, float))))


def sign_agreement(gold_sign, predicted_value):
    """Agreement of sign(predicted) with a non-zero gold sign; predicted exactly 0 counts as disagreement."""
    g = np.sign(np.asarray(gold_sign, float)); p = np.sign(np.asarray(predicted_value, float))
    mask = g != 0
    n = int(mask.sum())
    return {"n": n, "agreement": float(np.mean(p[mask] == g[mask])) if n else None,
            "positive_fraction": float(np.mean(g[mask] > 0)) if n else None}


def three_way_sign(values, dead_band):
    v = np.asarray(values, float)
    return np.where(v > dead_band, 1., np.where(v < -dead_band, -1., 0.))


def bootstrap(stat, n, *, samples=1000, seed=0):
    """Percentile 95% interval of `stat(index_array)` over item resamples; the same seed gives the same resamples."""
    rng = np.random.default_rng(seed)
    estimate = float(stat(np.arange(n)))
    draws = np.empty(samples)
    for s in range(samples):
        draws[s] = stat(rng.integers(0, n, n))
    return {"estimate": estimate, "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))], "samples": int(samples), "seed": int(seed), "n": int(n)}


# ----------------------------------------------------------------------------- deployable sensor

class Sensor:
    """A trained head plus its standardizer and the declared maps, saved as one JSON bundle."""
    SCHEMA_VERSION = 1

    def __init__(self, head, labels, mean, std, *, threshold=None, valence_map=VALENCE_MAP_NAME, valence_signs=None,
                 support_rule=SUPPORT_PROXY_NAME, support_labels=SUPPORT_PROXY_LABELS, metadata=None):
        self.head = head; self.labels = tuple(labels)
        self.mean = np.asarray(mean, float); self.std = np.asarray(std, float)
        if self.head.sizes[-1] != len(self.labels) or self.head.sizes[0] != len(self.mean) or self.mean.shape != self.std.shape:
            raise ValueError("head, labels and standardizer shapes disagree")
        self.threshold = None if threshold is None else float(threshold)
        self.valence_map = valence_map
        self.valence_signs = dict(valence_signs) if valence_signs is not None else {l: VALENCE_SIGNS.get(l) for l in self.labels}
        if set(self.valence_signs) != set(self.labels) or any(v is None for v in self.valence_signs.values()):
            raise ValueError("valence signs must cover exactly the head's labels")
        self.support_rule = support_rule; self.support_labels = tuple(support_labels)
        if any(l not in self.labels for l in self.support_labels):
            raise ValueError("support labels must be head labels")
        self.metadata = dict(metadata or {})

    @property
    def dimension(self):
        return int(self.head.sizes[0])

    @property
    def digest(self):
        core = {"head": self.head.to_dict()["digest"], "labels": list(self.labels), "mean": self.mean.tolist(), "std": self.std.tolist(),
                "valence_map": self.valence_map, "valence_signs": self.valence_signs, "support_rule": self.support_rule, "support_labels": list(self.support_labels)}
        return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()

    def probabilities(self, embeddings):
        x = np.atleast_2d(np.asarray(embeddings, float))
        if x.shape[1] != self.dimension:
            raise ValueError(f"embedding dimension {x.shape[1]} does not match the sensor's {self.dimension}")
        return self.head.predict(standardize(x, self.mean, self.std))

    def estimate(self, embedding):
        """Emotion probabilities, mapped valence in [-1, 1] and the support proxy in [0, 1] for one embedding."""
        p = self.probabilities(np.asarray(embedding, float).reshape(1, -1))[0]
        probabilities = {l: float(v) for l, v in zip(self.labels, p)}
        return {"emotion_probabilities": probabilities,
                "partner_affect": valence_from_probabilities(p, self.labels, self.valence_signs),
                "support_proxy": support_proxy_from_probabilities(p, self.labels, self.support_labels)}

    def to_dict(self):
        payload = {"schema_version": self.SCHEMA_VERSION, "kind": "regesh-affect-sensor", "head": self.head.to_dict(), "labels": list(self.labels),
                   "standardization": {"mean": self.mean.tolist(), "std": self.std.tolist(), "rule": "per-dimension (x - mean) / std from the training rows"},
                   "threshold": self.threshold, "valence_map": {"name": self.valence_map, "signs": self.valence_signs,
                   "rule": "sum(sign_l p_l) / sum(p_l), clipped to [-1, 1]"},
                   "support_proxy": {"name": self.support_rule, "labels": list(self.support_labels), "rule": "1 - prod(1 - p_l); a proxy, off by default"},
                   "metadata": self.metadata}
        payload["digest"] = self.digest
        return payload

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict()) + "\n")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        if d.get("schema_version") != cls.SCHEMA_VERSION or d.get("kind") != "regesh-affect-sensor":
            raise ValueError("unsupported sensor bundle")
        h = d["head"]
        head = MLP(h["sizes"], h["activation"], h["loss"], weights=h["weights"], metadata=h.get("metadata"))
        sensor = cls(head, d["labels"], d["standardization"]["mean"], d["standardization"]["std"], threshold=d.get("threshold"),
                     valence_map=d["valence_map"]["name"], valence_signs=d["valence_map"]["signs"],
                     support_rule=d["support_proxy"]["name"], support_labels=d["support_proxy"]["labels"], metadata=d.get("metadata"))
        if sensor.digest != d["digest"]:
            raise ValueError("sensor digest mismatch")
        return sensor


def annotate_event(event, sensor, embedding, *, fill_support=False):
    """Return a NEW Event whose data carries sensor-estimated evidence where the host left it absent.

    `partner_affect` is filled when absent. `support` is filled only when absent and the host passes
    `fill_support=True` (the value is a declared proxy). Hand-supplied fields are never overwritten.
    `data["sensor"]` records the model digest, probabilities and the valence map used.
    """
    estimate = sensor.estimate(embedding)
    data = deepcopy(event.data); filled = []
    if "partner_affect" not in data:
        data["partner_affect"] = estimate["partner_affect"]; filled.append("partner_affect")
    if fill_support and "support" not in data:
        data["support"] = estimate["support_proxy"]; filled.append("support")
    data["sensor"] = {"model_digest": sensor.digest, "emotion_probabilities": estimate["emotion_probabilities"], "valence_map": sensor.valence_map,
                      "partner_affect": estimate["partner_affect"], "support_proxy": estimate["support_proxy"], "support_rule": sensor.support_rule,
                      "filled": filled}
    return replace(event, data=data)

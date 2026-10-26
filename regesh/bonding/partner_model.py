"""Learned partner model: predict the partner's next affect from history and the agent's turn.

Frame. In a two-party conversation the PARTNER is speaker A (turns 1, 3, 5, ...)
and the AGENT role is speaker B (turns 2, 4, 6, ...). One example is the history
up to A's turn t (the previous A turn when it exists, and A's turn t), the
agent's intervening turn t+1, and the target A's affect at turn t+2: a seven-class
emotion label and its valence under a declared fixed map. This is the supervised,
observational analogue of the platform's online four-parameter identifier
(`dyad.RecursiveIdentifier`): P_{t+1} = b + r P_t + s_neg min(A_t,0) + s_pos max(A_t,0),
which sees only the two scalar valences. The learned model additionally sees
mean-pooled embeddings of the three utterances and the dialogue-act labels.

`LearnedPartnerModel.rank_candidates` is the latent-space analogue of
`dyad.steer`: the host supplies the embeddings (and, when known, the intended
valence and act) of several candidate replies and receives the predicted next
partner valence for each. The prediction is observational next-turn prediction
on a corpus of scripted human-human dialogues; it is not the causal effect of the
agent's expression, and speaker B in the corpus is a human, not an agent.

Everything here is deterministic given the inputs and declared seeds; nothing
calls an embedding endpoint (vectors come from `embeddings.EmbeddingCache`).
"""
from __future__ import annotations
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .dyad import RecursiveIdentifier
from .embeddings import DAILYDIALOG_EMOTIONS
from .mlp import MLP

ACTS = ("inform", "question", "directive", "commissive")
EMOTIONS = tuple(DAILYDIALOG_EMOTIONS)
EMOTION_INDEX = {e: i for i, e in enumerate(EMOTIONS)}
# Declared fixed valence map (see configs/bonding_partner_model.json for the justification).
VALENCE_MAP = {"no_emotion": 0., "happiness": .7, "surprise": .3, "anger": -.7, "disgust": -.6, "fear": -.5, "sadness": -.6}
HISTORY_SCALARS = ("a_prev_valence", "has_prev", "a_valence") + tuple(f"a_act_{a}" for a in ACTS) + ("exchange_index",)
CANDIDATE_SCALARS = ("b_valence",) + tuple(f"b_act_{a}" for a in ACTS)
SCALAR_FEATURES = HISTORY_SCALARS + CANDIDATE_SCALARS
FEATURE_SETS = ("embeddings+scalars", "scalars", "embeddings")
EXCHANGE_SCALE = 10.


# ----------------------------------------------------------------------------- dataset

def build_examples(dialogues, keep=None, split="train"):
    """Deterministic example list from DailyDialog dialogues (ParlAI JSON rows).

    Speakers alternate starting with A at index 0. For every A turn i (0, 2, 4, ...)
    with a B reply at i+1 and an A turn at i+2, one example is produced. The order
    is file order, then turn order, so examples of one dialogue are contiguous.
    `keep` is an optional boolean mask over dialogues (hash selection).
    """
    examples = []
    for di, row in enumerate(dialogues):
        if keep is not None and not keep[di]:
            continue
        turns = row["dialogue"]
        for i in range(0, len(turns) - 2, 2):
            prev = turns[i - 2] if i >= 2 else None
            examples.append({
                "split": split, "dialogue": di, "turn": i, "exchange": i // 2, "topic": row.get("topic"),
                "a_prev_text": prev["text"] if prev else None, "a_prev_emotion": prev["emotion"] if prev else None,
                "a_text": turns[i]["text"], "a_emotion": turns[i]["emotion"], "a_act": turns[i]["act"],
                "b_text": turns[i + 1]["text"], "b_emotion": turns[i + 1]["emotion"], "b_act": turns[i + 1]["act"],
                "target_emotion": turns[i + 2]["emotion"]})
    return examples


def act_onehot(act):
    v = np.zeros(len(ACTS))
    if act is not None:
        v[ACTS.index(act)] = 1.
    return v


def history_scalars(a_prev_emotion, a_emotion, a_act, exchange, valence_map=VALENCE_MAP):
    """Scalar context known before the agent speaks: previous/current partner valence, act, position."""
    has_prev = a_prev_emotion is not None
    return np.concatenate([[valence_map[a_prev_emotion] if has_prev else 0., 1. if has_prev else 0., valence_map[a_emotion]],
                           act_onehot(a_act), [exchange / EXCHANGE_SCALE]])


def candidate_scalars(b_emotion, b_act, valence_map=VALENCE_MAP):
    """Scalar description of the agent's (candidate) turn; None means unknown (neutral valence, no act)."""
    return np.concatenate([[valence_map[b_emotion] if b_emotion is not None else 0.], act_onehot(b_act)])


def scalar_features(examples, valence_map=VALENCE_MAP):
    """(n, 13) matrix: HISTORY_SCALARS then CANDIDATE_SCALARS."""
    if not examples:
        return np.zeros((0, len(SCALAR_FEATURES)))
    return np.vstack([np.concatenate([history_scalars(e["a_prev_emotion"], e["a_emotion"], e["a_act"], e["exchange"], valence_map),
                                      candidate_scalars(e["b_emotion"], e["b_act"], valence_map)]) for e in examples])


def targets(examples, valence_map=VALENCE_MAP):
    """Target valence (n,) and class index (n,) of A's turn t+2."""
    valence = np.array([valence_map[e["target_emotion"]] for e in examples], float)
    classes = np.array([EMOTION_INDEX[e["target_emotion"]] for e in examples], int)
    return valence, classes


def example_texts(examples):
    """Unique utterance texts an example set needs embeddings for, in first-seen order."""
    seen = {}
    for e in examples:
        for key in ("a_prev_text", "a_text", "b_text"):
            if e[key] is not None and e[key] not in seen:
                seen[e[key]] = None
    return list(seen)


def embedding_parts(examples, cache):
    """Raw embeddings (E_prev, E_a, E_b), each (n, d); E_prev rows are zero when there is no previous A turn.

    `cache` is anything with `get(texts) -> (len(texts), d)` (an EmbeddingCache or a stand-in).
    """
    texts = example_texts(examples)
    if not texts:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))
    vectors = np.asarray(cache.get(texts), float)
    index = {t: i for i, t in enumerate(texts)}
    e_a = vectors[[index[e["a_text"]] for e in examples]]
    e_b = vectors[[index[e["b_text"]] for e in examples]]
    e_prev = np.zeros_like(e_a)
    rows = [i for i, e in enumerate(examples) if e["a_prev_text"] is not None]
    if rows:
        e_prev[rows] = vectors[[index[examples[i]["a_prev_text"]] for i in rows]]
    return e_prev, e_a, e_b


def fit_standardizer(vectors, floor=1e-3):
    """Per-dimension mean and standard deviation; `floor` keeps near-constant (float16-quantized) dimensions small."""
    vectors = np.asarray(vectors, float)
    return vectors.mean(axis=0), np.maximum(vectors.std(axis=0), floor)


def fit_projection(z, components):
    """Principal components of standardized vectors: (d, k) matrix and the explained-variance fractions.

    Deterministic: eigendecomposition of the covariance, descending order, each component's
    sign fixed so its largest-magnitude entry is positive.
    """
    z = np.asarray(z, float); cov = z.T @ z / max(1, len(z))
    values, vectors = np.linalg.eigh(cov); order = np.argsort(values)[::-1][:components]
    comps = vectors[:, order]; values = np.maximum(values[order], 0.)
    signs = np.sign(comps[np.abs(comps).argmax(axis=0), np.arange(comps.shape[1])]); signs[signs == 0] = 1.
    return comps * signs, values / max(np.trace(cov), 1e-12)


def class_distribution(classes, n_classes=len(EMOTIONS)):
    counts = np.bincount(np.asarray(classes, int), minlength=n_classes)
    return {"counts": {EMOTIONS[i]: int(c) for i, c in enumerate(counts)}, "fractions": {EMOTIONS[i]: float(c / max(1, counts.sum())) for i, c in enumerate(counts)},
            "n": int(counts.sum()), "majority": EMOTIONS[int(np.argmax(counts))], "majority_fraction": float(counts.max() / max(1, counts.sum()))}


# ----------------------------------------------------------------------------- learned model

class LearnedPartnerModel:
    """Two small MLP heads over [standardized (optionally PCA-projected) embeddings of (A_{t-1}, A_t, B_{t+1}), scalar features].

    `feature_set`: 'embeddings+scalars' (full), 'scalars' (ablation without text) or
    'embeddings' (text, has_prev and position only: no gold affect or act labels).
    The valence head is trained with mean squared error, the class head with
    sigmoid cross-entropy on one-hot targets; class probabilities are the
    normalized sigmoids and the class decision is their argmax, or the argmax
    after division by the training prior ('balanced' decision rule).
    """
    SCHEMA = 1

    def __init__(self, feature_set, embedding_mean, embedding_sd, class_prior, valence_map=None, emotions=EMOTIONS,
                 valence_head=None, class_head=None, metadata=None, embedding_projection=None):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        self.feature_set = feature_set
        self.embedding_mean = np.asarray(embedding_mean, float); self.embedding_sd = np.asarray(embedding_sd, float)
        self.embedding_projection = None if embedding_projection is None else np.asarray(embedding_projection, float)
        self.class_prior = np.asarray(class_prior, float)
        self.valence_map = dict(valence_map or VALENCE_MAP); self.emotions = tuple(emotions)
        self.valence_head = valence_head; self.class_head = class_head
        self.metadata = metadata or {}

    # feature assembly -------------------------------------------------------
    @property
    def dim(self):
        return int(self.embedding_mean.shape[0])

    @property
    def latent(self):
        """Width of one utterance block after standardization and the optional projection."""
        return self.dim if self.embedding_projection is None else int(self.embedding_projection.shape[1])

    @property
    def input_size(self):
        d = self.latent
        return {"embeddings+scalars": 3 * d + len(SCALAR_FEATURES), "scalars": len(SCALAR_FEATURES), "embeddings": 3 * d + 2}[self.feature_set]

    def standardize(self, e):
        z = (np.asarray(e, float) - self.embedding_mean) / self.embedding_sd
        return z if self.embedding_projection is None else z @ self.embedding_projection

    def assemble(self, e_prev, e_a, e_b, h, c):
        """Input matrix from raw embeddings and the two scalar blocks; rows without a previous turn get zero E_prev."""
        h = np.atleast_2d(np.asarray(h, float)); c = np.atleast_2d(np.asarray(c, float))
        if self.feature_set == "scalars":
            return np.hstack([h, c])
        has_prev = h[:, HISTORY_SCALARS.index("has_prev")][:, None]
        z_prev = self.standardize(np.atleast_2d(e_prev)) * has_prev
        z_a = self.standardize(np.atleast_2d(e_a)); z_b = self.standardize(np.atleast_2d(e_b))
        if self.feature_set == "embeddings":
            return np.hstack([z_prev, z_a, z_b, has_prev, h[:, [HISTORY_SCALARS.index("exchange_index")]]])
        return np.hstack([z_prev, z_a, z_b, h, c])

    def features(self, examples, cache):
        s = scalar_features(examples, self.valence_map); nh = len(HISTORY_SCALARS)
        if self.feature_set == "scalars":
            return self.assemble(None, None, None, s[:, :nh], s[:, nh:])
        e_prev, e_a, e_b = embedding_parts(examples, cache)
        return self.assemble(e_prev, e_a, e_b, s[:, :nh], s[:, nh:])

    # training -----------------------------------------------------------------
    @classmethod
    def train(cls, feature_set, x_train, y_valence, y_class, x_dev, dev_valence, dev_class, *, embedding_mean, embedding_sd,
              hidden=(64,), activation="tanh", lr=1e-3, l2=1e-4, batch=64, epochs=80, patience=8, seed=0, valence_map=None,
              emotions=EMOTIONS, class_prior=None, metadata=None, embedding_projection=None):
        """Fit both heads with Adam and early stopping on the development split. Returns (model, histories)."""
        n_classes = len(emotions)
        prior = np.bincount(np.asarray(y_class, int), minlength=n_classes) / max(1, len(y_class)) if class_prior is None else np.asarray(class_prior, float)
        onehot = np.eye(n_classes)[np.asarray(y_class, int)]; dev_onehot = np.eye(n_classes)[np.asarray(dev_class, int)]
        sizes = [x_train.shape[1], *hidden]
        v_head = MLP(sizes + [1], activation=activation, loss="mse", seed=seed)
        c_head = MLP(sizes + [n_classes], activation=activation, loss="bce", seed=seed + 1)
        hv = v_head.fit(x_train, np.asarray(y_valence, float)[:, None], x_dev=x_dev, y_dev=np.asarray(dev_valence, float)[:, None],
                        epochs=epochs, batch=batch, lr=lr, l2=l2, patience=patience, seed=seed)
        hc = c_head.fit(x_train, onehot, x_dev=x_dev, y_dev=dev_onehot, epochs=epochs, batch=batch, lr=lr, l2=l2, patience=patience, seed=seed)
        meta = {"feature_set": feature_set, "hidden": list(hidden), "activation": activation, "lr": lr, "l2": l2, "batch": batch, "epochs": epochs,
                "patience": patience, "seed": seed, "n_train": int(len(x_train)), "n_dev": int(len(x_dev)), **(metadata or {})}
        model = cls(feature_set, embedding_mean, embedding_sd, prior, valence_map, emotions, v_head, c_head, meta, embedding_projection)
        return model, {"valence": hv, "class": hc}

    # inference ---------------------------------------------------------------
    def predict(self, x):
        """Predicted valence (clipped to [-1, 1]), normalized class probabilities and both class decisions."""
        x = np.atleast_2d(np.asarray(x, float))
        valence = np.clip(self.valence_head.predict(x)[:, 0], -1., 1.)
        sig = self.class_head.predict(x)
        probs = sig / np.maximum(sig.sum(axis=1, keepdims=True), 1e-12)
        balanced = probs / np.maximum(self.class_prior, 1e-6)
        return {"valence": valence, "class_probabilities": probs, "class": probs.argmax(axis=1), "class_balanced": balanced.argmax(axis=1)}

    def rank_candidates(self, history, candidate_embeddings, candidate_scalars=None):
        """Predicted next partner affect for each candidate agent reply (latent-space analogue of `dyad.steer`).

        `history`: dict with 'a_embedding' (d,), optional 'a_prev_embedding' (d,) or None, and
        'scalars' (the HISTORY_SCALARS vector from `history_scalars`). `candidate_embeddings`:
        (k, d) embeddings of the candidate replies. `candidate_scalars`: optional (k, 5)
        CANDIDATE_SCALARS rows (intended valence and act); when None the candidates are
        described as unknown (neutral valence, no act), which the 'embeddings' feature set ignores.
        Returns the `predict` dict with k rows plus the ranking (indices, best first by valence).
        """
        cands = np.atleast_2d(np.asarray(candidate_embeddings, float)); k = len(cands)
        h = np.tile(np.asarray(history["scalars"], float), (k, 1))
        c = np.tile(candidate_scalars_default(), (k, 1)) if candidate_scalars is None else np.atleast_2d(np.asarray(candidate_scalars, float))
        if self.feature_set == "scalars":
            x = self.assemble(None, None, None, h, c)
        else:
            prev = history.get("a_prev_embedding")
            e_prev = np.tile(np.zeros(self.dim) if prev is None else np.asarray(prev, float), (k, 1))
            e_a = np.tile(np.asarray(history["a_embedding"], float), (k, 1))
            x = self.assemble(e_prev, e_a, cands, h, c)
        out = self.predict(x)
        out["ranking"] = np.argsort(-out["valence"], kind="stable")
        return out

    # persistence --------------------------------------------------------------
    def to_dict(self):
        core = {"schema_version": self.SCHEMA, "feature_set": self.feature_set, "emotions": list(self.emotions), "valence_map": self.valence_map,
                "embedding_mean": self.embedding_mean.tolist(), "embedding_sd": self.embedding_sd.tolist(), "class_prior": self.class_prior.tolist(),
                "embedding_projection": None if self.embedding_projection is None else self.embedding_projection.tolist(),
                "valence_head": self.valence_head.to_dict(), "class_head": self.class_head.to_dict()}
        digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
        return {**core, "metadata": self.metadata, "digest": digest}

    def save(self, path):
        path = Path(path); payload = json.dumps(self.to_dict()) + "\n"
        if path.suffix == ".gz":
            with gzip.open(path, "wt") as f:
                f.write(payload)
        else:
            path.write_text(payload)
        return self.to_dict()["digest"]

    @classmethod
    def load(cls, path, verify=True):
        path = Path(path)
        text = gzip.open(path, "rt").read() if path.suffix == ".gz" else path.read_text()
        d = json.loads(text)
        if d.get("schema_version") != cls.SCHEMA:
            raise ValueError("unsupported partner model schema")
        model = cls(d["feature_set"], d["embedding_mean"], d["embedding_sd"], d["class_prior"], d["valence_map"], tuple(d["emotions"]),
                    _mlp_from_dict(d["valence_head"]), _mlp_from_dict(d["class_head"]), d.get("metadata", {}), d.get("embedding_projection"))
        if verify and model.to_dict()["digest"] != d["digest"]:
            raise ValueError("partner model digest mismatch")
        return model


def candidate_scalars_default():
    return candidate_scalars(None, None)


def _mlp_from_dict(d):
    if d["schema_version"] != 1:
        raise ValueError("unsupported MLP schema")
    return MLP(d["sizes"], d["activation"], d["loss"], weights=d["weights"], metadata=d["metadata"])


def make_history(a_embedding, a_prev_embedding, a_prev_emotion, a_emotion, a_act, exchange, valence_map=VALENCE_MAP):
    """Convenience for hosts: the `history` argument of `rank_candidates` from raw pieces."""
    return {"a_embedding": np.asarray(a_embedding, float), "a_prev_embedding": None if a_prev_embedding is None else np.asarray(a_prev_embedding, float),
            "scalars": history_scalars(a_prev_emotion, a_emotion, a_act, exchange, valence_map)}


# ----------------------------------------------------------------------------- baselines

def persistence_baseline(examples, valence_map=VALENCE_MAP):
    """Predict that A's affect at t+2 equals A's affect at t (valence and class)."""
    valence = np.array([valence_map[e["a_emotion"]] for e in examples], float)
    classes = np.array([EMOTION_INDEX[e["a_emotion"]] for e in examples], int)
    return valence, classes


def constant_baseline(train_valence, train_classes, n):
    """Mean training valence and the majority training class for every example."""
    counts = np.bincount(np.asarray(train_classes, int), minlength=len(EMOTIONS))
    return np.full(n, float(np.mean(train_valence))), np.full(n, int(np.argmax(counts)), int)


def pooled_bilinear_fit(a_valence, b_valence, target_valence):
    """Least squares for the identifier's four parameters on pooled examples: [1, P_t, min(A,0), max(A,0)]."""
    X = np.column_stack([np.ones(len(a_valence)), a_valence, np.minimum(b_valence, 0), np.maximum(b_valence, 0)])
    theta, *_ = np.linalg.lstsq(X, np.asarray(target_valence, float), rcond=None)
    return theta


def identifier_baseline(examples, valence_map=VALENCE_MAP, theta0=None, prior=1e2, lam=.98):
    """The platform's online identifier run within each dialogue on the (A valence, B valence) sequence.

    A fresh `RecursiveIdentifier` per dialogue (default settings unless `theta0`/`prior`/`lam`
    are given) predicts A_{t+2} from (A_t, B_{t+1}) BEFORE observing it, then updates.
    Returns the predictions (n,) and the parameter vector used for each prediction (n, 4),
    so the identifier can also rank candidate replies by their valence.
    """
    preds = np.zeros(len(examples)); thetas = np.zeros((len(examples), 4))
    ident = None; current = None
    for i, e in enumerate(examples):
        key = (e["split"], e["dialogue"])
        if key != current:
            state = None
            if theta0 is not None or prior != 1e2 or lam != .98:
                state = {"theta": list(map(float, theta0 if theta0 is not None else [0., .3, .2, .2])), "P": (np.eye(4) * prior).tolist(), "n": 0, "lam": lam, "sse": 0.}
            ident = RecursiveIdentifier(deepcopy(state) if state is not None else None); current = key
        p_now, a_now = valence_map[e["a_emotion"]], valence_map[e["b_emotion"]]
        thetas[i] = ident.state["theta"]
        preds[i] = ident.predict(p_now, a_now)
        ident.update(p_now, a_now, valence_map[e["target_emotion"]])
    return np.clip(preds, -1., 1.), thetas


def identifier_candidate_valence(thetas, a_valence, b_candidates):
    """Identifier prediction for alternative agent valences, using the per-example parameters."""
    feats = np.column_stack([np.ones(len(a_valence)), a_valence, np.minimum(b_candidates, 0), np.maximum(b_candidates, 0)])
    return np.clip((feats * thetas).sum(axis=1), -1., 1.)


class RidgeModel:
    """Pooled ridge regression on standardized features: a valence output and a one-hot class block (argmax)."""

    def __init__(self, lam=1., weights=None, bias=None, mean=None, sd=None):
        self.lam = lam; self.W = weights; self.b = bias; self.mean = mean; self.sd = sd

    def fit(self, x, y):
        x = np.asarray(x, float); y = np.atleast_2d(np.asarray(y, float).T).T
        self.mean = x.mean(axis=0); self.sd = np.maximum(x.std(axis=0), 1e-6)
        z = (x - self.mean) / self.sd
        A = z.T @ z + self.lam * np.eye(z.shape[1]); self.b = y.mean(axis=0)
        self.W = np.linalg.solve(A, z.T @ (y - self.b))
        return self

    def predict(self, x):
        return ((np.asarray(x, float) - self.mean) / self.sd) @ self.W + self.b

    @staticmethod
    def select(x, y, x_dev, y_dev, grid):
        """Choose the ridge penalty with the lowest development mean squared error (declared grid)."""
        scores = {}
        for lam in grid:
            m = RidgeModel(lam).fit(x, y); scores[lam] = float(np.mean((m.predict(x_dev) - np.atleast_2d(np.asarray(y_dev, float).T).T) ** 2))
        best = min(scores, key=lambda k: (scores[k], k))
        return RidgeModel(best).fit(x, y), {str(k): v for k, v in scores.items()}, best


# ----------------------------------------------------------------------------- evaluation

def confusion(target, pred, n_classes=len(EMOTIONS)):
    return np.bincount(np.asarray(target, int) * n_classes + np.asarray(pred, int), minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def macro_f1(target, pred, n_classes=len(EMOTIONS)):
    """Macro F1 over the classes present in the targets (declared convention)."""
    cm = confusion(target, pred, n_classes); tp = np.diag(cm); support = cm.sum(axis=1); predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), 0.); recall = np.where(support > 0, tp / np.maximum(support, 1), 0.)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.)
    present = support > 0
    return float(f1[present].mean()) if present.any() else float("nan"), f1


def evaluate(pred_valence, target_valence, pred_class=None, target_class=None, n_classes=len(EMOTIONS)):
    """Valence MAE/RMSE, sign accuracy on non-neutral targets (a zero prediction counts as wrong), and class metrics."""
    pv = np.asarray(pred_valence, float); tv = np.asarray(target_valence, float)
    err = pv - tv; nonneutral = tv != 0
    out = {"n": int(len(tv)), "mae": float(np.abs(err).mean()) if len(tv) else float("nan"), "rmse": float(np.sqrt((err ** 2).mean())) if len(tv) else float("nan"),
           "n_nonneutral": int(nonneutral.sum()),
           "sign_accuracy_nonneutral": float((np.sign(pv[nonneutral]) == np.sign(tv[nonneutral])).mean()) if nonneutral.any() else float("nan")}
    if pred_class is not None and target_class is not None:
        pc = np.asarray(pred_class, int); tc = np.asarray(target_class, int)
        f1, per_class = macro_f1(tc, pc, n_classes)
        out.update(accuracy=float((pc == tc).mean()) if len(tc) else float("nan"), macro_f1=f1,
                   per_class_f1={EMOTIONS[i]: float(per_class[i]) for i in range(n_classes)},
                   confusion=confusion(tc, pc, n_classes).tolist())
    return out


def _group_offsets(dialogue_keys):
    """Contiguous groups of examples by dialogue: (starts, sizes) in first-seen order."""
    keys = list(dialogue_keys); starts, sizes = [], []
    for i, k in enumerate(keys):
        if i == 0 or k != keys[i - 1]:
            starts.append(i); sizes.append(0)
        sizes[-1] += 1
    return np.array(starts, int), np.array(sizes, int)


def bootstrap_metrics(dialogue_keys, target_valence, target_class, predictions, *, seed, samples, pairs=(), metrics=("mae", "rmse", "sign_accuracy_nonneutral", "accuracy", "macro_f1")):
    """Dialogue-level bootstrap intervals for each prediction set and paired differences.

    `predictions`: {name: (valence (n,), class (n,) or None)}; `pairs`: [(name, baseline_name)]
    giving paired differences name - baseline on the same resamples. Examples must be grouped
    contiguously by dialogue (as `build_examples` produces them).
    """
    starts, sizes = _group_offsets(dialogue_keys); rng = np.random.default_rng(seed); G = len(starts)
    tv = np.asarray(target_valence, float); tc = np.asarray(target_class, int)
    names = list(predictions); stats = {n: {m: [] for m in metrics} for n in names}
    for _ in range(samples):
        pick = rng.integers(0, G, G); counts = sizes[pick]; total = int(counts.sum())
        idx = np.repeat(starts[pick], counts) + (np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts))
        for n in names:
            pv, pc = predictions[n]
            ev = evaluate(pv[idx], tv[idx], None if pc is None else pc[idx], None if pc is None else tc[idx])
            for m in metrics:
                stats[n][m].append(ev.get(m, float("nan")))
    out = {"samples": samples, "seed": seed, "dialogues": int(G), "estimates": {}, "paired_differences": {}}
    for n in names:
        pv, pc = predictions[n]
        point = evaluate(pv, tv, pc, None if pc is None else tc)
        out["estimates"][n] = {m: _interval(point.get(m, float("nan")), stats[n][m]) for m in metrics if not (pc is None and m in ("accuracy", "macro_f1"))}
    for a, b in pairs:
        pa, pb = predictions[a], predictions[b]
        pa_eval = evaluate(pa[0], tv, pa[1], None if pa[1] is None else tc); pb_eval = evaluate(pb[0], tv, pb[1], None if pb[1] is None else tc)
        out["paired_differences"][f"{a} - {b}"] = {
            m: _interval(pa_eval.get(m, float("nan")) - pb_eval.get(m, float("nan")), np.asarray(stats[a][m]) - np.asarray(stats[b][m]))
            for m in metrics if not ((pa[1] is None or pb[1] is None) and m in ("accuracy", "macro_f1"))}
    return out


def _interval(point, draws):
    draws = np.asarray(draws, float); draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return {"point": float(point), "ci95": [None, None], "n_draws": 0}
    lo, hi = np.quantile(draws, [.025, .975])
    return {"point": float(point), "ci95": [float(lo), float(hi)], "n_draws": int(len(draws))}


def preference_check(predict_fn, e_prev, e_a, e_b, h, c, target_valence, dialogue_keys, *, seed, samples=2000):
    """Within-item preference accuracy for `rank_candidates`-style use (a weak observational proxy).

    For each test example whose true next partner emotion is non-neutral, the actual agent
    turn is compared with one randomly drawn other agent turn from the test set (different
    example, declared seed). `predict_fn(e_prev, e_a, e_b, h, c) -> valence` is evaluated for
    both. 'Preference accuracy' is the fraction of items where sign(pred_actual - pred_other)
    equals the sign of the true target valence: for a positive outcome the model should
    prefer the actual turn, for a negative outcome it should prefer the alternative. Also
    reported for positive and negative targets separately, with dialogue bootstrap intervals.
    """
    tv = np.asarray(target_valence, float); items = np.where(tv != 0)[0]
    if len(items) == 0:
        return {"n_items": 0}
    rng = np.random.default_rng(seed); n = len(tv)
    other = rng.integers(0, n - 1, len(items)); other = other + (other >= items)   # uniform over the other examples
    pa = predict_fn(e_prev[items], e_a[items], e_b[items], h[items], c[items])
    po = predict_fn(e_prev[items], e_a[items], e_b[other], h[items], c[other])
    diff = pa - po; correct = (np.sign(diff) == np.sign(tv[items])).astype(float)
    pos, neg = tv[items] > 0, tv[items] < 0
    keys = [dialogue_keys[i] for i in items]
    out = {"n_items": int(len(items)), "n_positive": int(pos.sum()), "n_negative": int(neg.sum()),
           "preference_accuracy": _bootstrap_mean(correct, keys, rng, samples),
           "preference_accuracy_positive": _bootstrap_mean(correct[pos], [k for k, p in zip(keys, pos) if p], rng, samples) if pos.any() else None,
           "preference_accuracy_negative": _bootstrap_mean(correct[neg], [k for k, p in zip(keys, neg) if p], rng, samples) if neg.any() else None,
           "actual_above_other_fraction": float((diff > 0).mean()), "mean_abs_difference": float(np.abs(diff).mean()),
           "ties_fraction": float((diff == 0).mean())}
    return out


def _bootstrap_mean(values, keys, rng, samples):
    values = np.asarray(values, float)
    groups = {}
    for v, k in zip(values, keys):
        groups.setdefault(k, []).append(v)
    sums = np.array([np.sum(g) for g in groups.values()]); counts = np.array([len(g) for g in groups.values()]); G = len(sums)
    draws = []
    for _ in range(samples):
        pick = rng.integers(0, G, G); draws.append(sums[pick].sum() / counts[pick].sum())
    return _interval(float(values.mean()), draws)

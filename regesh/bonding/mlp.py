"""Small NumPy multilayer perceptron with Adam, early stopping and JSON persistence.

Deliberately minimal: one or two hidden layers, tanh or ReLU, mean-squared-error
or multilabel sigmoid cross-entropy. Weights are saved with a digest so an agent
can persist a trained head like the existing ridge regulator. No autograd
library is required; gradients are written out by hand and unit tested against
finite differences.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def _act(x, kind):
    return np.tanh(x) if kind == "tanh" else np.maximum(x, 0.)


def _act_grad(h, kind):
    return 1 - h ** 2 if kind == "tanh" else (h > 0).astype(h.dtype)


class MLP:
    def __init__(self, sizes, activation="tanh", loss="mse", seed=0, weights=None, metadata=None):
        self.sizes = list(sizes); self.activation = activation; self.loss = loss
        self.metadata = metadata or {}
        if weights is None:
            rng = np.random.default_rng(seed)
            self.W = [rng.normal(0, np.sqrt(2. / (a + b)), (a, b)) for a, b in zip(self.sizes[:-1], self.sizes[1:])]
            self.b = [np.zeros(b) for b in self.sizes[1:]]
        else:
            self.W = [np.asarray(w, float) for w in weights["W"]]; self.b = [np.asarray(v, float) for v in weights["b"]]

    def forward(self, x):
        h = [np.asarray(x, float)]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h[-1] @ W + b
            h.append(z if i == len(self.W) - 1 else _act(z, self.activation))
        return h

    def predict(self, x):
        out = self.forward(np.atleast_2d(x))[-1]
        if self.loss == "bce":
            out = 1 / (1 + np.exp(-np.clip(out, -40, 40)))
        return out

    def loss_value(self, x, y):
        out = self.forward(x)[-1]; y = np.asarray(y, float)
        if self.loss == "bce":
            z = np.clip(out, -40, 40)
            return float(np.mean(np.logaddexp(0, z) - y * z))
        return float(np.mean((out - y) ** 2))

    def gradients(self, x, y, l2=0.):
        h = self.forward(x); y = np.asarray(y, float); n = len(y)
        out = h[-1]
        if self.loss == "bce":
            delta = (1 / (1 + np.exp(-np.clip(out, -40, 40))) - y) / (n * y.shape[1])
        else:
            delta = 2 * (out - y) / (n * y.shape[1])
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = h[i].T @ delta + l2 * self.W[i]; gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * _act_grad(h[i], self.activation)
        return gW, gb

    def fit(self, x, y, *, x_dev=None, y_dev=None, epochs=50, batch=64, lr=1e-3, l2=1e-5, patience=5, seed=0, verbose=False):
        """Adam with mini-batches; early stopping on development loss when provided."""
        rng = np.random.default_rng(seed); x = np.asarray(x, float); y = np.asarray(y, float)
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps, t = .9, .999, 1e-8, 0
        best, best_state, bad, history = np.inf, None, 0, []
        for epoch in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch):
                idx = order[start:start + batch]
                gW, gb = self.gradients(x[idx], y[idx], l2); t += 1
                for i in range(len(self.W)):
                    mW[i] = b1 * mW[i] + (1 - b1) * gW[i]; vW[i] = b2 * vW[i] + (1 - b2) * gW[i] ** 2
                    mb[i] = b1 * mb[i] + (1 - b1) * gb[i]; vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
                    self.W[i] -= lr * (mW[i] / (1 - b1 ** t)) / (np.sqrt(vW[i] / (1 - b2 ** t)) + eps)
                    self.b[i] -= lr * (mb[i] / (1 - b1 ** t)) / (np.sqrt(vb[i] / (1 - b2 ** t)) + eps)
            train_loss = self.loss_value(x, y)
            dev_loss = self.loss_value(x_dev, y_dev) if x_dev is not None else train_loss
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "dev_loss": dev_loss})
            if verbose:
                print(f"epoch {epoch + 1} train {train_loss:.5f} dev {dev_loss:.5f}", flush=True)
            if dev_loss < best - 1e-9:
                best, bad = dev_loss, 0; best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.W, self.b = best_state
        self.metadata.update(epochs_run=len(history), best_dev_loss=float(best), parameters=int(sum(w.size for w in self.W) + sum(b.size for b in self.b)))
        return history

    def to_dict(self):
        payload = {"schema_version": 1, "sizes": self.sizes, "activation": self.activation, "loss": self.loss,
                   "weights": {"W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b]}, "metadata": self.metadata}
        payload["digest"] = hashlib.sha256(json.dumps({k: payload[k] for k in ("sizes", "activation", "loss", "weights")}, sort_keys=True).encode()).hexdigest()
        return payload

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict()) + "\n")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        if d["schema_version"] != 1:
            raise ValueError("unsupported MLP schema")
        return cls(d["sizes"], d["activation"], d["loss"], weights=d["weights"], metadata=d["metadata"])

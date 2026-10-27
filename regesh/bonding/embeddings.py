"""Embeddings from the local language model, with a content-addressed cache.

The same model that generates the agent's replies serves mean-pooled hidden-state
embeddings through llama.cpp's OpenAI-compatible endpoint when started with
`--embedding --pooling mean`. Vectors are cached as float16 NPZ files outside
git, keyed by the SHA-256 of the exact text, so training and evaluation never
re-query the server and are reproducible from the cache manifest.

Run: python -m regesh.bonding.embeddings --dataset goemotions --split test \
       --cache .cache/regesh-data --out .cache/embeddings --base-url http://127.0.0.1:8080/v1 --model regesh-qwen-coder
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import numpy as np

DAILYDIALOG_EMOTIONS = ("no_emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise")


def text_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingClient:
    def __init__(self, base_url=None, model=None, timeout=600):
        self.base_url = (base_url or os.environ.get("REGESH_MODEL_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("REGESH_MODEL_NAME")
        if not self.base_url or not self.model:
            raise ValueError("base_url and model are required")
        self.timeout = timeout; self.calls = 0; self.prompt_tokens = 0; self.elapsed = 0.

    def embed(self, texts, batch=32):
        out = []
        for start in range(0, len(texts), batch):
            chunk = [t if t.strip() else "." for t in texts[start:start + batch]]
            body = json.dumps({"model": self.model, "input": chunk}).encode()
            t0 = time.perf_counter()
            with urllib.request.urlopen(urllib.request.Request(self.base_url + "/embeddings", body, {"Content-Type": "application/json"}), timeout=self.timeout) as r:
                d = json.loads(r.read())
            self.elapsed += time.perf_counter() - t0; self.calls += 1
            self.prompt_tokens += int(d.get("usage", {}).get("prompt_tokens", 0) or 0)
            rows = sorted(d["data"], key=lambda x: x["index"])
            if len(rows) != len(chunk):
                raise ValueError("embedding count mismatch")
            out.extend(np.asarray(r["embedding"], dtype=np.float32) for r in rows)
        return np.vstack(out) if out else np.zeros((0, 0), np.float32)


class EmbeddingCache:
    """NPZ store: keys (sha256 of text), vectors (float16, n x d)."""
    def __init__(self, path):
        self.path = Path(path); self.keys = []; self.vectors = None; self.index = {}
        if self.path.exists():
            d = np.load(self.path, allow_pickle=False)
            self.keys = [str(k) for k in d["keys"]]; self.vectors = d["vectors"]
            self.index = {k: i for i, k in enumerate(self.keys)}

    def lookup(self, texts):
        keys = [text_key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.index]
        return keys, missing

    def add(self, texts, vectors):
        keys = [text_key(t) for t in texts]
        new, seen = [], set()
        for k, v in zip(keys, vectors):
            if k not in self.index and k not in seen:
                new.append((k, v)); seen.add(k)
        if not new:
            return 0
        block = np.vstack([v.astype(np.float16) for _, v in new])
        self.vectors = block if self.vectors is None else np.vstack([self.vectors, block])
        for k, _ in new:
            self.index[k] = len(self.keys); self.keys.append(k)
        return len(new)

    def get(self, texts):
        keys, missing = self.lookup(texts)
        if missing:
            raise KeyError(f"{len(missing)} texts not in cache")
        return self.vectors[[self.index[k] for k in keys]].astype(np.float32)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(self.keys), vectors=self.vectors if self.vectors is not None else np.zeros((0, 0), np.float16))
        os.replace(tmp, self.path)

    def ensure(self, texts, client, batch=32, checkpoint_every=2048, log=None):
        """Embed only texts missing from the cache; save periodically."""
        keys, missing = self.lookup(texts)
        unique = list(dict.fromkeys(texts[i] for i in missing))
        for start in range(0, len(unique), checkpoint_every):
            chunk = unique[start:start + checkpoint_every]
            self.add(chunk, client.embed(chunk, batch)); self.save()
            if log:
                log(f"{self.path.name}: {start + len(chunk)}/{len(unique)} new texts cached ({client.prompt_tokens} tokens, {client.elapsed:.0f}s)")
        return len(unique)


def goemotions_rows(cache, split):
    """Official GoEmotions split rows (id, text, labels) from the repository's pinned downloads."""
    from ..dataset_baseline import read_rows
    path = Path(cache) / f"goemotions-{split}-{split}.tsv"
    if not path.exists():
        candidates = sorted(Path(cache).glob(f"goemotions*{split}*.tsv"))
        if not candidates:
            raise FileNotFoundError(f"no GoEmotions {split} TSV under {cache}; run scripts/fetch_datasets.py")
        path = candidates[0]
    return path, list(read_rows(path))


def dailydialog_dialogues(cache, split):
    path = Path(cache) / f"{split}.json"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return path, rows


def select_by_hash(ids, fraction, salt):
    """Deterministic subset: keep ids whose salted SHA-256 falls below the fraction."""
    keep = []
    for i in ids:
        h = int(hashlib.sha256(f"{salt}:{i}".encode()).hexdigest()[:12], 16) / 16 ** 12
        keep.append(h < fraction)
    return np.array(keep)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["goemotions", "dailydialog"], required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--fraction", type=float, default=1.0, help="deterministic hash-selected fraction of rows/dialogues")
    p.add_argument("--cache", required=True, help="dataset cache directory")
    p.add_argument("--out", required=True, help="embedding cache directory")
    p.add_argument("--base-url", required=True); p.add_argument("--model", required=True)
    p.add_argument("--batch", type=int, default=32)
    a = p.parse_args()
    client = EmbeddingClient(a.base_url, a.model)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.dataset == "goemotions":
        path, rows = goemotions_rows(a.cache, a.split)
        ids = [r[0] for r in rows]
        keep = select_by_hash(ids, a.fraction, f"goemotions-{a.split}")
        texts = [r[1] for r, k in zip(rows, keep) if k]
        selected = [r[0] for r, k in zip(rows, keep) if k]
    else:
        path, dialogues = dailydialog_dialogues(a.cache, a.split)
        ids = [f"{a.split}-{i}" for i in range(len(dialogues))]
        keep = select_by_hash(ids, a.fraction, f"dailydialog-{a.split}")
        texts = [t["text"] for d, k in zip(dialogues, keep) if k for t in d["dialogue"]]
        selected = [i for i, k in zip(ids, keep) if k]
    cache = EmbeddingCache(out / f"{a.dataset}-{a.model}.npz")
    started = datetime.now(timezone.utc).isoformat(); t0 = time.perf_counter()
    new = cache.ensure(texts, client, a.batch, log=lambda m: print(m, flush=True))
    manifest_path = out / f"{a.dataset}-{a.split}-manifest.json"
    manifest = {"schema_version": 1, "dataset": a.dataset, "split": a.split, "fraction": a.fraction, "selection": "salted SHA-256 of id below fraction",
                "source_file": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "selected_units": len(selected), "texts": len(texts),
                "unique_texts_newly_embedded": new, "model": a.model, "endpoint": client.base_url + "/embeddings", "pooling": "mean (server flag)",
                "dimension": int(cache.vectors.shape[1]) if cache.vectors is not None else None, "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - t0,
                "prompt_tokens": client.prompt_tokens, "calls": client.calls, "cache_file": str(cache.path), "selected_ids_sha256": hashlib.sha256(json.dumps(selected).encode()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("dataset", "split", "texts", "unique_texts_newly_embedded", "elapsed_seconds")}))


if __name__ == "__main__":
    main()

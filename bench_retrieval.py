#!/usr/bin/env python3
"""Semantic retrieval benchmark for memory_store.py (Critical Issue 4).

Measures whether the configured embedding backend actually retrieves the
RIGHT memories, with labeled queries instead of assumptions:

  - exact-fact queries        ("what graphics card do I have")
  - paraphrase queries        ("how much VRAM is on my GPU")
  - preference/name queries   ("how do I take my coffee")
  - date queries              ("when is my birthday")
  - semantically-near-but-wrong distractors (ramen shop vs. city)
  - unrelated queries         (capital of France - measures junk scores)

Metrics: Recall@1 / Recall@3 / Recall@5 and MRR over the labeled queries,
plus the top-1 score of unrelated queries (calibration: lower is better).

Modes:
  python bench_retrieval.py --fake      # offline: verifies the metric
                                        # machinery with a deterministic
                                        # keyword embedder (safe anywhere)
  python bench_retrieval.py             # REAL benchmark through llama-server
                                        # /v1/embeddings (needs --embeddings)
  python bench_retrieval.py --compare   # llama-server vs sbert (needs
                                        # sentence-transformers), same dataset

The benchmark runs against a TEMPORARY database of synthetic facts - your
real memory.db is never touched. Report real numbers only for the mode you
actually ran; --fake numbers describe the harness, not embedding quality.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import iris_config as cfg  # noqa: E402

FACTS = [
    ("gpu", "My laptop has an NVIDIA RTX 4050 with 6 GB VRAM.", 0.9),
    ("editor", "My favorite python editor is a plain terminal with vim.", 0.6),
    ("coffee", "I take my coffee black with one sugar.", 0.7),
    ("wifi", "My wifi password is helloworld42.", 0.8),
    ("alarm", "My alarm tone is the ultron alarm sound.", 0.6),
    ("birthday", "My birthday is March 3rd.", 0.8),
    ("city", "I live in Kyoto near the central station.", 0.7),
    ("trip", "I am planning a trip to Hokkaido in winter.", 0.5),
    ("ramen", "The ramen shop near Kyoto station is great for late dinners.", 0.5),
    ("server", "I run llama.cpp llama-server with Qwen3.5 on port 8080.", 0.7),
]

QUERIES = [
    ("gpu", "what graphics card does my laptop have"),
    ("gpu", "how much VRAM is on my GPU"),            # paraphrase
    ("coffee", "how do I take my coffee"),
    ("wifi", "what is my wifi password"),
    ("birthday", "when is my birthday"),
    ("city", "which city do I live in"),
    ("alarm", "what sound wakes me up in the morning"),  # paraphrase-ish
    ("editor", "which editor do I prefer for python"),
    ("server", "what model server do I run locally"),
    ("trip", "where am I planning to travel"),         # paraphrase
]

UNRELATED = [
    "what is the capital of france",
    "explain the plot of moby dick",
    "how do photovoltaic cells work",
]


def fake_embedder(store):
    """Deterministic keyword embedder: 12-dim bag-of-concept vectors."""
    import numpy as np

    concepts = ["gpu", "vram", "graphics", "coffee", "sugar", "wifi", "password",
                "birthday", "march", "kyoto", "city", "station", "ramen",
                "editor", "vim", "terminal", "llama", "qwen", "server", "alarm",
                "ultron", "trip", "hokkaido", "travel", "winter", "sound",
                "morning", "wake", "laptop", "port"]

    def embed(texts):
        mat = []
        for t in texts:
            low = t.lower()
            v = [1.0 if c in low else 0.0 for c in concepts]
            if not any(v):
                v = [0.05] * len(concepts)
            mat.append(v)
        mat = np.asarray(mat, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    store._embed_batch = embed


def evaluate(store, k_max: int = 5, verbose=True) -> dict:
    rows = []
    for label, query in QUERIES:
        results = store.search(query, k=k_max)
        rank = None
        for i, h in enumerate(results, start=1):
            for lid, text, _imp in FACTS:
                if lid == label and h.content == text:
                    rank = i
                    break
            if rank:
                break
        rows.append((query, label, rank, results[0].score if results else 0.0))
        if verbose:
            mark = f"rank {rank}" if rank else "MISS"
            print(f"  [{mark:>8}] {query}  (top1 score {results[0].score:.3f})" if results
                  else f"  [   MISS ] {query}  (no results)")
    rr_sum = sum(1.0 / r for _, _, r, _ in rows if r)
    recall = {k: sum(1 for _, _, r, _ in rows if r and r <= k) / len(rows) for k in (1, 3, 5)}
    mrr = rr_sum / len(rows)

    unrelated_scores = []
    for q in UNRELATED:
        results = store.search(q, k=1)
        if results:
            unrelated_scores.append(results[0].score)
    cal = round(sum(unrelated_scores) / len(unrelated_scores), 3) if unrelated_scores else None
    return {
        "n_queries": len(rows),
        "recall@1": round(recall[1], 3),
        "recall@3": round(recall[3], 3),
        "recall@5": round(recall[5], 3),
        "mrr": round(mrr, 3),
        "unrelated_top1_score": cal,
    }


def build_store(backend: str | None):
    import memory_store

    tmp = Path(tempfile.mkdtemp(prefix="iris_bench_"))
    store = memory_store.MemoryStore(db_path=str(tmp / "bench.db"), embed_backend=backend)
    store.add_many(
        [{"content": text, "speaker": "user", "source": "bench", "importance": imp}
         for _lid, text, imp in FACTS]
    )
    return store


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fake", action="store_true", help="offline metric-machinery check")
    ap.add_argument("--compare", action="store_true",
                    help="benchmark llama-server vs sbert backends")
    args = ap.parse_args()

    print("IRIS retrieval benchmark")
    print("=" * 60)

    if args.fake:
        store = build_store(None)
        fake_embedder(store)
        print("mode: FAKE embedder (metric machinery verification only -")
        print("      these numbers say NOTHING about real embedding quality)\n")
        metrics = evaluate(store)
        store.close()
        print("\n" + json.dumps(metrics, indent=2))
        # the fake embedder is keyword-only: assert the HARNESS finds every
        # labeled fact (recall@5) and computes sane metrics, nothing more
        ok = metrics["recall@5"] == 1.0 and 0.0 < metrics["mrr"] <= 1.0
        print("\nmetric machinery:", "OK" if ok else "BROKEN")
        return 0 if ok else 1

    if not cfg.LLAMA_SERVER_URL:
        print("no LLAMA_SERVER_URL configured")
        return 1

    import requests

    try:
        requests.get(cfg.LLAMA_SERVER_URL + "/health", timeout=4)
    except Exception as e:
        print(f"llama-server not reachable at {cfg.LLAMA_SERVER_URL} ({type(e).__name__}).")
        print("start it with ./start_llama.sh (needs --embeddings) and re-run.")
        return 2

    print(f"mode: REAL embeddings via {cfg.EMBED_ENDPOINT}")
    print(f"      model field: {cfg.EMBED_MODEL}\n")

    backends = ["llamaserver"]
    if args.compare:
        try:
            import sentence_transformers  # noqa: F401

            backends.append("sbert")
        except ImportError:
            print("--compare requested but sentence-transformers is not installed;")
            print("pip install sentence-transformers to enable it.\n")

    all_metrics = {}
    for backend in backends:
        print(f"--- backend: {backend} ---")
        store = build_store(backend)
        metrics = evaluate(store)
        store.close()
        metrics["backend"] = backend
        all_metrics[backend] = metrics
        print()

    print(json.dumps(all_metrics, indent=2))
    print("\ninterpretation guide:")
    print("  recall@3 >= 0.8 and mrr >= 0.7  -> embeddings are fine for memory")
    print("  recall@3 < 0.5                  -> try EMBED_BACKEND=sbert, or add")
    print("                                     --pooling mean to llama-server,")
    print("                                     then re-run this benchmark")
    return 0


if __name__ == "__main__":
    sys.exit(main())

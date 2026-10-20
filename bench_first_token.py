#!/usr/bin/env python3
"""First-token latency and throughput benchmark for llama-server (CI-10).

Measures what the user actually perceives:
  TTFT      time from request start to the first visible text delta
  total     full generation wall time
  chars/sec visible output rate (chars, not tokens - no estimates here)

Run while llama-server is up:
  python bench_first_token.py              # 5 stock queries
  python bench_first_token.py --n 12       # more samples
  python bench_first_token.py --max-tokens 200

Reports mean / p50 / p95 / min / max. Needs ./start_llama.sh running.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tool_runtime as tr
import ultron_config as cfg

PROMPTS = [
    "Explain how rain forms, in two sentences.",
    "Give me three tips for better sleep.",
    "What are the main differences between TCP and UDP?",
    "Describe the color indigo to someone who cannot see.",
    "Summarize why the sky is blue.",
    "List three uses of hydrogen.",
    "Explain inertia like I am twelve.",
    "What makes a good espresso?",
]


def pct(values, p):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100.0 * (len(ordered) - 1))))
    return round(ordered[idx], 1)


def stats(values):
    if not values:
        return None
    return {
        "mean": round(sum(values) / len(values), 1),
        "p50": pct(values, 50),
        "p95": pct(values, 95),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
    }


def run(n: int, max_tokens: int, warmup: int) -> dict:
    rounds = []
    prompts = (PROMPTS * ((n // len(PROMPTS)) + 1))[: n + warmup]
    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ]
        t0 = time.perf_counter()
        ttft = None
        chars = 0
        events = 0
        for ev in tr.stream_completion(messages, max_tokens=max_tokens, temperature=0.6):
            if ev["type"] == "text":
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
                chars += len(ev["delta"])
                events += 1
        total = (time.perf_counter() - t0) * 1000.0
        if ttft is None:
            raise RuntimeError(
                "no text deltas received - check that the model answers plain prompts"
            )
        rec = {
            "ttft_ms": round(ttft, 1),
            "total_ms": round(total, 1),
            "chars": chars,
            "text_events": events,
            "chars_per_sec": round(chars / (total / 1000.0), 1) if total else 0,
        }
        if i >= warmup:
            rounds.append(rec)
        print(f"  [{len(rounds):>2}] ttft {rec['ttft_ms']:7.1f} ms   "
              f"total {rec['total_ms']:8.1f} ms   {rec['chars']:4d} chars   "
              f"{rec['chars_per_sec']:6.1f} c/s")
    return {
        "server": cfg.LLAMA_SERVER_URL,
        "model": cfg.LLM_MODEL_NAME,
        "max_tokens": max_tokens,
        "samples": len(rounds),
        "ttft_ms": stats([r["ttft_ms"] for r in rounds]),
        "total_ms": stats([r["total_ms"] for r in rounds]),
        "chars_per_sec": stats([r["chars_per_sec"] for r in rounds]),
        "note": "chars/sec is visible characters, not model tokens",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="llama-server TTFT benchmark")
    ap.add_argument("--n", type=int, default=5, help="measured prompts (default 5)")
    ap.add_argument("--warmup", type=int, default=1, help="unmeasured warmup prompts")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    if not tr.server_ok():
        print(f"llama-server is not reachable at {cfg.LLAMA_SERVER_URL}")
        print("start it with ./start_llama.sh and re-run.")
        return 2

    print(f"benchmarking {cfg.LLM_MODEL_NAME} at {cfg.LLAMA_SERVER_URL} "
          f"({args.n} samples + {args.warmup} warmup, max_tokens {args.max_tokens})")
    report = run(args.n, args.max_tokens, args.warmup)
    print()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"TTFT       : {report['ttft_ms']}")
        print(f"total      : {report['total_ms']}")
        print(f"chars/sec  : {report['chars_per_sec']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

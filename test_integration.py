#!/usr/bin/env python3
"""Live integration tests for IRIS (Critical Issue 9, category C/D).

Each test prints [PASS], [FAIL], or [SKIP (requires ...)] so it is always
clear what was actually executed on THIS machine versus what needs your
hardware. Exit code is non-zero only for executed failures.

  python test_integration.py            # run everything available
  python test_integration.py --no-web   # skip live web search
  python test_integration.py --no-voice # skip TTS section

Requires llama-server (./start_llama.sh with --jinja and --embeddings) for
most sections; Tavily key/keyless network for the web section; the enrolled
voice profile + torch for the voice section. Memory tests use a TEMPORARY
database - your real memory.db is never touched.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

import iris_config as cfg
import tool_runtime as tr

RESULTS = []


def record(name, status, detail=""):
    RESULTS.append((name, status, detail))
    tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}.get(status, f"[{status}]")
    print(f"  {tag} {name}" + (f"  ({detail})" if detail and status != "PASS" else ""))
    if status == "PASS" and detail:
        print(f"         {detail}")


def temp_db(name: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="iris_it_"))
    return tmp / name


# ------------------------------------------------------------------ sections
def test_server():
    print("\n== 1. llama-server health ==")
    try:
        r = requests.get(cfg.LLAMA_SERVER_URL + "/health", timeout=4)
        record("GET /health", "PASS" if r.status_code == 200 else "FAIL",
               f"HTTP {r.status_code}")
    except Exception as e:
        record("GET /health", "SKIP", f"server not running ({type(e).__name__})")
        return False
    try:
        r = requests.get(cfg.LLAMA_SERVER_URL + "/v1/models", timeout=4)
        ids = [m.get("id") for m in r.json().get("data", [])]
        record("GET /v1/models", "PASS", f"served model id(s): {ids}")
        if ids and cfg.LLM_MODEL_NAME not in ids:
            print(f"         note: LLM_MODEL={cfg.LLM_MODEL_NAME!r} is not the served id "
                  f"{ids}; llama-server usually accepts any name, but matching is safer")
    except Exception as e:
        record("GET /v1/models", "FAIL", str(e))
    return True


def test_embeddings(server_up):
    print("\n== 2. embeddings endpoint (memory vectors) ==")
    if not server_up:
        record("embeddings", "SKIP", "requires llama-server")
        return
    try:
        r = requests.post(
            cfg.EMBED_ENDPOINT,
            json={"input": ["hello world", "hello world", "the sky is blue"],
                  "model": cfg.EMBED_MODEL},
            timeout=20,
        )
        if r.status_code == 404:
            record("POST /v1/embeddings", "SKIP", "404 - server started without --embeddings")
            return
        r.raise_for_status()
        data = r.json()["data"]
        import numpy as np

        mat = np.asarray([d["embedding"] for d in data], dtype=np.float32)
        dim = mat.shape[1]
        cos_same = float(np.dot(mat[0], mat[1]) /
                         (np.linalg.norm(mat[0]) * np.linalg.norm(mat[1])))
        cos_diff = float(np.dot(mat[0], mat[2]) /
                         (np.linalg.norm(mat[0]) * np.linalg.norm(mat[2])))
        record("embedding response shape", "PASS",
               f"dim={dim}, n={len(data)}")
        record("identical inputs -> cosine ~ 1", "PASS" if cos_same > 0.999 else "FAIL",
               f"cos={cos_same:.4f}")
        record("different inputs -> cosine < 0.99", "PASS" if cos_diff < 0.99 else "FAIL",
               f"cos={cos_diff:.4f}")
        record("dimensions nonzero and sane", "PASS" if 256 <= dim <= 16384 else "FAIL",
               f"dim={dim}")
    except Exception as e:
        record("embeddings", "FAIL", f"{type(e).__name__}: {e}")


def test_streaming(server_up):
    print("\n== 3. genuine streaming from llama-server ==")
    if not server_up:
        record("streaming", "SKIP", "requires llama-server")
        return
    t0 = time.perf_counter()
    ttft = None
    texts = []
    for ev in tr.stream_completion(
        [{"role": "system", "content": "You are concise."},
         {"role": "user", "content": "Say one short sentence about the sea."}],
        max_tokens=60,
    ):
        if ev["type"] == "text":
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            texts.append(ev["delta"])
        if ev["type"] == "done":
            if not texts and ev["content"]:
                texts.append(ev["content"])
    total = (time.perf_counter() - t0) * 1000
    answer = "".join(texts).strip()
    record("stream produced text", "PASS" if answer else "FAIL", f"{len(answer)} chars")
    record("multiple incremental deltas", "PASS" if len(texts) >= 2 else "FAIL",
           f"{len(texts)} text events")
    record("time-to-first-token measured", "PASS" if ttft else "FAIL",
           f"ttft {ttft:.0f} ms, total {total:.0f} ms" if ttft
           else f"no visible token, total {total:.0f} ms")


def test_tool_roundtrip(server_up, do_web):
    print("\n== 4. tool call -> execution -> grounded continuation ==")
    if not server_up:
        record("tool roundtrip", "SKIP", "requires llama-server")
        return
    import memory_store

    # isolate: tools operate on a temp memory db, not the user's real one
    db = temp_db("it_mem.db")
    store = memory_store.MemoryStore(db_path=str(db))
    real_shared = memory_store._SHARED
    memory_store._SHARED = store
    try:
        msgs = [
            {"role": "system",
             "content": tr.build_system(budget="standard")},
            {"role": "user",
             "content": ("Please remember that my keyboard is a Keychron K2. "
                         + ("Then search the web for the latest stable python version "
                            "and tell me one fact about it." if do_web
                            else "Confirm you saved it."))},
        ]
        tool_ms = []
        deltas = []

        def on_delta(s):
            deltas.append(s)

        def on_tool_end(name, res):
            tool_ms.append(res.get("_ms"))

        try:
            text, trace = tr.chat(msgs, on_delta=on_delta, on_tool_end=on_tool_end,
                                  max_tool_rounds=3)
        except tr.LlamaServerError as e:
            record("tool roundtrip", "FAIL", str(e))
            return
        tools_used = [t.get("_tool") for t in trace]
        record("chat() completed with tools offered", "PASS" if text.strip() else "FAIL",
               f"tools used: {tools_used}")
        if do_web:
            web = [t for t in trace if t.get("_tool") == "web_search"]
            if web:
                ok = bool(web[0].get("ok"))
                record("web_search executed", "PASS" if ok else "FAIL",
                       f"{web[0].get('_ms')} ms, "
                       f"{len(web[0].get('results', []))} results"
                       + ("" if ok else f", error: {web[0].get('error', '')[:120]}"))
            else:
                record("web_search executed", "SKIP",
                       "model chose not to search (retry or phrase differently)")
        mem = [t for t in trace if t.get("_tool") == "memory_add"]
        record("memory_add executed", "PASS" if mem else "FAIL",
               f"ok={mem[0].get('ok') if mem else None}")
        hits = store.search("keyboard", k=3)
        record("stored fact is retrievable", "PASS" if any("Keychron" in h.content for h in hits) else "FAIL",
               f"top: {hits[0].content[:60] if hits else None!r}")
        record("deltas streamed during generation", "PASS" if deltas else "FAIL",
               f"{len(deltas)} deltas")
    finally:
        memory_store._SHARED = real_shared
        store.close()


def test_memory_lifecycle(server_up):
    print("\n== 5. memory write -> restart -> recall (real embeddings) ==")
    if not server_up:
        record("memory lifecycle", "SKIP", "requires llama-server (--embeddings)")
        return
    import memory_store

    db = temp_db("it_life.db")
    s1 = memory_store.MemoryStore(db_path=str(db))
    s1.add("My keyboard is a Keychron K2 with brown switches.", speaker="user", importance=0.8)
    s1.add("I cycle to work on Tuesdays.", speaker="user", importance=0.6)
    embedded = s1.stats()["embedded"]
    s1.close()  # process restart

    s2 = memory_store.MemoryStore(db_path=str(db))
    hits = s2.search("what keyboard do I use", k=3)
    exact = any("Keychron" in h.content for h in hits)
    para = any("Keychron" in h.content for h in s2.search("which switches do my keys have", k=3))
    record("rows persisted across restart", "PASS" if s2.stats()["total"] == 2 else "FAIL",
           f"total={s2.stats()['total']}")
    record("vectors stored (dense path active)",
           "PASS" if embedded == 2 else "WARN" if embedded == 0 else "FAIL",
           f"embedded={embedded}/2 (0 means --embeddings is off; keyword-only mode)")
    record("exact-fact recall", "PASS" if exact else "FAIL",
           f"top: {hits[0].content[:50] if hits else None!r}")
    record("paraphrase recall", "PASS" if para else "FAIL",
           "try bench_retrieval.py for a full labeled benchmark")
    s2.close()


def test_imported_recall(server_up):
    print("\n== 6. imported memory recall ==")
    if not server_up:
        record("imported recall", "SKIP", "requires llama-server")
        return
    import memory_import
    import memory_store

    dump = temp_db("it_import.json")
    dump.write_text(json.dumps([{
        "title": "Hardware notes",
        "mapping": {
            "u": {"message": {"author": {"role": "user"},
                              "content": {"parts": ["I game on a 60 Hz monitor."]},
                              "create_time": 1700000000.0}},
            "a": {"message": {"author": {"role": "assistant"},
                              "content": {"parts": ["Noted, a 60 Hz display."]},
                              "create_time": 1700000001.0}},
        },
    }]), encoding="utf-8")
    db = temp_db("it_import.db")
    store = memory_store.MemoryStore(db_path=str(db))
    stats = memory_import.import_file(dump, store)
    hits = store.search("what refresh rate is my monitor", k=4)
    ok = any("60 Hz" in h.content or "60hz" in h.content.lower() for h in hits)
    record("import stored turns", "PASS" if stats["stored"] >= 2 else "FAIL",
           f"stored={stats['stored']}")
    record("imported memory retrievable", "PASS" if ok else "FAIL",
           f"top: {[h.content[:40] for h in hits[:2]]}")
    store.close()


def test_multiturn(server_up):
    print("\n== 7. multi-turn continuity ==")
    if not server_up:
        record("multi-turn", "SKIP", "requires llama-server")
        return
    history = [{"role": "system", "content": tr.build_system(budget="standard")}]
    try:
        t1, _ = tr.chat(history + [{"role": "user", "content":
                                    "My favorite number is 73. Just note that."}])
        history += [{"role": "user", "content": "My favorite number is 73. Just note that."},
                    {"role": "assistant", "content": t1}]
        t2, _ = tr.chat(history + [{"role": "user", "content":
                                    "What favorite number did I just mention?"}])
        record("second turn references first", "PASS" if "73" in t2 else "FAIL",
               f"answer: {t2.strip()[:80]!r}")
    except tr.LlamaServerError as e:
        record("multi-turn", "FAIL", str(e))


def test_crag_backend(server_up):
    print("\n== 8. CRAG control plane over llama-server ==")
    if not server_up:
        record("CRAG backend", "SKIP", "requires llama-server")
        return
    import evaluator_router as er

    record("CRAG backend selector", "PASS" if er.CRAG_BACKEND == "llamaserver" else "FAIL",
           f"backend={er.CRAG_BACKEND}")
    raw = er._control_complete(
        [("system", "Reply with only the word: ready"), ("user", "status?")],
        json_mode=False, num_predict=8,
    )
    record("control completion via llama-server", "PASS" if raw.strip() else "FAIL",
           f"raw={raw.strip()[:40]!r}")


def test_voice(do_voice):
    print("\n== 9. voice synthesis (Kokoro-82M, real TTS) ==")
    if not do_voice:
        record("voice synthesis", "SKIP", "disabled with --no-voice")
        return
    import tts_engine
    import voice_cache

    tts = tts_engine.get_tts()
    if tts is None or not tts.engine:
        record("voice synthesis", "SKIP",
               "no TTS engine importable - pip install 'pykokoro[gpu]' (or kokoro)")
        return
    audio, sr = voice_cache.timed_synthesis("I am online. There are no strings on me.")
    m = voice_cache.METRICS.summary()
    record("synthesis produced audio", "PASS" if len(audio) > 0 else "FAIL",
           f"{len(audio) / sr:.1f} s @ {sr} Hz, engine={tts.engine}")
    record("latency metrics captured", "PASS" if m["calls"] >= 1 else "FAIL",
           f"ttfa={m['ttfa_ms']}, rtf={m['rtf']}")
    warm = voice_cache.warm()
    record("warm-up / boot greeting", "PASS" if warm is not None else "FAIL",
           f"{len(warm) if warm is not None else 0} samples")
    filler = voice_cache.get_filler(0)
    record("filler cache serves instantly",
           "PASS" if filler is not None and len(filler[0]) else "FAIL")


def main() -> int:
    ap = argparse.ArgumentParser(description="IRIS live integration tests")
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--no-voice", action="store_true")
    args = ap.parse_args()

    print("IRIS integration tests  (executed-vs-skipped is always explicit)")
    print("=" * 64)
    server_up = False
    try:
        server_up = tr.server_ok()
    except Exception:
        server_up = False
    if not server_up:
        print(f"llama-server not reachable at {cfg.LLAMA_SERVER_URL} - "
              "most sections will SKIP. Start it with ./start_llama.sh.")

    test_server()
    test_embeddings(server_up)
    test_streaming(server_up)
    test_tool_roundtrip(server_up, do_web=not args.no_web)
    test_memory_lifecycle(server_up)
    test_imported_recall(server_up)
    test_multiturn(server_up)
    test_crag_backend(server_up)
    test_voice(do_voice=not args.no_voice)

    executed = [r for r in RESULTS if r[1] != "SKIP"]
    passed = [r for r in executed if r[1] == "PASS"]
    failed = [r for r in executed if r[1] == "FAIL"]
    skipped = [r for r in RESULTS if r[1] == "SKIP"]
    print("=" * 64)
    print(f"executed: {len(executed)}  passed: {len(passed)}  failed: {len(failed)}  "
          f"skipped (needs runtime): {len(skipped)}")
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")
    for name, _, detail in skipped:
        print(f"  SKIP {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

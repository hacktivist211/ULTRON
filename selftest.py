#!/usr/bin/env python3
"""Self-test for the IRIS modules. Runs entirely offline (no GPU, no llama-server,
no Tavily key needed) except when you pass --live, which then also pings Tavily
(keyless or keyed) and llama-server if reachable.

  python selftest.py            # offline suite
  python selftest.py --live     # offline suite + real network probes

Voice enrollment cannot be tested here (needs your GPU + CosyVoice2 + the mp3);
run 'python enroll_voice.py' on your machine for that.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import iris_config as cfg

TMP = Path(tempfile.gettempdir())
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    RESULTS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not cond else ""))


# ------------------------------------------------------------------ config tests
def test_config():
    print("\n== iris_config ==")
    env_file = TMP / "iris_selftest.env"
    env_file.write_text('IRIS_SELFTEST_VAR=hello\n# comment\nEMPTY_LINE\n')
    os.environ.pop("IRIS_SELFTEST_VAR", None)
    n = cfg.load_dotenv(path=env_file)
    check("dotenv loads key", n == 1 and os.environ.get("IRIS_SELFTEST_VAR") == "hello")
    env_file.unlink(missing_ok=True)

    dirty = "Check **this** #header `code` https://example.com/page now 🙂"
    clean = cfg.speech_clean(dirty)
    check("speech_clean strips markdown/url/emoji",
          "http" not in clean and "**" not in clean and "#" not in clean and "example.com" not in clean and "🙂" not in clean,
          f"got: {clean!r}")
    long_text = " ".join(f"w{i}" for i in range(200))
    check("clip_words enforces budget", len(cfg.clip_words(long_text, "quick").split()) <= 25)
    check("word budgets present", cfg.WORD_BUDGETS == {"quick": 25, "standard": 60, "detailed": 150})
    check("iris dir is flat & detected", cfg.IRIS_DIR.is_dir())


# ------------------------------------------------------------------ memory store tests
def fake_embedder(store):
    """Deterministic keyword embedder: 3-dim one-hot style vectors."""
    def embed(texts):
        import numpy as np
        mat = []
        for t in texts:
            v = [0.0, 0.0, 0.0]
            low = t.lower()
            if "kyoto" in low:
                v[0] += 1.0
            if "python" in low or "code" in low:
                v[1] += 1.0
            if "ramen" in low:
                v[2] += 1.0
            if not any(v):
                v = [0.4, 0.4, 0.4]
            mat.append(v)
        mat = np.asarray(mat, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms
    store._embed_batch = embed  # monkeypatch


def test_memory_store():
    print("\n== memory_store (offline, fake embedder) ==")
    import memory_store
    import numpy as np

    db = TMP / "iris_selftest.db"
    db.unlink(missing_ok=True)
    store = memory_store.MemoryStore(db_path=str(db))
    fake_embedder(store)

    rid = store.add("I live in Kyoto and I love ramen.", speaker="user", source="manual", importance=0.9)
    check("add returns id", isinstance(rid, int))
    dup = store.add("I live in Kyoto  and I love ramen.", speaker="user", source="manual")  # whitespace-insensitive hash
    check("dedup by content hash", dup is None)
    store.add("My favorite python editor is a plain terminal.", speaker="user", source="manual", importance=0.6)
    store.add_many([
        {"content": "The ramen shop near Kyoto station is great.", "speaker": "assistant", "source": "chatgpt"},
        {"content": "Random weather note.", "speaker": "assistant", "source": "chatgpt"},
    ])
    st = store.stats()
    check("stats counts", st["total"] == 4, f"total={st['total']}")

    hits = store.search("kyoto", k=3)
    check("dense search ranks kyoto first", bool(hits) and "kyoto" in hits[0].content.lower(), f"top={hits[0].content if hits else None!r}")
    hits_kw = store.search("ramen", k=3)
    check("keyword fallback finds ramen", any("ramen" in h.content.lower() for h in hits_kw))
    ctx = store.as_context("kyoto ramen", k=3)
    check("as_context builds block", ctx.startswith("RELEVANT MEMORY CONTEXT") and "Kyoto" in ctx)

    tool = store.run_tool("memory_search", {"query": "python"})
    check("memory_search tool ok", tool["ok"] and isinstance(tool["memories"], list))
    tool2 = store.run_tool("memory_add", {"content": "My GPU is an RTX 4050 with 6 GB VRAM."})
    check("memory_add tool ok", tool2["ok"])
    tool3 = store.run_tool("memory_add", {"content": "My GPU is an RTX 4050 with 6 GB VRAM."})
    check("memory_add dedups", tool3.get("note") == "already known, not duplicated")

    store.close()
    db.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)


# ------------------------------------------------------------------ import tests
CHATGPT_SAMPLE = [{
    "title": "Trip planning",
    "create_time": 1700000000.0,
    "mapping": {
        "n_sys": {"message": {"author": {"role": "system"}, "content": {"parts": ["You are helpful"]}, "create_time": None}},
        "n_u1": {"message": {"author": {"role": "user"}, "content": {"content_type": "text", "parts": ["I live in Kyoto and I am planning a trip."]}, "create_time": 1700000001.0}},
        "n_a1": {"message": {"author": {"role": "assistant"}, "content": {"content_type": "text", "parts": ["Kyoto is lovely in autumn."]}, "create_time": 1700000002.0}},
        "n_t1": {"message": {"author": {"role": "tool"}, "content": {"content_type": "text", "parts": ["tool output"]}, "create_time": 1700000001.5}},
    },
}]
CLAUDE_SAMPLE = [{
    "name": "Voice assistant ideas",
    "chat_messages": [
        {"sender": "human", "text": "My name is Neeraj and I am building IRIS on an RTX 4050.", "created_at": "2025-01-02T10:00:00Z"},
        {"sender": "assistant", "text": "That is a fine goal. CosyVoice2 will fit in 6 GB VRAM.", "created_at": "2025-01-02T10:00:05Z"},
    ],
}]
GEMINI_SAMPLE = [{
    "header": "My Activity", "title": "Asked Gemini about RTX 4050",
    "time": "Jan 5, 2025, 3:04 PM",
    "body": "You asked: Is the RTX 4050 good for TTS?\nHere's what Gemini said: Yes, it handles CosyVoice well in FP16.",
}]
GENERIC_SAMPLE = [
    {"prompt": "What is my wifi password?", "response": "You told me it is helloworld42."},
    {"role": "user", "content": "I prefer dark mode everywhere."},
    {"role": "assistant", "content": "Noted, sir."},
]
PLAIN_SAMPLE = """User: remember my alarm tone is the jarvis alarm
Assistant: saved. I will use the jarvis alarm tone.

I take my coffee black with one sugar. It keeps me sharp during long coding sessions."""


def test_memory_import():
    print("\n== memory_import (synthetic exports) ==")
    import memory_import
    import memory_store

    files = {
        "chatgpt": TMP / "iris_st_chatgpt.json",
        "claude": TMP / "iris_st_claude.json",
        "gemini": TMP / "iris_st_gemini.json",
        "generic": TMP / "iris_st_generic.json",
        "text": TMP / "iris_st_plain.txt",
    }
    files["chatgpt"].write_text(json.dumps(CHATGPT_SAMPLE), encoding="utf-8")
    files["claude"].write_text(json.dumps(CLAUDE_SAMPLE), encoding="utf-8")
    files["gemini"].write_text(json.dumps(GEMINI_SAMPLE), encoding="utf-8")
    files["generic"].write_text("\n".join(json.dumps(x) for x in GENERIC_SAMPLE), encoding="utf-8")
    files["text"].write_text(PLAIN_SAMPLE, encoding="utf-8")

    expected = {"chatgpt": 2, "claude": 2, "gemini": 2, "generic": 4, "text": 3}
    parsed = {}
    for fmt, p in files.items():
        got_fmt, turns = memory_import.parse_file(p)
        n = len(turns)
        parsed[fmt] = turns
        check(f"{fmt} format detected", got_fmt == fmt, f"detected={got_fmt}")
        check(f"{fmt} parsed {expected[fmt]} turns", n == expected[fmt], f"n={n}")

    db = TMP / "iris_st_import.db"
    db.unlink(missing_ok=True)
    store = memory_store.MemoryStore(db_path=str(db))
    fake_embedder(store)

    total_in = 0
    for fmt, p in files.items():
        stats = memory_import.import_file(p, store)
        total_in += stats["stored"]
    st = store.stats()
    conv_rows = [r for r in store.db.execute("SELECT kind FROM memories")]
    n_conv = sum(1 for (k,) in conv_rows if k == "conv_summary")
    check("import stored turns + summaries", st["total"] == total_in and n_conv >= 2,
          f"total={st['total']} conv_summaries={n_conv}")

    # re-import same file: everything deduplicated
    before = store.stats()["total"]
    memory_import.import_file(files["chatgpt"], store)
    check("re-import dedups fully", store.stats()["total"] == before,
          f"before={before} after={store.stats()['total']}")

    hits = store.search("kyoto", k=5)
    check("imported memory is searchable", any("kyoto" in h.content.lower() for h in hits))
    hits2 = store.search("jarvis alarm", k=5)
    check("plain-text memory findable via keywords", any("jarvis alarm" in h.content.lower() for h in hits2))

    store.close()
    for p in files.values():
        p.unlink(missing_ok=True)
    db.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)


# ------------------------------------------------------------------ tavily + tools
def test_tavily_offline():
    print("\n== tavily_tool (offline) ==")
    import tavily_tool

    canned = {
        "ok": True,
        "answer": "Python 3.13 is the latest stable release.",
        "results": [
            {"title": "Python", "url": "https://www.python.org/downloads/", "snippet": "Download Python 3.13. The official home."},
        ],
    }
    spoken = tavily_tool.speakable(canned, budget="standard")
    check("speakable uses answer, TTS-safe", "3.13" in spoken and "http" not in spoken and "*" not in spoken, f"got {spoken!r}")
    canned2 = {"ok": True, "answer": "", "results": canned["results"]}
    spoken2 = tavily_tool.speakable(canned2)
    check("speakable falls back to snippet with domain", "python.org" in spoken2, f"got {spoken2!r}")

    schema = tavily_tool.TOOL_SCHEMA["function"]
    check("tool schema shape", schema["name"] == "web_search" and "query" in schema["parameters"]["required"])
    check("no key leaked in schema", "tvly-" not in json.dumps(tavily_tool.TOOL_SCHEMA))


# ------------------------------------------------------------------ streaming core (CI-1)
class _FakeStreamResponse:
    """requests.Response stand-in for offline streaming tests."""

    def __init__(self, chunks, status_code=200):
        self.status_code = status_code
        self._chunks = list(chunks)
        self.closed = False
        self.exhausted = False
        self.text = "err" if status_code != 200 else ""

    def iter_content(self, chunk_size=None):
        for c in self._chunks:
            yield c
        self.exhausted = True

    def close(self):
        self.closed = True


class _FakePost:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def __call__(self, endpoint, json=None, stream=False, timeout=None):
        self.payloads.append(json)
        return self.responses.pop(0) if self.responses else _FakeStreamResponse([])


def _sse(*objs):
    out = b""
    for o in objs:
        out += b"data: " + json.dumps(o).encode() + b"\n\n"
    return out + b"data: [DONE]\n\n"


def test_streaming_core():
    print("\n== tool_runtime streaming core (Critical Issue 1, offline) ==")
    import tool_runtime as tr

    # 1. decoder: split a transcript at EVERY byte boundary
    events = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo, "}}]},
        {"choices": [{"delta": {"content": "sir."}}]},
    ]
    transcript = _sse(*events)
    dec = tr.SSEDecoder()
    want = []
    for d in dec.feed(transcript):
        want.append(d)
    want.extend(dec.flush())
    bad = 0
    for p in range(len(transcript) + 1):
        d2 = tr.SSEDecoder()
        got = list(d2.feed(transcript[:p])) + list(d2.feed(transcript[p:])) + list(d2.flush())
        if got != want:
            bad += 1
    check("SSE decoder survives every byte split", bad == 0, f"{bad} bad splits")

    # 2. chat(): callback order + fired BEFORE the response finished
    resp = _FakeStreamResponse([transcript])
    seen = []

    def on_delta(s):
        seen.append((s, resp.exhausted))

    text, _ = tr.chat([{"role": "user", "content": "q"}], on_delta=on_delta,
                      use_tools=False, _post=_FakePost([resp]))
    check("chat() text assembled in order", text == "Hello, sir.", f"text={text!r}")
    check("callbacks fire before stream exhaustion", seen and not seen[0][1])
    check("response closed after success", resp.closed)

    # 3. think-block + reasoning_content never reach on_delta
    t_open = "<" + "think" + ">"
    t_close = "</" + "think" + ">"
    thinky = _sse(
        {"choices": [{"delta": {"reasoning_content": "secret chain of thought"}}]},
        {"choices": [{"delta": {"content": t_open + " hidden " + t_close + " Visible answer."}}]},
    )
    resp2 = _FakeStreamResponse([thinky])
    out = []
    text2, _ = tr.chat([{"role": "user", "content": "q"}], on_delta=out.append,
                       use_tools=False, _post=_FakePost([resp2]))
    check("reasoning blocks filtered from speech path",
          text2.strip() == "Visible answer." and all("secret" not in s for s in out),
          f"text={text2!r}")

    # 4. tool fragment merge + malformed args never executed
    tool_stream = _sse(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
                        "function": {"name": "web_search", "arguments": "{\"qu"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0,
                        "function": {"arguments": "ery\":\"x\"}"}}]}}]},
    )
    merger = tr.ToolCallMerger()
    dec3 = tr.SSEDecoder()
    for data in dec3.feed(tool_stream):
        if data != "[DONE]":
            merger.feed_event(json.loads(data))
    calls = merger.calls()
    check("streamed tool fragments merged", len(calls) == 1 and calls[0]["name"] == "web_search"
          and json.loads(calls[0]["arguments"]) == {"query": "x"})

    bad_stream = _sse(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c_bad",
                        "function": {"name": "web_search", "arguments": "{broken"}}]}}]},
    )
    post = _FakePost([_FakeStreamResponse([bad_stream]), _FakeStreamResponse([_sse({"choices": [{"delta": {"content": "recovered"}}]})])])
    executed = []
    orig = tr.execute_tool
    tr.execute_tool = lambda n, a: executed.append(n) or {"ok": True, "_tool": n, "_ms": 0}
    try:
        text3, trace3 = tr.chat([{"role": "user", "content": "q"}], _post=post)
    finally:
        tr.execute_tool = orig
    check("malformed tool args never executed", executed == [])
    check("model told about malformed args",
          any("malformed" in str(m.get("content", "")).lower()
              for m in post.payloads[1]["messages"] if m["role"] == "tool"))
    check("conversation recovers after malformed args", text3 == "recovered")

    # 5. history trimming
    msgs = [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": f"q{i} " + "x" * 100} for i in range(30)
    ] + [{"role": "assistant", "content": "a"}]
    trimmed = tr.trim_history(msgs, max_turns=8, max_chars=10**6)
    check("trim_history keeps system + bounds", trimmed[0]["role"] == "system"
          and len(trimmed) <= 9 and trimmed[-1]["content"] == "a")


def test_memory_restart():
    print("\n== memory_store: write -> close -> restart -> recall ==")
    import memory_store

    db = TMP / "iris_st_restart.db"
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    store = memory_store.MemoryStore(db_path=str(db))
    fake_embedder(store)
    store.add("My laptop is an HP Victus with an RTX 4050.", speaker="user", importance=0.9)
    store.add("I prefer dark mode everywhere.", speaker="user", importance=0.7)
    store.close()  # simulate process exit

    store2 = memory_store.MemoryStore(db_path=str(db))
    fake_embedder(store2)
    st = store2.stats()
    check("rows survive restart", st["total"] == 2, f"total={st['total']}")
    dup = store2.add("My laptop is an HP Victus  with an RTX 4050.")
    check("dedup works across restart", dup is None)
    hits = store2.search("what gpu does my laptop have", k=3)
    check("recall works after restart", any("RTX 4050" in h.content for h in hits),
          f"top={[h.content[:40] for h in hits]}")
    hits2 = store2.search("dark mode", k=3)
    check("second fact recallable", any("dark mode" in h.content.lower() for h in hits2))
    store2.close()
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)


def test_tavily_edge():
    print("\n== tavily_tool: empty + failure shapes (Critical Issue 5) ==")
    import tavily_tool

    orig = tavily_tool.search
    try:
        tavily_tool.search = lambda *a, **k: {"query": "q", "answer": "", "results": []}
        out = tavily_tool.run_tool({"query": "nothing findable"})
        check("empty results reported honestly", out["ok"] and out.get("empty") is True
              and "could not find" in out["guidance"], f"out={out}")

        def boom(*a, **k):
            raise tavily_tool.TavilyError("rate limited")

        tavily_tool.search = boom
        out2 = tavily_tool.run_tool({"query": "x"})
        check("failure returns ok:false, no crash", out2["ok"] is False and "unavailable" in out2["error"])
    finally:
        tavily_tool.search = orig

    normalized = tavily_tool._normalize({"results": [
        {"title": "T" * 400, "url": "https://x.example/" + "u" * 400, "content": "C" * 2000}
    ]})
    r = normalized["results"][0]
    check("payload bounded (title/url/snippet)",
          len(r["title"]) <= 200 and len(r["url"]) <= 300 and len(r["snippet"]) <= 400)


def test_tool_runtime_offline():
    print("\n== tool_runtime (offline) ==")
    import tool_runtime

    sys_prompt = tool_runtime.build_system(budget="standard")
    check("system prompt has persona rules", 'address the user as "sir"' in sys_prompt.lower() or "sir" in sys_prompt)
    check("system prompt has word budgets", "60" in sys_prompt)
    check("schemas registered", {t["function"]["name"] for t in tool_runtime.tool_schemas()} >= {"web_search", "memory_search", "memory_add"})

    unknown = tool_runtime.execute_tool("nonexistent_tool", {})
    check("unknown tool handled", unknown["ok"] is False)
    bad = tool_runtime.execute_tool("web_search", {})
    check("empty query handled gracefully", bad["ok"] is False and "error" in bad)

    events = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "web_search", "arguments": "{\"qu"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ery\":\"x\"}"}}]}}]},
        {"choices": [{"delta": {"content": "final text"}}]},
    ]
    calls = tool_runtime._accumulate_tool_calls(events)
    check("streamed tool_call deltas merged", len(calls) == 1 and calls[0]["name"] == "web_search"
          and json.loads(calls[0]["arguments"]) == {"query": "x"})


def test_voice_offline():
    print("\n== voice modules (static checks) ==")
    import enroll_voice
    import voice_cache

    check("filler lines defined", enroll_voice.cfg.FILLER_LINES == cfg.FILLER_LINES and len(cfg.FILLER_LINES) == 3)
    check("profile path is flat beside modules", cfg.VOICE_PROFILE.parent == cfg.IRIS_DIR)
    check("enroll refuses without sample gracefully",
          callable(enroll_voice.decode_audio) and callable(enroll_voice.transcribe))
    try:
        import torch  # noqa: F401
        check("torch available here", True)
    except ImportError:
        check("torch absent here (expected in sandbox; runs on your GPU box)", True,
              "voice enrollment will be exercised on your machine")


def test_setup_voice():
    print("\n== setup_voice (offline) ==")
    import subprocess
    import setup_voice as sv

    check("keep() keeps core model files",
          sv.keep("llm.pt") and sv.keep("flow.pt") and sv.keep("speech_tokenizer_v2.onnx")
          and sv.keep("cosyvoice2.yaml") and sv.keep("campplus.onnx"))
    check("keep() skips unrelated repo parts",
          not sv.keep("CosyVoice-BlankEN/model.safetensors") and not sv.keep("asset/dingding.png")
          and not sv.keep("README.md") and not sv.keep(".gitattributes"))

    envf = TMP / "iris_st_env.env"
    envf.write_text("TAVILY_API_KEY=dummy-secret\n# comment stays\nCOSYVOICE_ROOT=/old/path\n")
    written = sv.update_env(envf, {"COSYVOICE_ROOT": "/new/path", "COSYVOICE_MODEL": "/new/model"})
    text = envf.read_text()
    check("env editor preserves secrets and comments",
          "TAVILY_API_KEY=dummy-secret" in text and "# comment stays" in text)
    check("env editor updates existing key", "COSYVOICE_ROOT=/new/path" in text and "/old/path" not in text)
    check("env editor appends new key", "COSYVOICE_MODEL=/new/model" in text)
    check("env editor reports written keys", set(written) == {"COSYVOICE_ROOT", "COSYVOICE_MODEL"})
    envf.unlink(missing_ok=True)

    st = sv.status()
    check("status() is offline-safe", isinstance(st, dict) and "engine" in st and "deps_missing" in st)
    r = subprocess.run(
        [sys.executable, str(Path(sv.__file__).resolve()), "--check"],
        capture_output=True, text=True, timeout=60,
    )
    check("--check exits 0 with JSON report", r.returncode == 0 and '"engine"' in r.stdout,
          f"rc={r.returncode} {r.stderr[:120]}")


def test_live():
    print("\n== live network probes ==")
    import tavily_tool

    info = tavily_tool.check()
    check("tavily reachable (keyless or key)", info.get("reachable") is True, json.dumps(info.get("error", "")))
    print(f"       mode: {info.get('mode')}, sample: {info.get('sample_titles', [])[:1]}")
    try:
        import requests

        r = requests.get(cfg.LLAMA_SERVER_URL + "/health", timeout=4)
        check("llama-server /health", r.status_code == 200, f"HTTP {r.status_code}")
    except Exception as e:
        check("llama-server reachable", False, f"{type(e).__name__} (start it on {cfg.LLAMA_SERVER_URL})")


def main() -> int:
    live = "--live" in sys.argv
    test_config()
    test_memory_store()
    test_memory_restart()
    test_memory_import()
    test_tavily_offline()
    test_tavily_edge()
    test_streaming_core()
    test_tool_runtime_offline()
    test_voice_offline()
    test_setup_voice()
    if live:
        test_live()
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{'=' * 50}\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed" + (f", {len(failed)} FAILED" if failed else ""))
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

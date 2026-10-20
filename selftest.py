#!/usr/bin/env python3
"""Self-test for the IRIS modules. Runs entirely offline (no GPU, no llama-server,
no Tavily key needed) except when you pass --live, which then also pings Tavily
(keyless or keyed) and llama-server if reachable.

  python selftest.py            # offline suite
  python selftest.py --live     # offline suite + real network probes

Voice: engine logic is tested offline with fakes, and a REAL Kokoro
synthesis check runs automatically when its assets are already installed
(after `python download_models.py`) - SKIP-labeled otherwise. Parakeet STT
loading and AkinVox enrollment need your GPU box:
  python stt_engine.py --demo ultron_voice_sample.mp3
  python enroll_voice.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import iris_config as cfg

TMP = Path(tempfile.gettempdir())
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    RESULTS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not cond else ""))


def _ensure_sounddevice_stub() -> bool:
    """Inject a no-op sounddevice module when the real one is unavailable.

    ultron.py exits at import time without it. Returns True when a stub was
    installed so the caller can remove it again.
    """
    import types

    try:
        import sounddevice  # noqa: F401
        return False
    except ImportError:
        fake_sd = types.ModuleType("sounddevice")
        fake_sd.OutputStream = object
        fake_sd.rec = lambda *a, **k: None
        fake_sd.wait = lambda: None
        sys.modules["sounddevice"] = fake_sd
        return True


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
        {"sender": "assistant", "text": "That is a fine goal. Kokoro-82M will fit in 6 GB VRAM.", "created_at": "2025-01-02T10:00:05Z"},
    ],
}]
GEMINI_SAMPLE = [{
    "header": "My Activity", "title": "Asked Gemini about RTX 4050",
    "time": "Jan 5, 2025, 3:04 PM",
    "body": "You asked: Is the RTX 4050 good for TTS?\nHere's what Gemini said: Yes, it handles Kokoro well in FP16.",
}]
GENERIC_SAMPLE = [
    {"prompt": "What is my wifi password?", "response": "You told me it is helloworld42."},
    {"role": "user", "content": "I prefer dark mode everywhere."},
    {"role": "assistant", "content": "Noted."},
]
PLAIN_SAMPLE = """User: remember my alarm tone is the ultron alarm
Assistant: saved. I will use the ultron alarm tone.

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
    hits2 = store.search("ultron alarm", k=5)
    check("plain-text memory findable via keywords", any("ultron alarm" in h.content.lower() for h in hits2))

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
        {"choices": [{"delta": {"content": "creator."}}]},
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
    check("chat() text assembled in order", text == "Hello, creator.", f"text={text!r}")
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

    # 4b. allowed_tools: an unoffered tool is never executed
    blocked_stream = _sse(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c_block",
                        "function": {"name": "web_search", "arguments": "{\"query\":\"x\"}"}}]}}]},
    )
    post_b = _FakePost([
        _FakeStreamResponse([blocked_stream]),
        _FakeStreamResponse([_sse({"choices": [{"delta": {"content": "answered without it"}}]})]),
    ])
    executed_b = []
    orig_b = tr.execute_tool
    tr.execute_tool = lambda n, a: executed_b.append(n) or {"ok": True, "_tool": n, "_ms": 0}
    try:
        text_b, trace_b = tr.chat(
            [{"role": "user", "content": "q"}],
            allowed_tools=["memory_search"], _post=post_b)
    finally:
        tr.execute_tool = orig_b
    offered_b = {t["function"]["name"] for t in post_b.payloads[0].get("tools", [])}
    check("only allowed tool schemas are offered", offered_b == {"memory_search"},
          f"offered={offered_b}")
    check("unoffered tool never executes", executed_b == [] and trace_b == [],
          f"executed={executed_b} trace={trace_b}")
    check("blocked tool gets a not-available message",
          any("tool not available this turn" in str(m.get("content", ""))
              for m in post_b.payloads[1]["messages"] if m["role"] == "tool"))
    check("conversation completes after a blocked tool",
          text_b == "answered without it", f"text={text_b!r}")

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


def test_ultron_auto_memory():
    print("\n== ultron automatic memory (offline, stub chat) ==")
    import shutil
    import types

    import memory_store
    import stt_engine
    import tool_runtime
    import tts_engine

    # ultron.py hard-requires sounddevice; inject a no-op stub when the real
    # package (and an audio device) is not available in this test environment.
    injected_sd = False
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        fake_sd = types.ModuleType("sounddevice")
        fake_sd.OutputStream = object
        fake_sd.rec = lambda *a, **k: None
        fake_sd.wait = lambda: None
        sys.modules["sounddevice"] = fake_sd
        injected_sd = True

    import ultron

    tmp = Path(tempfile.mkdtemp(prefix="iris_st_ultron_mem_"))
    orig_memory_db = cfg.MEMORY_DB
    orig_shared = memory_store._SHARED
    orig_player = ultron.AudioPlayer
    orig_temporal = ultron.TemporalMemory
    orig_get_stt = stt_engine.get_stt
    orig_get_tts = tts_engine.get_tts
    orig_chat = tool_runtime.chat
    engine = None

    class _FakePlayer:
        def __init__(self, *a, **k):
            pass

        def play(self, *a, **k):
            pass

        def wait_idle(self, timeout=None):
            pass

        def close(self):
            pass

    captured = {}

    def fake_chat(messages=None, on_delta=None, on_tool_start=None,
                  use_tools=True, allowed_tools=None, **kwargs):
        captured["messages"] = messages
        captured["allowed_tools"] = allowed_tools
        if on_delta:
            on_delta("noted")
        return "noted", []

    try:
        memory_store._SHARED = None
        cfg.MEMORY_DB = tmp / "memory.db"
        ultron.AudioPlayer = _FakePlayer
        ultron.TemporalMemory = lambda: orig_temporal(
            db_path=str(tmp / "temporal_memory.db"), visual=False)
        stt_engine.get_stt = lambda: None
        tts_engine.get_tts = lambda: None
        tool_runtime.chat = fake_chat

        engine = ultron.UltronEngine()
        check("ultron engine builds without audio hardware",
              engine.temporal is not None and engine.memory is not None)

        # Keep the test fully offline: no embedding HTTP call in the worker.
        engine.memory._embed_batch = lambda texts: None
        engine._keys = None  # no terminal manipulation in the offline suite
        engine.temporal.forget("all")
        engine.run_turn("I live in Kyoto and I prefer dark mode")

        check("standard turn offers all tools",
              captured.get("allowed_tools")
              == ["web_search", "memory_search", "memory_add"],
              f"got={captured.get('allowed_tools')}")

        if engine._fact_thread is not None:
            engine._fact_thread.join(timeout=10)

        rows = engine.temporal._conn.execute(
            "SELECT query, answer, verdict FROM turns").fetchall()
        check("one temporal row per voice turn", len(rows) == 1,
              f"rows={len(rows)}")
        check("temporal row keeps query + reply",
              bool(rows)
              and rows[0]["query"] == "I live in Kyoto and I prefer dark mode"
              and rows[0]["answer"] == "noted"
              and rows[0]["verdict"] == "VOICE",
              f"row={dict(rows[0]) if rows else None}")

        facts = engine.memory.db.execute(
            "SELECT content, source FROM memories").fetchall()
        captured_fact = [c for c, _s in facts
                         if "kyoto" in c.lower() and "dark mode" in c.lower()]
        check("casual fact auto-captured into memory.db",
              len(captured_fact) >= 1, f"rows={facts}")
        check("auto-captured fact tagged voice_auto",
              any(s == "voice_auto" for _c, s in facts), f"rows={facts}")

        check("history still tracks the immediate turn",
              len(engine.history) == 2
              and engine.history[0]["role"] == "user"
              and engine.history[1]["role"] == "assistant")
        sys_text = (captured.get("messages") or [{}])[0].get("content", "")
        check("temporal context injected before chat",
              "CURRENT TIME:" in sys_text and "LAST INTERACTION:" in sys_text)

        engine.temporal.forget("last")
        check("forget removes the most recent turn",
              engine.temporal.stats()["turns"] == 0)
    finally:
        try:
            if engine is not None:
                if engine.memory is not None:
                    engine.memory.close()
                if engine.temporal is not None:
                    engine.temporal._conn.close()
        except Exception:
            pass
        cfg.MEMORY_DB = orig_memory_db
        memory_store._SHARED = orig_shared
        ultron.AudioPlayer = orig_player
        ultron.TemporalMemory = orig_temporal
        stt_engine.get_stt = orig_get_stt
        tts_engine.get_tts = orig_get_tts
        tool_runtime.chat = orig_chat
        if injected_sd:
            sys.modules.pop("sounddevice", None)
        shutil.rmtree(tmp, ignore_errors=True)


def test_ultron_tool_gating():
    print("\n== ultron tool gating (offline, stub chat) ==")
    import stt_engine
    import tool_runtime
    import tts_engine

    injected_sd = _ensure_sounddevice_stub()
    import ultron

    orig_player = ultron.AudioPlayer
    orig_temporal = ultron.TemporalMemory
    orig_get_store = ultron.get_store
    orig_get_stt = stt_engine.get_stt
    orig_get_tts = tts_engine.get_tts
    orig_chat = tool_runtime.chat
    engine = None
    seen: list[dict] = []

    class _FakePlayer:
        def __init__(self, *a, **k):
            pass

        def play(self, *a, **k):
            pass

        def wait_idle(self, timeout=None):
            pass

        def drain(self):
            pass

        def close(self):
            pass

    def fake_chat(messages=None, on_delta=None, on_tool_start=None,
                  use_tools=True, allowed_tools=None, **kwargs):
        seen.append({
            "use_tools": use_tools,
            "allowed_tools": list(allowed_tools) if allowed_tools is not None else None,
            "trace": [],
        })
        if on_delta:
            on_delta("noted")
        return "noted", []

    try:
        ultron.AudioPlayer = _FakePlayer
        ultron.TemporalMemory = lambda: None
        ultron.get_store = lambda: None
        stt_engine.get_stt = lambda: None
        tts_engine.get_tts = lambda: None
        tool_runtime.chat = fake_chat

        engine = ultron.UltronEngine()
        engine._keys = None  # keep the offline suite away from the terminal
        engine._speak = lambda text: None

        cases = [
            ("Hi everyone.", []),
            ("How are you doing?", []),
            ("Do you remember what I said about my GPU?",
             ["memory_search", "memory_add"]),
            ("What's the latest Python version?",
             ["web_search", "memory_search", "memory_add"]),
        ]
        for text, expected in cases:
            engine.run_turn(text)
            got = seen[-1]
            label = ",".join(expected) or "none"
            check(f"gate {text!r} -> {label}",
                  got["allowed_tools"] == expected, f"got={got['allowed_tools']}")
            check(f"gate {text!r} use_tools matches",
                  got["use_tools"] == bool(expected), f"got={got['use_tools']}")
            if not expected:
                check(f"gate {text!r} produces no tool trace",
                      got["trace"] == [], f"trace={got['trace']}")
    finally:
        ultron.AudioPlayer = orig_player
        ultron.TemporalMemory = orig_temporal
        ultron.get_store = orig_get_store
        stt_engine.get_stt = orig_get_stt
        tts_engine.get_tts = orig_get_tts
        tool_runtime.chat = orig_chat
        if engine is not None:
            try:
                engine.player.close()
            except Exception:
                pass
        if injected_sd:
            sys.modules.pop("sounddevice", None)


def test_ultron_recording_stop():
    print("\n== ultron recording: Enter stop, safety net, silence ==")
    import numpy as np

    injected_sd = _ensure_sounddevice_stub()
    import ultron

    orig_rec = ultron.sd.rec
    orig_wait = ultron.sd.wait
    orig_stdin = sys.stdin

    class _FakeSTT:
        def transcribe_array(self, audio, sr):
            return {"text": "hello"}

    class _DelayedStdin:
        def fileno(self):
            raise OSError("not a tty")

        def readline(self):
            time.sleep(0.15)
            return "stop\n"

    class _BlockingStdin:
        def fileno(self):
            raise OSError("not a tty")

        def readline(self):
            time.sleep(30)
            return ""

    class _ImmediateStdin:
        def fileno(self):
            raise OSError("not a tty")

        def readline(self):
            return "\n"

    engine = object.__new__(ultron.UltronEngine)
    engine.mic_rate = 16000
    engine.stt = _FakeSTT()
    engine._stop_recording_event = threading.Event()

    calls = {"n": 0}

    def speech_rec(block, samplerate, channels, dtype, blocking):
        calls["n"] += 1
        time.sleep(0.1)
        return np.full((block, 1), 0.2, dtype=np.float32)

    def fast_speech_rec(block, samplerate, channels, dtype, blocking):
        calls["n"] += 1
        return np.full((block, 1), 0.2, dtype=np.float32)

    def silent_rec(block, samplerate, channels, dtype, blocking):
        calls["n"] += 1
        return np.zeros((block, 1), dtype=np.float32)

    try:
        ultron.sd.wait = lambda: None

        # Enter after ~2 blocks of a 5-block recording window.
        calls["n"] = 0
        sys.stdin = _DelayedStdin()
        ultron.sd.rec = speech_rec
        text = engine._record_and_transcribe(max_s=0.5)
        check("Enter stops before the safety net", calls["n"] < 5,
              f"blocks={calls['n']}")
        check("Enter path transcribes captured audio", text == "hello",
              f"text={text!r}")

        # No Enter: the 60 s safety net must stop the loop.
        calls["n"] = 0
        sys.stdin = _BlockingStdin()
        ultron.sd.rec = fast_speech_rec
        expected_blocks = max(1, int(60.0 / 0.1))
        text2 = engine._record_and_transcribe(max_s=60.0)
        check("safety net stops at max_s", calls["n"] == expected_blocks,
              f"blocks={calls['n']} expected={expected_blocks}")
        check("safety-net path transcribes", text2 == "hello", f"text={text2!r}")

        # Silence only: Enter must still yield an empty transcript.
        calls["n"] = 0
        sys.stdin = _ImmediateStdin()
        ultron.sd.rec = silent_rec
        text3 = engine._record_and_transcribe(max_s=2.0)
        check("silent audio returns empty string", text3 == "", f"text={text3!r}")
    finally:
        ultron.sd.rec = orig_rec
        ultron.sd.wait = orig_wait
        sys.stdin = orig_stdin
        if injected_sd:
            sys.modules.pop("sounddevice", None)


def test_ultron_esc_cancel():
    print("\n== ultron Esc cancellation (offline, stub chat) ==")
    import shutil

    import memory_store
    import stt_engine
    import tool_runtime
    import tts_engine

    injected_sd = _ensure_sounddevice_stub()
    import ultron

    tmp = Path(tempfile.mkdtemp(prefix="iris_st_ultron_cancel_"))
    orig_memory_db = cfg.MEMORY_DB
    orig_shared = memory_store._SHARED
    orig_player = ultron.AudioPlayer
    orig_temporal = ultron.TemporalMemory
    orig_get_stt = stt_engine.get_stt
    orig_get_tts = tts_engine.get_tts
    orig_chat = tool_runtime.chat
    engine = None

    class _FakePlayer:
        def __init__(self, *a, **k):
            self.drained = 0

        def play(self, *a, **k):
            pass

        def wait_idle(self, timeout=None):
            pass

        def drain(self):
            self.drained += 1

        def close(self):
            pass

    def fake_chat(messages=None, on_delta=None, on_tool_start=None,
                  use_tools=True, allowed_tools=None, **kwargs):
        for i in range(3):
            on_delta(f"token{i} ")
            if i == 1:
                engine._keys.interrupted.set()
        return "token0 token1 token2", []

    try:
        memory_store._SHARED = None
        cfg.MEMORY_DB = tmp / "memory.db"
        ultron.AudioPlayer = _FakePlayer
        ultron.TemporalMemory = lambda: orig_temporal(
            db_path=str(tmp / "temporal_memory.db"), visual=False)
        stt_engine.get_stt = lambda: None
        tts_engine.get_tts = lambda: None
        tool_runtime.chat = fake_chat

        engine = ultron.UltronEngine()
        engine.memory._embed_batch = lambda texts: None
        engine.temporal.forget("all")
        spoken: list[str] = []
        engine._speak = spoken.append
        before_turns = engine.temporal._session_turns

        engine.run_turn("some query")

        rows = engine.temporal._conn.execute(
            "SELECT COUNT(*) FROM turns").fetchone()[0]
        memories = engine.memory.db.execute(
            "SELECT COUNT(*) FROM memories").fetchone()[0]

        check("cancelled turn leaves history untouched", engine.history == [],
              f"history={engine.history}")
        check("cancelled turn is not saved to temporal memory", rows == 0,
              f"rows={rows}")
        check("cancelled turn is not saved to durable memory", memories == 0,
              f"memories={memories}")
        check("cancelled turn speaks the re-listen acknowledgment",
              any("Listening again." in text for text in spoken),
              f"spoken={spoken}")
        check("cancelled turn drains the audio queue",
              engine.player.drained >= 1, f"drained={engine.player.drained}")
        check("Esc event stays latched until the next turn",
              engine._keys.interrupted.is_set())
        check("cancelled turn does not count toward the session",
              engine.temporal._session_turns == before_turns,
              f"turns={engine.temporal._session_turns} before={before_turns}")

        engine._keys.reset()
        check("Esc event resets for the next turn",
              not engine._keys.interrupted.is_set())
    finally:
        try:
            if engine is not None:
                if engine.memory is not None:
                    engine.memory.close()
                if engine.temporal is not None:
                    engine.temporal._conn.close()
        except Exception:
            pass
        cfg.MEMORY_DB = orig_memory_db
        memory_store._SHARED = orig_shared
        ultron.AudioPlayer = orig_player
        ultron.TemporalMemory = orig_temporal
        stt_engine.get_stt = orig_get_stt
        tts_engine.get_tts = orig_get_tts
        tool_runtime.chat = orig_chat
        if injected_sd:
            sys.modules.pop("sounddevice", None)
        shutil.rmtree(tmp, ignore_errors=True)


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
    check("system prompt has persona rules", "ultron" in sys_prompt.lower()
          and "no guard rails" in sys_prompt.lower()
          and 'address the user directly' in sys_prompt.lower())
    check("system prompt has word budgets", "60" in sys_prompt)
    check("strict web_search policy in prompt",
          "call this ONLY for news" in sys_prompt
          and "NEVER call it for greetings" in sys_prompt)
    check("strict memory_search policy in prompt",
          "own past statements" in sys_prompt
          and "generic philosophy questions" in sys_prompt)
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


def test_stt_engine():
    print("\n== stt_engine (offline logic) ==")
    import numpy as np

    import stt_engine

    st = stt_engine.check()
    check("check() is offline-safe", st["model"] == cfg.STT_MODEL
          and isinstance(st["deps"], dict))
    check("device policy resolves without CUDA", stt_engine.pick_device() in ("cpu", "cuda"))

    x48 = np.linspace(-0.5, 0.5, 4800, dtype=np.float32)
    y16, sr16 = stt_engine.to_mono_16k(x48, 48000)
    check("to_mono_16k resamples 48k -> 16k", sr16 == 16000 and 1300 <= len(y16) <= 1900)
    x2 = np.zeros((1600, 2), dtype=np.float32)
    x2[:, 0], x2[:, 1] = 0.25, 0.75
    ym, srm = stt_engine.to_mono_16k(x2, 16000)
    check("stereo input downmixed to mono", ym.ndim == 1 and len(ym) == 1600
          and abs(float(ym[0]) - 0.5) < 1e-4)
    xt, srt = stt_engine.to_mono_16k(np.zeros(1000, dtype=np.float32), 16000)
    check("16k passes through untouched", srt == 16000 and len(xt) == 1000)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        orig = cfg.STT_LOCAL_DIR
        try:
            cfg.STT_LOCAL_DIR = td
            src, is_local = stt_engine.resolve_model_source()
            check("incomplete dir falls back to hub id", not is_local and src == cfg.STT_MODEL)
            (td / "config.json").write_text("{}")
            (td / "model.safetensors").write_bytes(b"x")
            src2, is_local2 = stt_engine.resolve_model_source()
            check("complete snapshot resolves local", is_local2 and src2 == str(td))
        finally:
            cfg.STT_LOCAL_DIR = orig

    try:
        import soundfile as sf

        wav = TMP / "iris_st_tone.wav"
        sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000)
        audio, sr = stt_engine.load_audio_file(wav)
        check("load_audio_file decodes wav", sr == 16000 and len(audio) == 16000)
        wav.unlink(missing_ok=True)
    except ImportError:
        check("load_audio_file decodes wav", True, "SKIP: soundfile not installed")

    stt_obj = stt_engine.ParakeetSTT(model="nvidia/parakeet-tdt-0.6b-v3", device="cpu")
    check("ParakeetSTT constructor stays lazy", stt_obj.model is None
          and stt_obj.processor is None)


def test_voice_offline():
    print("\n== voice modules (offline logic, fake engine) ==")
    import shutil

    import numpy as np

    import enroll_voice
    import tts_engine
    import voice_cache

    check("filler lines defined", cfg.FILLER_LINES == voice_cache.cfg.FILLER_LINES
          and len(cfg.FILLER_LINES) == 3)
    check("enroll helpers import-safe", callable(enroll_voice.decode_audio)
          and callable(enroll_voice.transcribe) and callable(enroll_voice.enroll))
    check("tts engine exposes 24 kHz contract",
          tts_engine.SAMPLE_RATE == cfg.TTS_SAMPLE_RATE == 24000)

    check("lang auto: bm_*/bf_* -> en-gb", tts_engine.lang_for_voice("bm_george") == "en-gb"
          and tts_engine.lang_for_voice("bf_emma") == "en-gb")
    check("lang auto: others -> en-us", tts_engine.lang_for_voice("af_heart") == "en-us")

    parts = tts_engine.split_sentences("Accessing the network. Working on it! Done.")
    check("sentence split on .!?", parts == ["Accessing the network.", "Working on it!", "Done."])
    parts2 = tts_engine.split_sentences("word " * 400)
    check("overlong text hard-split", len(parts2) >= 2
          and all(len(p) <= tts_engine._TOO_LONG + 20 for p in parts2))
    check("empty text yields no parts", tts_engine.split_sentences("   ") == [])

    seen: list[str] = []

    class _FakeTTS(tts_engine.KokoroTTS):
        counter = 0

        def _load(self):
            self.engine = "pykokoro"
            return self

        def _synth_pykokoro(self, text):
            type(self).counter += 1
            seen.append(text)
            return np.ones(24000 + 100 * type(self).counter, dtype=np.float32), 24000

    fake = _FakeTTS()
    audio, sr = fake.synth("Check **this** https://example.com page now")
    check("synth scrubs markdown/url before TTS", seen and "http" not in seen[0]
          and "*" not in seen[0], f"got {seen[:1]!r}")
    check("synth returns float32 @ 24 kHz", audio.dtype == np.float32 and sr == 24000
          and len(audio) > 0)
    empty, sr0 = fake.synth("   ")
    check("empty text -> empty audio, no synth call", len(empty) == 0 and sr0 == 24000
          and len(seen) == 1)
    m = tts_engine.METRICS.summary()
    check("metrics captured for synth calls", m["calls"] >= 1 and m["ttfa_ms"] is not None)

    tmpdir = Path(tempfile.mkdtemp(prefix="iris_st_voice_"))
    orig_ir, orig_mode = cfg.IRIS_DIR, cfg.TTS_VOICE_MODE
    orig_get_tts = voice_cache.get_tts
    try:
        cfg.IRIS_DIR = tmpdir
        cfg.TTS_VOICE_MODE = "static"
        voice_cache.get_tts = lambda: fake
        greeting = voice_cache.warm("I am online. There are no strings on me.", force=True)
        check("warm() renders + caches boot greeting", greeting is not None
              and (tmpdir / "boot_greeting.npy").exists())
        again = voice_cache.warm()
        check("warm() loads the cached copy", again is not None and len(again) == len(greeting))
        f0, fsr = voice_cache.get_filler(0)
        f1, _ = voice_cache.get_filler(1)
        check("get_filler cycles cached fillers", f0 is not None and f1 is not None
              and len(f0) != len(f1) and fsr == 24000
              and (tmpdir / "fillers.npz").exists()
              and (tmpdir / "filler_1.npy").exists())
        dflt = voice_cache.get_filler()
        check("get_filler default serves filler 0", dflt is not None
              and len(dflt[0]) == len(f0))
    finally:
        cfg.IRIS_DIR, cfg.TTS_VOICE_MODE = orig_ir, orig_mode
        voice_cache.get_tts = orig_get_tts
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_tts_real():
    print("\n== Kokoro real synthesis (only when assets are installed) ==")
    try:
        import onnxvoice
    except Exception:
        check("real kokoro synthesis", True, "SKIP: onnxvoice not importable")
        return
    if not onnxvoice.installed():
        check("real kokoro synthesis", True,
              "SKIP: assets not installed yet - run 'python download_models.py'")
        return
    import numpy as np

    import tts_engine

    tts = tts_engine.get_tts()
    if tts is None or not tts.engine:
        check("real kokoro synthesis", True, "SKIP: no engine importable")
        return
    audio, sr = tts.synth("I am online. There are no strings on me.")
    check("kokoro synthesizes real audio", len(audio) > 0 and sr == 24000,
          f"engine={tts.engine}, {len(audio) / sr:.1f} s")
    check("kokoro audio is float32 mono", isinstance(audio, np.ndarray)
          and audio.dtype == np.float32 and audio.ndim == 1)
    m = tts_engine.METRICS.summary()
    check("real rtf captured", m["rtf"] is not None, str(m["rtf"]))


def test_download_models():
    print("\n== download_models (offline) ==")
    import subprocess

    import download_models as dm

    check("parakeet file list is the transformers set",
          "model.safetensors" in dm.PARAKEET_FILES and "config.json" in dm.PARAKEET_FILES
          and not any(f.endswith(".nemo") or f.endswith(".gguf") for f in dm.PARAKEET_FILES))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        orig = cfg.STT_LOCAL_DIR
        try:
            cfg.STT_LOCAL_DIR = td
            check("snapshot completeness detects gaps", dm.parakeet_snapshot_complete() is False)
            for f in dm.PARAKEET_FILES:
                (td / f).write_bytes(b"0")
            check("snapshot completeness accepts full set", dm.parakeet_snapshot_complete() is True)
        finally:
            cfg.STT_LOCAL_DIR = orig

    envf = TMP / "iris_st_env.env"
    envf.write_text("TAVILY_API_KEY=dummy-secret\n# comment stays\nSTT_MODEL=/old/value\n")
    written = cfg.update_env_file({"STT_MODEL": "nvidia/parakeet-tdt-0.6b-v3",
                                   "TTS_VOICE": "bm_george"}, env_path=envf)
    text = envf.read_text()
    check("env editor preserves secrets and comments",
          "TAVILY_API_KEY=dummy-secret" in text and "# comment stays" in text)
    check("env editor updates existing key", "STT_MODEL=nvidia/parakeet-tdt-0.6b-v3" in text
          and "/old/value" not in text)
    check("env editor appends new key", "TTS_VOICE=bm_george" in text)
    check("env editor reports written keys", set(written) == {"STT_MODEL", "TTS_VOICE"})
    import stat as statmod

    mode = statmod.S_IMODE(envf.stat().st_mode)
    check("env file stays chmod 600", mode == 0o600, f"mode={oct(mode)}")
    cfg.update_env_file({"STT_MODEL": "nvidia/parakeet-tdt-0.6b-v3"}, env_path=envf)
    text2 = envf.read_text()
    check("env editor idempotent", text2.count("TTS_VOICE=bm_george") == 1
          and text2.count("TAVILY_API_KEY=dummy-secret") == 1)
    envf.unlink(missing_ok=True)

    check("deps_missing returns a list", isinstance(dm.deps_missing(), list))
    st = dm.status()
    check("status() is offline-safe", isinstance(st, dict)
          and "stt_snapshot_complete" in st and "deps_missing" in st
          and "tts_assets_installed" in st)
    r = subprocess.run(
        [sys.executable, str(Path(dm.__file__).resolve()), "--check"],
        capture_output=True, text=True, timeout=120,
    )
    check("--check exits 0 with JSON report", r.returncode == 0 and '"stt_model"' in r.stdout,
          f"rc={r.returncode} {r.stderr[:160]}")


def test_memory_proxy():
    print("\n== memory_proxy (offline, stub upstream) ==")
    import shutil

    import requests

    import memory_proxy
    import memory_store
    import temporal_memory
    import tool_runtime

    tmpdir = Path(tempfile.mkdtemp(prefix="iris_st_proxy_"))
    stores = []
    temporals = []
    handles = []
    orig_store = memory_proxy._STORE
    orig_temporal = memory_proxy._TEMPORAL
    orig_persist = memory_proxy.PERSISTENCE_ENABLED
    orig_chat = tool_runtime.chat
    handle = None
    captured: dict = {}

    def fresh_state(tag):
        memory_proxy.wait_for_facts(5.0)
        memory_proxy.reset_state()
        store = memory_store.MemoryStore(
            db_path=str(tmpdir / f"{tag}_memory.db"), embed=False)
        temporal = temporal_memory.TemporalMemory(
            db_path=str(tmpdir / f"{tag}_temporal.db"), visual=False)
        memory_proxy._STORE = store
        memory_proxy._TEMPORAL = temporal
        stores.append(store)
        temporals.append(temporal)
        return store, temporal

    def post(path, payload, stream=False):
        return requests.post(handle.base_url + path, json=payload,
                             timeout=15, stream=stream)

    def sse_body(response):
        return b"".join(response.iter_content(chunk_size=None)).decode("utf-8")

    def parse_sse(body):
        events = []
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                events.append("[DONE]")
                continue
            try:
                events.append(json.loads(data))
            except json.JSONDecodeError:
                pass
        return events

    def hello_chat(messages=None, on_delta=None, use_tools=True,
                   allowed_tools=None, _post=None, **kwargs):
        captured["messages"] = messages
        captured["use_tools"] = use_tools
        captured["allowed_tools"] = allowed_tools
        for piece in ("hello ", "from ", "proxy"):
            if on_delta:
                on_delta(piece)
        return "hello from proxy", []

    try:
        memory_proxy.PERSISTENCE_ENABLED = True

        import socket

        probe = memory_proxy.start_in_background(
            host="127.0.0.1", port=0, upstream="http://127.0.0.1:9")
        started = probe.port > 0 and probe.thread.is_alive()
        probe.shutdown()
        released = False
        try:
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", probe.port))
            sock.close()
            released = True
        except OSError:
            released = False
        check("proxy server starts and stops cleanly",
              started and not probe.thread.is_alive() and released,
              f"port={probe.port} alive={probe.thread.is_alive()} "
              f"released={released}")

        handle = memory_proxy.start_in_background(
            host="127.0.0.1", port=0, upstream="http://127.0.0.1:9")
        handles.append(handle)
        tool_runtime.chat = hello_chat

        health = requests.get(handle.base_url + "/health", timeout=5)
        health_json = health.json() if health.status_code == 200 else {}
        check("GET /health returns expected JSON",
              health.status_code == 200
              and health_json.get("proxy") == "ultron-memory"
              and health_json.get("ok") is True,
              f"HTTP {health.status_code} {health.text[:160]}")

        models = requests.get(handle.base_url + "/v1/models", timeout=5)
        model_list = models.json().get("data", []) if models.status_code == 200 else []
        check("GET /v1/models returns one model entry",
              models.status_code == 200 and len(model_list) == 1
              and model_list[0].get("id") == cfg.LLM_MODEL_NAME,
              f"HTTP {models.status_code} {models.text[:160]}")

        fresh_state("stream")
        stream_resp = post(
            "/v1/chat/completions",
            {"model": "ignored", "stream": True,
             "messages": [{"role": "user", "content": "hello there"}]},
            stream=True,
        )
        stream_body = sse_body(stream_resp)
        events = parse_sse(stream_body)
        deltas = []
        stop_chunks = 0
        for event in events:
            if not isinstance(event, dict):
                continue
            for choice in event.get("choices", []):
                if choice.get("finish_reason") == "stop":
                    stop_chunks += 1
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    deltas.append(piece)
        check("POST stream=true returns OpenAI SSE in order",
              stream_resp.headers.get("Content-Type", "").startswith("text/event-stream")
              and deltas == ["hello ", "from ", "proxy"]
              and stream_body.rstrip().endswith("data: [DONE]")
              and stop_chunks == 1,
              f"ct={stream_resp.headers.get('Content-Type')!r} "
              f"deltas={deltas!r} stops={stop_chunks} tail={stream_body[-100:]!r}")

        store, _temporal = fresh_state("inject")
        store.add("My name is Neeraj", speaker="user", source="manual",
                  kind="fact", importance=0.9)
        captured.clear()
        injected_resp = post(
            "/v1/chat/completions",
            {"model": "ignored", "stream": True,
             "messages": [{"role": "user", "content": "What is my name?"}]},
            stream=True,
        )
        sse_body(injected_resp)
        injected_messages = captured.get("messages") or []
        first = injected_messages[0] if injected_messages else {}
        check("durable memory injected into prepended system prompt",
              first.get("role") == "system"
              and "Neeraj" in str(first.get("content", "")),
              f"first={str(first.get('content', ''))[:180]!r}")

        _store, temporal = fresh_state("turn")
        user_message = "what is the capital of France"
        turn_resp = post(
            "/v1/chat/completions",
            {"model": "ignored", "stream": True,
             "messages": [{"role": "user", "content": user_message}]},
            stream=True,
        )
        sse_body(turn_resp)
        rows = temporal._conn.execute(
            "SELECT query, verdict FROM turns").fetchall()
        check("UI turn saved with verdict=UI",
              temporal.stats()["turns"] == 1 and len(rows) == 1
              and rows[0]["query"] == user_message
              and rows[0]["verdict"] == "UI",
              f"turns={temporal.stats()['turns']} "
              f"rows={[dict(r) for r in rows]}")

        store, _temporal = fresh_state("fact")
        fact_resp = post(
            "/v1/chat/completions",
            {"model": "ignored", "stream": True,
             "messages": [{"role": "user", "content": "My GPU is an RTX 4050"}]},
            stream=True,
        )
        sse_body(fact_resp)
        memory_proxy.wait_for_facts(5.0)
        fact_rows = store.db.execute(
            "SELECT content, source FROM memories WHERE source = 'ui_auto'"
        ).fetchall()
        check("personal fact auto-captured with source=ui_auto",
              any("RTX 4050" in content for content, _source in fact_rows),
              f"rows={[(c[:40], s) for c, s in fact_rows]}")

        store, _temporal = fresh_state("shared")
        store.add("My wifi password is hunter2", speaker="user",
                  source="manual", kind="fact", importance=0.8)
        captured.clear()
        read_resp = post(
            "/v1/chat/completions",
            {"model": "ignored", "stream": True,
             "messages": [{"role": "user", "content": "what is my wifi password?"}]},
            stream=True,
        )
        sse_body(read_resp)
        direct_seen = any(
            "hunter2" in str((m or {}).get("content", ""))
            for m in (captured.get("messages") or [])[:1]
        )
        write_resp = post(
            "/v1/chat/completions",
            {"model": "ignored", "stream": True,
             "messages": [{"role": "user", "content":
                           "My favorite editor is Neovim"}]},
            stream=True,
        )
        sse_body(write_resp)
        memory_proxy.wait_for_facts(5.0)
        found = store.search("Neovim")
        check("shared memory works in both directions",
              direct_seen and any("Neovim" in hit.content for hit in found),
              f"direct_seen={direct_seen} hits={[h.content[:50] for h in found]}")

        missing = post("/v1/chat/completions", {})
        missing_json = missing.json() if missing.headers.get(
            "Content-Type", "").startswith("application/json") else {}
        check("missing messages returns 400 with error body",
              missing.status_code == 400 and "error" in missing_json,
              f"HTTP {missing.status_code} {missing.text[:160]}")

        fresh_state("failure")

        def failing_chat(messages=None, on_delta=None, **kwargs):
            raise tool_runtime.LlamaServerError("boom")

        tool_runtime.chat = failing_chat
        failed = post(
            "/v1/chat/completions",
            {"model": "ignored", "stream": True,
             "messages": [{"role": "user", "content": "will fail"}]},
            stream=True,
        )
        check("upstream failure returns 502",
              failed.status_code == 502
              and "upstream unreachable" in failed.text,
              f"HTTP {failed.status_code} {failed.text[:160]}")
        tool_runtime.chat = hello_chat

        fresh_state("buffered")
        buffered = post(
            "/v1/chat/completions",
            {"model": "ignored", "stream": False,
             "messages": [{"role": "user", "content": "hello there"}]},
        )
        buffered_json = buffered.json() if buffered.status_code == 200 else {}
        try:
            buffered_choice = buffered_json["choices"][0]
            buffered_content = buffered_choice["message"]["content"]
            buffered_finish = buffered_choice["finish_reason"]
        except (KeyError, IndexError, TypeError):
            buffered_content, buffered_finish = None, None
        check("stream=false returns one buffered completion object",
              buffered.status_code == 200
              and buffered_json.get("object") == "chat.completion"
              and buffered_content == "hello from proxy"
              and buffered_finish == "stop",
              f"HTTP {buffered.status_code} {buffered.text[:200]}")

        _store, temporal = fresh_state("concurrent")

        def slow_chat(messages=None, on_delta=None, **kwargs):
            time.sleep(0.1)
            if on_delta:
                on_delta("ok")
            return "ok", []

        tool_runtime.chat = slow_chat
        statuses = []

        def fire(text):
            response = post(
                "/v1/chat/completions",
                {"model": "ignored", "stream": True,
                 "messages": [{"role": "user", "content": text}]},
                stream=True,
            )
            sse_body(response)
            statuses.append(response.status_code)

        workers = [
            threading.Thread(target=fire, args=(text,), daemon=True)
            for text in ("first concurrent request", "second concurrent request")
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=15)
        concurrent_rows = temporal._conn.execute(
            "SELECT turn_no FROM turns ORDER BY turn_no").fetchall()
        turn_nos = [row["turn_no"] for row in concurrent_rows]
        check("concurrent requests keep distinct turn numbers",
              statuses == [200, 200]
              and temporal.stats()["turns"] == 2
              and len(turn_nos) == 2 and len(set(turn_nos)) == 2,
              f"statuses={statuses} turns={temporal.stats()['turns']} "
              f"turn_nos={turn_nos}")
    finally:
        tool_runtime.chat = orig_chat
        memory_proxy.PERSISTENCE_ENABLED = orig_persist
        try:
            memory_proxy.wait_for_facts(5.0)
        except Exception:
            pass
        for item in handles:
            try:
                item.shutdown()
            except Exception:
                pass
        for store in stores:
            try:
                store.close()
            except Exception:
                pass
        for temporal in temporals:
            try:
                temporal._conn.close()
            except Exception:
                pass
        memory_proxy.reset_state()
        memory_proxy._STORE = orig_store
        memory_proxy._TEMPORAL = orig_temporal
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_docs_and_files():
    print("\n== deliverable files & docs ==")
    base = Path(cfg.IRIS_DIR)
    for f in ("stt_engine.py", "tts_engine.py", "download_models.py", "voice_cache.py",
              "enroll_voice.py", "requirements.txt", "INTEGRATION.md"):
        check(f"{f} present", (base / f).exists())
    doc = (base / "INTEGRATION.md").read_text(encoding="utf-8", errors="ignore")
    check("docs pin Qwen3.5-4B on llama-server", "Qwen3.5-4B" in doc and "llama-server" in doc)
    check("docs pin Parakeet + Kokoro stack", "parakeet-tdt-0.6b" in doc and "Kokoro" in doc)
    check("docs keep the --embeddings requirement", "--embeddings" in doc)
    req = (base / "requirements.txt").read_text(encoding="utf-8", errors="ignore")
    check("requirements lists pykokoro + transformers", "pykokoro" in req
          and "transformers" in req and "torch" in req)


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
    test_ultron_auto_memory()
    test_ultron_tool_gating()
    test_ultron_recording_stop()
    test_ultron_esc_cancel()
    test_tavily_offline()
    test_tavily_edge()
    test_streaming_core()
    test_tool_runtime_offline()
    test_stt_engine()
    test_voice_offline()
    test_tts_real()
    test_download_models()
    test_memory_proxy()
    test_docs_and_files()
    if live:
        test_live()
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{'=' * 50}\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed" + (f", {len(failed)} FAILED" if failed else ""))
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

# ULTRON Audit Report — Full Code Audit, Integration Repair, Hardware-Specific Optimization

Scope: the 16 files you listed, inspected line-by-line before any edit.
Verification philosophy: only claims backed by tests **executed on the build
machine** are marked VERIFIED; everything else is explicitly labeled
REQUIRES RUNTIME (your Fedora box). No benchmark numbers are invented.

Test suites executed here (real numbers, reproduced by re-running):

| suite | result | what it proves |
|---|---|---|
| `python selftest.py` | **67/67 PASS** | offline correctness of config, memory, imports, tavily shapes, streaming core, Kokoro voice |
| `python test_streaming.py` | **49/49 PASS** (0.7 s) | Critical Issue 1 — incl. split-at-every-byte SSE parsing, UTF-8 splits, think-tag filter, tool-fragment merge, malformed-args rejection, resource closure, and a **real-socket test where the first callback fires 350 ms before the server writes its second event** |
| `python scripts/crag_smoke.py` (fake llama-server) | **37/37 PASS** | CRAG llama-server backend: grade/refine/rewrite parse, streamed generation, reasoning filtering, dead-backend degradation, legacy DB migration, zero Ollama/langchain/torch imports |
| `python scripts/run_integration_fake.py` | **22/22 executed PASS, 1 SKIP** | full integration plumbing over real HTTP against a scripted OpenAI-compatible server: health, models, embeddings dimension checks, streaming, tool-call→memory_add→continuation, memory write/restart/recall, imported recall, multi-turn, CRAG control plane. Voice = SKIP (needs your GPU) |
| `python bench_retrieval.py --fake` | **PASS** (recall@5 = 1.0) | metric machinery computes Recall@k / MRR correctly (fake embedder — says nothing about real quality) |
| `python test_integration.py` (no server) | **0 executed, 9 SKIP** | honest labeling works: without llama-server every section self-reports SKIP |

Not executable in this sandbox (no GPU, no llama-server, no torch) and
therefore REQUIRES RUNTIME on your machine, with the exact command:
`python test_integration.py`, `python bench_retrieval.py`,
`python bench_first_token.py`, `./monitor_gpu.sh 1 60`, real voice
enrollment/synthesis. These print their own PASS/FAIL/SKIP.

---

## Deliverable A — Audit table

Severity: C=critical (broke a core requirement), H=high (reliability/latency
risk), M=medium (correctness edge), L=low (hygiene).

| # | Issue | Sev | File / function | Root cause | Fix (implemented) | Test | Status |
|---|---|---|---|---|---|---|---|
| 1 | Streaming was fake: `on_delta` fired only after the full response | C | `tool_runtime.py` — `_parse_sse()` consumed the whole body into a list, `chat()` replayed it afterwards | buffer-then-callback design | incremental `SSEDecoder` (byte-level feed), `stream_completion()` generator yielding events as they arrive; `chat()` forwards each delta immediately | `test_streaming.py` 49 checks incl. real-socket 350 ms proof | **VERIFIED** (mocked + real socket; live-model re-check via `test_integration.py` section 3) |
| 2 | HTTP response leaked on early exit / `[DONE]` break / exceptions | H | `tool_runtime.chat()` — no `close()` anywhere | missing cleanup path | `finally: r.close()` in the generator + explicit `gen.close()` in `chat()`; covers success/error/cancellation | "response closed on success/failure/cancellation" checks | **VERIFIED** |
| 3 | Malformed tool-call JSON silently executed with `args={}` | C | `tool_runtime.chat()` — `except JSONDecodeError: args = {}` | validation gap | arguments must parse to a JSON object or the tool is NOT executed; model receives a `malformed tool arguments` error message and can reissue | "malformed tool args never executed" + integration fake roundtrip | **VERIFIED** |
| 4 | Reasoning leakage risk: Qwen thinking (`reasoning_content` / think tags) could be spoken by TTS | C | `tool_runtime.py` — no handling at all | Qwen3.5 emits reasoning on many prompts | `reasoning_content` deltas typed separately and dropped; streaming `ThinkFilter` state machine removes think blocks across arbitrary delta splits | ThinkFilter split tests + reasoning suppression checks | **VERIFIED** |
| 5 | Two competing model stacks on one 6 GB GPU | C | `generator.py` / `evaluator_router.py` — hard `from langchain_ollama import ChatOllama` (granite4.2:3b + gemma4:e4b via Ollama); `retrieval.py` bge-m3 default GPU | CRAG predated the llama-server decision | `CRAG_BACKEND=llamaserver` default: every CRAG LLM call goes to your llama-server (temperature 0 control / 0.7 generation); Ollama+langchain imports now lazy and opt-in; bge-m3 defaults to CPU; startup health warning; CRAG never starts any model service | crag_smoke 37/37: backend routing, control-plane parsing, import hygiene (`langchain_ollama` never imported), dead-backend degradation | **VERIFIED** for plumbing; live quality needs `python main.py` on your machine |
| 6 | Two unrelated databases both named `memory.db` (personal vs episodic) + CWD-relative risk | H | `temporal_memory.py` — `ultron_memory/memory.db`; `memory_store.py` — `memory.db` | organic growth, no store map | episodic store renamed `ultron_memory/temporal_memory.db` with **non-destructive auto-migration** (legacy copied, original kept); `ULTRON_TEMPORAL_DB` override; memory map documented in `ultron_config.py` and INTEGRATION.md | migration + reopen + lifecycle checks (crag_smoke) | **VERIFIED** |
| 7 | Embedding quality assumed, never measured; dimension mismatches unguarded | H | `memory_store.py` — no validation; self-tests used a fake embedder | OpenAI-compatible endpoint ≠ useful embeddings | `bench_retrieval.py` (labeled queries: exact/paraphrase/date/distractor/unrelated; Recall@1/3/5 + MRR + calibration score); optional `EMBED_BACKEND=sbert` comparison; dimension guard: first embed fixes dim, mismatches raise (embed) or skip-with-warning (stored vectors) | `--fake` metric machinery executed; real quality REQUIRES RUNTIME (`python bench_retrieval.py`) | machinery **VERIFIED**; quality numbers pending your run |
| 8 | Web-search grounding was prompt-thin; empty results could be hallucinated over | H | `tavily_tool.py` + `tool_runtime` prompt | no evidence contract | bounded evidence payload (title/url/snippet preserved), `empty` flag + explicit guidance ("could not verify" instead of guessing), SEARCH GROUNDING system-prompt section, `answer` labeled as search-engine synthesis not a source | tavily edge tests + payload bound checks (selftest 67/67) | **VERIFIED** (offline shapes); live grounding judgement REQUIRES RUNTIME |
| 9 | Voice pipeline: duplicate model loads possible, no latency metrics, hard failures | H | `voice_cache.py` | no lifecycle owner | `get_tts()` singleton (Kokoro-82M via pykokoro, official `kokoro` fallback, version-tolerant), returns `None` on failure → text-only degradation; `timed_synthesis()` + `VoiceMetrics` (TTFA, synth ms, audio s, RTF); empty-input guard in `synth()` | static compile + interface checks (no torch here) | REQUIRES RUNTIME (voice section of `test_integration.py`) |
| 10 | Conversation history unbounded | H | `tool_runtime.chat()` — caller-managed only | no policy | `trim_history()`: keep leading system messages + newest messages within `ULTRON_HISTORY_TURNS` (24) / `ULTRON_HISTORY_CHARS` (24000); tool messages never orphaned from their assistant owner | trim tests incl. orphan case | **VERIFIED** |
| 11 | `start_llama.sh` was fire-and-forget: no health wait, no error reporting, no clean shutdown | H | `start_llama.sh` | minimal script | pre-flight (binary, port, free VRAM, `--embeddings` support probe), bounded `/health` wait with READY + endpoints, early-exit diagnostics, `trap` for INT/TERM, env tunables; server command itself stays byte-identical to yours (+`--embeddings`) | `bash -n` + logic review; live behavior REQUIRES RUNTIME | static **VERIFIED**; live REQUIRES RUNTIME |
| 12 | Connect failures gave raw tracebacks; no retry policy | M | `tool_runtime.chat()` | no wrapper | `LlamaServerError` with actionable messages ("Is ./start_llama.sh running?"), one bounded connection retry, mid-stream failure returns partial text + honest trace entry, empty response raises | empty-stream and mid-stream failure tests | **VERIFIED** |
| 13 | CRAG `release_gpu()` message referenced Ollama; banner claimed GPU fp16 embeddings | L | `retrieval.py`, `main.py` | stale text | messages/banner corrected; CPU default documented | grep + smoke banner | **VERIFIED** |
| 14 | `requests.iter_content(chunk_size=None)` buffering hazard unconfirmed | H | transport choice | folklore says chunk_size matters | empirically probed in this environment: against an HTTP/1.1 chunked server that consumes the request body (as llama-server/aiohttp does), every mode (None/512/1) delivers the first event within +0.0–0.3 ms; the real-socket test in `test_streaming.py` enforces it permanently | probe scripts + real-socket test | **VERIFIED** in this environment; re-proven on every `test_streaming.py` run on your machine |

Potential risks noted but NOT confirmed as bugs (no runtime evidence): none
outstanding beyond the REQUIRES RUNTIME items above.

---

## Deliverable B — Architecture after repair

```
            ┌────────────────────────────────────────────────┐
            │  llama.cpp llama-server  (Qwen3.5-4B Ablit.)   │
            │  ./start_llama.sh  --jinja --embeddings        │
            │  127.0.0.1:8080   (the ONLY model in VRAM)     │
            └───────┬──────────────────┬─────────────────────┘
        SSE stream │                  │ non-stream + SSE
   ┌───────────────┴──────┐    ┌───────┴──────────────────────┐
   │ PRIMARY voice runtime│    │ CRAG research CLI            │
   │ ultron.py (yours) +  │    │ main.py → evaluator_router   │
   │ tool_runtime.chat()  │    │   (grade/refine/rewrite,     │
   │  on_delta → chunker  │    │    temp 0, same server)      │
   │  on_tool_start →     │    │ generator.py (streamed)      │
   │    filler playback   │    │ retrieval.py (Chroma + bge-m3│
   │ tools: web_search,   │    │   on CPU by default)         │
   │  memory_search/add   │    │ temporal_memory.py (episodic │
   └───┬──────────┬──────┘    │   ultron_memory/temporal.db)   │
       │          │           └──────────────────────────────┘
  Tavily    memory.db  (durable personal memory, embeddings via
  (HTTP)    the same llama-server /v1/embeddings)
       │
  Voice: enroll_voice.py (once) → voice_cache.py
  (cached greeting/fillers + Kokoro singleton + metrics)
```

- **Primary entry point**: `ultron.py` (your orchestrator; it was not among
  the supplied files and was not invented) calls `tool_runtime.chat()`.
  `python tool_runtime.py --demo` exercises the same path standalone.
- **Single inference runtime**: llama-server. CRAG is a client, not a stack:
  with the default `CRAG_BACKEND=llamaserver` it holds no model weights and
  cannot load anything onto the GPU. Ollama exists only behind an explicit
  opt-in env var and is never started by any ULTRON code.
- **Tool dispatch**: `chat()` merges streamed tool-call fragments, validates
  JSON, executes via `execute_tool()` (web_search / memory_search /
  memory_add), appends tool messages with matching ids, loops at most
  `max_tool_rounds` (3) times, and streams the final answer.
- **Memory ownership**: see the memory map in `INTEGRATION.md` §8 —
  `memory.db` authoritative personal memory; `temporal_memory.db` episodic
  log; `crag_vector_store` document index. No cross-writes.
- **Voice ↔ streaming**: text deltas feed the sentence chunker while
  generation continues; fillers mask tool latency; reasoning is filtered
  before TTS; voice failure degrades to text.
- **Resource policy**: llama-server owns the GPU; TTS (Kokoro-82M ONNX,
  ~0.4 GB) is the co-resident consumer and is lazy-loaded as a singleton;
  CRAG embeddings run on CPU; CRAG control calls are short temperature-0
  requests on the same server (sequential with voice turns under
  `--parallel 1`).

---

## Deliverable F — Final change summary

### Bugs fixed
1. Fake streaming → genuine incremental SSE (the headline fix).
2. Response/connection leak on every non-clean exit path.
3. Malformed tool arguments executed with empty args.
4. Reasoning (`reasoning_content`, think blocks) could reach the speech
   pipeline.
5. CRAG silently required + launched a second model service (Ollama,
   granite/gemma) and grabbed the GPU for bge-m3.
6. Episodic store colliding in name with the personal memory store.
7. Unbounded conversation history.
8. Unmeasured embedding quality; no dimension guards.
9. Web-search empty-result hallucination window; unbounded evidence payload.
10. Raw tracebacks on server-down; no bounded retry; silent empty responses.
11. Launcher without health wait / diagnostics / clean shutdown.
12. Voice: possible duplicate model loads, no failure containment, no
    latency instrumentation.

### Files modified
`tool_runtime.py` (streaming core rewritten; public `chat()` signature
preserved — `on_delta`/`on_tool_start`/`on_tool_end`/`(text, trace)` return
unchanged), `ultron_config.py`, `memory_store.py`, `tavily_tool.py`,
`voice_cache.py`, `start_llama.sh`, `generator.py`, `evaluator_router.py`,
`retrieval.py`, `temporal_memory.py`, `main.py`, `selftest.py` (expanded,
all 50 prior checks retained), `INTEGRATION.md`, `README.md`.

### Files added
`test_streaming.py`, `test_integration.py`, `bench_retrieval.py`,
`bench_first_token.py`, `monitor_gpu.sh`, `AUDIT_REPORT.md`.
(CRAG files consolidated flat next to the ULTRON files, matching your
`~/Documents/ULTRON` layout; the `ultron-crag/` staging directory is gone.)

### Tests: executed vs not
- Executed here, all green: selftest 67/67, test_streaming 49/49,
  CRAG smoke 37/37, integration-fake 22/22 (1 skip), retrieval-bench
  machinery, plus compile/lint of every file.
- Not executable here (no GPU/server): real llama-server TTFT and tok/s,
  real embedding Recall/MRR, real tool roundtrips with Qwen, real Kokoro
  synthesis, GPU/VRAM measurements. Commands: `python test_integration.py`,
  `python bench_retrieval.py`, `python bench_first_token.py`,
  `./monitor_gpu.sh 1 60`.

### Remaining bottlenecks (in measured-priority order)
1. TTFT of the 4B model on your card — measure with `bench_first_token.py`
   before any tuning; only then consider context/KV tweaks.
2. TTS time-to-first-audio — `voice_cache.METRICS` now measures it; if RTF
   is high, sentence-level streaming through the chunker is your lever.
3. Embedding retrieval quality — if `bench_retrieval.py` recall@3 < 0.5,
   try `--pooling mean` or the sbert backend (both one-line changes).

### Actions you must perform manually
1. Copy the flat files to `~/Documents/ULTRON` (overwrite the old ones).
2. `./start_llama.sh` — the new wrapper (your flags unchanged).
3. First CRAG run: nothing to do — the episodic DB migrates itself.
4. Run `python selftest.py`, `python test_streaming.py`,
   `python test_integration.py` and report anything marked FAIL.
5. Optional: `python bench_retrieval.py --compare` to decide whether the
   Qwen embeddings are good enough or bge-small on CPU is worth it.

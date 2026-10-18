# IRIS Integration Guide

Flat files only — drop everything into `~/Documents/IRIS`. This guide covers
the post-audit architecture (single llama-server runtime, genuine streaming,
unified memory map, CRAG behind the same server). See `AUDIT_REPORT.md` for
what was broken and how it was fixed, and `README.md` for the file map.

## The architecture in one paragraph

One inference runtime: **llama.cpp `llama-server` serving Qwen3.5-4B Q8_0**
(`./start_llama.sh`, your exact flags plus `--embeddings`). The voice
assistant (`jarvis.py` + `tool_runtime.chat()`) and the CRAG research CLI
(`main.py`) are both plain HTTP clients of that one server — no Ollama, no
second LLM ever loads. Web search goes through Tavily, durable personal
memory lives in `memory.db` (embeddings from the same server), the CRAG
episodic log lives in `iris_memory/temporal_memory.db`, and the document
index in `crag_vector_store/` (bge-m3, CPU by default). Voice is CosyVoice2
zero-shot from your one-time enrollment; text deltas stream to your chunker
the moment they leave the socket.

## 1. File map

| file | role |
|---|---|
| `iris_config.py` | shared config, `.env` loader, TTS-safe scrubbers, **memory map** |
| `start_llama.sh` | llama-server launcher: your flags + `--embeddings`, health wait, clean shutdown |
| `tool_runtime.py` | **genuinely incremental** SSE chat client + tool loop + JARVIS prompt |
| `tavily_tool.py` | evidence-grounded Tavily search (SDK/REST/keyless) + `web_search` tool |
| `memory_store.py` | durable personal memory (`memory.db`): hybrid search + tools |
| `memory_import.py` | import ChatGPT / Claude / Gemini / text exports into memory |
| `setup_voice.py` | one-time voice bootstrap: CosyVoice engine + 3.9 GB models + `.env` |
| `enroll_voice.py` | one-time voice enrollment from your JARVIS mp3 |
| `voice_cache.py` | profile loading, frontend memoization, TTS singleton, latency metrics |
| `selftest.py` | offline suite (67 checks) — syntax, parsers, memory, streaming |
| `test_streaming.py` | Critical-Issue-1 proof: every-byte splits + real-socket incremental (49 checks) |
| `test_integration.py` | live integration suite — executed vs SKIP always labeled |
| `bench_retrieval.py` | labeled embedding-quality benchmark (Recall@k, MRR) |
| `bench_first_token.py` | TTFT / generation-rate benchmark against llama-server |
| `monitor_gpu.sh` | VRAM/RAM sampler to run beside benchmarks |
| `generator.py` | CRAG answer generator — llama-server backend (Ollama opt-in) |
| `evaluator_router.py` | CRAG grading/refine/rewrite — llama-server backend, adaptive router |
| `retrieval.py` | CRAG ChromaDB document index (bge-m3, CPU default) |
| `temporal_memory.py` | CRAG episodic store (auto-migrated to `temporal_memory.db`) |
| `main.py` | CRAG CLI entry point (`python main.py`) |

Runtime directories created on your machine (third-party assets, the only
nesting): `CosyVoice/` (engine + model weights), `iris_memory/`,
`visual_memory/`, `crag_vector_store/`. Everything IRIS itself owns is flat.

## 2. One-time setup

```bash
cd ~/Documents/IRIS
cat > .env <<'EOF'
TAVILY_API_KEY=tvly-your-key-here
LLAMA_SERVER_URL=http://127.0.0.1:8080
LLM_MODEL=Qwen3.5-4B
EOF
chmod 600 .env

pip install requests numpy        # core deps

python setup_voice.py             # CosyVoice engine + models + .env keys
python enroll_voice.py            # one-time voice fingerprint (never re-learned)
python memory_import.py ~/Downloads/conversations.json   # optional memory import
```

## 3. Start the (single) model server

```bash
./start_llama.sh
```

Your command, byte-for-byte, plus `--embeddings` — the wrapper only adds
pre-flight checks (binary present, port free, VRAM shown), a bounded
`/health` wait with READY message, and clean Ctrl+C shutdown. Tunables via
env: `IRIS_LLAMA_BIN`, `IRIS_LLAMA_DIR`, `IRIS_MODEL`, `IRIS_PORT`,
`IRIS_HOST`, `IRIS_CTX`, `IRIS_WAIT_S`. Flag rationale:

- `--jinja` — required for `web_search` / `memory_*` tool calls (you had it)
- `--embeddings` — serves `/v1/embeddings` so memory vectors come from the
  same resident model; quality is measured, not assumed (`bench_retrieval.py`)
- `--cache-type-k/v q8_0`, `--n-gpu-layers all`, `--flash-attn on` — your
  VRAM discipline, untouched
- `--parallel 1` — one decode slot; CRAG and the voice assistant share it
  sequentially, which is exactly what a 6 GB card wants
- if VRAM gets tight with CosyVoice resident: optionally add
  `--ot per_layer_token_embd.weight=CPU` (~300 MB freed)

## 4. Wire into jarvis.py

```python
from tool_runtime import chat, build_system
from memory_store import get_store
import voice_cache

profile = voice_cache.load_profile()            # ms; never re-learns
cv, fp16 = voice_cache.get_cosyvoice()          # singleton; None if TTS down
if cv is not None:
    voice_cache.install_frontend_cache(cv, profile)
    greeting = profile.boot_greeting or voice_cache.warm(cv, profile)
    play(greeting)                              # text IRIS continues if this failed

# --- per turn ---
sys_prompt = build_system(budget="standard",
                          memory_query=user_text,   # auto memory injection
                          store=get_store())
messages = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_text}]

final_text, trace = chat(
    messages,                                   # history auto-bounded (24 turns / 24k chars)
    on_delta      = lambda s: chunker.feed(s),  # SOFT_FIRST 12-15 word path: fires DURING generation
    on_tool_start = lambda name: filler_player.fire(),
)
for sentence in chunker.drain():
    if cv is not None:
        for chunk in voice_cache.timed_inference(cv, sentence, profile):
            play(chunk["tts_speech"])
print(voice_cache.METRICS.summary())            # ttfa / rtf / synth stats
```

Streaming guarantees (verified in `test_streaming.py`): callbacks fire while
the response is still arriving, `reasoning_content` deltas and think-tag
reasoning blocks never reach TTS, malformed tool arguments never execute, and the HTTP
response is closed on success, error, and cancellation. History is trimmed
automatically (`IRIS_HISTORY_TURNS` / `IRIS_HISTORY_CHARS`, disable with
`IRIS_TRIM_HISTORY=0`).

## 5. CRAG (document research) — same server, zero extra VRAM

```bash
python main.py                     # interactive CRAG CLI
python main.py --ingest-dir ~/Documents/papers
```

All CRAG LLM calls (generator, grader, refiner, rewriter) now go to the same
llama-server at temperature 0/0.7 — `langchain_ollama`/Ollama are only
imported if you explicitly set `CRAG_BACKEND=ollama` (the original
experimental stack; do not run it while llama-server holds the GPU).
Embedding model bge-m3 runs on **CPU by default** so it cannot evict the LLM;
set `CRAG_EMBED_DEVICE=cuda` only for standalone sessions with the LLM
server stopped. On startup `main.py` warns if llama-server is down instead of
starting anything itself.

The episodic store moved from `iris_memory/memory.db` to
`iris_memory/temporal_memory.db` (same filename as the personal memory store
caused confusion). The first run migrates automatically — the old file is
copied, never modified or deleted.

## 6. Verify everything (in this order)

```bash
python selftest.py                          # 67 offline checks — safe anywhere
python test_streaming.py                    # 49 streaming checks incl. real-socket proof
python test_integration.py                  # live: PASS/FAIL/SKIP always labeled
python bench_retrieval.py --fake            # metric machinery (offline)
python bench_retrieval.py                   # REAL embedding quality (needs server)
python bench_retrieval.py --compare         # llama-server vs bge-small on CPU
python bench_first_token.py                 # TTFT + chars/sec (needs server)
./monitor_gpu.sh 1 60 > vram.csv &          # VRAM/RAM while you benchmark
python tavily_tool.py --smoke "latest stable python version"
python tool_runtime.py --demo "search for the newest nvidia driver and tell me briefly"
python memory_store.py stats
```

Everything that ran on the build machine is reported in `AUDIT_REPORT.md`
with its actual numbers; everything needing your GPU/server is SKIP-labeled
by the scripts themselves, not by me.

## 7. Configuration reference

| env / .env key | default | what it does |
|---|---|---|
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8080` | single model server both stacks call |
| `LLM_MODEL` | `Qwen3.5-4B` | model name field (server ignores it; match `/v1/models` if picky) |
| `TAVILY_API_KEY` | — | keyed search; keyless fallback otherwise |
| `EMBED_BACKEND` | `llamaserver` | `sbert` = sentence-transformers on CPU (comparison) |
| `EMBED_SBERT_MODEL` | `BAAI/bge-small-en-v1.5` | model for the sbert backend |
| `IRIS_HISTORY_TURNS` / `IRIS_HISTORY_CHARS` | 24 / 24000 | conversation bound in `chat()` |
| `IRIS_TRIM_HISTORY` | 1 | 0 disables auto-trimming |
| `IRIS_RETRY_CONNECT` | 1 | bounded connection retries |
| `IRIS_TIMEOUT_CONNECT` / `IRIS_TIMEOUT_READ` | 5 / 180 | connect + inter-byte silence timeouts |
| `IRIS_STREAM_USAGE` | 0 | request token usage in the SSE stream (newer builds) |
| `CRAG_BACKEND` | `llamaserver` | `ollama` = original experimental stack (opt-in) |
| `CRAG_TEMPERATURE` / `CRAG_MAX_TOKENS` | 0.7 / 1024 | CRAG generation settings |
| `CRAG_EMBED_DEVICE` | `cpu` | `cuda` only with the LLM server stopped |
| `IRIS_TEMPORAL_DB` | `iris_memory/temporal_memory.db` | CRAG episodic store path |
| `COSYVOICE_ROOT` / `COSYVOICE_MODEL` | written by `setup_voice.py` | TTS engine + weights |

## 8. Memory map (who owns what)

| store | owner | responsibility |
|---|---|---|
| `memory.db` | `memory_store.py` | **authoritative** durable personal memory: facts, preferences, imported history. Read by the voice runtime; written by `memory_add` + `memory_import.py`. |
| `iris_memory/temporal_memory.db` | `temporal_memory.py` | CRAG episodic turn log (timestamped Q/A with verdicts and sources). |
| `visual_memory/*.png` | `temporal_memory.py` | human-readable archive of the same turns. |
| `crag_vector_store/` | `retrieval.py` | chunked document index for CRAG retrieval. |

No component writes another component's store; all paths are resolved from
the module directory (never the CWD), so one `memory.db` can no longer appear
in random working directories.

## 9. Troubleshooting

- **llama-server not reachable** — `./start_llama.sh` prints why (missing
  binary, early exit code, health timeout). `python tool_runtime.py --health`
  for a quick probe.
- **embeddings 404** — server started without `--embeddings` (or an old
  build — the launcher warns). Memories store fine; keyword search works;
  run `python memory_store.py backfill` after restarting with the flag.
- **retrieval quality poor** — run `python bench_retrieval.py`; if
  recall@3 < 0.5, try `--pooling mean` on llama-server or
  `EMBED_BACKEND=sbert`, then re-benchmark.
- **tool calls never fire** — server must run with `--jinja`; check
  `python tool_runtime.py --demo ...`; malformed-argument rejections are
  visible in the trace (`"malformed tool arguments"`).
- **CRAG says backend unreachable** — start `./start_llama.sh`; CRAG never
  starts Ollama itself.
- **voice silent** — `voice_cache.get_cosyvoice()` returned `None`; the
  stderr message names the cause (run `setup_voice.py` / `enroll_voice.py`).
  Text IRIS keeps working.
- **`pynini` text-normalization errors** — `conda install -c conda-forge pynini`.
- **download interrupted** — re-run `python setup_voice.py`; completed files
  are skipped, `.part` files resume.
- **keyless rate limit** — put your key in `.env`.

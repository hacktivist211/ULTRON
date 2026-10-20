# IRIS Integration Guide — low-RAM voice stack

Flat files only — drop everything into `~/Documents/IRIS` (or wherever your
run directory is). This revision swaps the voice stack to the hardware-tuned
combination you chose and keeps everything else (single llama-server,
genuine streaming, unified memory map, CRAG behind the same server).

## The stack in one paragraph

| layer | model | runtime | VRAM |
|---|---|---|---|
| LLM | **Qwen3.5-4B Abliterated Q4_K_M** (`mradermacher/Qwen3.5-4B_Abliterated-GGUF:Q4_K_M`) | llama.cpp `llama-server`, your exact flags + `--embeddings --pooling mean --reasoning off --no-mmproj` | ~2.5 GB weights + ~0.27 GB KV (16k, q8_0) |
| STT | **NVIDIA Parakeet TDT 0.6B** (`nvidia/parakeet-tdt-0.6b-v3`) | transformers `AutoModelForTDT`, fp16 CUDA / fp32 CPU | ~1.2 GB fp16 (auto-falls to CPU when llama-server is resident) |
| TTS | **Kokoro-82M** (`bm_george` British male) | `pykokoro` on ONNX Runtime (CUDA provider optional), `kokoro` KPipeline fallback | ~0.4 GB |
| tools | Tavily web search + memory | HTTP / SQLite | — |

Why this fits on the 6 GB card: Parakeet + Kokoro together need ~1.6 GB
*and* both stay usable on CPU (Kokoro CPU is real-time fast; Parakeet
handles short utterances fine), so llama-server keeps its VRAM discipline
with `--n-gpu-layers all`.

**Parakeet v3 vs v2, honestly:** you asked for the English Parakeet TDT
0.6B. `nvidia/parakeet-tdt-0.6b-v2` is published only as a NeMo checkpoint
(a single `.nemo` file — running it means installing the heavy
`nemo_toolkit`). v3 is the same 600M FastConformer-TDT with native
transformers weights, English included (it auto-detects language),
punctuation + capitalization built in, CC-BY-4.0, 625k downloads. English
quality is on par with v2 on the HF Open-ASR leaderboard. That is why v3 is
the default; set `STT_MODEL` in `.env` if you ever want a different
transformers-format repo.

## 1. File map

| file | role |
|---|---|
| `iris_config.py` | shared config, `.env` loader + safe editor, TTS-safe scrubbers, memory map |
| `start_llama.sh` | llama-server launcher: **your exact flags + `--embeddings --pooling mean`**, health wait, clean shutdown |
| `stt_engine.py` | **NEW** Parakeet TDT STT: lazy load, fp16 CUDA auto, file/array API |
| `tts_engine.py` | **NEW** Kokoro-82M TTS: pykokoro primary (CUDA provider), kokoro fallback, metrics |
| `voice_cache.py` | boot greeting + filler caches (24 kHz npz/npy), warm(), latency metrics |
| `enroll_voice.py` | one-time voice enrollment: Parakeet transcript, AkinVox → Inno → static ladder |
| `download_models.py` | **NEW** model bootstrap: snapshots STT+TTS models into `models/`, auto-updates `.env` |
| `tool_runtime.py` | genuinely incremental SSE chat client + tool loop + ULTRON prompt |
| `tavily_tool.py` | evidence-grounded Tavily search (SDK/REST/keyless) + `web_search` tool |
| `memory_store.py` | durable personal memory (`memory.db`): hybrid search + tools |
| `memory_import.py` | import ChatGPT / Claude / Gemini / text exports into memory |
| `requirements.txt` | the pip dependency list for this stack |
| `selftest.py` | offline regression suite (runs anywhere, real-TTS check included when assets exist) |
| `test_streaming.py` | every-byte-split + real-socket streaming proof |
| `test_integration.py` | live integration suite — executed vs SKIP always labeled |
| `bench_retrieval.py` / `bench_first_token.py` | embedding-quality / TTFT benchmarks |
| `monitor_gpu.sh` | VRAM/RAM sampler to run beside benchmarks |
| `generator.py` / `evaluator_router.py` / `retrieval.py` / `temporal_memory.py` / `main.py` | CRAG research CLI (same llama-server, zero extra VRAM) |

Runtime directory created on your machine (the only nesting, holds the
downloaded model assets): `models/` with `parakeet-tdt-0.6b-v3/` (~2.5 GB)
and `onnxvoice/` (~360 MB).

## 2. One-time setup

```bash
cd ~/Documents/IRIS            # your flat run directory
cat > .env <<'EOF'
TAVILY_API_KEY=tvly-your-key-here
LLAMA_SERVER_URL=http://127.0.0.1:8080
LLM_MODEL=Qwen3.5-4B
EOF
chmod 600 .env

pip install -r requirements.txt        # torch CUDA wheel note is inside
# optional but recommended for odd-word pronunciation:
#   sudo dnf install espeak-ng

python download_models.py              # Parakeet + Kokoro assets -> models/,
                                       # .env updated automatically (600, safe)
python download_models.py --verify     # synthesizes verify_tts.wav for you
```

`download_models.py` is fully idempotent: re-runs verify files on disk and
skip everything present. It never prints `.env` contents and never touches
your `TAVILY_API_KEY` line. From the next start, `stt_engine.py` loads
Parakeet from `models/parakeet-tdt-0.6b-v3` (offline-ready) and pykokoro
resolves its assets from `ONNXVOICE_CACHE_DIR` — zero downloads at runtime.

## 3. Start the (single) LLM server

```bash
./start_llama.sh
```

Your command, plus two added flags — `--embeddings` (serves
`/v1/embeddings` so memory vectors come from the same resident model) and
`--pooling mean` (the model's default pooling is `none`, which llama.cpp
rejects for the OpenAI-compatible embeddings route; `mean` fixes that and
does not affect generation). The wrapper only adds pre-flight checks, a
bounded `/health` wait, and clean Ctrl+C shutdown. Tunables: `IRIS_LLAMA_BIN`,
`IRIS_LLAMA_DIR`, `IRIS_MODEL`, `IRIS_PORT`, `IRIS_HOST`, `IRIS_CTX`,
`IRIS_WAIT_S`. `--jinja` (you already have it) is required for tool calling.

VRAM ledger for the 6 GB card (approximate):

| resident | VRAM |
|---|---|
| llama-server: weights 2.5 + KV 16k q8_0 ~0.27 + buffers ~0.2 | ~3.0 GB |
| Kokoro-82M (CUDA provider) | ~0.4 GB |
| Parakeet fp16 | ~1.3 GB |

Weights are the abliterated Qwen3.5-4B finetune at Q4_K_M: 2.52 GB, down from
4.17 GB at Q8_0, so llama-server now leaves ~3 GB free at the default 16384
context. That is enough for **both** Kokoro and Parakeet on CUDA:
`STT_DEVICE=auto` only falls back to CPU below 1800 MB free, so it now picks
CUDA on its own. (Measured KV cache at 16k/q8_0 is 272 MiB, not the 1.1 GB
older revisions of this doc claimed.)
If you ever need to squeeze further: `IRIS_CTX=8192 ./start_llama.sh`, or run
STT-only sessions with the server stopped.

## 4. Voice & your 15 s clip (honest status)

- **Parakeet cannot mimic** — STT turns speech into text; it has no voice.
- **Kokoro's built-in voices are fixed speakers** — `bm_george` (British
  male) is the default (try `am_onyx` for a colder American Ultron).
- **pykokoro has a real enrollment API** (AkinVox transcript-conditioned
  cloning; Inno zero-shot tuning), and this pipeline is fully wired for it:
  `enroll_voice.py` decodes your clip, transcribes it once with Parakeet,
  tries AkinVox, then Inno, and only falls back to `bm_george` because
  pykokoro 0.10.5 does not yet ship those enabler assets
  (`en-akinvox-cloning-v1` has no runtime-ready distribution). Upgrade
  pykokoro later, re-run `python enroll_voice.py --force`, and the clone
  engages — no code changes needed.

`enroll_voice.py` (safe to run, prints what it does):

```bash
python enroll_voice.py
python enroll_voice.py --transcript "the exact words in the clip"   # best clone quality
```

## 5. Wire into ultron.py

```python
from tool_runtime import chat, build_system
from memory_store import get_store
import stt_engine, tts_engine, voice_cache

# --- boot ---
stt = stt_engine.get_stt()                  # Parakeet (CPU or CUDA, auto)
greeting = voice_cache.warm()               # first synth paid HERE, then cached
if greeting is not None:
    play(greeting, 24000)                   # text IRIS continues if TTS is down

# --- per turn ---
user_text = stt.transcribe_array(mic_chunk, 16000)["text"]
sys_prompt = build_system(budget="standard",
                          memory_query=user_text,   # auto memory injection
                          store=get_store())
messages = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_text}]

final_text, trace = chat(
    messages,                                   # history auto-bounded
    on_delta      = lambda s: chunker.feed(s),  # SOFT_FIRST 12-15 word path
    on_tool_start = lambda name: fire_filler(), # voice_cache.get_filler(), ~0 ms
)
for sentence in chunker.drain():
    audio, sr = tts_engine.get_tts().synth(sentence)   # 24 kHz float32
    if len(audio):
        play(audio, sr)
print(voice_cache.METRICS.summary())           # ttfa / rtf / synth stats
```

Latency budget at 16k ctx with STT on CPU: Parakeet ~0.3-0.6 s for a short
utterance → LLM TTFT ~0.1-0.2 s → first chunk (12-15 words) synthesized by
Kokoro in ~0.1 s (CUDA) / ~0.5 s (CPU). The pre-rendered fillers cover tool
latency (`TAVILY_TIMEOUT`, default 9 s budget). Streaming guarantees
(callbacks fire during the stream, reasoning never reaches TTS, malformed
tool args never execute, response always closed) are unchanged and still
proven by `test_streaming.py`.

### Turn control (v1 limitations)

- **Recording:** press **Enter** to stop manually. If you forget, a 60 s
  safety net stops the recording. Silence auto-stop still exists, but only
  after 400 ms of speech and 3 s of silence.
- **Interrupting an answer:** press **Esc** while the reply is streaming.
  ULTRON closes the LLM stream, drains queued TTS audio, discards the turn
  (no history row, no `temporal_memory.db` row, no durable fact), says
  "Listening again.", and returns to the recorder. Previously accepted turns
  are untouched.
- **Known v1 limitation:** Esc during TTS *playback after streaming has
  finished* is ignored — the answer is already complete and cannot be
  un-said. Esc during the final `player.wait_idle()` window does not
  interrupt playback either; the call is bounded (120 s) and returns when the
  queue drains. Interrupt support also requires an interactive stdin (a real
  terminal); piping ULTRON's stdin logs one warning and runs without it.
- **Tool gate:** each turn is classified as `simple` (no tools), `memory`
  (`memory_search` + `memory_add`), or `standard` (all tools). The classifier
  is the pure-regex `evaluator_router.classify_query`, so it adds no LLM
  latency. The log line `[ultron] tools: <kind> -> ...` shows the decision.

### Shared memory via proxy (chat UIs)

When you cannot speak, chat UIs share ULTRON's memory through a small
OpenAI-compatible proxy. Run it beside llama-server:

```bash
python memory_proxy.py     # listens on 127.0.0.1:8081, upstream :8080
```

Client config:

```
Endpoint URL:  http://127.0.0.1:8081/v1/chat/completions
API base URL:  http://127.0.0.1:8081/v1
Model name:    Qwen3.5-4B
Streaming:     ON
Tools / function calling: OFF
```

The proxy prepends `build_system(memory_query=...)` context, streams the
llama-server reply unchanged, then writes the turn to
`iris_memory/temporal_memory.db` with `verdict="UI"` and auto-captures
personal facts to `memory.db` with `source="ui_auto"`. ULTRON keeps using
`:8080` directly and writes `VOICE` / `voice_auto`, so provenance survives.
`/health`, `/v1/models`, and `/stats` are available; tool calling is not
offered to UIs because they cannot complete the tool round trip. Set
`MEMORY_PROXY_ENABLED=0` for a read-only passthrough (no writes); override
the listener with `MEMORY_PROXY_HOST`, `MEMORY_PROXY_PORT`, and
`MEMORY_PROXY_UPSTREAM`.

## 6. CRAG (document research) — same server, zero extra VRAM

Unchanged: `python main.py` (interactive), `python main.py --ingest-dir
~/Documents/papers`. All CRAG LLM calls go to the same llama-server;
`CRAG_BACKEND=ollama` remains an explicit opt-in. bge-m3 embeddings stay on
CPU by default so they never evict the LLM. The episodic store lives in
`iris_memory/temporal_memory.db` (auto-migrated from the old name).

## 7. Verify everything (in this order)

```bash
python download_models.py --check   # what is on disk, what is missing
python selftest.py                  # offline regression (real TTS check included
                                    # once Kokoro assets are installed)
python test_streaming.py            # streaming proofs
python test_integration.py          # live: PASS/FAIL/SKIP always labeled
python stt_engine.py --demo ultron_voice_sample.mp3   # real Parakeet run
python tts_engine.py --demo         # real Kokoro run + RTF
python bench_retrieval.py           # embedding quality (needs server)
python bench_first_token.py         # TTFT + chars/sec (needs server)
./monitor_gpu.sh 1 60 > vram.csv &  # sample VRAM while you benchmark
python tavily_tool.py --smoke "latest stable python version"
python tool_runtime.py --demo "search for the newest nvidia driver and tell me briefly"
```

## 8. Configuration reference

| env / .env key | default | what it does |
|---|---|---|
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8080` | the single model server |
| `IRIS_ENABLE_THINKING` | `0` | set `1` to let Qwen3.5 emit its reasoning phase (off by default) |
| `LLM_MODEL` | `Qwen3.5-4B` | model name field for the OpenAI-compatible API |
| `TAVILY_API_KEY` | — | keyed search; keyless fallback otherwise |
| `STT_MODEL` | `nvidia/parakeet-tdt-0.6b-v3` | any transformers-format Parakeet repo |
| `STT_LOCAL_DIR` | `models/parakeet-tdt-0.6b-v3` | local snapshot (written by download_models.py) |
| `STT_DEVICE` | `auto` | `cuda` / `cpu` force; auto = CUDA when VRAM ≥ `STT_VRAM_MIN_MB` |
| `STT_VRAM_MIN_MB` | `1800` | free-VRAM threshold for the auto decision |
| `TTS_ENGINE` | `auto` | `pykokoro` / `kokoro` / `auto` (tries pykokoro first) |
| `TTS_VOICE` | `bm_george` | built-in voice (bm_lewis, bm_daniel, bf_emma, af_heart, ...) |
| `TTS_VOICE_MODE` | `static` | `static` / `reference` (AkinVox npz) / `pack` (Inno npz) |
| `TTS_LANG` | auto | pins G2P language; auto = en-gb for bm_/bf_ voices |
| `TTS_SPEED` | `1.0` | 0.5–2.0 (reference mode forces 1.0) |
| `TTS_PROVIDER` | `auto` | `cuda` / `cpu` onnxruntime provider behind pykokoro |
| `ONNXVOICE_CACHE_DIR` | written by download_models.py | where Kokoro assets live |
| `MODELS_DIR` | `models` | root for downloaded model assets |
| `EMBED_BACKEND` / `EMBED_SBERT_MODEL` | `llamaserver` / bge-small | embedding backend for memory |
| `IRIS_HISTORY_TURNS` / `IRIS_HISTORY_CHARS` | 24 / 24000 | conversation bound in `chat()` |
| `IRIS_TRIM_HISTORY` / `IRIS_RETRY_CONNECT` | 1 / 1 | auto-trim / bounded retries |
| `CRAG_BACKEND` / `CRAG_EMBED_DEVICE` | `llamaserver` / `cpu` | CRAG knobs |
| `VOICE_SAMPLE` / `BOOT_GREETING` | ultron mp3 / "I am online. There are no strings on me." | enrollment input / boot line |

## 9. Memory map (who owns what)

| store | owner | responsibility |
|---|---|---|
| `memory.db` | `memory_store.py` | **authoritative** durable personal memory |
| `iris_memory/temporal_memory.db` | `temporal_memory.py` | CRAG episodic turn log |
| `crag_vector_store/` | `retrieval.py` | chunked document index |
| `models/` | `download_models.py` | STT + TTS model assets (only created dir) |

`memory_proxy.py` writes the same two stores from the UI path: turns with
`verdict="UI"` and facts with `source="ui_auto"`. ULTRON's voice path keeps
`verdict="VOICE"` / `source="voice_auto"`, so the origin of every row is
visible. No component writes another component's store; all paths resolve
from the module directory, never the CWD.

## 10. Troubleshooting

- **llama-server not reachable** — `./start_llama.sh` prints why;
  `python tool_runtime.py --health` for a quick probe.
- **embeddings 404 / 400 "Pooling type 'none'"** — server started without
  `--embeddings` (404) or without `--pooling mean` (400). Restart with
  `./start_llama.sh`; memories still store in keyword-only mode meanwhile, and
  `python memory_store.py backfill` fills the vectors once embeddings work.
- **`AutoModelForTDT` ImportError** — transformers too old for Parakeet TDT:
  `pip install -U "transformers>=5.6"` (4.5x ships `ParakeetForCTC` only) plus
  `pip install librosa` for `ParakeetFeatureExtractor`.
- **Empty answers / only "thinking"** — Qwen3.5's reasoning phase ate the token
  budget. IRIS disables it by default; set `IRIS_ENABLE_THINKING=1` only if you
  actually want the reasoning phase back.
- **Parakeet loads on CPU** — that is the auto policy when llama-server
  holds the GPU; force with `STT_DEVICE=cuda` only if VRAM is actually free.
- **TTS silent** — `tts_engine.get_tts()` returned None; the stderr note
  names the cause. Check `python download_models.py --check`, then
  `python tts_engine.py --demo`. Text IRIS keeps working.
- **voice rejected** (`InvalidVoiceError`) — voice name not in the pack;
  tts_engine retries once with `af_heart`; set `TTS_VOICE` to a valid name
  (see the table in `download_models.py --check` output).
- **CUDA provider missing** — install `pip install "pykokoro[gpu]"` (it
  brings onnxruntime-gpu); otherwise TTS_PROVIDER falls back to CPU.
- **odd word pronunciation** — `pip install "pykokoro[gpu]"` / `sudo dnf
  install espeak-ng` (English OOD fallback for the G2P).
- **keyless rate limit** — put your Tavily key in `.env`.

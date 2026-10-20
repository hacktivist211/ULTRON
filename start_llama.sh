#!/usr/bin/env bash
# IRIS llama-server launcher - Qwen3.5-4B via llama.cpp (never Ollama).
#
# This is YOUR exact working command with the flags below added:
#   --embeddings   -> serves /v1/embeddings so memory_store.py can embed
#                     memories through the same model (cheap: your
#                     --n-gpu-layers all setup already keeps it resident).
#   --pooling mean -> the model's default pooling is 'none', which llama.cpp
#                     rejects for the OpenAI-compatible /v1/embeddings route
#                     ("Pooling type 'none' is not OAI compatible"). 'mean'
#                     makes that endpoint work; it does not affect generation.
#   --reasoning off -> server-side default: Qwen3.5's thinking phase stays OFF
#                     for every request, so replies start speaking immediately
#                     instead of after hundreds of hidden reasoning tokens.
#                     (Request-level chat_template_kwargs in tool_runtime.py /
#                     evaluator_router.py are a second, independent belt.)
#
#   --no-mmproj  -> do NOT load the vision/CLIP projector. IRIS never sends
#                     images to the LLM (no image_url payloads anywhere; the
#                     --vision OCR flag in main.py is never read), but Qwen3.5
#                     is a VL model, so llama.cpp otherwise auto-loads a
#                     644 MiB projector AND warms it up with a dummy 1472x1472
#                     image. Total cost: ~900 MiB of VRAM out of 6 GB.
#                     Without this the server sits at ~104 MiB free and dies
#                     with "CUDA error: out of memory" in
#                     ggml_cuda_graph_evaluate_and_capture / cudaGraphInstantiate
#                     as soon as a new graph shape appears (measured: ~120 MiB
#                     more is needed as shapes accumulate). With it: ~1 GB free.
#                     Re-enable only if you start sending images to the model.
#
# Everything else is byte-for-byte your command. Weights come from the
# Hugging Face cache model downloaded by `-hf`; on a first run with a repo or
# quant you have never pulled, llama.cpp fetches it once (~2.5 GB for Q4_K_M)
# and then reuses it.
#
# Weights: mradermacher/Qwen3.5-4B_Abliterated-GGUF Q4_K_M (2.52 GiB).
# The abliterated (refusal-ablated) finetune of Qwen3.5-4B, same architecture,
# same chat template and same tool-calling markers, so --jinja tools,
# --reasoning off and --embeddings all behave exactly as before.
# Q4_K_M instead of the old Q8_0 drops resident VRAM from ~4.26 GiB to
# ~2.5 GiB, which is the main reason the 6 GB card now fits STT on CUDA as
# well (STT needs >= 1800 MiB free). The old unsloth Q8/Q4 cache was deleted.
#   --cache-type-k/v q8_0 below are the KV-CACHE types and are unchanged.
#
# This repo also ships mmproj-*.gguf files (it is a VL model too), so
# --no-mmproj below matters here: it prevents llama.cpp from downloading and
# loading a ~650 MB vision projector that IRIS never uses.
#
# What this wrapper adds around the server itself:
#   - pre-flight: binary exists, port not already serving, ~free VRAM shown
#   - waits up to 120 s for /health, then prints READY + endpoints
#   - Ctrl+C / SIGTERM: server shut down cleanly, exit code reported
#   - tunables via environment variables (defaults = your values):
#       IRIS_LLAMA_BIN  path to llama-server     (~/llama.cpp/build/bin/llama-server)
#       IRIS_LLAMA_DIR  llama.cpp checkout       (~/llama.cpp)
#       IRIS_MODEL      -hf model repo            (mradermacher/Qwen3.5-4B_Abliterated-GGUF:Q4_K_M)
#       IRIS_PORT       listen port               (8080)
#       IRIS_HOST       bind address              (127.0.0.1)
#       IRIS_CTX        context size              (16384)
#       IRIS_WAIT_S     health-wait timeout       (120)
#
# Memory tool calls need --jinja (you already have it).
# If VRAM ever gets tight with the STT/TTS models beside it, try adding:
#   --ot per_layer_token_embd.weight=CPU
# (offloads the token embedding layer to RAM, ~300 MB freed), or set
# STT_DEVICE=cpu / TTS_PROVIDER=cpu in .env - Kokoro on CPU is real-time fast
# and Parakeet tolerates CPU fine for short utterances.

set -u

LLAMA_DIR="${IRIS_LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_BIN="${IRIS_LLAMA_BIN:-$LLAMA_DIR/build/bin/llama-server}"
MODEL="${IRIS_MODEL:-mradermacher/Qwen3.5-4B_Abliterated-GGUF:Q4_K_M}"
PORT="${IRIS_PORT:-8080}"
HOST_="${IRIS_HOST:-127.0.0.1}"
CTX="${IRIS_CTX:-16384}"
WAIT_S="${IRIS_WAIT_S:-120}"

say() { printf '[llama] %s\n' "$*"; }
die() { printf '[llama] ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- pre-flight
[ -x "$LLAMA_BIN" ] || die "llama-server not found at $LLAMA_BIN
       build it first:  cd $LLAMA_DIR && cmake -B build && cmake --build build -j
       or override:     IRIS_LLAMA_BIN=/path/to/llama-server ./start_llama.sh"

if command -v curl >/dev/null 2>&1; then
    if curl -fsS -m 3 "http://$HOST_:$PORT/health" >/dev/null 2>&1; then
        say "a server is ALREADY healthy on $HOST_:$PORT - nothing to do."
        exit 0
    fi
elif command -v python3 >/dev/null 2>&1; then
    if python3 - "$HOST_" "$PORT" <<'PY' 2>/dev/null
import sys, urllib.request
try:
    urllib.request.urlopen(f"http://{sys.argv[1]}:{sys.argv[2]}/health", timeout=3)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    then
        say "a server is ALREADY healthy on $HOST_:$PORT - nothing to do."
        exit 0
    fi
fi

# Flag support probe (informational; old builds keep working, embeddings won't)
"$LLAMA_BIN" --help 2>&1 | grep -q -- '--embeddings' \
    || say "WARNING: this llama-server build does not list --embeddings;
       memory vectors will be unavailable (keyword search still works).
       Rebuild llama.cpp from a recent checkout if you want embeddings."

if command -v nvidia-smi >/dev/null 2>&1; then
    say "GPU status before start:"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>/dev/null | sed 's/^/       /' || true
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "${FREE_MB:-}" ] && [ "${FREE_MB}" -lt 1500 ] 2>/dev/null; then
        say "WARNING: only ${FREE_MB} MB VRAM free - another model may be resident.
       The server may fail to load all layers; close GPU apps or retry later."
    fi
fi

# ---------------------------------------------------------------- start + health wait
say "starting llama-server: $MODEL  (ctx $CTX, host $HOST_:$PORT)"
"$LLAMA_BIN" \
  -hf "$MODEL" \
  --no-mmproj \
  --jinja \
  --embeddings \
  --pooling mean \
  --reasoning off \
  --ctx-size "$CTX" \
  --n-gpu-layers all \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --batch-size 512 \
  --ubatch-size 128 \
  --threads 8 \
  --threads-batch 8 \
  --parallel 1 \
  --host "$HOST_" \
  --port "$PORT" &
SERVER_PID=$!

cleanup() {
    say "shutting down (pid $SERVER_PID) ..."
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    say "stopped."
}
trap cleanup INT TERM

t0=$SECONDS
READY=0
while [ $((SECONDS - t0)) -le "$WAIT_S" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        wait "$SERVER_PID" 2>/dev/null; RC=$?
        die "llama-server exited early (code $RC). Check the output above:
       - unknown flag?      compare '$LLAMA_BIN --help'
       - CUDA/Vulkan error? check nvidia-smi and free VRAM
       - model download?    -hf fetches on first use; retry once online"
    fi
    if command -v curl >/dev/null 2>&1; then
        curl -fsS -m 2 "http://$HOST_:$PORT/health" >/dev/null 2>&1 && READY=1
    else
        python3 - "$HOST_" "$PORT" <<'PY' 2>/dev/null && READY=1
import sys, urllib.request
urllib.request.urlopen(f"http://{sys.argv[1]}:{sys.argv[2]}/health", timeout=2)
PY
    fi
    [ "$READY" = "1" ] && break
    sleep 1
done

if [ "$READY" != "1" ]; then
    say "server did not become healthy within ${WAIT_S}s (large model loads take a while)."
    say "keeping it running - check http://$HOST_:$PORT/health in another terminal."
else
    say "READY in $((SECONDS - t0)) s  ->  http://$HOST_:$PORT"
    say "endpoints: /v1/chat/completions (tools)   /v1/embeddings (memory)   /health"
fi

# Foreground behavior: block until the server exits (Ctrl+C -> cleanup)
wait "$SERVER_PID"
RC=$?
say "llama-server exited (code $RC)."
exit $RC

#!/usr/bin/env bash
# IRIS llama-server launcher - Qwen3.5-4B via llama.cpp (never Ollama).
#
# This is YOUR exact working command with exactly ONE flag added:
#   --embeddings   -> serves /v1/embeddings so memory_store.py can embed
#                     memories through the same model (cheap: your
#                     --n-gpu-layers all setup already keeps it resident).
#
# Everything else is byte-for-byte your command. The model itself stays in
# llama.cpp's -hf cache (~/.cache/llama.cpp) - nothing is re-downloaded.
#
# What this wrapper adds around the server itself:
#   - pre-flight: binary exists, port not already serving, ~free VRAM shown
#   - waits up to 120 s for /health, then prints READY + endpoints
#   - Ctrl+C / SIGTERM: server shut down cleanly, exit code reported
#   - tunables via environment variables (defaults = your values):
#       IRIS_LLAMA_BIN  path to llama-server     (~/llama.cpp/build/bin/llama-server)
#       IRIS_LLAMA_DIR  llama.cpp checkout       (~/llama.cpp)
#       IRIS_MODEL      -hf model repo            (unsloth/Qwen3.5-4B-GGUF:Q8_0)
#       IRIS_PORT       listen port               (8080)
#       IRIS_HOST       bind address              (127.0.0.1)
#       IRIS_CTX        context size              (16384)
#       IRIS_WAIT_S     health-wait timeout       (120)
#
# Memory tool calls need --jinja (you already have it).
# If VRAM ever gets tight with CosyVoice running beside it, try adding:
#   --ot per_layer_token_embd.weight=CPU
# (offloads the token embedding layer to RAM, ~300 MB freed).

set -u

LLAMA_DIR="${IRIS_LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_BIN="${IRIS_LLAMA_BIN:-$LLAMA_DIR/build/bin/llama-server}"
MODEL="${IRIS_MODEL:-unsloth/Qwen3.5-4B-GGUF:Q8_0}"
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
  --jinja \
  --embeddings \
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

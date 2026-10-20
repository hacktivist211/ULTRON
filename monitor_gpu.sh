#!/usr/bin/env bash
# GPU / VRAM / system-RAM sampler for ULTRON benchmarks (Critical Issue 7/10).
#
#   ./monitor_gpu.sh              # 1 s interval, run until Ctrl+C
#   ./monitor_gpu.sh 5            # sample every 5 s
#   ./monitor_gpu.sh 1 60 > vram.csv   # 1 s interval, 60 samples, CSV
#
# Useful while running bench_first_token.py or a real voice conversation:
# it shows whether Kokoro TTS + llama-server + embeddings actually fit in
# the 6 GB budget, or whether something is swapping/evicting.
#
# Output columns (CSV):
#   timestamp, gpu_mem_used_mb, gpu_mem_total_mb, gpu_util_pct,
#   ram_used_mb, ram_total_mb, per-process GPU memory (top consumers)

set -u

INTERVAL="${1:-1}"
COUNT="${2:-0}"   # 0 = unlimited
i=0

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found - is the NVIDIA driver installed?" >&2
    exit 1
fi

echo "timestamp,gpu_mem_used_mb,gpu_mem_total_mb,gpu_util_pct,ram_used_mb,ram_total_mb,gpu_processes"

while true; do
    ts=$(date +%H:%M:%S)
    line=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -1)
    used=$(echo "$line" | cut -d, -f1 | tr -d ' ')
    total=$(echo "$line" | cut -d, -f2 | tr -d ' ')
    util=$(echo "$line" | cut -d, -f3 | tr -d ' ')
    ram=$(free -m | awk '/^Mem:/ {print $3 "," $2}')
    procs=$(nvidia-smi --query-compute-apps=pid,used_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1 "=" $2 "MB"}' \
        | paste -sd ';' -)
    echo "$ts,$used,$total,$util,$ram,\"$procs\""
    i=$((i + 1))
    if [ "$COUNT" != "0" ] && [ "$i" -ge "$COUNT" ]; then
        break
    fi
    sleep "$INTERVAL"
done

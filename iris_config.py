#!/usr/bin/env python3
"""IRIS shared configuration.

Design rules for this project:
  - FLAT files only: every artifact this module and its siblings create lives
    directly in the IRIS folder (the directory containing this file). No nested
    directories are ever created.
  - Secrets live in a `.env` file next to this module. They are loaded into the
    environment, never printed, and never written into code or logs.
  - Every module resolves paths relative to this file, so you can drop the whole
    set into ~/Documents/IRIS and it just works.

Override anything with environment variables or the `.env` file:
  TAVILY_API_KEY=tvly-...            (required for Tavily web search)
  LLAMA_SERVER_URL=http://127.0.0.1:8080
  LLM_MODEL=Qwen3.5-4B               (name field for the OpenAI-compatible API;
                                      the real model is served by llama.cpp via
                                      start_llama.sh: unsloth/Qwen3.5-4B-GGUF:Q8_0)
  EMBED_MODEL=Qwen3.5-4B              (embedding model name, if different)
  EMBED_ENDPOINT=...                 (defaults to llama-server /v1/embeddings)
  EMBED_BACKEND=llamaserver|sbert    (embeddings for memory_store; 'sbert' uses
                                      sentence-transformers on CPU, e.g. for
                                      quality comparison via bench_retrieval.py)
  EMBED_SBERT_MODEL=BAAI/bge-small-en-v1.5
  VOICE_SAMPLE=jarvis_alarm_15s031ms.mp3
  COSYVOICE_ROOT=/home/neeraj/Documents/IRIS/CosyVoice   (written by setup_voice.py)
  COSYVOICE_MODEL=/.../pretrained_models/CosyVoice2-0.5B (written by setup_voice.py)
  BOOT_GREETING="Systems online, sir."
  TAVILY_TIMEOUT=9.0                 (seconds, fillers cover this latency)
  TAVILY_INCLUDE_ANSWER=true         (Tavily's synthesized answer: great for speech)
  IRIS_HISTORY_TURNS=24              (conversation history bound: max messages)
  IRIS_HISTORY_CHARS=24000           (history bound: max total characters)
  IRIS_TRIM_HISTORY=1                (0 disables automatic trimming)
  IRIS_RETRY_CONNECT=1               (bounded connection retries in chat())
  IRIS_STREAM_USAGE=0                (1 = request token usage in the SSE stream;
                                      needs a llama.cpp build that supports it)

MEMORY MAP (who owns which store - all paths are module-relative, never CWD):
  memory.db               durable personal memory, owned by memory_store.py
                          (AUTHORITATIVE for user facts; used by the primary
                          voice runtime and memory_import.py)
  iris_memory/
    temporal_memory.db    episodic turn log, owned by temporal_memory.py in the
                          CRAG stack (legacy name iris_memory/memory.db is
                          migrated automatically, original kept as backup)
    visual_memory/*.png   human-readable visual archive of the same turns
  crag_vector_store/      ChromaDB document index, owned by retrieval.py in the
                          CRAG stack (your chunked documents, embeddings
                          BAAI/bge-m3, CPU by default)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

IRIS_DIR = Path(__file__).resolve().parent


def load_dotenv(path: str | Path | None = None, override: bool = False) -> int:
    """Tiny .env loader (no external dependency). Returns number of keys set."""
    p = Path(path) if path else IRIS_DIR / ".env"
    if not p.exists():
        return 0
    n = 0
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and (override or k not in os.environ):
            os.environ[k] = v
            n += 1
    return n


load_dotenv()

# ---------------------------------------------------------------- LLM (llama.cpp llama-server)
# The assistant talks pure HTTP to llama-server (never Ollama). Launch it with
# ./start_llama.sh - your Qwen3.5-4B Q8_0 flags + --jinja (tools) + --embeddings.
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL", "Qwen3.5-4B")
EMBED_ENDPOINT = os.environ.get("EMBED_ENDPOINT", LLAMA_SERVER_URL + "/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", LLM_MODEL_NAME)

# ---------------------------------------------------------------- Tavily
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
TAVILY_TIMEOUT = float(os.environ.get("TAVILY_TIMEOUT", "9.0"))
TAVILY_INCLUDE_ANSWER = os.environ.get("TAVILY_INCLUDE_ANSWER", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------- Voice / TTS
VOICE_SAMPLE = IRIS_DIR / os.environ.get("VOICE_SAMPLE", "jarvis_alarm_15s031ms.mp3")
VOICE_PROFILE = IRIS_DIR / "voice_profile.npz"          # one-time enrollment artifact
FILLER_LINES = [
    "Just a minute, sir.",
    "Processing, sir.",
    "Working on it, sir.",
]
BOOT_GREETING = os.environ.get("BOOT_GREETING", "Systems online, sir.")
TTS_SAMPLE_RATE = 24000                                  # CosyVoice2 native rate
COSYVOICE_FP16 = os.environ.get("COSYVOICE_FP16", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------- Spoken output budgets
WORD_BUDGETS = {"quick": 25, "standard": 60, "detailed": 150}

# ---------------------------------------------------------------- Memory
MEMORY_DB = IRIS_DIR / os.environ.get("MEMORY_DB", "memory.db")
EMBED_BATCH = 32

# ---------------------------------------------------------------- Streaming / history / retry

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


LLM_TIMEOUT = (
    _env_float("IRIS_TIMEOUT_CONNECT", 5.0),
    _env_float("IRIS_TIMEOUT_READ", 180.0),
)
HISTORY_TRIM = os.environ.get("IRIS_TRIM_HISTORY", "1").lower() not in ("0", "false", "off", "no")
try:
    HISTORY_TURNS = max(2, int(os.environ.get("IRIS_HISTORY_TURNS", "24")))
except ValueError:
    HISTORY_TURNS = 24
try:
    HISTORY_CHARS = max(1000, int(os.environ.get("IRIS_HISTORY_CHARS", "24000")))
except ValueError:
    HISTORY_CHARS = 24000
try:
    RETRY_CONNECT = max(0, int(os.environ.get("IRIS_RETRY_CONNECT", "1")))
except ValueError:
    RETRY_CONNECT = 1
STREAM_USAGE = os.environ.get("IRIS_STREAM_USAGE", "0").lower() in ("1", "true", "yes", "on")

# ---------------------------------------------------------------- Embedding backend
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "llamaserver").strip().lower()
EMBED_SBERT_MODEL = os.environ.get("EMBED_SBERT_MODEL", "BAAI/bge-small-en-v1.5")


def find_cosyvoice_root() -> Path | None:
    """Locate the CosyVoice repo checkout (importable as `cosyvoice`)."""
    candidates: list[Path] = []
    env_root = os.environ.get("COSYVOICE_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates += [
        IRIS_DIR / "CosyVoice",
        IRIS_DIR.parent / "CosyVoice",
        Path.home() / "CosyVoice",
        Path.home() / "Documents" / "CosyVoice",
        Path.home() / "Documents" / "IRIS" / "CosyVoice",
    ]
    for c in candidates:
        c = c.expanduser().resolve()
        if (c / "cosyvoice" / "cli" / "cosyvoice.py").exists():
            return c
    return None


def find_cosyvoice_model() -> Path | None:
    """Locate the CosyVoice2 model directory (contains campplus.onnx etc.)."""
    env_model = os.environ.get("COSYVOICE_MODEL")
    if env_model:
        p = Path(env_model).expanduser().resolve()
        return p if p.exists() else None
    root = find_cosyvoice_root()
    if root:
        for rel in ("pretrained_models/CosyVoice2-0.5B", "pretrained_models/CosyVoice-300M"):
            p = root / rel
            if p.exists():
                return p
    return None


# ---------------------------------------------------------------- TTS-safe text helpers
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MD_RE = re.compile(r"(\*\*|\*|__|_|`{1,3}|#{1,6}\s?|>+\s?|\||\[|\])")
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2190-\u21FF\u2600-\u27BF\uFE0F\u200D\u2B00-\u2BFF]"
)


def speech_clean(text: str, max_len: int = 900) -> str:
    """Scrub text so CosyVoice2 never trips on markdown, URLs, or emoji.

    Used everywhere text is about to be spoken: tool results, memory context,
    and any raw text destined for the speaker.
    """
    t = str(text or "")
    t = _URL_RE.sub(" ", t)
    t = _MD_RE.sub("", t)
    t = _EMOJI_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_len:
        t = t[:max_len].rsplit(" ", 1)[0] + "..."
    return t


def clip_words(text: str, budget: str | int = "standard") -> str:
    """Trim to a word budget ('quick' 25 / 'standard' 60 / 'detailed' 150)."""
    n = WORD_BUDGETS.get(budget, 60) if isinstance(budget, str) else int(budget)
    words = text.split()
    if len(words) <= n:
        return text
    return " ".join(words[:n]).rstrip(" ,;:") + "."


def human_date(ts: float | None) -> str:
    """Compact date for memory context lines."""
    if not ts:
        return "unknown date"
    import time
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def profile_summary() -> dict:
    """Snapshot of key config for logging (secrets are never included)."""
    return {
        "iris_dir": str(IRIS_DIR),
        "llama_server": LLAMA_SERVER_URL,
        "embed_endpoint": EMBED_ENDPOINT,
        "tavily_key_present": bool(TAVILY_API_KEY),
        "voice_sample": VOICE_SAMPLE.name if VOICE_SAMPLE.exists() else "MISSING",
        "voice_profile_present": VOICE_PROFILE.exists(),
        "cosyvoice_root": str(find_cosyvoice_root()) if find_cosyvoice_root() else "NOT FOUND",
        "cosyvoice_model": str(find_cosyvoice_model()) if find_cosyvoice_model() else "NOT FOUND",
        "memory_db": MEMORY_DB.name,
    }


if __name__ == "__main__":
    print(json.dumps(profile_summary(), indent=2))
    print("word budgets:", WORD_BUDGETS)

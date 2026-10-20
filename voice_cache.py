#!/usr/bin/env python3
"""Runtime voice cache for ULTRON - instant boot voice + zero-latency fillers.

The TTS engine itself is Kokoro-82M (tts_engine.py: pykokoro primary, the
official kokoro KPipeline as fallback). This module caches its two
latency-critical outputs on disk so startup never pays for them:

    boot_greeting.npy   spoken boot line, rendered once
    fillers.npz         f1..f3 + texts - the dead-air mask while tools run
    filler_1..3.npy     same fillers as loose files (drop-in for your player)

They are pre-rendered by enroll_voice.py and re-created automatically by
warm() whenever they are missing. All arrays are float32 at the native
Kokoro rate of 24 kHz, so any standard 24 kHz player works unchanged.

Startup (inside ultron.py):

    import voice_cache
    greeting = voice_cache.warm()               # np array or None; play it
    ...
    audio, sr = voice_cache.get_filler()        # on tool start, ~0 ms
    audio, sr = voice_cache.timed_synthesis(chunk)   # measured reply path
    print(voice_cache.METRICS.summary())        # ttfa / synth / rtf stats
"""
from __future__ import annotations

import sys
from pathlib import Path

import ultron_config as cfg
from tts_engine import METRICS, SAMPLE_RATE, get_tts  # re-exported for callers


def _cache_path(name: str) -> Path:
    return cfg.ULTRON_DIR / name


def warm(line: str | None = None, force: bool = False):
    """Return the boot greeting audio (np.float32 @ 24 kHz), rendering once.

    Also fills in any missing filler lines. Doubles as the TTS warm-up: the
    first (slowest) synthesis happens HERE, at startup, where it cannot hurt
    the first user query. Returns None when TTS is unavailable - the caller
    keeps going text-only.
    """
    import numpy as np

    tts = get_tts()
    if tts is None:
        return None
    greeting_path = _cache_path("boot_greeting.npy")
    text = line or cfg.BOOT_GREETING
    if greeting_path.exists() and not force:
        audio = np.load(str(greeting_path)).astype(np.float32)
    else:
        audio, _sr = tts.synth(text)
        if len(audio) == 0:
            return None
        np.save(str(greeting_path), audio)
    _ensure_fillers(tts)
    return audio


def _ensure_fillers(tts) -> None:
    """Render any missing filler line; persist fillers.npz + loose .npy files."""
    import json

    import numpy as np

    p = _cache_path("fillers.npz")
    have: dict = {}
    if p.exists():
        try:
            have = {k: np.asarray(v, dtype=np.float32)
                    for k, v in np.load(str(p)).items() if k != "texts"}
        except Exception:
            have = {}
    changed = False
    for i in range(1, len(cfg.FILLER_LINES) + 1):
        key = f"f{i}"
        if key in have and len(have[key]) > 0:
            continue
        audio, _sr = tts.synth(cfg.FILLER_LINES[i - 1])
        if len(audio) == 0:
            continue
        have[key] = audio
        np.save(str(_cache_path(f"filler_{i}.npy")), audio)
        changed = True
    if changed and have:
        np.savez_compressed(str(p), texts=np.array(json.dumps(cfg.FILLER_LINES)), **have)


def get_filler(idx: int | None = None):
    """(audio, sr) of filler #idx (cycles when idx is None). None if absent.

    On-the-fly rendering covers the very first boot (before enrollment or
    after cache deletion) - after that this is a pure np.load, ~0 ms.
    """
    import numpy as np

    p = _cache_path("fillers.npz")
    if not p.exists():
        tts = get_tts()
        if tts is None:
            return None
        _ensure_fillers(tts)
        if not p.exists():
            return None
    data = np.load(str(p))
    keys = sorted(k for k in data.files if k.startswith("f"))
    if not keys:
        return None
    key = keys[idx % len(keys)] if idx is not None else keys[0]
    return np.asarray(data[key], dtype=np.float32), SAMPLE_RATE


def timed_synthesis(text: str):
    """Measured synthesis for the reply path. Returns (audio, sr)."""
    import time as _t

    import numpy as np

    tts = get_tts()
    if tts is None:
        return np.zeros(0, dtype=np.float32), SAMPLE_RATE
    t0 = _t.perf_counter()
    audio, sr = tts.synth(text)
    dt_ms = (_t.perf_counter() - t0) * 1000.0
    if len(audio):
        METRICS.note(dt_ms, dt_ms, len(audio), sr)
    return audio, sr


def status() -> dict:
    """Offline-safe snapshot (instantiates the engine, never synthesizes)."""
    from tts_engine import available

    return {
        "engines": available(),
        "voice": cfg.TTS_VOICE,
        "voice_mode": cfg.TTS_VOICE_MODE,
        "reference_voice_file": cfg.TTS_REFERENCE_VOICE.exists(),
        "pack_file": cfg.TTS_VOICE_PACK.exists(),
        "fillers_cached": _cache_path("fillers.npz").exists(),
        "boot_greeting_cached": _cache_path("boot_greeting.npy").exists(),
        "metrics": METRICS.summary(),
    }


def _cli() -> int:
    import json

    if "--warm" in sys.argv:
        a = warm()
        print("boot greeting:", f"{len(a) / SAMPLE_RATE:.1f} s of audio" if a is not None
              else "UNAVAILABLE (no TTS engine)")
        f = get_filler(0)
        print("filler 0:", f"{len(f[0]) / SAMPLE_RATE:.1f} s" if f else "missing")
        return 0 if a is not None else 1
    print(json.dumps(status(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())

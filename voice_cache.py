#!/usr/bin/env python3
"""Runtime voice profile loading for IRIS (the "never re-learn" half).

At startup, instead of decoding the mp3, running Whisper, and re-extracting
speaker embeddings, jarvis.py does:

    profile = voice_cache.load_profile()          # ~ms, loads the fingerprint
    voice_cache.install_frontend_cache(cv, profile)
    greeting = voice_cache.warm(cv, profile)      # optional: play instantly

and synthesizes every reply with:

    for chunk in voice_cache.cached_inference(cv, sentence, profile, stream=True):
        play(chunk["tts_speech"])

How the caching works (and why it is version-tolerant):
  CosyVoice2's frontend recomputes speaker embeddings, acoustic tokens, and
  spectral features from the prompt wav on EVERY inference call. This module
  wraps those frontend methods with a memoizer: the first call computes and
  caches, every later call returns the cached tensors instantly. Because the
  prompt wav is always the same profile tensor, the hit rate is 100%.

  The caches are pre-seeded from the tensors saved during enrollment. If a
  seeded tensor mismatches (CosyVoice version drift, fp16 flag change), the
  warm-up inference catches the failure, drops the seeds, recomputes once,
  and continues - correctness is never at risk.

Together with enroll_voice.py this keeps Time-To-First-Speech low: no
re-decode, no re-transcribe, no repeated prompt feature extraction.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import iris_config as cfg

ARTIFACT_METHODS = ("_extract_spk_embedding", "_extract_speech_token", "_extract_speech_feat")


# ------------------------------------------------------------------ model singleton
_CV_INSTANCE = None
_CV_ERROR = ""


def get_cosyvoice(fp16: bool | None = None, cpu: bool = False):
    """Load the CosyVoice2 model ONCE per process and reuse it everywhere.

    jarvis.py, the chunker, and this module must never construct the model
    twice - on a 6 GB GPU a duplicate TTS model would evict the LLM. The
    enrollment loader (enroll_voice.load_cosyvoice) is reused verbatim so
    version-signature quirks are handled in exactly one place.

    Returns (model, fp16_used) or (None, False) when TTS is unavailable -
    the assistant then degrades to text-only output instead of crashing.
    """
    global _CV_INSTANCE, _CV_ERROR
    if _CV_INSTANCE is not None:
        return _CV_INSTANCE, _CV_FP16
    if _CV_ERROR:
        return None, False
    try:
        from enroll_voice import load_cosyvoice

        if fp16 is None:
            fp16 = cfg.COSYVOICE_FP16
        model, used_fp16 = load_cosyvoice(cpu=cpu)
        globals()["_CV_INSTANCE"] = model
        globals()["_CV_FP16"] = used_fp16
        return model, used_fp16
    except Exception as e:
        _CV_ERROR = f"{type(e).__name__}: {e}"
        print(
            f"[voice] CosyVoice unavailable ({_CV_ERROR}). "
            "IRIS continues text-only; run 'python setup_voice.py' then "
            "'python enroll_voice.py' to enable speech.",
            file=sys.stderr,
        )
        return None, False


_CV_FP16 = False


def cosyvoice_available() -> bool:
    """True if a TTS model is (or can be) loaded in this process."""
    return _CV_INSTANCE is not None or not _CV_ERROR and cfg.find_cosyvoice_model() is not None


# ------------------------------------------------------------------ voice metrics
class VoiceMetrics:
    """Latency instrumentation for the voice pipeline (Critical Issue 6/10).

    Tracks, per synthesis call: time to first playable audio (ttfa_ms),
    total synthesis duration (synth_ms), audio duration (audio_s from the
    24 kHz sample count), and real-time factor (synth time / audio time;
    below 1.0 means faster than real time). summary() aggregates them.
    """

    def __init__(self):
        self.calls = 0
        self.ttfa_ms: list[float] = []
        self.synth_ms: list[float] = []
        self.audio_s: list[float] = []
        self.rtf: list[float] = []
        self.errors = 0

    def note(self, ttfa_ms: float, synth_ms: float, n_samples: int, sr: int = 24000):
        self.calls += 1
        self.ttfa_ms.append(ttfa_ms)
        self.synth_ms.append(synth_ms)
        audio_s = n_samples / float(sr)
        self.audio_s.append(audio_s)
        self.rtf.append(synth_ms / 1000.0 / audio_s if audio_s > 0 else 0.0)

    def note_error(self):
        self.errors += 1

    def summary(self) -> dict:
        def _agg(xs):
            if not xs:
                return None
            ordered = sorted(xs)
            return {
                "mean": round(sum(xs) / len(xs), 1),
                "p50": round(ordered[len(ordered) // 2], 1),
                "max": round(ordered[-1], 1),
            }

        return {
            "calls": self.calls,
            "errors": self.errors,
            "ttfa_ms": _agg(self.ttfa_ms),
            "synth_ms": _agg(self.synth_ms),
            "audio_s": _agg(self.audio_s),
            "rtf": _agg(self.rtf),
        }


METRICS = VoiceMetrics()


class VoiceProfile:
    """The learned voice fingerprint. Loading it never touches the mp3 again."""

    def __init__(self, path: Path | None = None):
        import numpy as np
        import torch

        self.path = Path(path or cfg.VOICE_PROFILE)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run 'python enroll_voice.py' once to create it."
            )
        data = np.load(str(self.path), allow_pickle=False)
        self.prompt_text = str(data["prompt_text"])
        self.wav16k = torch.from_numpy(data["wav16k"].astype(np.float32))  # (1, T)
        self.artifacts: dict[str, dict[str, "torch.Tensor"]] = {}
        for key in data.files:
            if key.startswith("art__"):
                _, method, field = key.split("__", 2)
                self.artifacts.setdefault(method, {})[field] = torch.from_numpy(data[key])
        self.meta: dict = {}
        if "meta" in data.files:
            import json

            try:
                self.meta = json.loads(str(data["meta"]))
            except Exception:
                self.meta = {}
        self.fillers: list = self._load_fillers()
        self.boot_greeting = self._load_one("boot_greeting.npy")

    def _load_fillers(self):
        import numpy as np

        p = cfg.IRIS_DIR / "fillers.npz"
        if not p.exists():
            return []
        try:
            data = np.load(str(p))
            return [np.asarray(data[f], dtype=np.float32) for f in sorted(data.files) if f.startswith("f")]
        except Exception:
            return []

    def _load_one(self, name: str):
        import numpy as np

        p = cfg.IRIS_DIR / name
        if not p.exists():
            return None
        try:
            return np.load(str(p)).astype(np.float32)
        except Exception:
            return None

    def info(self) -> str:
        dur = self.wav16k.shape[-1] / 16000.0
        arts = ", ".join(self.artifacts) or "partial"
        return (
            f"voice profile: {self.path.name} ({self.path.stat().st_size / 1e6:.1f} MB, "
            f"{dur:.1f} s prompt, artifacts: {arts}, {len(self.fillers)} fillers, "
            f"{'boot greeting' if self.boot_greeting is not None else 'no boot greeting'})"
        )


class FrontendCache:
    """Memoizing wrapper around CosyVoice frontend prompt-extraction methods."""

    def __init__(self):
        self._store: dict[tuple, dict] = {}
        self.hits = 0
        self.misses = 0
        self.seeded_from_profile = 0

    def get(self, method: str, key: tuple):
        entry = self._store.get((method, key))
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(self, method: str, key: tuple, value: dict) -> None:
        self._store[(method, key)] = value

    def seed(self, method: str, key: tuple, value: dict) -> None:
        self._store[(method, key)] = value
        self.seeded_from_profile += 1

    def drop_seeded(self) -> None:
        """Forget seeded values (kept computed ones are re-added by first use)."""
        self._store.clear()
        self.seeded_from_profile = 0

    def stats(self) -> str:
        total = self.hits + self.misses
        return f"frontend cache: {self.hits}/{total} hits, {self.seeded_from_profile} seeded from profile"


def install_frontend_cache(cv, profile: VoiceProfile, seed: bool = True) -> FrontendCache:
    """Wrap cv.frontend extraction methods with memoization. Returns the cache."""
    cache = FrontendCache()
    front = cv.frontend
    if seed and not profile.meta.get("partial"):
        _seed_from_profile(cache, front, profile)
    for name in ARTIFACT_METHODS:
        orig = getattr(front, name, None)
        if not callable(orig):
            continue
        cache.__dict__.setdefault("_origs", {})[name] = orig

        def make_wrapper(method_name: str, original):
            def wrapper(wav, *a, **k):
                try:
                    key = (id(wav), int(wav.shape[-1]))
                except Exception:
                    key = (id(wav), 0)
                hit = cache.get(method_name, key)
                if hit is None:
                    hit = original(wav, *a, **k)
                    cache.put(method_name, key, dict(hit))
                return hit

            wrapper.__name__ = method_name
            return wrapper

        setattr(front, name, make_wrapper(name, orig))
    return cache


def _seed_from_profile(cache: FrontendCache, front, profile: VoiceProfile) -> None:
    """Pre-load cached tensors saved at enrollment. Skips dtype-incompatible ones."""
    fp16_profile = bool(profile.meta.get("fp16"))
    try:
        import torch

        model_fp16 = bool(getattr(front, "fp16", False))
    except Exception:
        model_fp16 = fp16_profile
    key = (id(profile.wav16k), int(profile.wav16k.shape[-1]))
    for method, fields in profile.artifacts.items():
        if fp16_profile != model_fp16 and "spk" not in method:
            continue  # half/full precision mismatch: let first call recompute
        try:
            value = {k: v for k, v in fields.items()}
            cache.seed(method, key, value)
        except Exception:
            continue


def cached_inference(cv, text: str, profile: VoiceProfile, stream: bool = True, speed: float = 1.0):
    """Zero-shot synthesis through the memoized frontend. Generator of chunks."""
    text = str(text or "").strip()
    if not text:
        return  # empty input: no audio, no error
    yield from cv.inference_zero_shot(
        text, profile.prompt_text, profile.wav16k, stream=stream, speed=speed
    )


def timed_inference(cv, text: str, profile: VoiceProfile, stream: bool = True, speed: float = 1.0):
    """cached_inference + latency instrumentation. Generator of chunks.

    Measures time-to-first-chunk, total synthesis time, and audio duration
    (sample count / 24 kHz), recording them in voice_cache.METRICS. Wrap your
    existing player loop with this to get real RTF numbers on your machine:

        for chunk in voice_cache.timed_inference(cv, sentence, profile):
            play(chunk["tts_speech"])
        print(voice_cache.METRICS.summary())
    """
    t0 = time.perf_counter()
    ttfa = None
    n_samples = 0
    try:
        for chunk in cached_inference(cv, text, profile, stream=stream, speed=speed):
            if ttfa is None:
                ttfa = (time.perf_counter() - t0) * 1000.0
            wav = chunk.get("tts_speech")
            if wav is not None and hasattr(wav, "shape"):
                n_samples += int(wav.shape[-1])
            yield chunk
        if ttfa is not None:
            METRICS.note(ttfa, (time.perf_counter() - t0) * 1000.0, n_samples, sr=cfg.TTS_SAMPLE_RATE)
    except Exception:
        METRICS.note_error()
        raise


def warm(cv, profile: VoiceProfile, cache: FrontendCache | None = None, line: str | None = None):
    """Warm-up pass that doubles as a fallback safety net.

    Renders one short line ('Systems online, sir.' by default), which forces
    every prompt-extraction path to run once. If seeded tensors are wrong the
    exception is caught, seeds are dropped, and the line is recomputed - so
    the FIRST user query never pays the failure cost. Returns 24 kHz float32
    numpy audio (usable as the spoken boot greeting), or None on failure.
    """
    import numpy as np

    line = line or cfg.BOOT_GREETING
    try:
        chunks = [c["tts_speech"] for c in cached_inference(cv, line, profile, stream=False)]
        import torch

        wav = torch.cat(chunks, dim=1)
        return wav.squeeze(0).float().numpy().astype(np.float32)
    except Exception as e:
        print(f"[voice] warm-up failed with cached seeds ({type(e).__name__}); recomputing once", file=sys.stderr)
        if cache is not None:
            cache.drop_seeded()
        try:
            chunks = [c["tts_speech"] for c in cached_inference(cv, line, profile, stream=False)]
            import torch

            wav = torch.cat(chunks, dim=1)
            return wav.squeeze(0).float().numpy().astype(np.float32)
        except Exception as e2:
            print(f"[voice] warm-up failed entirely: {e2}", file=sys.stderr)
            return None


def load_profile() -> VoiceProfile | None:
    try:
        p = VoiceProfile()
        print(f"[voice] {p.info()}", file=sys.stderr)
        return p
    except FileNotFoundError as e:
        print(f"[voice] {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[voice] profile load failed ({type(e).__name__}: {e})", file=sys.stderr)
        return None


if __name__ == "__main__":
    import json

    profile = load_profile()
    if profile is None:
        sys.exit(1)
    print("prompt transcript:", profile.prompt_text)
    print("filler lines pre-rendered:", len(profile.fillers))
    print("boot greeting ready:", profile.boot_greeting is not None)
    print("tts model:", "loaded (singleton)" if _CV_INSTANCE is not None else
          (f"not loaded ({_CV_ERROR})" if _CV_ERROR else "loadable via get_cosyvoice()"))
    print("metrics so far:", json.dumps(METRICS.summary()))

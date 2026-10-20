#!/usr/bin/env python3
"""ULTRON STT - NVIDIA Parakeet TDT 0.6B (transformers-native, v3).

FastConformer-TDT, ~600M params: RTFx in the thousands on GPU, punctuation
and capitalization baked in, robust to noise. Loaded through transformers
``AutoModelForTDT`` - no NeMo dependency, which is the whole point on a 6 GB
card next to llama-server.

Why v3 and not v2: ``nvidia/parakeet-tdt-0.6b-v2`` is published only as a
NeMo checkpoint (a single .nemo file, needs nemo_toolkit); v3 is the same
600M FastConformer-TDT with native transformers weights, English included
among its 25 languages, and identical latency behavior. Override with
STT_MODEL + any transformers-format repo if you prefer.

Device policy ('auto'): CUDA when at least STT_VRAM_MIN_MB (default 1800)
is free - llama-server normally holds most of the 4050's 6 GB, so auto
settles on CPU unless you shrink the context or force STT_DEVICE=cuda.
Both devices work; fp16 on CUDA, fp32 on CPU.

Offline loading: download_models.py snapshots the repo into
models/parakeet-tdt-0.6b-v3 and writes STT_LOCAL_DIR into .env; this module
prefers that directory whenever it contains config.json + model.safetensors.

Usage inside ultron.py:

    from stt_engine import get_stt
    stt = get_stt()                                  # singleton, lazy load
    text = stt.transcribe_array(mic_float32, 16000)["text"]
    text = stt.transcribe_file("clip.mp3")["text"]

CLI:
    python stt_engine.py --check                     # status, no model load
    python stt_engine.py --demo ultron_voice_sample.mp3
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import ultron_config as cfg

_STT = None   # singleton ParakeetSTT
_ERR = ""     # cached load failure (reported once, degrades to text-only)


# ------------------------------------------------------------------ helpers
def _free_vram_mb() -> int:
    """Free VRAM in MB, or 0 when it cannot be determined.

    Returns 0 on any failure (CUDA unavailable, or the driver refuses
    ``cudaMemGetInfo`` under memory pressure - which itself raises "CUDA
    error: out of memory"). 0 makes the auto policy fall back to CPU, which
    is the safe choice when another model already owns the card.
    """
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return int(free // (1024 * 1024))
    except Exception:
        pass
    return 0


def pick_device() -> str:
    """Resolve STT_DEVICE: auto -> cuda when VRAM allows, else cpu."""
    if cfg.STT_DEVICE in ("cuda", "gpu"):
        return "cuda"
    if cfg.STT_DEVICE == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available() and _free_vram_mb() >= cfg.STT_VRAM_MIN_MB:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_model_source() -> tuple[str, bool]:
    """(model_path_or_id, is_local). The local snapshot wins when complete."""
    p = Path(cfg.STT_LOCAL_DIR)
    if (p / "config.json").exists() and (p / "model.safetensors").exists():
        return str(p), True
    return cfg.STT_MODEL, False


def to_mono_16k(audio, sr: int):
    """-> (np.float32 mono @16k, 16000). Accepts mono/stereo numpy arrays."""
    import numpy as np

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    if sr != cfg.STT_SAMPLE_RATE:
        try:
            import soxr

            x = soxr.resample(x, sr, cfg.STT_SAMPLE_RATE)
        except Exception:
            # linear fallback (soxr ships with pykokoro; this covers bare envs)
            n = int(round(len(x) * cfg.STT_SAMPLE_RATE / sr))
            idx = np.arange(n) * (sr / cfg.STT_SAMPLE_RATE)
            x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return np.asarray(x, dtype=np.float32), cfg.STT_SAMPLE_RATE


def load_audio_file(path: str | Path):
    """Decode any soundfile-readable file (wav/flac/mp3) -> (mono float32, sr).

    Falls back to ffmpeg for codecs libsndfile cannot handle.
    """
    p = Path(path)
    try:
        import soundfile as sf

        audio, sr = sf.read(str(p), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio, sr
    except Exception:
        pass
    import subprocess
    import tempfile

    tmp = Path(tempfile.gettempdir()) / f"{p.stem}_ultron16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(p),
         "-ac", "1", "-ar", str(cfg.STT_SAMPLE_RATE), str(tmp)],
        check=True,
    )
    import soundfile as sf

    audio, sr = sf.read(str(tmp), dtype="float32")
    return audio, sr


# ------------------------------------------------------------------ engine
class ParakeetSTT:
    """Lazy Parakeet TDT wrapper; torch/transformers import on first load()."""

    def __init__(self, model: str | None = None, device: str | None = None):
        if model:
            self.model_source, self.is_local = model, False
        else:
            self.model_source, self.is_local = resolve_model_source()
        self.device = device or pick_device()
        self.model = None
        self.processor = None
        self.load_s = 0.0
        self.last: dict = {}

    def load(self):
        if self.model is not None:
            return self
        try:
            import torch
            from transformers import AutoModelForTDT, AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                f"STT dependencies missing ({e}). Install with:\n"
                "  pip install -U 'transformers>=5.6' torch soundfile soxr librosa"
            ) from e
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        # transformers >= 5 renamed the loader kwarg torch_dtype -> dtype
        try:
            import transformers

            major = int(str(transformers.__version__).split(".")[0])
        except Exception:
            major = 5
        dtype_kw = "dtype" if major >= 5 else "torch_dtype"
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_source)
        try:
            self.model = (
                AutoModelForTDT.from_pretrained(self.model_source, **{dtype_kw: dtype})
                .to(self.device)
                .eval()
            )
        except Exception as e:
            # Safety net: if CUDA was chosen but the card is full, fall back
            # to CPU rather than leaving STT unusable for the whole session.
            if self.device == "cuda" and "out of memory" in str(e).lower():
                print("[stt] CUDA out of memory loading Parakeet; retrying on CPU",
                      file=sys.stderr)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.device = "cpu"
                self.model = (
                    AutoModelForTDT.from_pretrained(
                        self.model_source, **{dtype_kw: torch.float32})
                    .to("cpu")
                    .eval()
                )
            else:
                raise
        self.load_s = time.perf_counter() - t0
        return self

    def transcribe_array(self, audio, sr: int) -> dict:
        """Transcribe a float array at any sample rate -> stats dict."""
        import numpy as np
        import torch

        out = {"text": "", "device": self.device, "audio_s": 0.0,
               "decode_s": 0.0, "rtf": None}
        x, sr16 = to_mono_16k(audio, sr)
        if len(x) < 400:  # under 25 ms there is nothing to recognize
            return out
        self.load()
        t0 = time.perf_counter()
        with torch.inference_mode():
            inputs = self.processor([x], sampling_rate=sr16)
            inputs.to(self.model.device, dtype=self.model.dtype)
            generated = self.model.generate(
                **inputs, return_dict_in_generate=True, max_new_tokens=256)
            text = self.processor.decode(generated.sequences, skip_special_tokens=True)
        dt = time.perf_counter() - t0
        # transformers >= 5's ParakeetProcessor.decode returns a list (one
        # transcript per input); earlier code did str(text) and spoke "['...']".
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        out["text"] = str(text).strip()
        out["audio_s"] = round(len(x) / sr16, 2)
        out["decode_s"] = round(dt, 3)
        out["rtf"] = round(dt / (len(x) / sr16), 3) if len(x) else None
        self.last = out
        return out

    def transcribe_file(self, path: str | Path) -> dict:
        audio, sr = load_audio_file(path)
        return self.transcribe_array(audio, sr)


def get_stt() -> ParakeetSTT | None:
    """Process-wide singleton. Returns None (one stderr note) when the model
    cannot load - ULTRON degrades to text input, it never crashes."""
    global _STT, _ERR
    if _STT is not None:
        return _STT
    if _ERR:
        return None
    stt = ParakeetSTT()
    try:
        stt.load()
        _STT = stt
        print(
            f"[stt] Parakeet TDT ready on {stt.device} "
            f"({stt.model_source}, load {stt.load_s:.1f} s)",
            file=sys.stderr,
        )
    except Exception as e:
        _ERR = f"{type(e).__name__}: {e}"
        print(
            f"[stt] unavailable ({_ERR}). Run 'python download_models.py' "
            "and check the deps listed there. STT stays off until then.",
            file=sys.stderr,
        )
        return None
    return _STT


def check() -> dict:
    """Offline status: no model load, no network, just facts."""
    deps = {}
    for mod in ("torch", "transformers", "soundfile", "soxr", "librosa"):
        try:
            __import__(mod)
            deps[mod] = True
        except Exception:
            deps[mod] = False
    src, local = resolve_model_source()
    return {
        "model": cfg.STT_MODEL,
        "source": src,
        "local_snapshot": local,
        "device_pref": cfg.STT_DEVICE,
        "vram_min_mb": cfg.STT_VRAM_MIN_MB,
        "deps": deps,
    }


def _cli() -> int:
    import json

    if "--check" in sys.argv:
        print(json.dumps(check(), indent=2))
        return 0
    target = next((a for a in sys.argv[2:] if not a.startswith("-")), None)
    if target:
        stt = get_stt()
        if stt is None:
            return 1
        t0 = time.perf_counter()
        out = stt.transcribe_file(target)
        out["total_s"] = round(time.perf_counter() - t0, 2)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    print(__doc__)
    print(json.dumps(check(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())

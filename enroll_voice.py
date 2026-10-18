#!/usr/bin/env python3
"""ONE-TIME voice enrollment for IRIS (CosyVoice2 zero-shot cloning).

Answers the key question: does the assistant re-learn the sample audio on
every startup?  NO. Enrollment runs once, extracts everything CosyVoice2
needs from your 15 second JARVIS sample, and saves it as a single fingerprint
file (voice_profile.npz). Every later startup just loads that fingerprint
(milliseconds) and never touches the mp3, ffmpeg, or Whisper again.

What enrollment does, exactly once:
  1. decode jarvis_alarm_15s031ms.mp3 to clean 16 kHz mono (ffmpeg fallback)
  2. transcribe it (faster-whisper or whisper; or pass --transcript yourself)
  3. run the CosyVoice2 frontend on it ONCE and cache the speaker embedding,
     acoustic speech tokens, and spectral features
  4. pre-render your three fillers AND the boot greeting in the cloned voice,
     so dead-air masking audio matches your JARVIS timbre exactly
  5. render a voice check line and measure speaker similarity (cosine)

Artifacts created (all flat, beside this file):
  voice_profile.npz    the learned voice fingerprint (prompt text + wav + tensors)
  filler_1.npy ...     pre-rendered fillers, float32 24 kHz (drop-in for your player)
  fillers.npz          same fillers in one file, with texts
  boot_greeting.npy    spoken greeting for instant startup playback
  voice_check.wav      audition clip so you can hear the clone yourself
  voice_prompt_16k.wav decoded wav cache of the mp3 (reusable, optional)

Usage:
  python enroll_voice.py
  python enroll_voice.py --force
  python enroll_voice.py --transcript "exact words spoken in the sample"
  python enroll_voice.py --cpu                # if the GPU is busy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import iris_config as cfg

ARTIFACT_METHODS = ("_extract_spk_embedding", "_extract_speech_token", "_extract_speech_feat")


# ------------------------------------------------------------------ audio decode
def decode_audio(path: Path, sr: int = 16000):
    """Return torch tensor (1, T) float32 mono at sr. ffmpeg fallback for mp3."""
    import torch
    import torchaudio

    wav_cache = cfg.IRIS_DIR / "voice_prompt_16k.wav"
    try:
        wav, got = torchaudio.load(str(path))
        if got != sr:
            wav = torchaudio.functional.resample(wav, orig_freq=got, new_freq=sr)
    except Exception:
        import subprocess

        if not wav_cache.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                 "-ac", "1", "-ar", str(sr), str(wav_cache)],
                check=True,
            )
        wav, got = torchaudio.load(str(wav_cache))
        if got != sr:
            wav = torchaudio.functional.resample(wav, orig_freq=got, new_freq=sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.clamp(-1.0, 1.0)
    try:  # light peak normalization
        peak = wav.abs().max()
        if 0.1 < peak < 0.98:
            wav = wav / peak * 0.95
    except Exception:
        pass
    return wav.float()


# ------------------------------------------------------------------ transcription
def transcribe(path: Path, wav) -> str:
    """Transcribe the voice sample once. Returns the prompt transcript."""
    try:
        from faster_whisper import WhisperModel

        device = "cuda" if _cuda() else "cpu"
        model = WhisperModel("base.en", device=device, compute_type="int8" if device == "cuda" else "int8")
        segments, _ = model.transcribe(str(path), language="en", beam_size=1, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            return text
    except ImportError:
        print("[enroll] faster-whisper not installed, trying openai-whisper")
    except Exception as e:
        print(f"[enroll] faster-whisper failed ({type(e).__name__}: {e}); trying openai-whisper")
    try:
        import whisper

        model = whisper.load_model("base.en", device="cuda" if _cuda() else "cpu")
        text = model.transcribe(str(path), language="en")["text"].strip()
        if text:
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"[enroll] whisper failed ({type(e).__name__}: {e})")
    return ""


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# ------------------------------------------------------------------ cosyvoice loading
def load_cosyvoice(cpu: bool = False):
    """Import and load CosyVoice2 from the detected checkout. Robust to versions."""
    import sys as _sys

    root = cfg.find_cosyvoice_root()
    if root is None:
        raise RuntimeError(
            "CosyVoice checkout not found. Run 'python setup_voice.py' once: it clones "
            "the engine, downloads CosyVoice2-0.5B, and writes COSYVOICE_ROOT / "
            "COSYVOICE_MODEL into .env automatically."
        )
    if str(root) not in _sys.path:
        _sys.path.insert(0, str(root))
    model_dir = cfg.find_cosyvoice_model()
    if model_dir is None:
        raise RuntimeError(
            "CosyVoice2 model files not found. Run 'python setup_voice.py' once: it "
            "downloads CosyVoice2-0.5B (about 3.9 GB) and updates .env automatically."
        )
    fp16 = cfg.COSYVOICE_FP16 and not cpu
    from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice  # type: ignore

    for ctor, kwargs in ((CosyVoice2, {"fp16": fp16, "cpu": cpu}), (CosyVoice, {"fp16": fp16, "cpu": cpu})):
        try:
            return ctor(str(model_dir), **kwargs), fp16
        except TypeError:
            try:
                return ctor(str(model_dir)), fp16
            except TypeError:
                continue
    raise RuntimeError("could not construct a CosyVoice model (unexpected signature)")


# ------------------------------------------------------------------ rendering
def synth(cv, text: str, prompt_text: str, wav16k, stream: bool = False):
    """Render text in the cloned voice. Returns torch tensor (1, T) at 24 kHz."""
    import torch

    chunks = []
    for out in cv.inference_zero_shot(text, prompt_text, wav16k, stream=stream):
        chunks.append(out["tts_speech"])
    if not chunks:
        raise RuntimeError("CosyVoice produced no audio")
    return torch.cat(chunks, dim=1)


def save_npy(path: Path, tensor) -> None:
    """Save a float32 24 kHz mono numpy array for the instant-playback pipeline."""
    import numpy as np

    np.save(str(path), tensor.squeeze(0).float().numpy().astype(np.float32))


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="One-time IRIS voice enrollment")
    ap.add_argument("--force", action="store_true", help="re-enroll even if a profile exists")
    ap.add_argument("--transcript", default="", help="manual transcript of the sample (bypass ASR)")
    ap.add_argument("--cpu", action="store_true", help="run enrollment on CPU")
    ap.add_argument("--greeting", default=cfg.BOOT_GREETING, help="boot greeting line to pre-render")
    ap.add_argument("--skip-fillers", action="store_true", help="do not (re)render filler npy files")
    args = ap.parse_args()

    if cfg.VOICE_PROFILE.exists() and not args.force:
        print(f"voice profile already exists: {cfg.VOICE_PROFILE}")
        print("startup will load this fingerprint directly (no mp3 decode, no ASR, no re-learning).")
        print("use --force to re-enroll.")
        return 0

    if not cfg.VOICE_SAMPLE.exists():
        print(f"ERROR: voice sample not found: {cfg.VOICE_SAMPLE}")
        print("put the sample next to this file or set VOICE_SAMPLE in .env")
        return 1

    t0 = time.time()

    # 1) decode
    print(f"[1/5] decoding {cfg.VOICE_SAMPLE.name} ...")
    wav16k = decode_audio(cfg.VOICE_SAMPLE)
    dur = wav16k.shape[-1] / 16000.0
    print(f"      {dur:.1f} s of 16 kHz audio")
    if dur < 5 or dur > 30:
        print("      WARNING: CosyVoice zero-shot works best with 5-30 s of clean speech")

    # 2) transcript
    print("[2/5] transcribing sample (one time only) ...")
    prompt_text = args.transcript.strip() or transcribe(cfg.VOICE_SAMPLE, wav16k)
    if len(prompt_text) < 3:
        print("ERROR: could not get a transcript (>= 3 chars required by CosyVoice).")
        print("Run again with:  python enroll_voice.py --transcript \"words spoken in the clip\"")
        return 1
    print(f"      transcript: {prompt_text}")

    # 3) load CosyVoice and extract the fingerprint
    print("[3/5] loading CosyVoice2 and extracting the voice fingerprint ...")
    artifacts, fp16_used = {}, False
    try:
        cv, fp16_used = load_cosyvoice(args.cpu)
        front = cv.frontend
        for name in ARTIFACT_METHODS:
            fn = getattr(front, name, None)
            if not callable(fn):
                continue
            try:
                out = fn(wav16k)
                artifacts[name] = {k: v.detach().cpu().numpy() for k, v in dict(out).items()}
                print(f"      cached {name} -> {list(artifacts[name])}")
            except Exception as e:
                print(f"      (skipped {name}: {type(e).__name__})")
    except Exception as e:
        print(f"      CosyVoice unavailable ({e}).")
        print("      Saving a PARTIAL profile (wav + transcript). Zero-shot will still work,")
        print("      but fillers cannot be pre-rendered now. Re-run enroll later with the")
        print("      CosyVoice repo available for the full fingerprint.")
        _save_partial(prompt_text, wav16k)
        return 0

    # 4) save the profile
    import numpy as np
    import torch

    flat = {"prompt_text": np.array(prompt_text), "wav16k": wav16k.numpy().astype(np.float16)}
    for m, d in artifacts.items():
        for k, v in d.items():
            flat[f"art__{m}__{k}"] = v
    flat["meta"] = np.array(
        json.dumps(
            {
                "sample": cfg.VOICE_SAMPLE.name,
                "duration_s": round(dur, 2),
                "fp16": fp16_used,
                "created": time.time(),
                "transcript": prompt_text,
            }
        )
    )
    np.savez_compressed(str(cfg.VOICE_PROFILE), **flat)
    print(f"      saved {cfg.VOICE_PROFILE.name} ({cfg.VOICE_PROFILE.stat().st_size / 1e6:.1f} MB)")

    # 5) pre-render fillers + greeting in the cloned voice, then verify
    print("[4/5] pre-rendering fillers and boot greeting in YOUR cloned voice ...")
    if not args.skip_fillers:
        try:
            for i, line in enumerate(cfg.FILLER_LINES, 1):
                audio = synth(cv, line, prompt_text, wav16k)
                save_npy(cfg.IRIS_DIR / f"filler_{i}.npy", audio)
                print(f"      filler_{i}.npy  <- {line!r}")
            np.savez_compressed(
                str(cfg.IRIS_DIR / "fillers.npz"),
                texts=np.array(json.dumps(cfg.FILLER_LINES)),
                **{f"f{i}": np.load(str(cfg.IRIS_DIR / f"filler_{i}.npy")) for i in range(1, len(cfg.FILLER_LINES) + 1)},
            )
            boot = synth(cv, args.greeting, prompt_text, wav16k)
            save_npy(cfg.IRIS_DIR / "boot_greeting.npy", boot)
            import torchaudio

            torchaudio.save(str(cfg.IRIS_DIR / "voice_check.wav"), boot, cfg.TTS_SAMPLE_RATE)
        except Exception as e:
            print(f"      filler pre-render failed ({type(e).__name__}: {e}); continuing")

    print("[5/5] verifying the clone ...")
    try:
        import torchaudio

        check = synth(cv, "Voice check complete, sir. All systems nominal.", prompt_text, wav16k)
        torchaudio.save(str(cfg.IRIS_DIR / "voice_check.wav"), check, cfg.TTS_SAMPLE_RATE)
        if "_extract_spk_embedding" in artifacts:
            import torch.nn.functional as F

            check16 = torchaudio.functional.resample(check, cfg.TTS_SAMPLE_RATE, 16000)
            ref = front._extract_spk_embedding(wav16k)["flow_spk_embedding"]
            new = front._extract_spk_embedding(check16)["flow_spk_embedding"]
            cos = float(F.cosine_similarity(ref, new, dim=-1))
            print(f"      speaker similarity (cosine): {cos:.3f}  (>= 0.6 is a solid clone)")
            print("      listen to voice_check.wav to audition the voice yourself")
    except Exception as e:
        print(f"      verification skipped ({type(e).__name__}: {e})")

    print(f"\nenrollment finished in {time.time() - t0:.1f} s.")
    print("From now on, startup only does: np.load(voice_profile.npz). The mp3, ffmpeg,")
    print("whisper, and speaker extraction are NEVER run again unless you pass --force.")
    return 0


def _save_partial(prompt_text: str, wav16k) -> None:
    import numpy as np

    np.savez_compressed(
        str(cfg.VOICE_PROFILE),
        prompt_text=np.array(prompt_text),
        wav16k=wav16k.numpy().astype(np.float16),
        meta=np.array(json.dumps({"partial": True})),
    )
    print(f"      saved partial profile {cfg.VOICE_PROFILE}")


if __name__ == "__main__":
    sys.exit(main())

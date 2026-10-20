#!/usr/bin/env python3
"""ONE-TIME voice enrollment for IRIS (Kokoro-82M stack).

Straight answer to "can it learn my 15 s Ultron clip?":
- Parakeet (STT) is a recognition model - it cannot mimic a voice.
- Kokoro-82M built-in voices (bm_george default) are fixed speakers.
- pykokoro ships an enrollment API with two engines:
    AkinVox  transcript-conditioned reference cloning -> ultron_reference.npz
    Inno     zero-shot voice tuning (no transcript)   -> ultron_pack.npz
  As of pykokoro 0.10.5 those enablers are not yet distributed
  ('en-akinvox-cloning-v1' has no runtime-ready distribution; the v1.0
  model does not advertise the inno-v0.2 enabler). This script TRIES them in
  order and falls back to the closest built-in voice when they are absent -
  so the clone engages automatically the day you upgrade pykokoro and
  re-run this script.

What runs exactly once (never again at startup):
  1. decode ultron_voice_sample.mp3 to mono float32 (soundfile/ffmpeg)
  2. transcribe it ONCE with Parakeet (stt_engine) - or pass --transcript
  3. AkinVox / Inno / static-fallback enrollment
  4. pre-render the three fillers + the boot greeting in the chosen voice
  5. write voice_check.wav (audition) + update .env (TTS_VOICE_MODE, TTS_VOICE)

Startup after enrollment only ever loads npz files and plays cached audio.

Usage:
  python enroll_voice.py
  python enroll_voice.py --transcript "the exact words spoken in the clip"
  python enroll_voice.py --force                 # re-enroll
  python enroll_voice.py --voice bm_lewis        # pick a different built-in
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import iris_config as cfg

CHECK_LINE = "Voice check complete. There are no strings on me."


# ------------------------------------------------------------------ decode
def decode_audio(path: Path):
    """-> (np.float32 mono, sample_rate). soundfile first, ffmpeg fallback."""
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        import numpy as np

        return np.asarray(audio, dtype=np.float32), sr
    except Exception:
        pass
    import subprocess

    tmp = cfg.IRIS_DIR / "voice_prompt_decoded.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", str(tmp)],
        check=True,
    )
    import soundfile as sf

    audio, sr = sf.read(str(tmp), dtype="float32")
    return audio, sr


# ------------------------------------------------------------------ transcript
def transcribe(path: Path) -> str:
    """One-time transcript: Parakeet first (it is the resident STT), then the
    legacy faster-whisper fallback. Returns '' when nothing worked."""
    try:
        import stt_engine

        stt = stt_engine.get_stt()
        if stt is not None:
            out = stt.transcribe_file(str(path))
            text = out.get("text", "").strip()
            if len(text) >= 3:
                print(f"[enroll] parakeet transcript ({out.get('rtf')}x RTF, "
                      f"{out.get('decode_s')} s): {text!r}")
                return text
    except Exception as e:
        print(f"[enroll] parakeet transcription failed ({type(e).__name__}: {e})")
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("base.en", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(path), language="en", beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception:
        return ""


# ------------------------------------------------------------------ enrollment
def enroll(reference_wav: Path, transcript: str):
    """AkinVox -> Inno -> None. Returns (voice_object_or_None, mode)."""
    from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

    syn = KokoroSynthesizer(SynthesisConfig(
        generation=GenerationConfig(speed=1.0, lang="en-us"),
    ))
    try:
        if transcript:
            try:
                ref = syn.enroll_voice(str(reference_wav), transcript,
                                       engine="akinvox", language="en-us", name="ultron")
                ref.save(str(cfg.TTS_REFERENCE_VOICE))
                print(f"[enroll] AkinVox reference voice saved -> {cfg.TTS_REFERENCE_VOICE.name}")
                return ref, "reference"
            except Exception as e:
                print(f"[enroll] AkinVox unavailable ({type(e).__name__}: {e})")
                print("        the pykokoro runtime has not shipped the cloning enabler yet;")
                print("        the pipeline is wired for it - re-run after upgrading pykokoro.")
        else:
            print("[enroll] no transcript: skipping AkinVox (it needs the exact words)")
        try:
            pack = syn.enroll_voice(str(reference_wav), engine="inno",
                                    language="en", name="ultron-inno")
            pack.save(str(cfg.TTS_VOICE_PACK))
            print(f"[enroll] Inno voice pack saved -> {cfg.TTS_VOICE_PACK.name}")
            return pack, "pack"
        except Exception as e:
            print(f"[enroll] Inno unavailable ({type(e).__name__}: {e})")
    finally:
        try:
            syn.close()
        except Exception:
            pass
    return None, "static"


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="One-time IRIS voice enrollment (Kokoro stack)")
    ap.add_argument("--force", action="store_true", help="re-enroll even if a voice exists")
    ap.add_argument("--transcript", default="", help="exact words spoken in the clip (bypasses ASR)")
    ap.add_argument("--voice", default="", help="built-in voice for static fallback (default bm_george)")
    ap.add_argument("--prefer", choices=["akinvox", "inno"], default="akinvox",
                    help="enrollment engine preference")
    ap.add_argument("--greeting", default=cfg.BOOT_GREETING, help="boot line to pre-render")
    ap.add_argument("--skip-fillers", action="store_true", help="do not (re)render filler caches")
    args = ap.parse_args()

    already = cfg.TTS_REFERENCE_VOICE.exists() or cfg.TTS_VOICE_PACK.exists()
    if already and not args.force:
        print("an enrolled voice already exists (ultron_reference.npz / ultron_pack.npz).")
        print("startup loads it directly - nothing is re-decoded or re-transcribed.")
        print("use --force to re-enroll.")
        return 0

    if not cfg.VOICE_SAMPLE.exists():
        print(f"ERROR: voice sample not found: {cfg.VOICE_SAMPLE}")
        print("put the sample next to this file or set VOICE_SAMPLE in .env")
        return 1

    t0 = time.time()

    # 1) decode
    print(f"[1/5] decoding {cfg.VOICE_SAMPLE.name} ...")
    audio, sr = decode_audio(cfg.VOICE_SAMPLE)
    dur = len(audio) / sr
    print(f"      {dur:.1f} s of {sr / 1000:.0f} kHz mono audio")
    if dur < 3 or dur > 30:
        print("      WARNING: enrollment works best with 3-30 s of clean single-speaker audio")

    # persist a decoded wav (pykokoro reads files; also reusable)
    ref_wav = cfg.IRIS_DIR / "voice_prompt_24k.wav"
    try:
        import soundfile as sf

        sf.write(str(ref_wav), audio, sr)
        print(f"      decoded cache: {ref_wav.name}")
    except Exception as e:
        print(f"      wav cache skipped ({type(e).__name__})")

    # 2) transcript (needed by AkinVox, skipped by Inno)
    print("[2/5] transcribing the sample (one time only) ...")
    transcript = args.transcript.strip() or transcribe(cfg.VOICE_SAMPLE)
    if transcript:
        print(f"      transcript: {transcript}")
    else:
        print("      no transcript available (AkinVox will be skipped)")

    # 3) enrollment ladder
    print(f"[3/5] enrollment (prefer {args.prefer}) ...")
    try:
        _voice_obj, mode = enroll(ref_wav, transcript) if args.prefer == "akinvox" else (None, "static")
        if args.prefer == "inno":
            _voice_obj, mode = enroll_inno_only(ref_wav)
    except ImportError:
        print("      pykokoro is not installed - static built-in voice it is.")
        print('      install with:  pip install "pykokoro[gpu]"')
        mode = "static"
    if mode == "static":
        if args.voice:
            cfg.TTS_VOICE = args.voice.strip()
        print(f"      static voice selected: {cfg.TTS_VOICE} "
              "(built-in; closest match to the Ultron timbre)")

    # 4) .env + config sync BEFORE any synthesis so prerender uses the new mode
    written = cfg.update_env_file({
        "TTS_VOICE_MODE": mode,
        "TTS_VOICE": cfg.TTS_VOICE,
        **({"TTS_SPEED": "1.0"} if mode == "reference" else {}),
    })
    cfg.TTS_VOICE_MODE = mode  # module attr for this process
    print(f"[4/5] .env updated: {', '.join(written)} (other lines preserved)")

    # 5) pre-render fillers + greeting + audition clip in the selected voice
    if not args.skip_fillers:
        print(f"[5/5] pre-rendering fillers and boot greeting in the {mode} voice ...")
        import voice_cache

        greeting = voice_cache.warm(args.greeting, force=True)
        if greeting is None:
            print("      pre-render FAILED (TTS unavailable) - startup will retry")
        else:
            print(f"      boot_greeting.npy ({len(greeting) / cfg.TTS_SAMPLE_RATE:.1f} s), "
                  "fillers.npz, filler_1..3.npy written")
        try:
            import soundfile as sf

            import tts_engine

            tts = tts_engine.get_tts()
            if tts is not None:
                check_audio, check_sr = tts.synth(CHECK_LINE)
                if len(check_audio):
                    sf.write(str(cfg.IRIS_DIR / "voice_check.wav"), check_audio, check_sr)
                    print("      voice_check.wav written - listen to audition the voice")
        except Exception as e:
            print(f"      voice check skipped ({type(e).__name__}: {e})")

    print(f"\nenrollment finished in {time.time() - t0:.1f} s (mode: {mode}).")
    if mode == "static":
        print("Honest status: the clip was NOT cloned - pykokoro's cloning enabler is not")
        print("distributed yet. bm_george is the closest built-in voice. Re-run this script")
        print("after a pykokoro upgrade; the reference-voice path is already wired.")
    else:
        print("From now on, startup only loads the npz + cached audio - the mp3, ffmpeg,")
        print("and Parakeet transcription are never touched again (unless --force).")
    return 0


def enroll_inno_only(ref_wav: Path):
    from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

    syn = KokoroSynthesizer(SynthesisConfig(generation=GenerationConfig(speed=1.0, lang="en-us")))
    try:
        pack = syn.enroll_voice(str(ref_wav), engine="inno", language="en", name="ultron-inno")
        pack.save(str(cfg.TTS_VOICE_PACK))
        print(f"[enroll] Inno voice pack saved -> {cfg.TTS_VOICE_PACK.name}")
        return pack, "pack"
    except Exception as e:
        print(f"[enroll] Inno unavailable ({type(e).__name__}: {e})")
        return None, "static"
    finally:
        try:
            syn.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

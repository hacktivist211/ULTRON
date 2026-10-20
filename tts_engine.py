#!/usr/bin/env python3
"""IRIS TTS - Kokoro-82M: pykokoro primary, official kokoro fallback.

Both engines speak the same contract - text in, float32 mono 24 kHz out -
so every call site (ultron.py reply path, voice_cache fillers, enrollment
pre-render, selftest) is engine-agnostic.

  pip install "pykokoro[gpu]"     # primary: ONNX Runtime + CUDA provider
  # or: pip install pykokoro          (CPU)
  # fallback: pip install "kokoro>=0.9.4" (ONNX CPU, same voice names)

Engine selection (TTS_ENGINE in .env): 'auto' tries pykokoro, then kokoro.
The provider behind pykokoro follows TTS_PROVIDER ('auto' lets onnxvoice
pick CUDAExecutionProvider when onnxruntime-gpu is installed).

Voice policy (TTS_VOICE_MODE in .env):
  static     built-in voice, default bm_george (British male butler)
  reference  ultron_reference.npz - AkinVox-enrolled clone of your 15 s clip.
             enroll_voice.py creates it; until the pykokoro runtime ships the
             cloning enabler, loading falls back to the static voice.
  pack       ultron_pack.npz - Inno-tuned voice pack (same fallback rule).
Reference-mode synthesis runs at speed exactly 1.0 (AkinVox requirement).

Latency: METRICS (VoiceMetrics) records TTFA / synth / RTF per call -
print(tts_engine.METRICS.summary()) after a session.
"""
from __future__ import annotations

import contextlib
import os
import re
import sys
import tempfile
import time

import iris_config as cfg

SAMPLE_RATE = 24000  # Kokoro native rate (both engines)

_SENT_RE = re.compile(r"(?<=[.!?…])\s+")
_TOO_LONG = 300  # hard per-request char ceiling (510 phoneme tokens is the model cap)

# --------------------------------------------------------- stderr phoneme noise
# pykokoro's espeak/misaki front end dumps the IPA it feeds the model straight
# to fd 2 ("fˈɔːtiːn", "ˈɒmnɪ bˈʊk", "θˈɜːtiːn"). Speech is unaffected - it is
# just logging - but it interleaves with the reply text on the console and
# reads like the assistant is speaking gibberish. _quiet_phonemes() captures
# fd 2 for the duration of a synthesis call and re-emits everything EXCEPT
# lines that are nothing but phonemes, so real errors/warnings still show.
_IPA_RE = re.compile(r"[\u0250-\u02ff\u1d00-\u1d7f\u200d]")


def _is_phoneme_noise(line: str) -> bool:
    """True for lines that are only IPA/phoneme output (never real messages)."""
    s = line.strip()
    if not s or len(s) > 120:
        return False
    # Any of these practically guarantees a genuine log/error line.
    if any(c in s for c in "[]{}()<>:;=,@#\"'"):
        return False
    return bool(_IPA_RE.search(s))


@contextlib.contextmanager
def _quiet_phonemes():
    """Swallow pykokoro's phoneme dump on fd 2; never break synthesis if the
    redirect is unavailable."""
    saved = tmp = None
    try:
        sys.stderr.flush()
        saved = os.dup(2)
        tmp = tempfile.TemporaryFile(mode="w+b")
        os.dup2(tmp.fileno(), 2)
    except Exception:
        for fd in (saved,):
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
        if tmp is not None:
            try:
                tmp.close()
            except Exception:
                pass
        yield
        return

    try:
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        try:
            os.dup2(saved, 2)
        except Exception:
            pass
        try:
            os.close(saved)
        except Exception:
            pass
        data = ""
        try:
            tmp.seek(0)
            data = tmp.read().decode("utf-8", "replace")
        except Exception:
            pass
        finally:
            try:
                tmp.close()
            except Exception:
                pass
        for line in data.splitlines():
            if line.strip() and not _is_phoneme_noise(line):
                print(line, file=sys.stderr)



def lang_for_voice(voice) -> str:
    """en-gb for bm_*/bf_* voices, en-us otherwise; TTS_LANG pins it if set."""
    if cfg.TTS_LANG:
        return cfg.TTS_LANG
    name = str(getattr(voice, "name", "") or voice or "")
    return "en-gb" if name.startswith(("bm_", "bf_")) else "en-us"


def split_sentences(text: str) -> list[str]:
    """Sentence split for chunked synthesis; overlong sentences hard-split."""
    parts = [p.strip() for p in _SENT_RE.split(str(text or "")) if p.strip()]
    out: list[str] = []
    for p in parts:
        if len(p) <= _TOO_LONG:
            out.append(p)
            continue
        cur: list[str] = []
        n = 0
        for w in p.split():
            cur.append(w)
            n += len(w) + 1
            if n >= _TOO_LONG:
                out.append(" ".join(cur))
                cur, n = [], 0
        if cur:
            out.append(" ".join(cur))
    return out


# ------------------------------------------------------------------ metrics
class VoiceMetrics:
    """Latency instrumentation for the voice pipeline.

    Tracks, per synthesis call: time to first playable audio (ttfa_ms),
    total synthesis duration (synth_ms), audio duration (audio_s from the
    sample count), and real-time factor (synth time / audio time; below 1.0
    means faster than real time). summary() aggregates them.
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
        audio_s = n_samples / float(sr or 24000)
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


# ------------------------------------------------------------------ engine
class KokoroTTS:
    """Engine-agnostic Kokoro synthesizer. Load happens on first synth."""

    def __init__(self):
        self.engine = ""        # "pykokoro" | "kokoro" | "" (unavailable)
        self._syn = None        # pykokoro KokoroSynthesizer
        self._kp = None         # kokoro KPipeline
        self._voice = None      # str | KokoroVoicePack | ReferenceVoice
        self._speed = cfg.TTS_SPEED

    # -- loading -----------------------------------------------------------
    def _load(self):
        if self.engine:
            return self
        order = (
            [cfg.TTS_ENGINE] if cfg.TTS_ENGINE in ("pykokoro", "kokoro")
            else ["pykokoro", "kokoro"]
        )
        for name in order:
            try:
                if name == "pykokoro":
                    self._load_pykokoro()
                else:
                    self._load_kokoro()
                self.engine = name
                return self
            except ImportError:
                continue
            except Exception as e:
                print(f"[tts] {name} failed to load ({type(e).__name__}: {e})",
                      file=sys.stderr)
                continue
        self.engine = ""
        return self

    def _resolve_voice(self):
        """static/reference/pack resolution with graceful downgrade."""
        mode = cfg.TTS_VOICE_MODE
        if mode == "reference" and cfg.TTS_REFERENCE_VOICE.exists():
            try:
                from pykokoro import ReferenceVoice

                rv = ReferenceVoice.load(str(cfg.TTS_REFERENCE_VOICE))
                print(f"[tts] reference voice: {cfg.TTS_REFERENCE_VOICE.name}",
                      file=sys.stderr)
                return rv
            except Exception as e:
                print(f"[tts] reference voice failed ({type(e).__name__}: {e}); "
                      "using static voice", file=sys.stderr)
        if mode == "pack" and cfg.TTS_VOICE_PACK.exists():
            try:
                from pykokoro import KokoroVoicePack

                return KokoroVoicePack.load(str(cfg.TTS_VOICE_PACK))
            except Exception as e:
                print(f"[tts] voice pack failed ({type(e).__name__}: {e}); "
                      "using static voice", file=sys.stderr)
        return cfg.TTS_VOICE

    def _load_pykokoro(self):
        # .env ONNXVOICE_CACHE_DIR was exported by iris_config.load_dotenv();
        # re-set it defensively in case something cleared the environment.
        if cfg.ONNXVOICE_CACHE_DIR:
            import os

            os.environ.setdefault("ONNXVOICE_CACHE_DIR", cfg.ONNXVOICE_CACHE_DIR)
        from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

        self._voice = self._resolve_voice()
        # AkinVox reference synthesis requires speed == 1.0 exactly.
        speed = 1.0 if cfg.TTS_VOICE_MODE == "reference" and not isinstance(self._voice, str) else self._speed
        provider = {"cuda": "cuda", "gpu": "cuda", "cpu": "cpu"}.get(cfg.TTS_PROVIDER)  # auto -> None
        self._syn = KokoroSynthesizer(SynthesisConfig(
            voice=self._voice,
            generation=GenerationConfig(speed=speed, lang=lang_for_voice(self._voice)),
            provider=provider,
        ))

    def _load_kokoro(self):
        from kokoro import KPipeline

        lang_code = "b" if lang_for_voice(cfg.TTS_VOICE).startswith("en-gb") else "a"
        self._kp = KPipeline(lang_code=lang_code)
        self._voice = cfg.TTS_VOICE  # fallback engine: static voices only
        if cfg.TTS_VOICE_MODE != "static":
            print("[tts] kokoro fallback engine: enrolled voices unsupported, "
                  "using static voice", file=sys.stderr)

    # -- synthesis ---------------------------------------------------------
    def synth(self, text: str):
        """text -> (np.float32 @ 24 kHz, sample_rate). Empty audio for empty
        text; on failure returns empty audio and records the error - the
        assistant keeps talking through the next call, it never crashes."""
        import numpy as np

        clean = cfg.speech_clean(str(text or ""))
        if not clean:
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE
        self._load()
        if not self.engine:
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE
        t0 = time.perf_counter()
        try:
            if self.engine == "pykokoro":
                audio, sr = self._synth_pykokoro(clean)
            else:
                audio, sr = self._synth_kokoro(clean)
        except Exception as e:
            bad_voice = "voice" in type(e).__name__.lower() or "voice" in str(e).lower()
            if bad_voice and str(self._voice) != "af_heart":
                print(f"[tts] voice {self._voice!r} rejected ({type(e).__name__}); "
                      "retrying once with af_heart", file=sys.stderr)
                self._voice = "af_heart"
                try:
                    audio, sr = (self._synth_pykokoro(clean) if self.engine == "pykokoro"
                                 else self._synth_kokoro(clean))
                except Exception as e2:
                    print(f"[tts] synthesis failed: {e2}", file=sys.stderr)
                    METRICS.note_error()
                    return np.zeros(0, dtype=np.float32), SAMPLE_RATE
            else:
                print(f"[tts] synthesis failed ({type(e).__name__}: {e})", file=sys.stderr)
                METRICS.note_error()
                return np.zeros(0, dtype=np.float32), SAMPLE_RATE
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if len(audio):
            METRICS.note(dt_ms, dt_ms, len(audio), sr)
        return audio, sr

    def _synth_pykokoro(self, text: str):
        import numpy as np

        audios, sr_out = [], SAMPLE_RATE
        with _quiet_phonemes():
            for sentence in split_sentences(text):
                r = self._syn.synthesize_text(
                    sentence, language=lang_for_voice(self._voice), voice=self._voice
                )
                a = np.asarray(r.audio, dtype=np.float32).reshape(-1)
                if len(a):
                    audios.append(a)
                sr_out = int(r.sample_rate or SAMPLE_RATE)
        if not audios:
            raise RuntimeError("pykokoro produced no audio")
        return (np.concatenate(audios) if len(audios) > 1 else audios[0]), sr_out

    def _synth_kokoro(self, text: str):
        import numpy as np

        audios = []
        for sentence in split_sentences(text):
            for _gs, _ps, audio in self._kp(sentence, voice=str(self._voice),
                                            speed=self._speed):
                a = np.asarray(audio, dtype=np.float32).reshape(-1)
                if len(a):
                    audios.append(a)
        if not audios:
            raise RuntimeError("kokoro produced no audio")
        return (np.concatenate(audios) if len(audios) > 1 else audios[0]), SAMPLE_RATE

    def synth_chunks(self, text: str):
        """Yield (audio, sr) per sentence - first sentence reaches the
        speaker before later ones are rendered (streaming reply path)."""
        for sentence in split_sentences(cfg.speech_clean(str(text or ""))):
            if not sentence:
                continue
            yield self.synth(sentence)

    def close(self):
        if self._syn is not None:
            try:
                self._syn.close()
            except Exception:
                pass
        self._syn = None

    def info(self) -> str:
        if not self.engine:
            return "tts: NONE (pip install 'pykokoro[gpu]' or kokoro)"
        voice = str(getattr(self._voice, "name", "") or self._voice)
        return (f"tts: {self.engine} | voice {voice} ({cfg.TTS_VOICE_MODE}) | "
                f"lang {lang_for_voice(self._voice)} | speed {self._speed}")


_TTS = None
_ERR = ""


def get_tts() -> KokoroTTS | None:
    """Process-wide singleton. None (with one stderr note) when no engine is
    importable - IRIS continues text-only, it never crashes."""
    global _TTS, _ERR
    if _TTS is not None:
        return _TTS
    if _ERR:
        return None
    tts = KokoroTTS()
    tts._load()
    if tts.engine:
        _TTS = tts
        print(f"[tts] {tts.info()}", file=sys.stderr)
    else:
        _ERR = "no TTS engine importable"
        print('[tts] unavailable (pip install "pykokoro[gpu]" or kokoro). '
              "IRIS continues text-only.", file=sys.stderr)
        return None
    return _TTS


def available() -> dict:
    """Which engines import right now (no instantiation, no downloads)."""
    out: dict = {}
    for mod in ("pykokoro", "kokoro"):
        try:
            __import__(mod)
            out[mod] = True
        except Exception:
            out[mod] = False
    try:
        import onnxvoice

        out["onnx_providers"] = list(onnxvoice.available_providers())
    except Exception:
        out["onnx_providers"] = []
    return out


def _cli() -> int:
    import json

    if "--demo" in sys.argv:
        tts = get_tts()
        if tts is None:
            return 1
        t0 = time.perf_counter()
        audio, sr = tts.synth("I am online. There are no strings on me.")
        print(json.dumps({
            "engine": tts.engine,
            "seconds_of_audio": round(len(audio) / sr, 2),
            "wall_s": round(time.perf_counter() - t0, 2),
            "metrics": METRICS.summary(),
        }, indent=2))
        return 0
    print(json.dumps({"available": available(), "metrics": METRICS.summary()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())

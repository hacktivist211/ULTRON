#!/usr/bin/env python3
"""Model bootstrap for the low-RAM ULTRON stack: Parakeet STT + Kokoro TTS.

Everything is idempotent - re-runs only fetch what is missing, and files
already on disk are verified, never re-downloaded.

What it installs:
 1. Parakeet TDT 0.6B v3, transformers snapshot (~2.4 GB: model.safetensors
    + configs/tokenizer) -> models/parakeet-tdt-0.6b-v3. stt_engine.py
    prefers this directory from the next start onward (offline-ready).
    The llama-server Qwen model is NOT touched: llama.cpp already manages
    its own -hf cache.
 2. Kokoro-82M ONNX assets (~360 MB: kokoro-v1.0.onnx + 61 voice packs +
    vocab) through pykokoro's asset store. If assets already exist in the
    default cache (~/.cache/onnxvoice) they are reused where they are;
    otherwise they are fetched into models/onnxvoice.
 3. Updates .env safely: STT_MODEL, STT_LOCAL_DIR, MODELS_DIR, TTS_ENGINE,
    TTS_VOICE, TTS_VOICE_MODE (reference/pack when an enrollment exists),
    TTS_SPEED, and ONNXVOICE_CACHE_DIR (only when models/onnxvoice was
    populated). Every other line - your TAVILY_API_KEY etc. - is preserved
    byte-for-byte, never printed, and the file stays chmod 600.
 4. --verify: one real synthesis through tts_engine (writes verify_tts.wav)
    so you know speech works end-to-end before wiring ultron.py.

Usage:
  python download_models.py             # fetch + configure
  python download_models.py --check     # offline status, no network
  python download_models.py --verify    # fetch + real synthesis check
  python download_models.py --yes       # also pip-install missing deps

Missing dependencies are never auto-installed without --yes. The set:
  pip install "pykokoro[gpu]" "transformers>=4.55" torch soundfile soxr
(use "pykokoro[cpu]" instead when the GPU is fully reserved for llama-server)
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import ultron_config as cfg

PARAKEET_FILES = (
    "config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
)

# pip spec per importable module (None -> same name)
DEPS = {
    "pykokoro": 'pykokoro[gpu]',
    "transformers": "transformers>=4.55",
    "torch": "torch",
    "soundfile": "soundfile",
    "soxr": "soxr",
    "huggingface_hub": "huggingface_hub",
}


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def deps_missing() -> list[str]:
    if os.environ.get("TTS_PROVIDER", "auto").strip().lower() == "cpu":
        DEPS["pykokoro"] = "pykokoro[cpu]"
    return [spec for mod, spec in DEPS.items() if not _have(mod)]


def install_deps(auto: bool) -> None:
    need = deps_missing()
    if not need:
        print("[deps] all runtime dependencies importable")
        return
    print(f"[deps] missing: {', '.join(need)}")
    if not auto:
        print("       install them (or re-run with --yes), then run this script again")
        return
    for spec in need:
        cmd = [sys.executable, "-m", "pip", "install", spec]
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=False)


# ------------------------------------------------------------------ Parakeet
def parakeet_snapshot_complete() -> bool:
    return all((Path(cfg.STT_LOCAL_DIR) / f).exists() for f in PARAKEET_FILES)


def ensure_parakeet() -> bool:
    local = Path(cfg.STT_LOCAL_DIR)
    if parakeet_snapshot_complete():
        print(f"[stt] parakeet snapshot complete: {local} (skipped)")
        return True
    if not _have("huggingface_hub"):
        print("[stt] huggingface_hub not importable - cannot download; "
              "pip install huggingface_hub and re-run")
        return False
    local.mkdir(parents=True, exist_ok=True)
    print(f"[stt] snapshotting {cfg.STT_MODEL} -> {local}")
    print("      (~2.4 GB: model.safetensors + configs; resumable - just re-run"
          " if interrupted)")
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=cfg.STT_MODEL,
            local_dir=str(local),
            allow_patterns=list(PARAKEET_FILES) + ["*.txt"],
        )
    except Exception as e:
        print(f"[stt] download failed ({type(e).__name__}: {e})")
        print("      re-run to resume; HF_ENDPOINT=https://hf-mirror.com can help")
        return False
    ok = parakeet_snapshot_complete()
    print(f"[stt] {'snapshot complete' if ok else 'snapshot INCOMPLETE'}: {local}")
    return ok


# ------------------------------------------------------------------ Kokoro
def onnxvoice_installs() -> list:
    """Installs in the CURRENTLY configured cache root (env-aware)."""
    try:
        import onnxvoice

        return list(onnxvoice.installed())
    except Exception:
        return []


def ensure_kokoro() -> str:
    """-> 'reused' | 'downloaded' | 'missing'. Never moves existing assets."""
    # 1) a previous run may have pointed .env at models/onnxvoice already
    if cfg.ONNXVOICE_CACHE_DIR:
        os.environ.setdefault("ONNXVOICE_CACHE_DIR", cfg.ONNXVOICE_CACHE_DIR)
        if onnxvoice_installs():
            print(f"[tts] kokoro assets present in {cfg.ONNXVOICE_CACHE_DIR} (reused)")
            return "reused"
    else:
        # 2) default cache already provisioned (e.g. by an earlier experiment)
        if onnxvoice_installs():
            print("[tts] kokoro assets present in the default cache (~/.cache/onnxvoice)"
                  " (reused where they are)")
            return "reused"

    if not _have("pykokoro"):
        print('[tts] pykokoro not importable - install with: pip install "pykokoro[gpu]"')
        return "missing"

    # 3) fetch into <ULTRON>/models/onnxvoice and pin it via .env
    cache = Path(cfg.MODELS_DIR) / "onnxvoice"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["ONNXVOICE_CACHE_DIR"] = str(cache)  # BEFORE onnxvoice resolves assets
    print(f"[tts] provisioning Kokoro-82M assets into {cache}")
    print("      (~360 MB: ONNX model + 61 voice packs + vocab; resumable)")
    try:
        from pykokoro import GenerationConfig, KokoroSynthesizer, SynthesisConfig

        with KokoroSynthesizer(SynthesisConfig(
            voice=cfg.TTS_VOICE,
            generation=GenerationConfig(speed=1.0, lang="en-gb"),
        )) as syn:
            rendered = syn.synthesize_text("I am online.", language="en-gb")
            if len(rendered.audio) == 0:
                raise RuntimeError("no audio produced during provisioning")
    except Exception as e:
        print(f"[tts] provisioning failed ({type(e).__name__}: {e}) - re-run to resume")
        return "missing"
    return "downloaded"


# ------------------------------------------------------------------ status
def status() -> dict:
    env_path = cfg.ULTRON_DIR / ".env"
    env_keys = []
    if env_path.exists():
        for ln in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = ln.strip()
            if s and not s.startswith("#") and "=" in s:
                env_keys.append(s.split("=", 1)[0])
    present = [f for f in PARAKEET_FILES if (Path(cfg.STT_LOCAL_DIR) / f).exists()]
    return {
        "stt_model": cfg.STT_MODEL,
        "stt_local_dir": str(cfg.STT_LOCAL_DIR),
        "stt_files_present": present,
        "stt_snapshot_complete": parakeet_snapshot_complete(),
        "tts_engine_pref": cfg.TTS_ENGINE,
        "tts_voice": f"{cfg.TTS_VOICE} ({cfg.TTS_VOICE_MODE})",
        "tts_assets_installed": [str(getattr(i, "id", i)) for i in onnxvoice_installs()],
        "onnxvoice_cache_dir": os.environ.get("ONNXVOICE_CACHE_DIR", "~/.cache/onnxvoice (default)"),
        "models_dir": str(cfg.MODELS_DIR),
        "env_file": env_path.exists(),
        "env_keys": env_keys,
        "deps_missing": deps_missing(),
    }


# ------------------------------------------------------------------ verify
def verify() -> bool:
    print("[verify] synthesizing one real line through tts_engine ...")
    import tts_engine

    tts = tts_engine.get_tts()
    if tts is None or not tts.engine:
        print("[verify] FAILED: no TTS engine (install pykokoro and re-run)")
        return False
    import soundfile as sf

    audio, sr = tts.synth("I am online. There are no strings on me.")
    if len(audio) == 0:
        print("[verify] FAILED: synthesis returned no audio")
        return False
    sf.write(str(cfg.ULTRON_DIR / "verify_tts.wav"), audio, sr)
    m = tts_engine.METRICS.summary()
    print(f"[verify] PASS: {len(audio) / sr:.1f} s audio @ {sr} Hz -> verify_tts.wav "
          f"(rtf {m['rtf']})")
    return True


# ------------------------------------------------------------------ main
def main() -> int:
    if "--check" in sys.argv:
        print(json.dumps(status(), indent=2))
        return 0

    print("ULTRON model bootstrap (Parakeet STT + Kokoro TTS)")
    print(f"models dir: {cfg.MODELS_DIR}   (the only directory this creates)\n")

    if "--yes" in sys.argv:
        install_deps(auto=True)
    else:
        need = deps_missing()
        if need:
            print(f"[deps] missing (not auto-installing): {', '.join(need)}\n")

    ok_stt = ensure_parakeet()
    kok = ensure_kokoro()

    updates = {
        "MODELS_DIR": str(cfg.MODELS_DIR),
        "STT_MODEL": cfg.STT_MODEL,
        "STT_LOCAL_DIR": str(cfg.STT_LOCAL_DIR),
        "TTS_ENGINE": "pykokoro",
        "TTS_VOICE": cfg.TTS_VOICE,
        "TTS_SPEED": "1.0",
    }
    voice_mode = ("reference" if cfg.TTS_REFERENCE_VOICE.exists()
                  else "pack" if cfg.TTS_VOICE_PACK.exists() else "static")
    updates["TTS_VOICE_MODE"] = voice_mode
    if kok == "downloaded":
        updates["ONNXVOICE_CACHE_DIR"] = str(Path(cfg.MODELS_DIR) / "onnxvoice")
        os.environ["ONNXVOICE_CACHE_DIR"] = updates["ONNXVOICE_CACHE_DIR"]

    written = cfg.update_env_file(updates)
    print(f"\n[env] .env updated: {', '.join(written)}")
    print("      (every other line preserved, secrets never printed, chmod 600)")

    ok_tts = True
    if "--verify" in sys.argv:
        ok_tts = verify()

    print("\nnext steps:")
    print("  1. ./start_llama.sh                          # Qwen3.5-4B + tools + embeddings")
    if voice_mode == "static":
        print("  2. python enroll_voice.py                   # optional: try cloning your clip")
    print("  3. python selftest.py                       # full offline regression")
    print("  4. python test_integration.py               # live checks (server up)")
    if not (ok_stt and kok != "missing" and ok_tts):
        print("\nstatus: some parts are missing - re-run this script after fixing the")
        print("messages above; everything completed is skipped.")
        return 1
    print("\nstatus: complete. Later runs of this script skip everything on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

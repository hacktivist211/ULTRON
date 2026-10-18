#!/usr/bin/env python3
"""One-time voice stack bootstrap for IRIS.

You said you do not have the voice models yet - this script installs them.

What it installs (and only what is missing):
  1. The CosyVoice engine: `git clone --recursive` of FunAudioLLM/CosyVoice
     into  <IRIS>/CosyVoice  (this is the TTS codebase itself - installing it
     is like installing a package; it is the only directory ever created).
     The recursive clone is required: CosyVoice vendors Matcha-TTS as a
     submodule under third_party/.
  2. The CosyVoice2-0.5B voice model files (~3.9 GB) from HuggingFace into
     <IRIS>/CosyVoice/pretrained_models/CosyVoice2-0.5B - the canonical path
     CosyVoice expects. Every file is size-verified, so interrupted downloads
     are resumed by simply re-running this script.
  3. Writes COSYVOICE_ROOT and COSYVOICE_MODEL into .env (other lines such as
     your TAVILY_API_KEY are preserved byte-for-byte, never printed). From the
     next run onward, everything loads locally - zero downloads.
  4. Optionally installs missing python dependencies (torch, torchaudio,
     onnxruntime-gpu, transformers, ...) after asking, or immediately with
     --yes.

Usage:
  python setup_voice.py             # interactive
  python setup_voice.py --yes       # non-interactive, auto pip install
  python setup_voice.py --check     # status report only, no changes

If huggingface.co is slow from your network, point at a mirror first:
  HF_ENDPOINT=https://hf-mirror.com python setup_voice.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

import iris_config as cfg

ENGINE_URL = "https://github.com/FunAudioLLM/CosyVoice"
ENGINE_DIR = cfg.IRIS_DIR / "CosyVoice"
MODEL_DIR = ENGINE_DIR / "pretrained_models" / "CosyVoice2-0.5B"
HF_BASE = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
HF_REPO = "FunAudioLLM/CosyVoice2-0.5B"

# Parts of the HF repo that are NOT needed for zero-shot cloning.
SKIP_PREFIXES = ("CosyVoice-BlankEN/", "asset/")
SKIP_FILES = {".gitattributes", "README.md", ".gitignore"}

# Runtime python dependencies (module -> pip package; None = same name).
RUNTIME_DEPS = {
    "torch": None,
    "torchaudio": None,
    "transformers": None,
    "hyperpyyaml": "HyperPyYAML",
    "conformer": None,
    "librosa": None,
    "inflect": None,
    "onnxruntime": "onnxruntime-gpu",
    "tqdm": None,
}
OPTIONAL_DEPS = {"faster_whisper": "faster-whisper"}  # enrollment transcription


# ------------------------------------------------------------------ .env editor
def update_env(env_path: Path, updates: dict) -> list:
    """Insert/update keys in .env, preserving every other line untouched.

    Returns the list of keys written. Never prints file contents (the file
    may contain secrets).
    """
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out, seen = [], set()
    for ln in lines:
        stripped = ln.strip()
        key = stripped.split("=", 1)[0].strip() if ("=" in stripped and not stripped.startswith("#")) else None
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(ln)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}={v}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass
    return sorted(seen.union(updates))


# ------------------------------------------------------------------ engine
def engine_ok() -> bool:
    return (ENGINE_DIR / "cosyvoice" / "cli" / "cosyvoice.py").exists()


def submodule_ok() -> bool:
    return (ENGINE_DIR / "third_party" / "Matcha-TTS" / "matcha" / "models" / "components" / "transformer.py").exists()


def ensure_engine() -> bool:
    if engine_ok():
        if not submodule_ok():
            print("[engine] Matcha-TTS submodule missing, initializing ...")
            r = subprocess.run(
                ["git", "-C", str(ENGINE_DIR), "submodule", "update", "--init", "--recursive"],
                check=False,
            )
            if r.returncode != 0:
                print("[engine] WARN: submodule init failed; CosyVoice may fail to import")
        return True
    if shutil.which("git") is None:
        print("ERROR: git not found. Install it (sudo dnf install git) and re-run.")
        return False
    print(f"[engine] cloning CosyVoice (with Matcha-TTS submodule) into {ENGINE_DIR} ...")
    r = subprocess.run(
        ["git", "clone", "--recursive", "--depth", "1", ENGINE_URL, str(ENGINE_DIR)],
        check=False,
    )
    return r.returncode == 0 and engine_ok()


# ------------------------------------------------------------------ model files
def keep(rel: str) -> bool:
    """True if a repo file belongs to the zero-shot voice stack."""
    if rel in SKIP_FILES:
        return False
    return not any(rel.startswith(p) for p in SKIP_PREFIXES)


def repo_files() -> list:
    """[(relative path, size)] for the HF model repo, skipping unneeded parts."""
    r = requests.get(f"{HF_BASE}/api/models/{HF_REPO}?blobs=true", timeout=20)
    r.raise_for_status()
    out = []
    for s in r.json().get("siblings", []):
        name = s.get("rfilename", "")
        size = int(s.get("size") or 0)
        if keep(name):
            out.append((name, size))
    return out


def download_file(rel: str, size: int) -> bool:
    dest = MODEL_DIR / rel
    if dest.exists() and (size == 0 or dest.stat().st_size == size):
        return False  # already present, verified
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE}/{HF_REPO}/resolve/main/{requests.utils.quote(rel)}"
    tmp = Path(str(dest) + ".part")
    for attempt in (1, 2):
        try:
            with requests.get(url, stream=True, timeout=(10, 60), allow_redirects=True) as r:
                r.raise_for_status()
                done = 0
                t_last = 0.0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        if time.time() - t_last > 1.0:
                            total = size / 1e6 if size else done / 1e6
                            print(f"\r  {rel}: {done / 1e6:8.1f} / {total:8.1f} MB", end="", flush=True)
                            t_last = time.time()
            print(f"\r  {rel}: {done / 1e6:8.1f} / {(size or done) / 1e6:8.1f} MB - done")
            if size and done != size:
                raise IOError(f"size mismatch: got {done}, expected {size}")
            tmp.replace(dest)
            return True
        except Exception as e:
            print(f"\n  {rel}: attempt {attempt} failed ({type(e).__name__}: {e})")
            if attempt == 2:
                print(f"  MANUAL FALLBACK: download {url} into {MODEL_DIR}")
                return False
            time.sleep(2)
    return False


def ensure_models() -> bool:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        files = repo_files()
    except Exception as e:
        print(f"[models] cannot list {HF_BASE}/{HF_REPO} ({e}).")
        existing = sorted(p.name for p in MODEL_DIR.glob("*") if p.is_file())
        if "llm.pt" in existing and "campplus.onnx" in existing:
            print("[models] local files already present - continuing with those.")
            return True
        print("[models] no network and no local model files. Re-run later, or")
        print(f"          manually download from {HF_BASE}/{HF_REPO} into {MODEL_DIR}")
        return False
    total = sum(s for _, s in files)
    print(f"[models] {len(files)} files, {total / 1e9:.2f} GB total (skip already-downloaded)")
    got = 0
    for rel, size in files:
        if download_file(rel, size):
            got += 1
    print(f"[models] downloaded {got} new file(s); model dir: {MODEL_DIR}")
    core = {"llm.pt", "flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v2.onnx"}
    return core.issubset({p.name for p in MODEL_DIR.glob("*") if p.is_file()})


# ------------------------------------------------------------------ dependencies
def missing_deps(extra: dict | None = None) -> list:
    deps = dict(RUNTIME_DEPS)
    deps.update(extra or {})
    out = []
    for module, pip_name in deps.items():
        if importlib.util.find_spec(module) is None:
            out.append(pip_name or module)
    return out


def install_deps(auto: bool) -> None:
    need = missing_deps()
    if not need:
        print("[deps] all runtime dependencies already importable")
        return
    print(f"[deps] missing: {', '.join(need)}")
    if not auto:
        reply = input("install them now with pip? [y/N] ").strip().lower()
        if reply != "y":
            print("[deps] skipped. install manually, then re-run enroll_voice.py")
            return
    torch_part = [p for p in ("torch", "torchaudio") if p in need]
    rest = [p for p in need if p not in ("torch", "torchaudio")]
    cmds = []
    if torch_part:
        cmds.append([sys.executable, "-m", "pip", "install", *torch_part,
                     "--index-url", "https://download.pytorch.org/whl/cu124"])
    if rest:
        cmds.append([sys.executable, "-m", "pip", "install", *rest])
    for cmd in cmds:
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=False)
    opt = [p for m, p in OPTIONAL_DEPS.items() if importlib.util.find_spec(m) is None]
    if opt:
        print(f"[deps] optional (enrollment transcription): pip install {' '.join(opt)}")
    print("[deps] note: if text normalization errors appear later:")
    print("       conda install -c conda-forge pynini")


# ------------------------------------------------------------------ status / main
def status() -> dict:
    env_path = cfg.IRIS_DIR / ".env"
    env_keys = []
    if env_path.exists():
        for ln in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = ln.strip()
            if s and not s.startswith("#") and "=" in s:
                env_keys.append(s.split("=", 1)[0])
    files = [p.name for p in MODEL_DIR.glob("*") if p.is_file()] if MODEL_DIR.exists() else []
    return {
        "engine": engine_ok() and submodule_ok(),
        "model_files": len(files),
        "model_dir": str(MODEL_DIR) if files else None,
        "env_file": env_path.exists(),
        "env_keys": env_keys,
        "deps_missing": missing_deps(),
    }


def main() -> int:
    auto = "--yes" in sys.argv
    if "--check" in sys.argv:
        print(json.dumps(status(), indent=2))
        return 0

    print(f"IRIS voice setup  (engine: {ENGINE_DIR.name}/  models: {HF_REPO})")
    print(f"flat rule: only the CosyVoice engine checkout is created; all IRIS files stay flat.\n")

    ok = ensure_engine()
    if not ok:
        return 1
    print(f"[engine] ready: {ENGINE_DIR}")

    ok = ensure_models()
    if not ok:
        return 1

    written = update_env(
        cfg.IRIS_DIR / ".env",
        {"COSYVOICE_ROOT": str(ENGINE_DIR), "COSYVOICE_MODEL": str(MODEL_DIR)},
    )
    os.environ["COSYVOICE_ROOT"] = str(ENGINE_DIR)
    os.environ["COSYVOICE_MODEL"] = str(MODEL_DIR)
    print(f"[env] .env updated with: {', '.join(written)} (other lines preserved)")

    install_deps(auto)

    print("\nsetup complete. next steps:")
    print("  1. python enroll_voice.py        # one-time: learn your JARVIS voice")
    print("  2. re-running THIS script will skip everything already downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

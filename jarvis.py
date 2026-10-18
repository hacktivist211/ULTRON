#!/usr/bin/env python3
"""
JARVIS Voice Assistant - Core Entry Point
Engine: CosyVoice2 TTS + llama-server (Qwen 3.5)
"""
import sys
import time
import queue
import threading
import numpy as np

import iris_config as cfg
import voice_cache
import tool_runtime
from memory_store import get_store

try:
    import sounddevice as sd
except ImportError:
    sys.exit("sounddevice is required. Install with: pip install sounddevice")


class ThinkFilter:
    """Strips <think>...</think> reasoning blocks from live streaming text."""
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.pending = ""
        self.in_think = False

    def _hold(self, tag):
        for size in range(min(len(tag) - 1, len(self.pending)), 0, -1):
            if self.pending.endswith(tag[:size]):
                return size
        return 0

    def feed(self, text):
        self.pending += text
        visible = []
        while True:
            if self.in_think:
                idx = self.pending.find(self.CLOSE)
                if idx == -1:
                    keep = self._hold(self.CLOSE)
                    self.pending = self.pending[len(self.pending) - keep:] if keep else ""
                    break
                self.pending = self.pending[idx + len(self.CLOSE):]
                self.in_think = False
            else:
                idx = self.pending.find(self.OPEN)
                if idx == -1:
                    keep = self._hold(self.OPEN)
                    cut = len(self.pending) - keep
                    visible.append(self.pending[:cut])
                    self.pending = self.pending[cut:]
                    break
                visible.append(self.pending[:idx])
                self.pending = self.pending[idx + len(self.OPEN):]
                self.in_think = True
        return "".join(visible)

    def flush(self):
        tail = "" if self.in_think else self.pending
        self.pending = ""
        return tail


class SpeechChunker:
    """Groups streaming text into natural clauses and sentences for TTS synthesis."""
    def __init__(self, soft_chars=35):
        self.soft_chars = soft_chars
        self.buffer = ""

    def feed(self, text):
        self.buffer += text
        pieces = []
        import re
        while True:
            match = re.search(r"[.!?]+\s+|\n+|,|;|:", self.buffer)
            if not match:
                break
            idx = match.end()
            raw, self.buffer = self.buffer[:idx], self.buffer[idx:]
            clean = cfg.speech_clean(raw)
            if clean:
                pieces.append(clean)
        return pieces

    def flush(self):
        clean = cfg.speech_clean(self.buffer)
        self.buffer = ""
        return [clean] if clean else []


class VoicePlayer:
    """Threaded non-blocking audio output player using sounddevice."""
    def __init__(self, cv_model, profile):
        self.cv = cv_model
        self.profile = profile
        self.sr = cfg.TTS_SAMPLE_RATE
        self.q = queue.Queue()
        self.stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32")
        self.stream.start()
        self.last_filler = -1
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            audio = self.q.get()
            if audio is None:
                break
            try:
                self.stream.write(audio.reshape(-1, 1))
            finally:
                self.q.task_done()

    def speak_text(self, text):
        if not text.strip():
            return
        for chunk in voice_cache.cached_inference(self.cv, text, self.profile, stream=True):
            pcm = chunk["tts_speech"].squeeze(0).cpu().numpy().astype(np.float32)
            self.q.put(pcm)

    def play_filler(self):
        if not self.profile or not getattr(self.profile, "fillers", None):
            return
        import random
        choices = [i for i in range(len(self.profile.fillers)) if i != self.last_filler] or [0]
        self.last_filler = random.choice(choices)
        self.q.put(self.profile.fillers[self.last_filler])

    def wait_idle(self):
        self.q.join()


class JarvisEngine:
    """Main orchestrator connecting memory retrieval, LLM streaming, tools, and speech."""
    def __init__(self):
        print("[jarvis] Loading voice profile fingerprint...")
        self.profile = voice_cache.load_profile()
        if not self.profile:
            sys.exit("[jarvis] Voice profile missing. Run 'python setup_voice.py' and 'python enroll_voice.py' first.")

        print("[jarvis] Loading CosyVoice2 model...")
        cv_root = cfg.find_cosyvoice_root()
        cv_model_dir = cfg.find_cosyvoice_model()
        if not cv_root or not cv_model_dir:
            sys.exit("[jarvis] CosyVoice directory not found. Run 'python setup_voice.py' first.")

        sys.path.insert(0, str(cv_root))
        sys.path.insert(0, str(cv_root / "third_party" / "Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import CosyVoice2

        self.cv = CosyVoice2(str(cv_model_dir), load_jit=False, load_trt=False, fp16=cfg.COSYVOICE_FP16)
        self.cache = voice_cache.install_frontend_cache(self.cv, self.profile)
        self.player = VoicePlayer(self.cv, self.profile)
        self.memory = get_store()
        self.history = []

        print("[jarvis] Warming up audio pipeline...")
        boot_audio = voice_cache.warm(self.cv, self.profile, self.cache)
        if boot_audio is not None:
            self.player.q.put(boot_audio)

    def run_turn(self, user_text):
        if not user_text.strip():
            return

        sys_prompt = tool_runtime.build_system(budget="standard", memory_query=user_text, store=self.memory)
        messages = [{"role": "system", "content": sys_prompt}] + self.history[-6:] + [{"role": "user", "content": user_text}]

        chunker = SpeechChunker()
        think = ThinkFilter()
        filler_played = False
        full_response = []

        def on_delta(piece):
            nonlocal filler_played
            visible = think.feed(piece)
            if visible:
                full_response.append(visible)
                for clause in chunker.feed(visible):
                    print(clause, end=" ", flush=True)
                    self.player.speak_text(clause)

        def on_tool_start(name):
            nonlocal filler_played
            if not filler_played:
                self.player.play_filler()
                filler_played = True

        try:
            final_text, trace = tool_runtime.chat(
                messages=messages,
                on_delta=on_delta,
                on_tool_start=on_tool_start,
                use_tools=True,
            )

            tail = think.flush()
            if tail:
                full_response.append(tail)
                for clause in chunker.feed(tail):
                    print(clause, end=" ", flush=True)
                    self.player.speak_text(clause)

            for clause in chunker.flush():
                print(clause, end=" ", flush=True)
                self.player.speak_text(clause)

            self.player.wait_idle()
            print()

            reply = "".join(full_response).strip() or cfg.speech_clean(final_text)
            if reply:
                self.history.append({"role": "user", "content": user_text})
                self.history.append({"role": "assistant", "content": reply})

        except Exception as err:
            print(f"\n[jarvis] Error during processing: {err}")
            self.player.speak_text("I encountered an issue reaching my language model, sir.")
            self.player.wait_idle()

    def loop(self):
        print("\n[jarvis] Systems online, sir. Type your query below.")
        while True:
            try:
                user_input = input("\nyou> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "goodbye"):
                    self.player.speak_text("Goodbye, sir.")
                    self.player.wait_idle()
                    break
                self.run_turn(user_input)
            except (KeyboardInterrupt, EOFError):
                print("\n[jarvis] Systems offline.")
                break


if __name__ == "__main__":
    jarvis = JarvisEngine()
    jarvis.loop()
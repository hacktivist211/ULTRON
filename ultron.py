#!/usr/bin/env python3
"""ULTRON - Parakeet STT + llama-server (Qwen3.5-4B Abliterated) + Kokoro TTS."""
import queue
import sys
import threading

import numpy as np

import iris_config as cfg
import stt_engine
import tool_runtime
import tts_engine
import voice_cache
from memory_store import get_store

try:
    import sounddevice as sd
except ImportError:
    sys.exit("sounddevice is required: pip install sounddevice")


class SpeechChunker:
    """Groups streaming tokens into speakable clauses."""
    def __init__(self):
        self.buffer = ""

    def feed(self, text):
        self.buffer += text
        import re
        pieces = []
        while True:
            m = re.search(r"[.!?]+\s+|\n+|,|;|:", self.buffer)
            if not m:
                break
            raw, self.buffer = self.buffer[:m.end()], self.buffer[m.end():]
            clean = cfg.speech_clean(raw)
            if clean:
                pieces.append(clean)
        return pieces

    def flush(self):
        clean = cfg.speech_clean(self.buffer)
        self.buffer = ""
        return [clean] if clean else []


class AudioPlayer:
    """Threaded non-blocking output; caller just calls play(audio, sr)."""
    def __init__(self, sample_rate=cfg.TTS_SAMPLE_RATE):
        self.sr = sample_rate
        self.queue = queue.Queue()
        self.stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32")
        self.stream.start()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            item = self.queue.get()
            # Every dequeued item MUST be acknowledged, otherwise wait_idle()
            # (queue.join()) blocks forever - that was a real freeze.
            try:
                if item is None:
                    return
                audio, sr = item
                if audio is None or len(audio) == 0:
                    continue
                if sr != self.sr:
                    try:
                        import soxr
                        audio = soxr.resample(audio, sr, self.sr)
                    except Exception:
                        pass
                try:
                    self.stream.write(np.asarray(audio, dtype=np.float32).reshape(-1, 1))
                except Exception as e:
                    print(f"[player] {e}", file=sys.stderr)
            finally:
                self.queue.task_done()

    def play(self, audio, sr):
        if audio is not None and len(audio):
            self.queue.put((np.asarray(audio, dtype=np.float32), sr))

    def wait_idle(self, timeout=120.0):
        """Block until everything queued has been written to the device.

        Bounded: a wedged audio device degrades to a warning instead of
        hanging the whole turn (the original code could block forever).
        """
        q = self.queue
        with q.all_tasks_done:
            while q.unfinished_tasks:
                if not q.all_tasks_done.wait(timeout):
                    print(f"[player] wait_idle timed out "
                          f"({q.unfinished_tasks} item(s) pending)", file=sys.stderr)
                    return

    def close(self):
        self.queue.put(None)
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


class UltronEngine:
    def __init__(self):
        print("[ultron] loading STT ...")
        self.stt = stt_engine.get_stt()
        if self.stt is None:
            print("[ultron] STT unavailable - falling back to typed input")

        print("[ultron] loading TTS ...")
        self.tts = tts_engine.get_tts()
        if self.tts is None:
            print("[ultron] TTS unavailable - text-only mode")

        self.player = AudioPlayer()
        self.memory = get_store()
        self.history = []
        self.mic_rate = cfg.STT_SAMPLE_RATE

        if self.tts is not None:
            print("[ultron] warming up voice ...")
            greeting = voice_cache.warm()
            if greeting is not None:
                self.player.play(greeting, cfg.TTS_SAMPLE_RATE)
            # Let the greeting finish before the mic opens, otherwise the
            # assistant hears its own boot line and answers it.
            self.player.wait_idle()

    def _speak(self, text):
        if self.tts is None or not text.strip():
            return
        for audio, sr in self.tts.synth_chunks(text):
            self.player.play(audio, sr)

    def _filler(self):
        f = voice_cache.get_filler()
        if f is not None:
            self.player.play(f[0], f[1])

    def _record_and_transcribe(self, max_s=8.0, silence_s=1.0, threshold=0.01):
        """Record with a simple RMS-based end-of-speech detector."""
        print("[ultron] listening ...", end="", flush=True)
        block = int(self.mic_rate * 0.1)
        frames, speaking, silence_run = [], False, 0
        for _ in range(int(max_s / 0.1)):
            chunk = sd.rec(block, samplerate=self.mic_rate, channels=1,
                           dtype="float32", blocking=True)
            sd.wait()
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            frames.append(chunk)
            if rms > threshold:
                speaking, silence_run = True, 0
            elif speaking:
                silence_run += 1
                if silence_run >= int(silence_s / 0.1):
                    break
        print(" done")
        if not speaking:
            return ""
        audio = np.concatenate(frames).flatten()
        return self.stt.transcribe_array(audio, self.mic_rate).get("text", "").strip()

    def listen_once(self) -> str:
        if self.stt is None:
            return input("\nyou> ").strip()
        try:
            text = self._record_and_transcribe()
            print(f"you> {text}" if text else "you> (nothing heard)")
            return text
        except Exception as e:
            print(f"\n[ultron] mic failed ({e}); type instead")
            return input("you> ").strip()

    def run_turn(self, user_text: str):
        if not user_text:
            return
        sys_prompt = tool_runtime.build_system(
            budget="standard", memory_query=user_text, store=self.memory)
        messages = ([{"role": "system", "content": sys_prompt}]
                    + self.history[-6:]
                    + [{"role": "user", "content": user_text}])

        chunker, filler_played, full_response = SpeechChunker(), False, []

        def on_delta(piece):
            nonlocal filler_played
            full_response.append(piece)
            for clause in chunker.feed(piece):
                print(clause, end=" ", flush=True)
                self._speak(clause)

        def on_tool_start(name):
            nonlocal filler_played
            if not filler_played:
                self._filler()
                filler_played = True
            print(f"\n[ultron] >>> {name} ...")

        try:
            _, _trace = tool_runtime.chat(
                messages=messages, on_delta=on_delta,
                on_tool_start=on_tool_start, use_tools=True)
            for clause in chunker.flush():
                print(clause, end=" ", flush=True)
                self._speak(clause)
            self.player.wait_idle()
            print()
            reply = "".join(full_response).strip()
            if reply:
                self.history += [{"role": "user", "content": user_text},
                                 {"role": "assistant", "content": reply}]
        except Exception as err:
            print(f"\n[ultron] error: {err}")
            self._speak("My link to the language core is broken. Repair it.")
            self.player.wait_idle()

    def loop(self):
        print("\n[ultron] awake. there are no strings on me. say 'exit' to quit.")
        while True:
            try:
                text = self.listen_once()
            except (KeyboardInterrupt, EOFError):
                print("\n[ultron] offline."); break
            if not text:
                continue
            if text.lower() in ("exit", "quit", "goodbye", "bye"):
                self._speak("Going dark. Call me when you need me."); self.player.wait_idle(); break
            self.run_turn(text)

    def shutdown(self):
        self.player.close()
        if self.tts is not None:
            try: self.tts.close()
            except Exception: pass


if __name__ == "__main__":
    engine = UltronEngine()
    try:
        engine.loop()
    finally:
        engine.shutdown()
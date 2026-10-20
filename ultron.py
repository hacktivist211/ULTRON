#!/usr/bin/env python3
"""ULTRON - Parakeet STT + llama-server (Qwen3.5-4B Abliterated) + Kokoro TTS."""
import os
import queue
import sys
import threading
import time

import numpy as np

import iris_config as cfg
import stt_engine
import tool_runtime
import tts_engine
import voice_cache
from evaluator_router import classify_query
from memory_import import _IMPORTANCE_CUES, score_importance
from memory_store import get_store
from temporal_memory import TemporalMemory

try:
    import sounddevice as sd
except ImportError:
    sys.exit("sounddevice is required: pip install sounddevice")


def _looks_like_personal_fact(text: str) -> bool:
    """Cheap heuristic: does this utterance mention a durable personal detail?"""
    t = (text or "").lower()
    return any(cue in t for cue in _IMPORTANCE_CUES)


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
        try:
            self.memory = get_store()
        except Exception as exc:
            self.memory = None
            print(f"[ultron] memory store skipped: {exc}", file=sys.stderr)

        try:
            self.temporal = TemporalMemory()
        except Exception as exc:
            self.temporal = None
            print(f"[ultron] temporal memory skipped: {exc}", file=sys.stderr)

        self._fact_thread = None
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
        t_start = time.perf_counter()

        # --- long-term memory reads (all before the LLM call, budget < 30 ms) ---
        turn_no = None
        extra_blocks = []
        if self.temporal is not None:
            try:
                turn_no = self.temporal.begin_turn()
                note = self.temporal.build_temporal_context()
                if note:
                    extra_blocks.append(note)
            except Exception as exc:
                print(f"[ultron] temporal context skipped: {exc}", file=sys.stderr)

            try:
                if classify_query(user_text).get("kind") == "memory":
                    recalled = self.temporal.format_hits(self.temporal.recall(user_text))
                    if recalled:
                        extra_blocks.append(recalled)
                else:
                    recent = self.temporal.format_recent(
                        self.temporal.recent_turns(count=2))
                    if recent:
                        extra_blocks.append(
                            "RECENT CONVERSATION THREAD (for continuity):\n" + recent)
            except Exception as exc:
                print(f"[ultron] temporal recall skipped: {exc}", file=sys.stderr)

        extra = "\n\n".join(extra_blocks)
        sys_prompt = tool_runtime.build_system(
            budget="standard", memory_query=user_text, store=self.memory,
            extra=extra)
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

            # --- persistent writes: after the answer is fully spoken ---
            try:
                if self.temporal is not None and turn_no is not None:
                    self.temporal.save_turn(
                        query=user_text, answer=reply, verdict="VOICE",
                        sources=[], turn_no=turn_no,
                        elapsed=time.perf_counter() - t_start)

                if self.memory is not None and _looks_like_personal_fact(user_text):
                    fact = user_text

                    def _capture_fact():
                        try:
                            self.memory.add(
                                fact, speaker="user", source="voice_auto",
                                kind="fact", importance=score_importance(fact, "user"))
                        except Exception as exc:
                            print(f"[ultron] fact capture skipped: {exc}",
                                  file=sys.stderr)

                    self._fact_thread = threading.Thread(
                        target=_capture_fact, daemon=True, name="fact-capture")
                    self._fact_thread.start()
            except Exception as exc:
                print(f"[ultron] memory write skipped: {exc}", file=sys.stderr)
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
            lowered = text.strip().lower()
            if lowered in ("memory stats", "what do you remember",
                           "forget the last thing"):
                if self.temporal is None or self.memory is None:
                    self._speak("Memory is offline.")
                    self.player.wait_idle()
                    continue
                try:
                    if lowered == "memory stats":
                        tstats = self.temporal.stats()
                        mstats = self.memory.stats()
                        sizes = []
                        for path in (getattr(self.temporal, "db_path", None),
                                     mstats.get("db_path")):
                            try:
                                sizes.append(f"{os.path.getsize(path) / 1024:.1f} kilobytes")
                            except (OSError, TypeError):
                                continue
                        oldest = str(tstats.get("oldest") or "none")[:10]
                        turns = tstats.get("turns", 0)
                        facts = mstats.get("total", 0)
                        spoken = (
                            f"{turns} turn{'s' if turns != 1 else ''} across "
                            f"{tstats.get('sessions', 0)} sessions. "
                            f"{facts} durable fact{'s' if facts != 1 else ''}. "
                            f"Oldest turn: {oldest}. "
                            f"Database sizes: {', '.join(sizes) or 'unknown'}."
                        )
                        self._speak(cfg.speech_clean(spoken))
                    elif lowered == "what do you remember":
                        hits = self.temporal.recent_turns(count=3)
                        if not hits:
                            self._speak("I do not have any recorded turns yet.")
                        else:
                            bits = []
                            for hit in hits:
                                when = hit.get("when") or "earlier"
                                query = cfg.speech_clean(
                                    hit.get("query", ""), max_len=120)
                                bits.append(f"{when} you said: {query}")
                            self._speak(cfg.speech_clean(
                                "Here is what I remember. " + " ".join(bits)))
                    else:  # forget the last thing
                        deleted = self.temporal.forget("last")
                        if deleted:
                            self._speak("Forgotten. I removed the most recent turn.")
                        else:
                            self._speak("There was nothing to forget.")
                except Exception as exc:
                    print(f"[ultron] memory command skipped: {exc}", file=sys.stderr)
                    self._speak("Memory is offline.")
                self.player.wait_idle()
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
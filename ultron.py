#!/usr/bin/env python3
"""ULTRON - Parakeet STT + llama-server (Qwen3.5-4B Abliterated) + Kokoro TTS."""
import os
import queue
import re
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


class TurnCancelled(Exception):
    """Raised inside the stream callback when the user presses Esc."""


class KeyInterruptMonitor:
    """Poll stdin for a single Esc keypress without extra dependencies.

    POSIX uses cbreak mode + select; Windows uses msvcrt. If stdin is not an
    interactive terminal the monitor declines to start and the turn simply
    runs without interrupt support.
    """

    def __init__(self):
        self.interrupted = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._fd = None
        self._saved = None
        self._cbreak = False
        self._warned = False

    def start(self) -> bool:
        """Enter cbreak mode and start the polling thread. Never raises."""
        if self._thread is not None and self._thread.is_alive():
            return True
        self.reset()
        self._stop.clear()
        if not self._enter():
            if not self._warned:
                print("[ultron] interrupt key unavailable (non-interactive stdin)",
                      file=sys.stderr)
                self._warned = True
            return False
        try:
            self._thread = threading.Thread(
                target=self._poll, daemon=True, name="esc-monitor")
            self._thread.start()
        except Exception:
            self._restore()
            self._thread = None
            if not self._warned:
                print("[ultron] interrupt key unavailable (non-interactive stdin)",
                      file=sys.stderr)
                self._warned = True
            return False
        return True

    def _enter(self) -> bool:
        if os.name == "nt":
            try:
                import msvcrt  # noqa: F401
                return True
            except Exception:
                return False
        try:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            if not os.isatty(self._fd):
                return False
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._cbreak = True
            return True
        except Exception:
            return False

    def _poll(self) -> None:
        if os.name == "nt":
            try:
                import msvcrt

                while not self._stop.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b"\x1b", "\x1b"):
                            self.interrupted.set()
                            return
                    time.sleep(0.02)
            except Exception:
                return
            return

        try:
            import select

            while not self._stop.is_set():
                ready, _, _ = select.select([self._fd], [], [], 0.02)
                if not ready:
                    continue
                if os.read(self._fd, 1) == b"\x1b":
                    self.interrupted.set()
                    return
        except Exception:
            return

    def _restore(self) -> None:
        if self._cbreak and self._saved is not None and self._fd is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:
                pass
        self._cbreak = False
        self._saved = None

    def stop(self) -> None:
        """Stop polling and restore the terminal. Never raises."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            try:
                thread.join(timeout=0.2)
            except Exception:
                pass
        self._thread = None
        self._restore()

    def reset(self) -> None:
        self.interrupted.clear()

    def wait_for_esc(self, timeout: float | None = None) -> bool:
        """Test helper: block until Esc is seen (or timeout)."""
        return self.interrupted.wait(timeout)


# --- lightweight pre-retrieval gate (Adaptive-RAG style, zero LLM calls) ---
_TOOL_SETS = {
    "simple": [],
    "memory": ["memory_search", "memory_add"],
    "standard": ["web_search", "memory_search", "memory_add"],
}

# classify_query's CASUAL_RE does not cover "how are you doing today", and a
# farewell can be prefixed ("Auntie, goodbye."). Keep the supplement narrow so
# real knowledge questions still go through the standard path.
_CASUAL_EXTRA_RE = re.compile(
    r"^(?:how\s+are\s+you(?:\s+doing)?(?:\s+today)?|"
    r"how(?:'s| is)\s+it\s+going|how(?:'s| is)\s+your\s+day|"
    r"what(?:'s| is)\s+up|you\s+good|you\s+okay|"
    r"are\s+you\s+(?:there|awake|real))"
    r"\b[?.!\s]*$",
    re.IGNORECASE,
)
_FAREWELL_START_RE = re.compile(
    r"^(?:bye|goodbye|farewell|see\s+ya|see\s+you(?:\s+later)?|"
    r"talk\s+to\s+you\s+later|catch\s+you\s+later)\b",
    re.IGNORECASE,
)
_FAREWELL_VOCATIVE_RE = re.compile(
    r"[,;]\s*(?:bye|goodbye|farewell|see\s+ya|see\s+you(?:\s+later)?)\b"
    r"[.!?~\s]*$",
    re.IGNORECASE,
)


def _looks_casual(text: str) -> bool:
    """True for short greetings/farewells that must never trigger tools."""
    compact = " ".join((text or "").split())
    if not compact:
        return True
    if _CASUAL_EXTRA_RE.match(compact):
        return True
    if len(compact.split()) > 6:
        return False
    return bool(_FAREWELL_START_RE.match(compact)
                or _FAREWELL_VOCATIVE_RE.search(compact))


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

    def drain(self):
        """Discard queued audio without playing it (used on turn cancellation).

        Every dequeued item is acknowledged so wait_idle() cannot deadlock.
        A shutdown sentinel is put back for the worker.
        """
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                return
            try:
                if item is None:
                    self.queue.put(None)
                    return
            finally:
                self.queue.task_done()

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
        self._keys = KeyInterruptMonitor()
        self._stop_recording_event = threading.Event()
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

    def _record_and_transcribe(self, max_s=60.0, silence_s=3.0, threshold=0.01):
        """Record until Enter, a long silence, or the max_s safety net.

        Enter is read by a daemon thread that only consumes a single line and
        exits, so it cannot swallow later typed input.  The old RMS silence
        auto-stop is retained as a fallback but only after at least 400 ms of
        actual speech, and only after three seconds of silence.
        """
        print("[ultron] listening ... (press Enter when done)", end="", flush=True)
        block = int(self.mic_rate * 0.1)
        frames, speaking, silence_run, speech_blocks = [], False, 0, 0
        stop = self._stop_recording_event
        stop.clear()
        monitor_done = threading.Event()

        def _watch_enter():
            try:
                stream = sys.stdin
                if stream is None:
                    return
                try:
                    fd = stream.fileno()
                except Exception:
                    fd = None
                if fd is not None:
                    import select

                    while not monitor_done.is_set():
                        ready, _, _ = select.select([fd], [], [], 0.1)
                        if ready:
                            stream.readline()
                            stop.set()
                            return
                    return
                # Non-file stdin (tests, StringIO): one blocking read.  The
                # daemon exits after that single line and cannot be cancelled
                # portably.
                stream.readline()
                stop.set()
            except Exception:
                return

        monitor = None
        if sys.stdin is None or not hasattr(sys.stdin, "readline"):
            print("\n[ultron] Enter-to-stop unavailable (no stdin); "
                  "using silence auto-stop", file=sys.stderr)
        else:
            monitor = threading.Thread(target=_watch_enter, daemon=True,
                                       name="enter-stop-monitor")
            try:
                monitor.start()
            except Exception as exc:
                monitor = None
                print(f"\n[ultron] Enter-to-stop unavailable ({exc}); "
                      "using silence auto-stop", file=sys.stderr)

        try:
            for _ in range(max(1, int(max_s / 0.1))):
                if stop.is_set():
                    break
                chunk = sd.rec(block, samplerate=self.mic_rate, channels=1,
                               dtype="float32", blocking=True)
                sd.wait()
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                frames.append(chunk)
                if rms > threshold:
                    speaking, silence_run = True, 0
                    speech_blocks += 1
                elif speaking:
                    silence_run += 1
                    if (speech_blocks >= 4
                            and silence_run >= int(silence_s / 0.1)):
                        break
        finally:
            monitor_done.set()
            stop.clear()
        print(" done")
        if not speaking:
            return ""
        audio = np.concatenate(frames).flatten()
        return self.stt.transcribe_array(audio, self.mic_rate).get("text", "").strip()

    def _abort_turn(self):
        """Discard the in-flight turn and acknowledge the re-listen."""
        try:
            self.player.drain()
        except Exception as exc:
            print(f"[ultron] audio drain skipped: {exc}", file=sys.stderr)
        print("[ultron] turn cancelled — listening again")
        self._speak("Listening again.")
        try:
            self.player.wait_idle(timeout=2.0)
        except TypeError:  # very old/stub player without a timeout argument
            self.player.wait_idle()

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

        # --- lightweight retrieval gate: classify once, reuse everywhere ---
        try:
            kind = classify_query(user_text).get("kind", "standard")
        except Exception as exc:
            print(f"[ultron] query classifier failed ({exc}); offering all tools",
                  file=sys.stderr)
            kind = "standard"
        if kind == "standard" and _looks_casual(user_text):
            kind = "simple"
        allowed_tools = list(_TOOL_SETS.get(kind, _TOOL_SETS["standard"]))
        print(f"[ultron] tools: {kind} -> {','.join(allowed_tools) or 'none'}")

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
                if kind == "memory":
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
        # Simple greetings do not need durable-memory retrieval either; skip
        # the embedding lookup and keep the turn cheap.
        sys_prompt = tool_runtime.build_system(
            budget="standard",
            memory_query=None if kind == "simple" else user_text,
            store=None if kind == "simple" else self.memory,
            extra=extra)
        messages = ([{"role": "system", "content": sys_prompt}]
                    + self.history[-6:]
                    + [{"role": "user", "content": user_text}])

        chunker, filler_played, full_response = SpeechChunker(), False, []

        def on_delta(piece):
            nonlocal filler_played
            if self._keys is not None and self._keys.interrupted.is_set():
                raise TurnCancelled()
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
            try:
                if self._keys is not None:
                    self._keys.start()
                _, _trace = tool_runtime.chat(
                    messages=messages, on_delta=on_delta,
                    on_tool_start=on_tool_start,
                    use_tools=bool(allowed_tools),
                    allowed_tools=allowed_tools)
            finally:
                if self._keys is not None:
                    self._keys.stop()
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
        except TurnCancelled:
            self._abort_turn()
            return
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
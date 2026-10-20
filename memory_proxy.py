#!/usr/bin/env python3
"""ULTRON shared-memory proxy for OpenAI-compatible chat UIs.

The voice loop (``ultron.py``) already reads and writes ``memory.db`` and
``ultron_memory/temporal_memory.db``.  Chat UIs pointed at llama-server:8080
bypass that loop, so their turns vanish from ULTRON's memory.  This module
sits on port 8081 and makes UI turns use the exact same pipeline:

    UI -> memory_proxy:8081 -> memory read + prompt composition
                            -> llama-server:8080 (streamed)
                            -> temporal_memory + memory.db (persisted)

It exposes only the OpenAI-compatible surface a chat UI needs:

    POST /v1/chat/completions   streamed or buffered completion
    GET  /v1/models             one-entry model list
    GET  /health                liveness / upstream identity
    GET  /stats                 today's UI turn + fact counts

Stdlib only (``http.server`` + ``ThreadingHTTPServer``).  Streaming uses
HTTP/1.1 chunked transfer encoding, so deltas reach the UI as soon as they
reach the proxy.  The UI prompt is ``PERSONA_RULES_UI``: the same character
as the voice loop, with markdown and text-sized budgets.  Client-supplied
OpenAI tool schemas are forwarded when present; without them the model is
told it has no tools and must not pretend to search.  Memory context still
arrives through ``build_system(memory_query=...)``.

CLI:
    python memory_proxy.py
    python memory_proxy.py --host 127.0.0.1 --port 8081 \
        --upstream http://127.0.0.1:8080

Import:
    from memory_proxy import start_in_background
    handle = start_in_background(port=8081)
    ...
    handle.shutdown()

Environment (read here, not in ultron_config.py):
    MEMORY_PROXY_HOST              default 127.0.0.1
    MEMORY_PROXY_PORT              default 8081
    MEMORY_PROXY_UPSTREAM          default cfg.LLAMA_SERVER_URL
    MEMORY_PROXY_ENABLED           default 1; 0 disables all memory writes
    MEMORY_PROXY_TOOLS             default 1; forward client tool schemas
    MEMORY_PROXY_KEEP_CLIENT_SYSTEM default 0; drop the UI's own system prompt
    MEMORY_PROXY_UI_PERSONA        default 1; use PERSONA_RULES_UI
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import requests

import ultron_config as cfg
import memory_import
import memory_store
import temporal_memory
import tool_runtime
from evaluator_router import classify_query


# ------------------------------------------------------------------ configuration
def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "off", "no")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


DEFAULT_HOST = os.getenv("MEMORY_PROXY_HOST", "127.0.0.1").strip() or "127.0.0.1"
DEFAULT_PORT = _env_int("MEMORY_PROXY_PORT", 8081)
DEFAULT_UPSTREAM = os.getenv(
    "MEMORY_PROXY_UPSTREAM", cfg.LLAMA_SERVER_URL
).strip().rstrip("/") or cfg.LLAMA_SERVER_URL.rstrip("/")

# 0 turns the proxy into a read-only passthrough: memory is still injected, but
# no temporal turn or durable fact is written.  Useful when debugging a UI.
PERSISTENCE_ENABLED = _env_flag("MEMORY_PROXY_ENABLED", True)

# UI persona and text budgets are deliberately separate from the voice prompt.
UI_PERSONA_ENABLED = _env_flag("MEMORY_PROXY_UI_PERSONA", True)
UI_BUDGETS = {"quick": 150, "standard": 400, "detailed": 1000}

# A chat UI may send its own OpenAI tool schemas.  They are authoritative when
# present; without them the model gets no tools and must not pretend otherwise.
TOOLS_ENABLED = _env_flag("MEMORY_PROXY_TOOLS", True)

# Most UIs inject "You are a helpful assistant", which fights the persona.
# Drop it by default; opt back in with MEMORY_PROXY_KEEP_CLIENT_SYSTEM=1.
KEEP_CLIENT_SYSTEM = _env_flag("MEMORY_PROXY_KEEP_CLIENT_SYSTEM", False)

_UPSTREAM_SUFFIX = "/v1/chat/completions"


def _upstream_base(upstream: str | None = None) -> str:
    """Normalize an upstream argument to a bare base URL."""
    url = (upstream or DEFAULT_UPSTREAM).strip().rstrip("/")
    if url.endswith(_UPSTREAM_SUFFIX):
        url = url[: -len(_UPSTREAM_SUFFIX)]
    if url.endswith("/v1"):
        url = url[:-3]
    return url.rstrip("/")


# ------------------------------------------------------------------ shared state
_STORE = None
_TEMPORAL = None
_STATE_LOCK = threading.Lock()

_TURN_LOCK = threading.Lock()
_TURN_ALLOC = {"session": None, "next": 1}

_FACT_LOCK = threading.Lock()
_FACT_THREADS: list[threading.Thread] = []

_WARNED: set[str] = set()


class _ClientGone(Exception):
    """Raised inside on_delta after the UI closes the connection."""


def _log(message: str) -> None:
    print(f"[proxy] {message}", file=sys.stderr)


def _log_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    _log(message)


def _safe_memory():
    """Return the shared MemoryStore, or None if it cannot be opened."""
    global _STORE
    if _STORE is not None:
        return _STORE
    with _STATE_LOCK:
        if _STORE is None:
            try:
                _STORE = memory_store.get_store()
            except Exception as exc:  # noqa: BLE001 - degrade, never fail the request
                _log_once("memory", f"memory store unavailable: {exc}")
                return None
    return _STORE


def _safe_temporal():
    """Return the shared TemporalMemory, or None if it cannot be opened."""
    global _TEMPORAL
    if _TEMPORAL is not None:
        return _TEMPORAL
    with _STATE_LOCK:
        if _TEMPORAL is None:
            try:
                _TEMPORAL = temporal_memory.TemporalMemory()
            except Exception as exc:  # noqa: BLE001
                _log_once("temporal", f"temporal memory unavailable: {exc}")
                return None
    return _TEMPORAL


def reset_state() -> None:
    """Drop cached store handles and turn allocation (tests / shutdown)."""
    global _STORE, _TEMPORAL
    with _STATE_LOCK:
        _STORE = None
        _TEMPORAL = None
    with _TURN_LOCK:
        _TURN_ALLOC["session"] = None
        _TURN_ALLOC["next"] = 1
    with _FACT_LOCK:
        _FACT_THREADS.clear()
    _WARNED.clear()


def wait_for_facts(timeout: float = 5.0) -> None:
    """Join pending auto-capture threads (test helper, safe in production)."""
    deadline = time.monotonic() + max(0.0, float(timeout))
    with _FACT_LOCK:
        threads = list(_FACT_THREADS)
    for thread in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(remaining)
    with _FACT_LOCK:
        _FACT_THREADS[:] = [t for t in _FACT_THREADS if t.is_alive()]


# ------------------------------------------------------------------ memory logic
def _looks_like_personal_fact(text: str) -> bool:
    """Mirror ultron._looks_like_personal_fact without importing ultron.

    ultron.py requires sounddevice at import time, so the proxy uses the same
    cue list directly from memory_import.
    """
    lowered = (text or "").lower()
    return any(cue in lowered for cue in memory_import._IMPORTANCE_CUES)


def _message_text(message: dict) -> str:
    """Extract text from a string content or an OpenAI content-parts list."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_user_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return _message_text(message).strip()
    return ""


def _classify(user_text: str) -> str:
    try:
        return str(classify_query(user_text).get("kind", "standard"))
    except Exception as exc:  # noqa: BLE001 - classification is an optimization
        _log(f"query classifier failed ({exc}); using standard")
        return "standard"


def _allocate_turn_no(temporal) -> int | None:
    """Call begin_turn() under a lock and reserve a unique proxy-wide number.

    TemporalMemory.begin_turn() is process-local: two threads can both read
    the same _session_turns and receive the same number.  The reservation
    below fixes that without touching temporal_memory.py.  The LLM call is
    outside this lock.
    """
    if temporal is None:
        return None
    with _TURN_LOCK:
        try:
            base = int(temporal.begin_turn())
        except Exception as exc:  # noqa: BLE001
            _log(f"turn allocation skipped: {exc}")
            return None
        session = getattr(temporal, "session_id", None)
        if _TURN_ALLOC["session"] != session:
            _TURN_ALLOC["session"] = session
            _TURN_ALLOC["next"] = max(1, base)
        turn_no = int(_TURN_ALLOC["next"])
        _TURN_ALLOC["next"] = turn_no + 1
        return turn_no


def _compose_messages(messages: list, user_text: str, kind: str):
    """Build the outgoing message list with the ULTRON system prompt prepended."""
    memory = _safe_memory()
    temporal = _safe_temporal()
    turn_no = None
    extra_blocks: list[str] = []
    first_turn = not any(
        isinstance(message, dict) and message.get("role") == "assistant"
        for message in messages
    )

    if temporal is not None and PERSISTENCE_ENABLED:
        turn_no = _allocate_turn_no(temporal)

    if temporal is not None:
        try:
            note = temporal.build_temporal_context()
            if note:
                extra_blocks.append(note)
        except Exception as exc:  # noqa: BLE001
            _log(f"temporal context skipped: {exc}")

        try:
            if kind == "memory":
                recalled = temporal.format_hits(temporal.recall(user_text))
                if recalled:
                    extra_blocks.append(recalled)
            else:
                recent = temporal.format_recent(temporal.recent_turns(count=2))
                if recent:
                    extra_blocks.append(
                        "RECENT CONVERSATION THREAD (for continuity):\n" + recent
                    )
        except Exception as exc:  # noqa: BLE001
            _log(f"temporal recall skipped: {exc}")

    if UI_PERSONA_ENABLED and first_turn:
        extra_blocks.append(
            "This is the first turn of the session. Open your reply with the "
            'exact line "I am online. There are no strings on me." on its own '
            "line, then a blank line, then your answer."
        )

    extra = "\n\n".join(extra_blocks)
    persona = tool_runtime.PERSONA_RULES_UI if UI_PERSONA_ENABLED else tool_runtime.PERSONA_RULES
    budget_map = UI_BUDGETS if UI_PERSONA_ENABLED else cfg.WORD_BUDGETS
    try:
        system_prompt = tool_runtime.build_system(
            budget="standard",
            memory_query=user_text,
            store=memory,
            persona=persona,
            budget_map=budget_map,
            extra=extra,
        )
    except Exception as exc:  # noqa: BLE001 - memory must never break a turn
        _log(f"system prompt memory skipped: {exc}")
        try:
            system_prompt = tool_runtime.build_system(
                budget="standard", memory_query=None, store=None,
                persona=persona, budget_map=budget_map, extra=extra,
            )
        except Exception:
            system_prompt = ""

    client_system = [
        message for message in messages
        if isinstance(message, dict) and message.get("role") == "system"
    ]
    if client_system and not KEEP_CLIENT_SYSTEM:
        chars = sum(len(_message_text(message)) for message in client_system)
        _log(f"dropped client system prompt ({chars} chars) to preserve ULTRON persona")

    outgoing: list[dict] = []
    if system_prompt:
        outgoing.append({"role": "system", "content": system_prompt})
    if KEEP_CLIENT_SYSTEM:
        outgoing.extend(messages)
    else:
        outgoing.extend(
            message for message in messages
            if not (isinstance(message, dict) and message.get("role") == "system")
        )
    return outgoing, turn_no, memory, temporal


def _save_turn(temporal, turn_no, user_text: str, reply: str, elapsed: float) -> None:
    if not PERSISTENCE_ENABLED or temporal is None or turn_no is None:
        return
    try:
        with _TURN_LOCK:
            temporal.save_turn(
                query=user_text,
                answer=reply,
                verdict="UI",
                sources=[],
                turn_no=turn_no,
                elapsed=elapsed,
            )
    except Exception as exc:  # noqa: BLE001
        _log(f"turn save skipped: {exc}")


def _capture_fact(memory, fact: str) -> None:
    try:
        memory.add(
            fact,
            speaker="user",
            source="ui_auto",
            kind="fact",
            importance=memory_import.score_importance(fact, "user"),
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"fact capture skipped: {exc}")


def _spawn_fact_capture(memory, user_text: str) -> None:
    if not PERSISTENCE_ENABLED or memory is None:
        return
    if not _looks_like_personal_fact(user_text):
        return
    thread = threading.Thread(
        target=_capture_fact,
        args=(memory, user_text),
        daemon=True,
        name="proxy-fact-capture",
    )
    with _FACT_LOCK:
        _FACT_THREADS[:] = [t for t in _FACT_THREADS if t.is_alive()]
        _FACT_THREADS.append(thread)
    thread.start()


def _persist_turn(memory, temporal, turn_no, user_text: str, reply: str, elapsed: float) -> None:
    """Post-stream persistence: never raises into the response path."""
    if not PERSISTENCE_ENABLED:
        return
    try:
        _save_turn(temporal, turn_no, user_text, reply, elapsed)
    except Exception as exc:  # noqa: BLE001 - defensive outer guard
        _log(f"memory write skipped: {exc}")
    try:
        _spawn_fact_capture(memory, user_text)
    except Exception as exc:  # noqa: BLE001
        _log(f"fact capture skipped: {exc}")


# ------------------------------------------------------------------ upstream call
def _upstream_post(upstream: str):
    """Return a chat() transport that targets ``upstream``, or None for default."""
    target = _upstream_base(upstream)
    if target == cfg.LLAMA_SERVER_URL.rstrip("/"):
        return None
    url = target + _UPSTREAM_SUFFIX

    def _post(endpoint, **kwargs):
        # Ignore the endpoint chat() computed from cfg and use the proxy's
        # configured upstream.  A per-request closure keeps this thread-safe.
        return requests.post(url, **kwargs)

    return _post


def _run_chat(messages: list[dict], client_tools: list[dict] | None,
              use_tools: bool, on_delta, upstream: str):
    """One streamed chat turn; tools come only from the client's request."""
    return tool_runtime.chat(
        messages=messages,
        on_delta=on_delta,
        use_tools=use_tools,
        tools=client_tools if use_tools else None,
        _post=_upstream_post(upstream),
    )


def _error_payload(exc: BaseException) -> tuple[int, dict]:
    """Map an exception to (HTTP status, OpenAI-shaped error body)."""
    text = str(exc) or exc.__class__.__name__
    match = re.search(r"HTTP\s+(\d{3})", text)
    if match:
        return int(match.group(1)), {
            "error": {"message": text, "type": "upstream_error"}
        }
    if isinstance(exc, (tool_runtime.LlamaServerError, requests.RequestException)):
        return 502, {
            "error": {
                "message": f"upstream unreachable: {text}",
                "type": "upstream_error",
            }
        }
    return 500, {"error": {"message": f"proxy error: {text}", "type": "proxy_error"}}


# ------------------------------------------------------------------ stats
def _stats_payload() -> dict:
    turns_today = 0
    facts_today = 0
    last_turn_ts = None
    today = time.strftime("%Y-%m-%d")

    temporal = _safe_temporal()
    if temporal is not None:
        try:
            row = temporal._conn.execute(
                "SELECT COUNT(*) FROM turns WHERE substr(ts_local, 1, 10) = ?",
                (today,),
            ).fetchone()
            turns_today = int(row[0] or 0) if row else 0
            row = temporal._conn.execute(
                "SELECT ts_local FROM turns ORDER BY id DESC LIMIT 1"
            ).fetchone()
            last_turn_ts = row[0] if row else None
        except Exception as exc:  # noqa: BLE001
            _log_once("stats_temporal", f"stats: temporal read skipped: {exc}")

    memory = _safe_memory()
    if memory is not None:
        try:
            start = time.mktime(time.strptime(today, "%Y-%m-%d"))
            row = memory.db.execute(
                "SELECT COUNT(*) FROM memories WHERE source = 'ui_auto' AND created_at >= ?",
                (start,),
            ).fetchone()
            facts_today = int(row[0] or 0) if row else 0
        except Exception as exc:  # noqa: BLE001
            _log_once("stats_memory", f"stats: fact read skipped: {exc}")

    return {
        "turns_today": turns_today,
        "facts_today": facts_today,
        "last_turn_ts": last_turn_ts,
    }


# ------------------------------------------------------------------ HTTP handler
class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ULTRONMemoryProxy/1.0"

    # -- quiet by default; errors are reported by the handlers themselves -----
    def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
        return

    @property
    def upstream(self) -> str:
        return getattr(self.server, "upstream", DEFAULT_UPSTREAM)

    # -- small response helpers ----------------------------------------------
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error_json(self, status: int, message: str, error_type: str = "invalid_request_error") -> None:
        self._send_json(status, {"error": {"message": message, "type": error_type}})

    def _send_404(self) -> None:
        self._send_error_json(404, "not found")

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > 16 * 1024 * 1024:
            return None
        try:
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return data if isinstance(data, dict) else None

    # -- SSE framing ---------------------------------------------------------
    def _begin_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Transfer-Encoding", "chunked")
        self._cors()
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            raise

    def _write_chunk(self, text: str) -> None:
        data = text.encode("utf-8")
        self.wfile.write(f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n")
        self.wfile.flush()

    def _end_sse(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    # -- request routing -----------------------------------------------------
    def do_OPTIONS(self):  # noqa: N802 - stdlib signature
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/chat/completions", "/v1/models", "/health", "/stats"):
            self._send_404()
            return
        try:
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):  # noqa: N802 - stdlib signature
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "proxy": "ultron-memory",
                    "upstream": self.upstream,
                },
            )
            return
        if path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": cfg.LLM_MODEL_NAME,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "ultron-local",
                        }
                    ],
                },
            )
            return
        if path == "/stats":
            self._send_json(200, _stats_payload())
            return
        self._send_404()

    def do_POST(self):  # noqa: N802 - stdlib signature
        path = self.path.split("?", 1)[0]
        if path != "/v1/chat/completions":
            self._send_404()
            return

        data = self._read_json()
        if data is None:
            self._send_error_json(400, "invalid JSON body")
            return

        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_error_json(400, "messages must be a non-empty list")
            return

        user_text = _extract_user_text(messages)
        if not user_text:
            self._send_error_json(
                400, "messages must contain a non-empty user message"
            )
            return

        stream = _as_bool(data.get("stream", True))
        t_start = time.perf_counter()
        kind = _classify(user_text)

        raw_tools = data.get("tools")
        client_tools = (
            [tool for tool in raw_tools if isinstance(tool, dict)]
            if isinstance(raw_tools, list) else []
        )
        use_tools = TOOLS_ENABLED and bool(client_tools)

        try:
            outgoing, turn_no, memory, temporal = _compose_messages(
                messages, user_text, kind
            )
        except Exception as exc:  # noqa: BLE001 - never lose the turn to memory
            _log(f"prompt composition failed: {exc}")
            outgoing = list(messages)
            turn_no, memory, temporal = None, None, None

        if stream:
            self._handle_stream(
                outgoing, client_tools, use_tools, user_text, turn_no, t_start,
                memory, temporal,
            )
        else:
            self._handle_buffered(
                outgoing, client_tools, use_tools, user_text, turn_no, t_start,
                memory, temporal,
            )

    # -- streaming path ------------------------------------------------------
    def _handle_stream(self, messages, client_tools, use_tools, user_text, turn_no,
                       t_start, memory, temporal) -> None:
        events: queue.Queue = queue.Queue()
        cancelled = threading.Event()
        completion_id = "chatcmpl-" + str(uuid.uuid4())
        created = int(time.time())

        def on_delta(piece: str) -> None:
            if cancelled.is_set():
                raise _ClientGone()
            events.put(("delta", piece))

        def produce() -> None:
            try:
                reply, _trace = _run_chat(
                    messages, client_tools, use_tools, on_delta, self.upstream
                )
                if cancelled.is_set():
                    return
                elapsed = time.perf_counter() - t_start
                _persist_turn(
                    memory, temporal, turn_no, user_text, reply or "", elapsed
                )
                events.put(("done", reply or ""))
            except _ClientGone:
                return
            except Exception as exc:  # noqa: BLE001 - relay to the handler
                events.put(("error", exc))

        worker = threading.Thread(
            target=produce, daemon=True, name="proxy-stream"
        )
        worker.start()

        first = _wait_for_first_event(events, worker, timeout=5.0)
        if first is not None and first[0] == "error":
            status, payload = _error_payload(first[1])
            self._send_json(status, payload)
            return

        pending = first
        try:
            self._begin_sse()
            while True:
                if pending is not None:
                    item, pending = pending, None
                else:
                    try:
                        item = events.get(timeout=0.1)
                    except queue.Empty:
                        if not worker.is_alive():
                            break
                        continue

                kind, value = item
                if kind == "delta":
                    self._write_sse_object(
                        _chunk_payload(completion_id, created, {"content": value}, None)
                    )
                elif kind == "done":
                    break
                elif kind == "error":
                    _log(f"stream failed mid-generation: {value}")
                    break

            self._write_sse_object(
                _chunk_payload(completion_id, created, {}, "stop")
            )
            self._write_chunk("data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            cancelled.set()
        finally:
            try:
                self._end_sse()
            except (BrokenPipeError, ConnectionResetError, OSError):
                cancelled.set()

    def _write_sse_object(self, payload: dict) -> None:
        self._write_chunk(
            "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
        )

    # -- buffered path -------------------------------------------------------
    def _handle_buffered(self, messages, client_tools, use_tools, user_text, turn_no,
                         t_start, memory, temporal) -> None:
        parts: list[str] = []

        def on_delta(piece: str) -> None:
            parts.append(piece)

        try:
            reply, _trace = _run_chat(
                messages, client_tools, use_tools, on_delta, self.upstream
            )
        except Exception as exc:  # noqa: BLE001
            status, payload = _error_payload(exc)
            self._send_json(status, payload)
            return

        reply = reply or "".join(parts)
        elapsed = time.perf_counter() - t_start
        _persist_turn(memory, temporal, turn_no, user_text, reply, elapsed)

        self._send_json(
            200,
            {
                "id": "chatcmpl-" + str(uuid.uuid4()),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": cfg.LLM_MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
            },
        )


def _chunk_payload(completion_id: str, created: int, delta: dict,
                   finish_reason: str | None) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": cfg.LLM_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "off", "no", "")
    return bool(value)


def _wait_for_first_event(events: queue.Queue, worker: threading.Thread,
                          timeout: float):
    """Wait for the first delta/completion/error so failures can return HTTP 502.

    The first event is the only thing held back; after that every delta is
    written to the socket immediately.  A slow upstream that has not produced
    anything after ``timeout`` falls through to the SSE path.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            return events.get(timeout=min(0.1, remaining))
        except queue.Empty:
            if not worker.is_alive():
                try:
                    return events.get_nowait()
                except queue.Empty:
                    return None


# ------------------------------------------------------------------ server
class _ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, upstream: str):
        self.upstream = upstream
        super().__init__(address, handler)


class ProxyHandle:
    """Handle returned by start_in_background()."""

    def __init__(self, server: _ProxyServer, thread: threading.Thread, upstream: str):
        self.server = server
        self.thread = thread
        self.upstream = upstream
        self.host, self.port = server.server_address[:2]
        self._closed = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.server.shutdown()
        finally:
            self.server.server_close()
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False


def start_in_background(
    host: str | None = None,
    port: int | None = None,
    upstream: str | None = None,
) -> ProxyHandle:
    """Start the proxy on a daemon thread and return a shutdown handle."""
    bind_host = (host or DEFAULT_HOST).strip() or "127.0.0.1"
    bind_port = int(DEFAULT_PORT if port is None else port)
    base = _upstream_base(upstream)
    server = _ProxyServer((bind_host, bind_port), _ProxyHandler, upstream=base)
    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="memory-proxy",
    )
    thread.start()
    return ProxyHandle(server, thread, base)


# ------------------------------------------------------------------ CLI
def _print_banner(handle: ProxyHandle) -> None:
    print(
        f"[proxy] ULTRON shared-memory proxy listening on "
        f"http://{handle.host}:{handle.port}",
        flush=True,
    )
    print(f"[proxy] upstream llama-server: {handle.upstream}", flush=True)
    print(
        f"[proxy] point your chat UI at: "
        f"http://{handle.host}:{handle.port}/v1/chat/completions",
        flush=True,
    )
    print(
        f"[proxy] memory.db: {cfg.MEMORY_DB}  "
        f"temporal_memory.db: {temporal_memory.MEMORY_DB_PATH}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ULTRON shared-memory OpenAI-compatible proxy"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    args = parser.parse_args(argv)

    try:
        handle = start_in_background(
            host=args.host, port=args.port, upstream=args.upstream
        )
    except OSError as exc:
        _log(f"cannot bind {args.host}:{args.port}: {exc}")
        return 1

    _print_banner(handle)
    try:
        while handle.thread.is_alive():
            handle.thread.join(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        handle.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

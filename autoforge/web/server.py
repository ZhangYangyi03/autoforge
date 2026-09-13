"""autoforge.web — the harness web UI.

One command (`auto web`) starts a local server that serves a single-page UI
over the same agent the CLI runs. No build step, no npm, no extra dependency:
`http.server` from the standard library plus one HTML file.

Two ideas are borrowed from DeepSeek Harness because they are the right ones:

  * **Every run is traceable.** Each session keeps an append-only event log of
    everything the model saw and did — system prompt, replies, tool calls,
    tool results, forge attempts, errors — and the UI can filter it by source.
  * **Runtime modes.** The same machinery runs in `standard` mode (the full
    forging agent) or `minimal` mode (a two-tool coding agent), so you can see
    what the extra capability actually buys.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

__all__ = ["serve", "Harness"]

STATIC = Path(__file__).parent / "static"
DEFAULT_PORT = 8765


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


def _event(source: str, **payload: Any) -> dict[str, Any]:
    return {"seq": 0, "t": time.time(), "source": source, **payload}


#: How a forge-pipeline / agent-loop event kind is presented in the
#: trajectory view. Kinds come from ForgeAgent.trace (agent.py) and from
#: ForgePipeline._emit (forge/pipeline.py).
_SOURCE_OF_KIND = {
    "call": "tool_call",
    "result": "tool_result",
    "finish": "system",
    "forge_attempt": "forge",
    "forge_done": "forge",
    "forge_error": "error",
    "auto_quarantine": "tool",
    "evolve": "forge",
    "spawn": "subagent",
    "amend": "selfmod",
}


def _normalise(raw: dict[str, Any]) -> dict[str, Any]:
    kind = str(raw.get("kind") or "event")
    return {
        "seq": 0,
        "t": float(raw.get("t") or time.time()),
        "source": _SOURCE_OF_KIND.get(kind, kind),
        "text": _summarise(kind, raw),
        "raw": _jsonable(raw),
    }


def _summarise(kind: str, raw: dict[str, Any]) -> str:
    if kind == "call":
        return f"{raw.get('tool')}({_short(raw.get('args'))})"
    if kind == "result":
        mark = "ok" if raw.get("ok") else "failed"
        return f"{raw.get('tool')} -> {mark}"
    if kind == "finish":
        return (f"turn {raw.get('turns')} · tools {raw.get('tools')} · "
                f"self-terminated={raw.get('self_terminated')}")
    if kind == "forge_attempt":
        return f"attempt {raw.get('round', '?')}: {str(raw.get('need', ''))[:160]}"
    if kind == "forge_done":
        return (f"{'sealed' if raw.get('ok') else 'rejected'} after "
                f"{raw.get('rounds', '?')} round(s)")
    if kind == "forge_error":
        return f"forge error: {str(raw.get('error') or raw)[:200]}"
    if kind == "auto_quarantine":
        return (f"quarantined {raw.get('name', '?')} "
                f"(success_rate={raw.get('success_rate')})")
    return str(raw.get("text") or raw.get("summary") or raw)[:300]


def _short(value: Any, limit: int = 90) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _jsonable(obj: Any, depth: int = 0) -> Any:
    """Best-effort conversion so arbitrary agent objects survive JSON."""
    if depth > 4:
        return str(obj)[:400]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj if not isinstance(obj, str) else obj[:4000]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in list(obj.items())[:40]}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in list(obj)[:40]]
    for attr in ("to_json", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn(), depth + 1)
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        return _jsonable(
            {k: v for k, v in vars(obj).items() if not k.startswith("_")}, depth + 1
        )
    return str(obj)[:400]


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------


class Session:
    """One conversation, plus the event log that makes it inspectable."""

    def __init__(self, sid: str, cfg: dict, mode: str, factory: Callable) -> None:
        self.id = sid
        self.cfg = cfg
        self.mode = mode
        self.title = "new session"
        self.created = time.time()
        self.updated = time.time()
        self.history: list = []
        self.events: list[dict[str, Any]] = []
        self.busy = False
        self.error: str | None = None
        self.turns = 0
        self._factory = factory
        self._agent = None
        self._lock = threading.Lock()
        self._feed: list[queue.Queue] = []

    # -- lazy build so listing sessions is instant ------------------------

    @property
    def agent(self):
        if self._agent is None:
            self._agent = self._factory(self.cfg, self.mode)
            self.push(_event("system", text=f"mode={self.mode}  "
                                           f"model={self.cfg.get('model')}  "
                                           f"base={self.cfg.get('base')}"))
        return self._agent

    # -- event log --------------------------------------------------------

    def push(self, ev: dict[str, Any]) -> dict[str, Any]:
        ev["seq"] = len(self.events) + 1
        ev.setdefault("t", time.time())
        self.events.append(ev)
        self.updated = time.time()
        for q in list(self._feed):
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass
        return ev

    def drain(self, after: int = 0, timeout: float = 0.0) -> list[dict[str, Any]]:
        fresh = [e for e in self.events if e["seq"] > after]
        if fresh or timeout <= 0:
            return fresh
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.05)
            fresh = [e for e in self.events if e["seq"] > after]
            if fresh:
                return fresh
        return []

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        self._feed.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        try:
            self._feed.remove(q)
        except ValueError:
            pass

    # -- introspection ----------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "mode": self.mode,
            "created": self.created, "updated": self.updated,
            "busy": self.busy, "turns": self.turns, "events": len(self.events),
            "error": self.error,
        }

    def detail(self, after: int = 0) -> dict[str, Any]:
        return {**self.summary(), "events": [e for e in self.events if e["seq"] > after]}


# --------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------


class Harness:
    """Owns the sessions and knows how to build an agent for a given mode."""

    def __init__(self, cfg: dict, factory: Callable, *, default_mode: str = "standard"):
        self.cfg = cfg
        self.factory = factory
        self.default_mode = default_mode
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()
        self.started = time.time()

    def create(self, mode: str | None = None, title: str = "") -> Session:
        sid = uuid.uuid4().hex[:12]
        s = Session(sid, self.cfg, mode or self.default_mode, self.factory)
        if title:
            s.title = title[:80]
        with self.lock:
            self.sessions[sid] = s
        return s

    def get(self, sid: str) -> Session | None:
        return self.sessions.get(sid)

    def remove(self, sid: str) -> bool:
        with self.lock:
            s = self.sessions.pop(sid, None)
        if s is not None and s._agent is not None:
            try:
                s._agent.registry.unregister("terminate")
            except Exception:
                pass
        return s is not None

    def list(self) -> list[dict[str, Any]]:
        return [s.summary() for s in sorted(
            self.sessions.values(), key=lambda x: x.updated, reverse=True)]

    # -- work -------------------------------------------------------------

    def _guard(self, session: Session, label: str) -> bool:
        """One run at a time per session; the browser must not double-submit."""
        if session.busy:
            session.push(_event("error", text=f"{label} refused: session is busy"))
            return False
        session.busy = True
        session.error = None
        return True

    def _run(self, session: Session, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:                                  # noqa: BLE001
            session.error = f"{type(exc).__name__}: {exc}"
            session.push(_event("error", text=session.error,
                                detail=traceback.format_exc()[-2000:]))
        finally:
            session.busy = False
            session.updated = time.time()

    def send(self, session: Session, text: str) -> None:
        if not self._guard(session, "message"):
            return
        if session.title == "new session":
            session.title = text.strip().split("\n")[0][:60] or "session"
        session.push(_event("user", text=text))

        def work() -> None:
            mark = len(session.agent.trace)
            result = session.agent.run(text, history=session.history)
            self._pump_trace(session, mark)
            if result.content:
                session.push(_event("assistant", text=result.content))
            if result.self_terminated:
                session.push(_event("system", text=f"self-terminated: {result.termination_reason}"))
            session.history = result.messages
            session.turns += 1

        threading.Thread(target=lambda: self._run(session, work), daemon=True).start()

    def forge(self, session: Session, need: str) -> None:
        if not self._guard(session, "forge"):
            return
        session.push(_event("user", text=f"[forge] {need}"))

        def work() -> None:
            agent = session.agent
            pipeline = getattr(agent, "pipeline", None)
            if pipeline is None:
                raise RuntimeError("this mode cannot forge (no pipeline mounted)")
            mark = len(agent.trace)
            result = pipeline.forge(need)
            self._pump_trace(session, mark)
            ok = bool(getattr(result, "ok", False))
            spec = getattr(result, "spec", None)      # ForgeResult.spec: ToolSpec | None
            rounds = getattr(result, "rounds", 0)
            session.push(_event(
                "forge",
                text=(f"{'sealed' if ok else 'rejected'}: "
                      f"{getattr(spec, 'name', None) or '(no tool)'} "
                      f"· {rounds} round(s)"),
                ok=ok, rounds=rounds, checks=self._checks(result),
            ))
            if ok and spec is not None and getattr(agent, "store", None) is not None:
                try:
                    agent.store.save_tool(spec)
                    session.push(_event("system", text=f"saved {spec.name} to the store"))
                except Exception as exc:                          # noqa: BLE001
                    session.push(_event("error", text=f"save failed: {exc}"))

        threading.Thread(target=lambda: self._run(session, work), daemon=True).start()

    def _pump_trace(self, session: Session, mark: int) -> None:
        for raw in session.agent.trace[mark:]:
            session.push(_normalise(raw))

    @staticmethod
    def _checks(result: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for attempt in getattr(result, "attempts", []) or []:
            if getattr(attempt, "error", None):
                out.append({"round": getattr(attempt, "round", 0),
                            "name": "error", "passed": False,
                            "detail": str(attempt.error)[:300]})
            report = getattr(attempt, "report", None)
            for check in getattr(report, "checks", []) or []:
                out.append({
                    "round": getattr(attempt, "round", 0),
                    "name": getattr(check, "name", "?"),
                    "passed": bool(getattr(check, "passed", False)),
                    "detail": str(getattr(check, "detail", ""))[:300],
                })
        return out

    def tools(self) -> dict[str, Any]:
        """The live tool table, taken from the most recent session.

        Prefer a session that has already run; otherwise build the newest
        session's agent so the panel can show a fresh session's baseline
        toolset (minimal mode always has two tools).
        """
        sessions = sorted(self.sessions.values(), key=lambda s: s.updated, reverse=True)
        if not sessions:
            return {"tools": [], "total": 0, "by_state": {}}
        session = next((s for s in sessions if s._agent is not None), sessions[0])
        try:
            rep = _jsonable(session.agent.registry.report())
        except Exception as exc:                               # noqa: BLE001
            return {"tools": [], "total": 0, "error": f"{type(exc).__name__}: {exc}"}
        return rep if isinstance(rep, dict) else {"tools": []}

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "model": self.cfg.get("model"),
            "base": self.cfg.get("base"),
            "fast": self.cfg.get("fast"),
            "max_tokens": self.cfg.get("max_tokens"),
            "proxy": self.cfg.get("proxy"),
            "modes": ["standard", "minimal"],
            "default_mode": self.default_mode,
            "sessions": len(self.sessions),
            "uptime_s": round(time.time() - self.started, 1),
            "cwd": os.getcwd(),
        }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "autoforge-harness"

    harness: Harness
    token: str | None = None

    # -- helpers ----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:      # quieter server
        if os.environ.get("AUTOFORGE_WEB_VERBOSE"):
            super().log_message(fmt, *args)

    def _authorised(self) -> bool:
        if not self.token:
            return True
        got = (self.headers.get("X-Auth-Token")
               or parse_qs(urlparse(self.path).query).get("token", [""])[0])
        return got == self.token

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> dict[str, Any]:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8")) or {}
        except Exception:
            return {}

    @property
    def parts(self) -> list[str]:
        return [p for p in urlparse(self.path).path.split("/") if p]

    def _session(self, sid: str) -> Session | None:
        s = self.harness.get(sid)
        if s is None:
            self._json({"error": f"no session {sid!r}"}, 404)
        return s

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:                                  # noqa: N802
        if not self._authorised():
            return self._json({"error": "unauthorised"}, 401)
        parts = self.parts
        query = parse_qs(urlparse(self.path).query)

        if not parts or parts[0] in ("index.html", "ui"):
            page = (STATIC / "index.html").read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")

        if parts[0] == "api":
            rest = parts[1:]
            if rest == ["state"]:
                return self._json(self.harness.status())
            if rest == ["sessions"]:
                return self._json({"sessions": self.harness.list()})
            if rest == ["tools"]:
                return self._json(self.harness.tools())
            if len(rest) == 3 and rest[0] == "sessions" and rest[2] == "stream":
                return self._stream(rest[1], query)
            if len(rest) == 2 and rest[0] == "sessions":
                s = self._session(rest[1])
                if s is None:
                    return None
                after = int(query.get("after", ["0"])[0] or 0)
                return self._json(s.detail(after))
        return self._json({"error": "not found", "path": self.path}, 404)

    def do_POST(self) -> None:                                 # noqa: N802
        if not self._authorised():
            return self._json({"error": "unauthorised"}, 401)
        parts = self.parts
        body = self._body()
        if parts[:1] != ["api"]:
            return self._json({"error": "not found", "path": self.path}, 404)
        rest = parts[1:]

        if rest == ["sessions"]:
            s = self.harness.create(body.get("mode"), body.get("title", ""))
            return self._json(s.detail(), 201)
        if rest == ["tools", "reload"]:
            return self._json(self.harness.tools())
        if len(rest) == 3 and rest[0] == "sessions":
            s = self._session(rest[1])
            if s is None:
                return None
            action = rest[2]
            if action == "send":
                text = str(body.get("text") or "").strip()
                if not text:
                    return self._json({"error": "text is required"}, 400)
                self.harness.send(s, text)
                return self._json(s.detail(), 202)
            if action == "forge":
                need = str(body.get("need") or "").strip()
                if not need:
                    return self._json({"error": "need is required"}, 400)
                self.harness.forge(s, need)
                return self._json(s.detail(), 202)
        return self._json({"error": "not found", "path": self.path}, 404)

    def do_DELETE(self) -> None:                               # noqa: N802
        if not self._authorised():
            return self._json({"error": "unauthorised"}, 401)
        parts = self.parts
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "sessions":
            return self._json({"removed": self.harness.remove(parts[2])})
        return self._json({"error": "not found", "path": self.path}, 404)

    # -- SSE --------------------------------------------------------------

    def _stream(self, sid: str, query: dict[str, list[str]]) -> None:
        session = self._session(sid)
        if session is None:
            return None
        after = int((query.get("after") or ["0"])[0] or 0)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            while True:
                fresh = session.drain(after, timeout=1.0)
                if fresh:
                    for ev in fresh:
                        after = ev["seq"]
                        chunk = f"id: {ev['seq']}\ndata: {json.dumps(ev, default=str)}\n\n"
                        self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
                else:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                if not session.busy and not fresh and _idle(session):
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def _idle(session: Session, seconds: float = 30.0) -> bool:
    """Close the stream once nothing has happened for a while."""
    return (time.time() - session.updated) > seconds


def serve(
    cfg: dict,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    factory: Callable | None = None,
    mode: str = "standard",
    token: str | None = None,
) -> None:
    """Start the harness UI. Blocks until Ctrl-C."""
    if factory is None:                                        # avoid import cycle
        from ..cli import _build_mode
        factory = _build_mode

    harness = Harness(cfg, factory, default_mode=mode)
    handler = type("BoundHandler", (_Handler,), {"harness": harness, "token": token})

    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    actual = httpd.server_address[1]

    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{shown}:{actual}/"
    if token:
        url += f"?token={token}"

    print(f"\n  autoforge harness  →  {url}")
    print(f"  model : {cfg.get('model')}")
    print(f"  base  : {cfg.get('base')}")
    print(f"  mode  : {mode}   ({'FAST' if cfg.get('fast') else 'full'} checks)")
    if host in ("0.0.0.0", "") and not token:
        print("  warning: bound to all interfaces with no token — anyone on the "
              "network can drive this agent")
    print("  ctrl-c to stop\n")

    if open_browser:
        def _open() -> None:
            time.sleep(0.6)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        httpd.server_close()

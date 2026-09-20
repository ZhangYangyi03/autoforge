"""autoforge node service -- a reachable execution endpoint on this machine.

Why it exists: autoforge on Windows can only be driven from the machine it runs
on. `distributed` means another box (the KOS Linux machine, a WSL distro, a
cloud GPU host) can reach THIS node over TCP -- not loopback -- and get an
honest answer, run a command, or submit a python job.

Three design decisions, each because the alternative is a lie:

1. Plain stdlib http.server. Not FastAPI. This service must start as a bare
   background process on a machine where nothing is installed, no uvicorn, no
   venv. A relay with a dependency list is a relay that fails exactly when it
   is needed.

2. Every mutating route needs a bearer token (node.token, 0600). Binding
   0.0.0.0 with /exec unauthenticated is a remote code execution endpoint on
   the LAN with my name on it. Health is open because a health check that
   needs a secret cannot be used by a monitor; everything else is gated.

3. Job state lives in files under jobs/, not in memory. A restart must not
   lose a submitted job, for the same reason the toolmarket store is sqlite
   and not :memory: -- the first restart that eats the shelf is the lesson.

Routes
    GET  /node/health          open. liveness + what this node is.
    GET  /node/info            token. paths, tools, market reachability.
    POST /node/exec            token. {cmd, timeout, cwd, shell} -> rc/out/err.
    POST /node/submit          token. {code|task, name} -> job id (async).
    GET  /node/job/{id}        token. status + result of a submitted job.
    GET  /node/jobs            token. list.
"""
from __future__ import annotations

import base64
import json
import os
import platform
import secrets
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(ROOT, "jobs")
TOKEN_FILE = os.path.join(ROOT, "node.token")
NODE_NAME = os.environ.get("AUTOFORGE_NODE_NAME") or platform.node()
DEFAULT_TIMEOUT = float(os.environ.get("AUTOFORGE_NODE_TIMEOUT", "120"))
MAX_TIMEOUT = 3600.0


def _token() -> str:
    env = os.environ.get("AUTOFORGE_NODE_TOKEN")
    if env:
        return env.strip()
    if os.path.exists(TOKEN_FILE):
        return open(TOKEN_FILE, encoding="utf-8").read().strip()
    tok = secrets.token_urlsafe(32)
    with open(TOKEN_FILE, "w", encoding="utf-8") as fh:
        fh.write(tok)
    try:  # best effort on POSIX; a no-op on Windows, where ACLs rule instead
        os.chmod(TOKEN_FILE, 0o600)
    except OSError:
        pass
    return tok


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")


def _job_path(jid: str) -> str:
    return os.path.join(JOBS, jid + ".json")


def _save_job(job: dict) -> None:
    os.makedirs(JOBS, exist_ok=True)
    tmp = _job_path(job["id"]) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(job, fh, ensure_ascii=False)
    os.replace(tmp, _job_path(job["id"]))


def _load_job(jid: str):
    p = _job_path(jid)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _run_job(jid: str, code: str, timeout: float) -> None:
    job = _load_job(jid) or {}
    job.update(state="running", started=time.time())
    _save_job(job)
    started = time.time()
    path = os.path.join(JOBS, jid + ".py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(code)
    try:
        proc = subprocess.run([sys.executable, "-I", path], capture_output=True,
                              timeout=timeout)
        job.update(
            state="done" if proc.returncode == 0 else "failed",
            rc=proc.returncode,
            stdout=proc.stdout.decode("utf-8", "replace")[-200000:],
            stderr=proc.stderr.decode("utf-8", "replace")[-20000:],
        )
    except subprocess.TimeoutExpired:
        job.update(state="timeout", rc=None,
                   stderr=f"killed after {timeout}s")
    except Exception:
        job.update(state="failed", rc=None, stderr=traceback.format_exc()[-4000:])
    job["ms"] = int((time.time() - started) * 1000)
    job["finished"] = time.time()
    _save_job(job)


class Handler(BaseHTTPRequestHandler):
    server_version = "autoforge-node/1.0"
    # HTTP/1.0 on purpose. With HTTP/1.1 the handler promises keep-alive, and
    # this server closes after every reply; a proxy in front (ngrok, a reverse
    # proxy, curl with a pool) then reads a reply that never comes and turns it
    # into 502 -- intermittent, because it only bites when a second request
    # lands on the socket that was already closed. Sending `Connection: close`
    # without setting close_connection is the same bug with a header on top.
    # HTTP/1.0 has no such ambiguity: one request, one reply, one close.
    protocol_version = "HTTP/1.0"

    # --- plumbing ---------------------------------------------------------
    def log_message(self, fmt, *args):  # keep stdout quiet; run detached
        pass

    def _trace(self, note: str, code: int = 0) -> None:
        try:
            with open(os.path.join(ROOT, "requests.log"), "a", encoding="utf-8") as fh:
                fh.write("%.3f %s %s -> %s %s\n" % (time.time(), self.command,
                                                     self.path, code, note))
        except OSError:
            pass

    def _send(self, code: int, obj) -> None:
        self._trace("", code)
        body = _json_bytes(obj)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {"_raw": base64.b64encode(raw).decode("ascii")}

    def _authed(self) -> bool:
        hdr = self.headers.get("Authorization") or ""
        tok = hdr[7:].strip() if hdr.lower().startswith("bearer ") else ""
        if not tok:
            tok = (self.headers.get("X-Node-Token") or "").strip()
        return bool(tok) and secrets.compare_digest(tok, _token())

    def _need_auth(self) -> bool:
        if self._authed():
            return False
        self._send(401, {"ok": False, "error": "missing or bad bearer token"})
        return True

    # --- market reverse proxy --------------------------------------------
    def _proxy(self, sub: str) -> None:
        """Reverse-proxy /market/<path> to the local toolmarket.

        Why: one public URL has to serve both halves. ngrok's free tier gives a
        single HTTP tunnel, and a second tunnel needs a card, so rather than
        choose between "the shelf is reachable" and "the node is reachable" the
        node -- which already speaks HTTP -- carries the market under a path
        prefix. The market keeps its own surface and its own auth; this only
        moves bytes.
        """
        import urllib.error as _e
        import urllib.request as _u
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else None
        url = "http://127.0.0.1:8000/" + sub
        if "?" in self.path:
            url += "?" + self.path.split("?", 1)[1]
        hdr = {
            "Content-Type": self.headers.get("Content-Type") or "application/json",
            "User-Agent": "autoforge-node-proxy/1.0",
            "ngrok-skip-browser-warning": "true",
        }
        if self.headers.get("Authorization"):
            hdr["Authorization"] = self.headers["Authorization"]
        self._trace("proxy->" + url)
        req = _u.Request(url, data=raw, headers=hdr, method=self.command)
        try:
            with _u.urlopen(req, timeout=120) as r:
                body, code = r.read(), r.status
                ctype = r.headers.get("Content-Type") or "application/json"
        except _e.HTTPError as exc:
            body, code = exc.read(), exc.code
            ctype = exc.headers.get("Content-Type") or "application/json"
        except Exception as exc:
            body = _json_bytes({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
            code, ctype = 502, "application/json"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    # --- routes -----------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path.startswith("/market"):
            return self._proxy(path[len("/market"):].lstrip("/") or "health")
        if path in ("/node/health", "/node", "/"):
            self._send(200, {
                "ok": True, "node": NODE_NAME,
                "host": platform.node(), "platform": platform.platform(),
                "python": platform.python_version(),
                "pid": os.getpid(), "ts": time.time(),
                "role": "autoforge execution node",
            })
            return
        if path == "/node/info":
            if self._need_auth():
                return
            market = None
            try:
                import urllib.request
                with urllib.request.urlopen(
                        "http://127.0.0.1:8000/health", timeout=4) as r:
                    market = json.loads(r.read().decode())
            except Exception as exc:
                market = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self._send(200, {
                "ok": True, "node": NODE_NAME, "root": ROOT,
                "jobs_dir": JOBS, "python": sys.executable,
                "cwd": os.getcwd(), "toolmarket_local": market,
                "endpoints": ["/node/health", "/node/info", "/node/exec",
                              "/node/submit", "/node/job/{id}", "/node/jobs"],
            })
            return
        if path == "/node/jobs":
            if self._need_auth():
                return
            ids = sorted((f[:-5] for f in os.listdir(JOBS)
                          if f.endswith(".json")), reverse=True) \
                if os.path.isdir(JOBS) else []
            self._send(200, {"ok": True, "count": len(ids), "jobs": ids[:100]})
            return
        if path.startswith("/node/job/"):
            if self._need_auth():
                return
            job = _load_job(path.rsplit("/", 1)[-1])
            if job is None:
                self._send(404, {"ok": False, "error": "no such job"})
                return
            self._send(200, {"ok": True, "job": job})
            return
        self._send(404, {"ok": False, "error": f"no route {path}"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path.startswith("/market"):
            return self._proxy(path[len("/market"):].lstrip("/"))
        if self._need_auth():
            return
        body = self._body()
        if path == "/node/exec":
            cmd = body.get("cmd") or body.get("command")
            if not cmd:
                self._send(400, {"ok": False, "error": "need 'cmd'"})
                return
            timeout = min(float(body.get("timeout") or DEFAULT_TIMEOUT), MAX_TIMEOUT)
            cwd = body.get("cwd") or ROOT
            started = time.time()
            try:
                # A string is a shell line, and only a shell can run it: passing
                # one with shell=False makes subprocess look for a file whose
                # name is the entire line and fail with WinError 2. A list is
                # argv and must NOT go through a shell, or quoting becomes a
                # second, accidental interpreter.
                if isinstance(cmd, str):
                    proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                                          cwd=cwd, shell=True)
                else:
                    proc = subprocess.run(list(cmd), capture_output=True,
                                          timeout=timeout, cwd=cwd, shell=False)
                self._send(200, {
                    "ok": proc.returncode == 0, "rc": proc.returncode, "node": NODE_NAME,
                    "cwd": cwd, "ms": int((time.time() - started) * 1000),
                    "stdout": proc.stdout.decode("utf-8", "replace")[-200000:],
                    "stderr": proc.stderr.decode("utf-8", "replace")[-20000:],
                })
            except subprocess.TimeoutExpired:
                self._send(200, {"ok": False, "rc": None, "node": NODE_NAME,
                                 "error": f"timeout after {timeout}s",
                                 "ms": int((time.time() - started) * 1000)})
            except Exception as exc:
                self._send(200, {"ok": False, "rc": None, "node": NODE_NAME,
                                 "error": f"{type(exc).__name__}: {exc}"})
            return
        if path == "/node/submit":
            code = body.get("code")
            task = body.get("task")
            if not code and task:
                code = ("# task submitted to node %s for the operator to inspect\n"
                        "print(%r)\n" % (NODE_NAME, task))
            if not code:
                self._send(400, {"ok": False, "error": "need 'code' or 'task'"})
                return
            timeout = min(float(body.get("timeout") or DEFAULT_TIMEOUT), MAX_TIMEOUT)
            jid = body.get("id") or uuid.uuid4().hex[:12]
            _save_job({"id": jid, "state": "queued", "code": code,
                       "submitted": time.time(), "node": NODE_NAME,
                       "task": task, "timeout": timeout,
                       "submitter": body.get("submitter") or "unknown"})
            threading.Thread(target=_run_job, args=(jid, code, timeout),
                             daemon=True).start()
            self._send(202, {"ok": True, "job": jid, "node": NODE_NAME,
                             "poll": f"/node/job/{jid}"})
            return
        self._send(404, {"ok": False, "error": f"no route {path}"})


def serve(host: str = "0.0.0.0", port: int = 8077) -> None:
    os.makedirs(JOBS, exist_ok=True)
    _token()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"autoforge node {NODE_NAME} on {host}:{port} root={ROOT}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("AUTOFORGE_NODE_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("AUTOFORGE_NODE_PORT", "8077")))
    a = ap.parse_args()
    serve(a.host, a.port)

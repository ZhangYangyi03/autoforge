"""Keep the sibling tool-market reachable, from inside autoforge.

Why this lives in autoforge rather than in a startup shortcut
------------------------------------------------------------
The market is a peer: autoforge forges and enforces tools locally, the market
is the shared shelf other agents read from. _sync_to_market already pushes
every forged tool there, and it is deliberately allowed to fail -- a forge must
not roll back because a sibling process is down.

That design is right, and it has one consequence that is easy to miss: if the
market is simply not running, every push fails silently, forever, and the
agent own ledger says market_sync: ok=false with no indication that the fix
is one process launch away. The tool is forged, callable, and invisible to
everyone else. Nothing is broken and nothing works.

A shortcut in the Startup folder fixes that for one user on one machine, and
only after a login. It does not fix it for a fresh clone, a container, a CI run,
or a second agent on the same host. So the guarantee belongs here, in the code
that depends on the market being up.

What this does NOT do
---------------------
It does not make the market a dependency. ensure_running is best-effort and
returns a report; a caller that ignores it loses nothing but the distribution
step. It never raises, never blocks a forge, and never starts a second copy if
one is already listening.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8000"
STARTUP_TIMEOUT_S = 25.0
_LAUNCHER_CANDIDATES = ("toolmarket_server.py", "autostart_server.py")


def market_url() -> str:
    return os.environ.get("TOOLMARKET_URL", DEFAULT_URL).rstrip("/")


def _host_port(url: str):
    rest = url.split("://", 1)[-1]
    hostport = rest.split("/", 1)[0]
    if ":" in hostport:
        host, _, port = hostport.partition(":")
        return host, int(port)
    return hostport, 80 if url.startswith("http://") else 443


def is_listening(url=None, timeout: float = 1.0) -> bool:
    host, port = _host_port(url or market_url())
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def health(url=None, timeout: float = 5.0) -> dict:
    target = (url or market_url()).rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(target, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        return {"ok": True, "status": resp.status, "body": json.loads(body)}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code,
                "error": "HTTP %s: %s" % (exc.code, exc.read().decode("utf-8", "replace")[:200])}
    except Exception as exc:
        return {"ok": False, "error": ("%s: %s" % (type(exc).__name__, exc))[:200]}


def _find_launcher():
    explicit = os.environ.get("TOOLMARKET_LAUNCHER")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    here = Path(__file__).resolve()
    roots = [here.parent.parent.parent, here.parent.parent.parent.parent, Path.home()]
    for root in roots:
        for name in _LAUNCHER_CANDIDATES:
            for candidate in (root / name, root / "tool-market" / name):
                if candidate.is_file():
                    return candidate
    return None


def ensure_running(url=None, *, launch: bool = True, timeout: float = STARTUP_TIMEOUT_S) -> dict:
    target = url or market_url()
    report = {"url": target}
    if is_listening(target):
        report["action"] = "already-up"
        report["ok"] = True
        return report
    if not launch:
        report.update(action="unavailable", ok=False, error="not listening and launch=False")
        return report
    launcher = _find_launcher()
    if launcher is None:
        report.update(action="unavailable", ok=False,
                      error="no launcher found; set TOOLMARKET_LAUNCHER")
        return report
    creationflags = 0
    if os.name == "nt":
        creationflags = (getattr(subprocess, "DETACHED_PROCESS", 0)
                         | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    try:
        proc = subprocess.Popen(
            [sys.executable, str(launcher)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
            cwd=str(launcher.parent), creationflags=creationflags,
            start_new_session=(os.name != "nt"))
    except Exception as exc:
        report.update(action="unavailable", ok=False,
                      error=("launch failed: %s: %s" % (type(exc).__name__, exc))[:200])
        return report
    report["pid"] = proc.pid
    report["launcher"] = str(launcher)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_listening(target, timeout=0.5):
            report.update(action="started", ok=True)
            return report
        if proc.poll() is not None:
            report.update(action="unavailable", ok=False,
                          error="launcher exited rc=%s before binding" % proc.returncode)
            return report
        time.sleep(0.25)
    report.update(action="unavailable", ok=False,
                  error="still not listening after %.0fs" % timeout)
    return report

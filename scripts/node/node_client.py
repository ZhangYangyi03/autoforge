"""Talk to an autoforge node from anywhere -- this is the half that runs on the
OTHER machine.

Usage:
    python node_client.py health
    python node_client.py info
    python node_client.py exec  "nproc" --url http://192.168.1.107:8077
    python node_client.py submit-file job.py --name nightly
    python node_client.py wait <job-id>

Token comes from --token, $AUTOFORGE_NODE_TOKEN, or ../node.token next to this
file. Exits non-zero when the node is unreachable, so a script that pipes this
fails loudly instead of treating silence as success.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("AUTOFORGE_NODE_URL", "http://127.0.0.1:8077")


def _token(explicit: str = "") -> str:
    if explicit:
        return explicit
    env = os.environ.get("AUTOFORGE_NODE_TOKEN")
    if env:
        return env.strip()
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "node.token"),
                 os.path.join(os.path.dirname(here), "node.token")):
        if os.path.exists(cand):
            return open(cand, encoding="utf-8").read().strip()
    return ""


class Node:
    def __init__(self, url: str = DEFAULT_URL, token: str = "", timeout: float = 180.0):
        self.url = url.rstrip("/")
        self.token = _token(token)
        self.timeout = timeout

    def call(self, path: str, payload=None, method: str = None):
        url = self.url + path
        data = json.dumps(payload).encode() if payload is not None else None
        hdr = {"Content-Type": "application/json",
               "User-Agent": "autoforge-node-client/1.0"}
        if self.token:
            hdr["Authorization"] = "Bearer " + self.token
        req = urllib.request.Request(url, data=data, headers=hdr,
                                     method=method or ("POST" if data else "GET"))
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                return json.loads(body)
            except Exception:
                return {"ok": False, "http": exc.code, "error": body[:500]}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "url": url}

    def health(self):
        return self.call("/node/health")

    def info(self):
        return self.call("/node/info")

    def exec(self, cmd, timeout=120, cwd=None):
        return self.call("/node/exec", {"cmd": cmd, "timeout": timeout, "cwd": cwd})

    def submit(self, code, name=None, timeout=600):
        return self.call("/node/submit", {"code": code, "task": name, "timeout": timeout})

    def job(self, jid):
        return self.call("/node/job/" + jid)

    def wait(self, jid, poll=2.0, deadline=900):
        end = time.time() + deadline
        while time.time() < end:
            r = self.job(jid)
            st = (r.get("job") or {}).get("state")
            if st in ("done", "failed", "timeout"):
                return r
            time.sleep(poll)
        return {"ok": False, "error": f"job {jid} still running after {deadline}s"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["health", "info", "exec", "submit", "submit-file", "job", "wait", "jobs"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--token", default="")
    ap.add_argument("--name", default=None)
    ap.add_argument("--timeout", type=float, default=180.0)
    a = ap.parse_args(argv)
    n = Node(a.url, a.token, timeout=max(a.timeout, 30))
    if a.action == "health":
        r = n.health()
    elif a.action == "info":
        r = n.info()
    elif a.action == "exec":
        r = n.exec(a.arg, timeout=a.timeout)
    elif a.action in ("submit", "submit-file"):
        code = open(a.arg, encoding="utf-8").read() if a.action == "submit-file" else a.arg
        r = n.submit(code, a.name, a.timeout)
    elif a.action == "job":
        r = n.job(a.arg)
    elif a.action == "wait":
        r = n.wait(a.arg, deadline=a.timeout)
    else:
        r = n.call("/node/jobs")
    print(json.dumps(r, ensure_ascii=False, indent=2)[:200000])
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

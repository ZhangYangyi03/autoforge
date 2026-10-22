#!/usr/bin/env python3
r"""af_ctl.py -- controller side of the af-link bridge.
  af_ctl.py status
  af_ctl.py send "command" [--node N] [--wait 45]
  af_ctl.py push --file LOCAL --path REMOTE [--name NAME]
  af_ctl.py pull --path REMOTE --out LOCAL
  af_ctl.py listen [--secs 60]
Config: %USERPROFILE%\.af-link.json  ({"topic":..,"node":..,"secret":..})
"""
import argparse, base64, hashlib, hmac, json, os, sys, time, urllib.request

CFG = json.load(open(os.path.expanduser("~/.af-link.json"), encoding="utf-8"))
T = CFG["topic"]; S = CFG["secret"]; BASE = "https://ntfy.sh/" + T


def sig(o):
    return hmac.new(S.encode(), json.dumps({k: v for k, v in o.items() if k != "h"},
                                            sort_keys=True).encode(), hashlib.sha256).hexdigest()


def pub(o, title=""):
    o = dict(o); o["h"] = sig(o)
    h = {"User-Agent": "af-ctl/1"}
    if title:
        h["X-Title"] = title
    urllib.request.urlopen(urllib.request.Request(BASE, data=json.dumps(o).encode(),
                           method="POST", headers=h), timeout=25).read()


def events(since):
    raw = urllib.request.urlopen(urllib.request.Request(
        "%s/json?poll=1&since=%d" % (BASE, since), headers={"User-Agent": "af-ctl/1"}),
        timeout=30).read().decode("utf-8", "replace")
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def wait_for(ids, node, secs, verbose=False):
    since = int(time.time()) - secs - 15
    got, seen = {}, set()
    end = time.time() + secs
    while time.time() < end and len(got) < len(ids):
        for m in events(since):
            if m.get("time", 0) > since:
                since = max(since, m["time"] - 1)
            if m.get("id") in seen:
                continue
            seen.add(m.get("id"))
            try:
                p = json.loads(m.get("message", "{}"))
            except Exception:
                continue
            if verbose and p.get("kind") in ("hello", "ping", "error"):
                print("[%s] %s" % (p.get("from"), json.dumps(p, ensure_ascii=False)), flush=True)
            if p.get("from") != node:
                continue
            if p.get("kind") == "out" and p.get("id") in ids:
                got.setdefault(p["id"], {})[p.get("i", 1)] = base64.b64decode(p["b64"]).decode("utf-8", "replace")
        if len(got) < len(ids):
            time.sleep(2)
    return {k: "".join(v[i] for i in sorted(v)) for k, v in got.items()}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("action")
    a.add_argument("cmd", nargs="?")
    a.add_argument("--node", default=CFG["node"])
    a.add_argument("--wait", type=int, default=45)
    a.add_argument("--file"); a.add_argument("--path"); a.add_argument("--name")
    a.add_argument("--out"); a.add_argument("--secs", type=int, default=60)
    a.add_argument("-v", action="store_true")
    ns = a.parse_args()
    if ns.action == "status":
        seen = {}
        for m in events(int(time.time()) - 600):
            try:
                p = json.loads(m.get("message", "{}"))
            except Exception:
                continue
            if p.get("kind") in ("hello", "ping"):
                seen[p.get("from")] = p
        print(json.dumps({"topic": T, "node": ns.node, "seen_recently": seen,
                          "now": time.time()}, indent=1, ensure_ascii=False))
    elif ns.action == "send":
        cid = "c%d" % int(time.time())
        pub({"to": ns.node, "cmd": ns.cmd, "id": cid})
        for _, v in wait_for({cid}, ns.node, ns.wait, ns.v).items():
            print(v)
    elif ns.action == "listen":
        since = int(time.time()) - ns.secs
        end = time.time() + ns.secs
        while time.time() < end:
            for m in events(since):
                since = max(since, m.get("time", 0) - 1)
                try:
                    p = json.loads(m.get("message", "{}"))
                except Exception:
                    continue
                if p.get("kind") == "out":
                    print("[%s %s] %s" % (p.get("from"), p.get("id"),
                          base64.b64decode(p["b64"]).decode("utf-8", "replace")), flush=True)
                else:
                    print("[%s] %s" % (p.get("from"), json.dumps(p, ensure_ascii=False)), flush=True)
            time.sleep(2)
    elif ns.action == "push":
        data = open(ns.file, "rb").read()
        name = ns.name or os.path.basename(ns.file)
        t0 = int(time.time()) - 3
        for i in range(0, len(data), 1200):
            pub(base64.b64encode(data[i:i + 1200]).decode(), title="AFDATA:%s:%d" % (name, i // 1200 + 1))
        cid = "f%d" % int(time.time())
        pub({"to": ns.node, "pull": {"name": name, "path": ns.path,
                                     "sha": hashlib.sha256(data).hexdigest(), "since": t0}, "id": cid})
        for _, v in wait_for({cid}, ns.node, ns.wait, ns.v).items():
            print(v)
    elif ns.action == "pull":
        cid = "g%d" % int(time.time())
        pub({"to": ns.node, "put": {"path": ns.path, "name": ns.name or os.path.basename(ns.path)}, "id": cid})
        for _, v in wait_for({cid}, ns.node, ns.wait, ns.v).items():
            print(v)
    else:
        print(__doc__); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

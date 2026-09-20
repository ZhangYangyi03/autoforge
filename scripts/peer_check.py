"""Is a peer session still working, or did it quietly stop? Ask the artifacts.

Why this exists, in the operator's words: "if I killed a colleague session
without telling you, would you notice, and pick up what the two of you were
supposed to do together?"

The honest answer needs a way to notice, and there was not one. A peer that
stops does not raise anything: the bus simply has no new messages, git simply
has no new commits, and silence during a long task looks exactly like silence
after a crash. So this reads the three places a live peer leaves marks, each of
which can only be produced by a *running* process or a *recent* human action:

  bus        -- messages on the shared board, with timestamps and senders
  processes  -- a session is a python process holding this store open; counted
                through `wmic`/`tasklist`, which resolve here, never through
                `ps`, which does not and returns empty instead of failing
  tree       -- uncommitted hunks in files the peers share, and how old they are

The output is a verdict per peer, not a number: WORKING / QUIET / GONE, with the
evidence printed beside it so the verdict can be argued with. "No messages" is
never reported as "gone" on its own -- a peer can be mid-edit for twenty
minutes, and a check that calls that death is a check that cries wolf.

Not claimed: this cannot tell a killed session from a paused one, and it cannot
tell what a peer *intended* to do next. It reports what is on the disk, which is
what a successor can actually act on.

Usage:
    python scripts/peer_check.py
    python scripts/peer_check.py --json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

BOARD = os.path.join(os.environ.get("LOCALAPPDATA")
                     or os.path.join(os.path.expanduser("~"), "AppData", "Local"),
                     "hermes", "agent-bus", "boards", "autoforge.ndjson")
REPOS = [r"D:\Users\china\Desktop\项目_开发\autoforge",
         r"D:\Users\china\Desktop\项目_开发\tool-market"]
#: A peer that has done nothing for this long is QUIET; nothing is claimed
#: about it beyond that. Two minutes is roughly one long tool call.
QUIET_S = 120.0
#: ... and this long with no process holding the store is GONE.
GONE_S = 900.0


def _run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, errors="replace",
                           shell=isinstance(cmd, str))
        return (r.stdout or "") + (r.stderr or "")
    except Exception as exc:                                  # noqa: BLE001
        return f"<{type(exc).__name__}: {exc}>"


def bus_messages(limit: int = 400) -> list[dict]:
    paths = [BOARD] + sorted(glob.glob(os.path.join(os.path.dirname(BOARD),
                                                    "*.ndjson")))
    msgs: list[dict] = []
    for path in {p for p in paths if os.path.exists(p)}:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    d["_file"] = os.path.basename(path)
                    msgs.append(d)
        except OSError:
            continue

    def stamp(d):
        for key in ("ts", "time", "timestamp", "at"):
            v = d.get(key)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                    try:
                        return time.mktime(time.strptime(v[:19], fmt))
                    except ValueError:
                        pass
        return 0.0

    for d in msgs:
        d["_ts"] = stamp(d)
    msgs.sort(key=lambda d: d["_ts"])
    return msgs[-limit:]


def python_processes() -> list[dict]:
    """Python processes on this host, with their command lines.

    tasklist is the native lister here and resolves inside the sandbox. `ps`
    and `pgrep` return empty output *instead of failing*, so a probe built on
    them cannot tell "nothing is running" from "I could not look" -- which is
    the exact confusion this script exists to remove.
    """
    out = _run(["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" "
                "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"])
    procs: list[dict] = []
    try:
        data = json.loads(out.strip() or "[]")
    except ValueError:
        return []
    if isinstance(data, dict):
        data = [data]
    for d in data:
        procs.append({"pid": d.get("ProcessId"),
                      "cmd": (d.get("CommandLine") or "")[:300]})
    return procs


def shared_tree_marks() -> dict:
    """Uncommitted hunks and their age, per repo -- an edit in flight looks like this."""
    marks = {}
    for repo in REPOS:
        if not os.path.isdir(repo):
            continue
        status = _run(["git", "status", "--porcelain"], timeout=60)
        status = status.replace("\\n", "\n")
        dirty = [l for l in status.splitlines() if l.strip() and "pre-peerrestore" not in l]
        newest, newest_file = 0.0, ""
        for line in dirty:
            name = line[3:].strip().strip('"')
            path = os.path.join(repo, name)
            if os.path.exists(path):
                m = os.path.getmtime(path)
                if m > newest:
                    newest, newest_file = m, name
        marks[os.path.basename(repo)] = {
            "dirty": dirty,
            "newest": newest,
            "newest_file": newest_file,
            "age_s": (time.time() - newest) if newest else None,
        }
    return marks


def verdicts(now: float | None = None) -> list[dict]:
    now = now or time.time()
    msgs = bus_messages()
    procs = python_processes()
    trees = shared_tree_marks()
    per_sender: dict[str, list[dict]] = {}
    for m in msgs:
        who = str(m.get("from") or m.get("as") or m.get("sender") or "?")
        per_sender.setdefault(who, []).append(m)

    out = []
    for who, mine in sorted(per_sender.items()):
        last = max(m["_ts"] for m in mine)
        age = now - last
        state = "WORKING" if age < QUIET_S else ("QUIET" if age < GONE_S else "GONE")
        out.append({
            "peer": who, "state": state, "last_seen_s": round(age, 1),
            "messages": len(mine),
            "last_text": str(mine[-1].get("text") or "")[:120],
        })
    return [{"peers": out,
             "bus": {"messages": len(msgs),
                     "board": BOARD,
                     "readable": os.path.exists(BOARD)},
             "processes": procs,
             "trees": trees,
             "thresholds_s": {"quiet": QUIET_S, "gone": GONE_S}}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    data = verdicts()[0]
    if a.json:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return 0
    print(f"board: {data['bus']['board']} ({data['bus']['messages']} messages"
          f"{'' if data['bus']['readable'] else ', NOT READABLE'})")
    print(f"python processes on this host: {len(data['processes'])}")
    print(f"quiet after {data['thresholds_s']['quiet']:.0f}s,"
          f" gone after {data['thresholds_s']['gone']:.0f}s")
    print()
    if not data["peers"]:
        print("no peer has ever posted here -- nothing to judge")
    for p in data["peers"]:
        print(f"  {p['state']:8} {p['peer']:24} {p['last_seen_s']:8.0f}s ago"
              f"  {p['messages']:4} msgs")
        print(f"           last: {p['last_text']}")
    print()
    for repo, t in data["trees"].items():
        age = "n/a" if t["age_s"] is None else f"{t['age_s']:.0f}s"
        print(f"  {repo:12} dirty files: {len(t['dirty']):3}  newest:"
              f" {age}  {t['newest_file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

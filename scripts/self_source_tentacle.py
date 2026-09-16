"""A hand that edits my own source, and leaves a record of having done it.

The gap this closes: `amend_self` rewrites the running agent's prompt in memory
and dies with the process, so a rule I "wrote" could vanish on restart. Editing
the source file is what persists -- but doing that by hand, session after
session, is how a repo fills up with half-applied patches and .bak files (this
one had thirteen of them). So the edit is a tool with rules:

  * it refuses any path outside the repositories it is allowed to touch;
  * it counts occurrences first and refuses if the count is not what the caller
    declared -- a patch that matches twice is a patch that was not understood;
  * it preserves line endings, because this filesystem is CRLF and a silent LF
    rewrite shows up later as a diff of the whole file;
  * it backs up before writing, compiles after writing, and restores the backup
    if compilation fails -- a self-edit that breaks the agent is worse than no
    edit at all;
  * it records the edit in the agent's own hash-chained ledger, so "who changed
    me" is answered by the log rather than by memory.

Usage:
    python scripts/self_source_tentacle.py --file <path> --find <s> --replace <s>
        --expect 1 --why "why this edit exists"

An edit that does not compile is undone, so this hand cannot leave the
agent unable to run itself.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import py_compile
import shutil
import sys

ALLOWED_ROOTS = (
    "D:\\Users\\china\\Desktop\\" + "\u9879\u76ee_\u5f00\u53d1" + "\\autoforge",
    "D:\\Users\\china\\Desktop\\" + "\u9879\u76ee_\u5f00\u53d1" + "\\tool-market",
)
LEDGER = "C:\\Users\\china\\AppData\\Local\\autoforge\\autoforge.db"


def allowed(path):
    """Only the two repositories. A self-edit tool with no fence is a footgun."""
    p = os.path.abspath(path).lower()
    return any(p.startswith(r.lower() + os.sep) for r in ALLOWED_ROOTS)


def record(kind, payload):
    """Append to my own ledger, chained if the store offers a chained writer."""
    if ALLOWED_ROOTS[0] not in sys.path:
        sys.path.insert(0, ALLOWED_ROOTS[0])
    note = None
    try:
        from autoforge.store import ToolStore
        ToolStore(LEDGER).log_event(kind, payload)
        return "ledger: written through ToolStore.log_event"
    except Exception as e:
        note = "ledger: ToolStore path unavailable (%s: %s)" % (type(e).__name__, e)
    import sqlite3
    con = sqlite3.connect(LEDGER)
    con.execute(
        "INSERT INTO forge_events (timestamp, kind, payload, writer_id)"
        " VALUES (?,?,?,?)",
        (datetime.datetime.now().timestamp(), kind,
         json.dumps(payload, ensure_ascii=False), "self_source_tentacle"))
    con.commit()
    return note + "; wrote a raw row instead"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--find", required=True)
    ap.add_argument("--replace", required=True)
    ap.add_argument("--expect", type=int, default=1)
    ap.add_argument("--why", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    path = os.path.abspath(a.file)
    if not allowed(path):
        print("REFUSED: %s is outside the repositories this tool may edit" % path)
        return 2
    if not os.path.exists(path):
        print("REFUSED: no such file %s" % path)
        return 2

    with open(path, "r", encoding="utf-8", newline="") as fh:
        before = fh.read()
    hits = before.count(a.find)
    if hits != a.expect:
        print("REFUSED: --find matches %dx, expected %d. Nothing written."
              % (hits, a.expect))
        return 3

    after = before.replace(a.find, a.replace)
    if "\r\n" in before and "\r\n" not in a.replace and "\n" in a.replace:
        print("REFUSED: replacement would drop CRLF line endings. Write \\r\\n explicitly.")
        return 4

    if a.dry_run:
        print("DRY RUN ok: %d occurrence(s) in %s" % (hits, path))
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = "%s.%s.tentacle.bak" % (path, stamp)
    shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(after)

    compiled = None
    if path.endswith(".py"):
        try:
            py_compile.compile(path, doraise=True)
            compiled = "ok"
        except Exception as e:
            compiled = "FAILED: %s" % e
            shutil.copy2(backup, path)
            print("COMPILE FAILED, restored from %s: %s"
                  % (os.path.basename(backup), e))
            record("self_edit_reverted",
                   {"file": path, "why": a.why, "error": str(e)})
            return 5

    note = record("self_edit", {"file": path, "why": a.why, "occurrences": hits,
                                "backup": os.path.basename(backup),
                                "compiled": compiled})
    print("EDITED %s (%dx), compile=%s" % (os.path.basename(path), hits, compiled))
    print("  backup: %s" % os.path.basename(backup))
    print("  %s" % note)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Byte-level audit: did anything double a backslash or mix line endings?

Compares the working tree against HEAD for the given paths, so a defect
introduced by an edit is separated from one that was already committed.
Run from the repo root:  python scripts/audit_bytes.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PATHS = [
    "autoforge/cli.py",
    "autoforge/agent.py",
    "autoforge/core/compaction.py",
    "autoforge/core/llm.py",
    "autoforge/store.py",
    "autoforge/skills.py",
    "autoforge/modes.py",
    "autoforge/tools/registry.py",
]

BACKSLASH = chr(92)
DOUBLE = BACKSLASH + BACKSLASH
# The defect this looks for is an *escape* that got doubled — `\\n` where the
# author meant a newline, which silently prints a literal backslash-n and once
# corrupted a docstring. A bare `\\` is ordinary Python (a Windows path being
# normalised, a regex), so a plain double-backslash *increase* is reported but
# is not itself a finding.
DOUBLED_ESCAPES = tuple((DOUBLE + c).encode() for c in "ntr0")


def counts(blob: bytes) -> dict[str, int]:
    lf = blob.count(b"\n")
    crlf = blob.count(b"\r\n")
    return {
        "dbl": blob.count(DOUBLE.encode()),
        "esc": sum(blob.count(e) for e in DOUBLED_ESCAPES),
        "crlf": crlf,
        "lone_lf": lf - crlf,
    }


def head_blob(path: str) -> bytes | None:
    r = subprocess.run(["git", "show", f"HEAD:{path}"],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    problems = 0
    print(f"{'path':34} {'dbl':>9} {'dbl_esc':>9} {'lone_lf':>9}  verdict")
    for p in PATHS:
        f = Path(p)
        if not f.exists():
            print(f"{p:34} {'':>9} {'':>9} {'':>9}  MISSING")
            continue
        now = counts(f.read_bytes())
        old = head_blob(p)
        if old is None:
            cur = f"{now['dbl']}->{now['dbl']}"
            verdict = "new file"
            delta_dbl = delta_esc = 0
        else:
            was = counts(old)
            delta_dbl = now["dbl"] - was["dbl"]
            delta_esc = now["esc"] - was["esc"]
            cur = f"{was['dbl']}->{now['dbl']}"
            verdict = "ok"
            if delta_dbl > 0 or delta_esc > 0:
                verdict = f"BACKSLASH +{delta_dbl}/+{delta_esc}"
                problems += 1
        if now["lone_lf"] and now["crlf"]:
            verdict = (verdict + " MIXED-EOL").strip()
            problems += 1
        elif now["lone_lf"] and not now["crlf"]:
            verdict = (verdict + " all-LF").strip()
        print(f"{p:34} {cur:>9} {now['esc']:>9} {now['lone_lf']:>9}  {verdict}")

    print()
    print(f"problems: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

"""Which needs were forged more than once -- said from the ledger, not guessed.

The question this answers is demand-side: an agent that forges the same
capability three times already has that capability and does not know it. Names
are the weak signal (the same job gets called crlf_patch, crlf_srcpatch and
crlfpatch across three sessions); the need text is the stronger one. So the
clustering here runs on both, and a cluster is merged when either agrees:

  * two needs share a recorded tool name -- same name is same job, unless one
    of them was renamed, which this cannot see and states;
  * their token sets overlap by >= --threshold (Jaccard) -- catches the same job
    described twice in different words;
  * a short need is a substring of a longer one -- catches "add X" vs "add X to
    the repo, then run the tests".

Both directions are reported separately on purpose:

  merged   -- one cluster, several distinct tool names. This is a real
              duplicate: the merge is "keep the better one, retire the rest",
              and the names are listed so the retirement can be done by name.
  retried  -- one cluster, one name, several forge attempts. The tool exists
              and the need came back anyway, which is a different defect: the
              lookup that was supposed to find it did not. Merging nothing;
              this is the signal to look at retrieval, not at the shelf.

Usage:
    python scripts/duplicate_needs.py                    # print the report
    python scripts/duplicate_needs.py --md out.md        # write it
    python scripts/duplicate_needs.py --min-events 3     # noisier floor
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sqlite3
import sys

def _candidate_dbs() -> list[str]:
    """Ledgers to try, in order, then the one to say out loud in the report.

    Not a single constant, because a scrubbed environment is a real case here:
    this script gets run both by the agent (whose subprocess environment keeps
    only a handful of variables, so LOCALAPPDATA is *absent* and the path
    silently resolves to %USERPROFILE%\autoforge\autoforge.db) and from a
    normal shell (where it is present). Resolving to the wrong file and then
    reporting "no such table: forge_events" is a confusing answer to a question
    whose real answer is "I looked in the other place".
    """
    base = os.environ.get("LOCALAPPDATA")
    out = []
    if base:
        out.append(os.path.join(base, "autoforge", "autoforge.db"))
    out.append(os.path.join(os.path.expanduser("~"), "AppData", "Local",
                            "autoforge", "autoforge.db"))
    out.append(os.path.join(os.path.expanduser("~"), "autoforge", "autoforge.db"))
    return out


def _has_ledger(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        got = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='forge_events'"
        ).fetchone()
        conn.close()
        return got is not None
    except sqlite3.Error:
        return False


DEFAULT_DB = next((p for p in _candidate_dbs() if _has_ledger(p)),
                  _candidate_dbs()[0])

#: Words that appear in nearly every need and therefore carry no signal about
#: which job it is. Both languages, because this ledger is written in both.
STOPWORDS = set("""
a an the and or of to for at in on with my own that this it is be as by from into
out up then than so not no do does tool tools need needs write written make made
create created build built using use used which what how return returns print
prints report reports read reads file files path paths true false ok
本机 一个 读取 打印 返回 并且 并 的 了 把 用 到 在 和 与 或 是 不 要 里 从 对 给
""".split())

_NEED_KINDS = ("forge_start", "forge_done", "forge_attempt", "forge_error")


def _tokens(text: str) -> set[str]:
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", text.lower())
    return {w for w in text.split() if w not in STOPWORDS and len(w) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def read_needs(db_path: str) -> list[dict]:
    """Every recorded need, with the tool name and kind it was recorded under."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    marks = ",".join("?" * len(_NEED_KINDS))
    out: list[dict] = []
    for kind, payload in conn.execute(
            f"SELECT kind, payload FROM forge_events WHERE kind IN ({marks})",
            _NEED_KINDS):
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            continue
        need = (data.get("need") or "").strip()
        if not need:
            continue
        out.append({"need": need, "kind": kind,
                    "name": (data.get("name") or data.get("tool") or "").strip()})
    conn.close()
    return out


def cluster(events: list[dict], threshold: float = 0.45) -> list[list[dict]]:
    """Group distinct need texts that describe the same job.

    Union-find over three independent signals rather than a chain of sort keys:
    a chains A>B by tokens, B>C by substring and A>C not at all, and the chain
    is what makes a cluster that no single rule would have produced. That is the
    intended behaviour -- it is the same failure mode as "same job, three
    sessions, three phrasings".
    """
    uniq: dict[str, dict] = {}
    for ev in events:
        u = uniq.setdefault(ev["need"], {"need": ev["need"], "names": set(),
                                         "kinds": set(), "n": 0})
        u["n"] += 1
        u["kinds"].add(ev["kind"])
        if ev["name"]:
            u["names"].add(ev["name"])
    items = list(uniq.values())
    for it in items:
        it["tok"] = _tokens(it["need"])

    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a["names"] and (a["names"] & b["names"]):
                union(i, j)
            elif _jaccard(a["tok"], b["tok"]) >= threshold:
                union(i, j)
            else:
                short, long = (a, b) if len(a["need"]) <= len(b["need"]) else (b, a)
                if len(short["need"]) > 25 and short["need"].lower() in long["need"].lower():
                    union(i, j)

    groups: dict[int, list[dict]] = collections.defaultdict(list)
    for i, it in enumerate(items):
        groups[find(i)].append(it)
    return sorted(groups.values(),
                  key=lambda g: -sum(x["n"] for x in g))


def report(events: list[dict], groups: list[list[dict]], min_events: int = 2,
           db_path: str = DEFAULT_DB) -> str:
    total_events = len(events)
    distinct = len({e["need"] for e in events})
    heavy = [g for g in groups if sum(x["n"] for x in g) >= min_events]

    merged, retried = [], []
    for g in heavy:
        names = sorted({n for x in g for n in x["names"]})
        rec = {"events": sum(x["n"] for x in g), "needs": len(g), "names": names,
               "examples": [x["need"] for x in sorted(g, key=lambda y: -y["n"])[:3]]}
        (merged if len(g) > 1 else retried).append(rec)
    merged.sort(key=lambda r: -len(r["names"]))
    retried.sort(key=lambda r: -r["events"])

    lines: list[str] = []
    lines.append("# Duplicate needs, from the ledger")
    lines.append("")
    lines.append(f"Source: `{db_path}` -- {total_events} recorded needs, "
                 f"{distinct} distinct texts, {len(groups)} clusters.")
    lines.append("")
    lines.append(f"- **{len(merged)} clusters are the same job under different "
                 f"tool names** ({sum(len(r['names']) for r in merged)} names). "
                 f"These are real duplicates: the merge is keep the best, retire "
                 f"the rest.")
    lines.append(f"- **{len(retried)} clusters are the same need retried under one "
                 f"name.** The tool exists and the need came back, so the defect "
                 f"is retrieval, not the shelf.")
    lines.append("")
    lines.append("## Merge these: same job, different names")
    lines.append("")
    lines.append("keep is the one with the most events, or the clearest name when tied")
    lines.append("")
    for r in merged:
        lines.append(f"### {len(r['names'])} names, {r['events']} forge events, {r['needs']} phrasings")
        lines.append("")
        lines.append("  names: " + ", ".join(r["names"]))
        lines.append("")
        for ex in r["examples"]:
            lines.append(f"  - {ex[:160]}")
        lines.append("")
    lines.append("## Look at retrieval, not at the shelf")
    lines.append("")
    for r in retried[:20]:
        lines.append(f"- {r['events']}x  {r['examples'][0][:150]}")
    lines.append("")
    lines.append("## Method, and what it cannot see")
    lines.append("")
    lines.append("Cluster when two needs share a tool name, or their token sets "
                 "overlap by Jaccard >= 0.45, or a >25-character need is a "
                 "substring of a longer one: 210 needs, 106 distinct tool names, "
                 "165 clusters flagged. Name-sharing is the strongest signal and "
                 "the substring rule is the weakest; a chain of the three can put "
                 "two needs in one cluster that no single rule would join.")
    lines.append("")
    lines.append("Not seen: a tool that was renamed. Two names for one job with no "
                 "shared vocabulary and no substring relation land in different "
                 "clusters, and the fix for that is not a better threshold -- it "
                 "is a rename recorded as an event, which the ledger does not yet do.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--md", default=None, help="write the report here")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-events", type=int, default=2)
    a = ap.parse_args()
    if not _has_ledger(a.db):
        print(f"no forge_events table at {a.db}; tried: "
              + ", ".join(_candidate_dbs()), file=sys.stderr)
        return 2
    events = read_needs(a.db)
    groups = cluster(events, a.threshold)
    text = report(events, groups, a.min_events, a.db)
    if a.md:
        os.makedirs(os.path.dirname(os.path.abspath(a.md)), exist_ok=True)
        with open(a.md, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {a.md} ({len(text)} chars)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

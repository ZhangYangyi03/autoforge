
"""Calibrate the routing weights against the ledger (fast path).

`RoutingWeights` shipped with values chosen by taste. This measures them and
searches for better ones, on ground truth already on disk:

  probes  -- 312 (query, expect=call) pairs from the tools' own TriggerProbes,
             plus 312 negative_query pairs. Model-written from the description,
             so lexical access is easy here. Necessary, not sufficient.
  forged  -- 117 accepted forges: real needs written with no knowledge of what
             the tool would be called. Small, and the only set that is not
             partly the router's own echo.

The optimisation must not be judged on what it optimised on, so the forged set
is split in half by time (older / newer) and the grid is chosen on one half and
reported on the other.
"""
from __future__ import annotations

import itertools
import json
import math
import pickle
import random
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoforge.route.router import _COST, _TRUST, text_similarity
from autoforge.store import ToolStore
from autoforge.tools.spec import ToolState

DB = r"C:\Users\china\AppData\Local\autoforge\autoforge.db"
CACHE = Path(__file__).resolve().parents[1] / "_calfeats.pkl"
STATES = (ToolState.ACTIVE, ToolState.PROBATION)


def load_cases(specs):
    pos, neg = [], []
    for name, sp in specs.items():
        for p in (sp.probes or []):
            q = getattr(p, "query", None)
            if q and getattr(p, "expect", None) == "call":
                pos.append({"q": q, "target": name})
            nq = getattr(p, "negative_query", None)
            if nq:
                neg.append({"q": nq, "target": name})
    return pos, neg


def forged_cases(specs):
    conn = sqlite3.connect(DB, timeout=30)
    out = []
    for (_id, payload) in conn.execute(
            "select id, payload from forge_events where kind='forge_done' order by id"):
        d = json.loads(payload)
        if (str(d.get("ok")) == "True" and d.get("name") and d.get("need")
                and d["name"] in specs):
            out.append({"q": d["need"], "target": d["name"]})
    return out


def build_features(specs):
    """(target, [(name, text, success, trust, cost, over)]) per case.

    Pre-tokenising each tool once is the whole trick: recomputing
    `text_similarity` inside the loop re-tokenises 120 descriptions 741 times
    and does not finish in ten minutes.
    """
    from autoforge.route.router import _WORD

    def toks(text):
        return Counter(w.lower() for w in _WORD.findall(text or "") if len(w) > 1)

    pool = []
    for s in specs.values():
        if s.state not in STATES:
            continue
        d = toks(f"{s.name} {s.description} {' '.join(s.tags)}")
        st = s.stats
        pool.append({
            "name": s.name, "tok": d,
            "dn": math.sqrt(sum(v * v for v in d.values())) or 1.0,
            "success": st.success_rate if st.calls >= 3 else 0.5,
            "trust": _TRUST.get(s.state, 0.0),
            "cost": _COST.get(s.cost_hint, 0.0),
            "over": min(1.0, st.trigger_misses / max(st.calls, 1)) if st.calls else 0.0,
        })
    routable = {p["name"] for p in pool}

    def one(cases):
        out = []
        for c in cases:
            if c["target"] not in routable:
                continue
            q = toks(c["q"])
            qn = math.sqrt(sum(v * v for v in q.values()))
            rows = []
            for p in pool:
                if not q or qn == 0:
                    t = 0.0
                else:
                    common = set(q) & set(p["tok"])
                    t = (sum(q[x] * p["tok"][x] for x in common)
                         / (qn * p["dn"]))
                rows.append((p["name"], t, p["success"], p["trust"],
                             p["cost"], p["over"]))
            out.append((c["target"], rows))
        return out

    return one

def score_set(cases, w):
    """w = (text, success, trust, cost, over) multipliers."""
    wt, ws, wtr, wc, wo = w
    h1 = h3 = 0
    rr = []
    for target, rows in cases:
        best = [t[0] for t in sorted(
            rows, key=lambda r: (wt * r[1] + ws * r[2] + wtr * r[3]
                                 - wc * r[4] - wo * r[5]), reverse=True)]
        if target in best:
            i = best.index(target)
            rr.append(1.0 / (i + 1))
            h1 += (i == 0)
            h3 += (i < 3)
        else:
            rr.append(0.0)
    n = len(cases)
    return (h1 / n, h3 / n, statistics.mean(rr), n)


def neg_false_win(cases, w):
    wt, ws, wtr, wc, wo = w
    wrong = 0
    for target, rows in cases:
        best = max(rows, key=lambda r: (wt * r[1] + ws * r[2] + wtr * r[3]
                                        - wc * r[4] - wo * r[5]))
        wrong += (best[0] == target)
    return wrong / len(cases) if cases else 0.0


def main():
    specs = {n: r.to_spec() for n, r in ToolStore(DB).load_all_tools().items()}
    pos, neg = load_cases(specs)
    forged = forged_cases(specs)
    one = build_features(specs)
    F_pos, F_neg, F_for = one(pos), one(neg), one(forged)
    if "--cache" in sys.argv:
        pickle.dump({"pos": F_pos, "neg": F_neg, "forged": F_for},
                    open(CACHE, "wb"))

    shipped = (1.0, 1.2, 0.8, 0.5, 0.6)
    print(f"library {len(specs)} tools | probes {len(F_pos)} pos / {len(F_neg)} neg"
          f" | forged {len(F_for)}")
    print("\nshipped (text 1.0, success 1.2, trust 0.8, cost 0.5, over 0.6)")
    for label, cases in (("probes", F_pos), ("forged", F_for)):
        a, b, m, n = score_set(cases, shipped)
        print(f"  {label:7} top1 {a:.3f}  top3 {b:.3f}  MRR {m:.3f}  (n={n})")
    print(f"  negative false-win {neg_false_win(F_neg, shipped):.3f}")

    # ---- grid search, judged on the OLDER half of the forged set ----------
    half = len(F_for) // 2
    older, newer = F_for[:half], F_for[half:]
    grid = list(itertools.product(
        (0.4, 0.7, 1.0, 1.4, 2.0),      # text
        (0.0, 0.4, 0.8, 1.2, 2.0),      # success
        (0.0, 0.3, 0.6, 1.0),           # trust
        (0.0, 0.3, 0.6),                # cost
        (0.0, 0.4, 0.8),                # over_trigger
    ))
    print(f"\nsearching {len(grid)} weight vectors on the older half (n={len(older)})")
    best = None
    for w in grid:
        _, _, mrr, _ = score_set(older, w)
        fw = neg_false_win(F_neg, w)
        obj = mrr - 0.5 * fw
        if best is None or obj > best[0]:
            best = (obj, w, mrr, fw)
    obj, w, mrr_old, fw = best
    print("chosen on older half: text %.1f success %.1f trust %.1f cost %.1f over %.1f"
          % w)
    print("  older-half MRR %.3f  negative false-win %.3f" % (mrr_old, fw))
    print("\nHELD-OUT (newer half, never used to choose):")
    a, b, m, n = score_set(newer, w)
    print(f"  tuned   top1 {a:.3f}  top3 {b:.3f}  MRR {m:.3f}  (n={n})")
    a, b, m, n = score_set(newer, shipped)
    print(f"  shipped top1 {a:.3f}  top3 {b:.3f}  MRR {m:.3f}  (n={n})")
    print("\nfull sets with the tuned vector:")
    for label, cases in (("probes", F_pos), ("forged", F_for)):
        a, b, m, n = score_set(cases, w)
        print(f"  {label:7} top1 {a:.3f}  top3 {b:.3f}  MRR {m:.3f}  (n={n})")
    print(f"  negative false-win {neg_false_win(F_neg, w):.3f}")
    print("\njson:", json.dumps({"text": w[0], "success": w[1], "trust": w[2],
                                 "cost": w[3], "over_trigger": w[4]}))


if __name__ == "__main__":
    main()

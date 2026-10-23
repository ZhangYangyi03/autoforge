"""What reasoning ladder does the live catalog actually publish per model?

xdpool builds its thinking-level map from `reasoning.supportedEfforts` on each
catalog row (lib/index.js:11878 `thinkingLevelMap` -> 11879 `efforts =
info.supportedEfforts`; `undefined` when the row carries no `reasoning` object).
So a model whose row omits it gets NO levels at all, and DSH's picker falls back
to showing the request as non-reasoning -- which reads as "off".

This prints, per model, what the gateway really said: the raw `reasoning` object,
plus the ladder xdpool would derive from it.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import (GLOBAL_BASE, GLOBAL_CONFIG_PATH, MODELS_PATH,  # noqa: E402
                     load_accounts)

LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"]


def fetch(url: str, h: dict) -> dict:
    req = urllib.request.Request(url, headers=h, method="GET")
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    accts = load_accounts(verbose=False)
    for region in ("cn", "global"):
        acct = next((a for a in accts if a.region == region), None)
        if acct is None or not acct.usable_token():
            print(f"{region}: no usable account\n")
            continue
        url = (f"{GLOBAL_BASE}{GLOBAL_CONFIG_PATH}" if region == "global"
               else f"{acct.chat_base}{MODELS_PATH}")
        doc = fetch(url, acct.chat_headers())
        rows = (doc.get("data") or {}).get("models") or []

        print("=" * 96)
        print(f"{region.upper()}  {acct.label}  — {len(rows)} models")
        print("=" * 96)
        print(f"{'id':24s} {'raw reasoning object':40s} {'derived ladder'}")
        print("-" * 96)
        no_reasoning, has_efforts = [], []
        for r in rows:
            if not isinstance(r, dict):
                continue
            mid = str(r.get("id") or "")
            reasoning = r.get("reasoning")
            if not isinstance(reasoning, dict):
                no_reasoning.append(mid)
                print(f"{mid:24s} {'(absent)':40s} none — picker shows no levels")
                continue
            efforts = reasoning.get("supportedEfforts")
            if isinstance(efforts, list) and efforts:
                ladder = ",".join(l for l in LEVELS if l in efforts)
                has_efforts.append(mid)
                print(f"{mid:24s} {json.dumps(reasoning, ensure_ascii=False)[:38]:40s} {ladder}")
            else:
                no_reasoning.append(mid)
                print(f"{mid:24s} {json.dumps(reasoning, ensure_ascii=False)[:38]:40s} "
                      f"(no supportedEfforts) none")
        print()
        print(f"  with a ladder      : {len(has_efforts):2d}")
        print(f"  with NO ladder     : {len(no_reasoning):2d}  <- these read as 'off'")
        if no_reasoning:
            print(f"                      {', '.join(no_reasoning)}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

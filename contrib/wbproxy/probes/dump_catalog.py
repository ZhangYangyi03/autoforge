"""Dump the full model catalog with the metadata a chooser needs.

Every model row carries more than an id: context window, max output, credit
multiplier, whether it supports images, and bilingual descriptions. A user
facing 34 ids needs those to pick, so this prints them and reports which fields
are actually present per region.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import GLOBAL_BASE, GLOBAL_CONFIG_PATH, MODELS_PATH, load_accounts  # noqa


def get(url: str, h: dict) -> str:
    req = urllib.request.Request(url, headers=h, method="GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read(400_000).decode("utf-8", "replace")


def main() -> int:
    accounts = load_accounts()
    for region in ("cn", "global"):
        acct = next((a for a in accounts if a.region == region), None)
        if acct is None or not acct.usable_token():
            print(f"{region}: no usable account")
            continue
        url = (f"{GLOBAL_BASE}{GLOBAL_CONFIG_PATH}" if region == "global"
               else f"{acct.chat_base}{MODELS_PATH}")
        doc = json.loads(get(url, acct.chat_headers()))
        data = doc.get("data") or {}
        rows = data.get("models") or []
        print("=" * 100)
        print(f"{region.upper()}   account={acct.label}   models={len(rows)}")
        # Which keys exist at all, so a chooser knows what it can display.
        keys: set[str] = set()
        for r in rows:
            if isinstance(r, dict):
                keys |= set(r)
        print(f"fields: {', '.join(sorted(keys))}")
        print()
        print(f"{'id':26s} {'ctx':>9s} {'maxOut':>8s} {'mult':>7s} {'img':>4s}  description")
        for r in rows:
            if not isinstance(r, dict):
                continue
            mid = str(r.get("id") or r.get("model") or "")
            ctx = r.get("contextWindow") or r.get("maxInputTokens") or ""
            mo = r.get("maxOutputTokens") or r.get("maxAllowedSize") or ""
            mult = r.get("multiplier", r.get("credits", ""))
            img = "y" if r.get("supportsImages") else ""
            desc = str(r.get("descriptionEn") or "")[:44]
            print(f"{mid:26s} {str(ctx):>9s} {str(mo):>8s} {str(mult):>7s} {img:>4s}  {desc}")
        # Which agent uses which models -- the roster is per agent, not global.
        agents = data.get("agents") or []
        if agents:
            print(f"\nagents ({len(agents)}):")
            for a in agents[:6]:
                if not isinstance(a, dict):
                    continue
                print(f"  {str(a.get('description'))[:20]:22s} "
                      f"models={len(a.get('models') or [])} "
                      f"{(a.get('models') or [])[:5]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

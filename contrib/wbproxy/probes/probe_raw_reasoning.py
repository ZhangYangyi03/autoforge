"""Dump the raw reasoning field for the two models the user named.

The whole chain is: upstream catalog row -> parseReasoning -> thinkingLevelMap
-> model descriptor -> picker. Every link except the first has been verified
(lib/index.js:800 folds the singular form into a 5-level ladder; the built-in
fallback table has no ladders at all, but that table is not what a live CN
catalog serves).

So this prints the first link verbatim, for `deepseek-v4.1-flash` and
`step-5-preview`, exactly as the CN gateway sent it -- not normalised, not
summarised. If the reasoning object is not the shape assumed anywhere upstream,
that is where the answer is.
"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import load_accounts  # noqa: E402

TARGETS = ["deepseek-v4.1-flash", "step-5-preview"]


def main() -> int:
    accts = load_accounts(verbose=False)
    acct = next((a for a in accts if a.region == "cn"), None)
    if acct is None or not acct.usable_token():
        print("no usable CN account")
        return 1

    url = f"{acct.chat_base}/v2/enterprises/personal/models"
    req = urllib.request.Request(url, headers=acct.chat_headers(), method="GET")
    with urllib.request.urlopen(req, timeout=40) as r:
        doc = json.loads(r.read().decode("utf-8"))
    rows = (doc.get("data") or {}).get("models") or []
    print(f"CN catalog: {len(rows)} models via {acct.label}\n")

    for mid in TARGETS:
        row = next((x for x in rows if isinstance(x, dict)
                    and str(x.get("id")) == mid), None)
        print("=" * 78)
        print(mid)
        print("=" * 78)
        if row is None:
            print("  NOT IN CATALOG")
            continue
        print("  full row (verbatim):")
        print(json.dumps(row, indent=2, ensure_ascii=False))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

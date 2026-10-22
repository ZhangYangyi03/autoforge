"""Compare the two accounts' product configs field by field.

The global catalog returns 21 models but its `productFeatures` block is nearly
empty -- three sharing toggles, against the CN account's fourteen flags including
`RequestBodyGzip`, `EnableAutoModelTiers` and `DeferToolLoading`. An account
whose product features were never provisioned would answer catalog reads and
refuse chat, which is exactly what this one does.

This prints both blocks side by side, plus the agent rosters, so "the account is
not enabled for the CLI product" can be distinguished from "the request is wrong".
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import (GLOBAL_BASE, GLOBAL_CONFIG_PATH, MODELS_PATH,  # noqa: E402
                     load_accounts)


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
        try:
            doc = fetch(url, acct.chat_headers())
        except Exception as exc:                        # noqa: BLE001
            print(f"{region}: catalog failed {type(exc).__name__}: {exc}\n")
            continue
        data = doc.get("data") or {}
        feats = data.get("productFeatures") or {}
        print("=" * 72)
        print(f"{region.upper()}  {acct.label}")
        print(f"  enterpriseId   : {data.get('enterpriseId')!r}")
        print(f"  mergeStrategy  : {data.get('mergeStrategy')!r}")
        print(f"  models         : {len(data.get('models') or [])}")
        print(f"  agents         : {len(data.get('agents') or [])}")
        enabled = sorted(k for k, v in feats.items() if v)
        disabled = sorted(k for k, v in feats.items() if not v)
        print(f"  features ON ({len(enabled)}): {', '.join(enabled) or '-'}")
        print(f"  features OFF({len(disabled)}): {', '.join(disabled) or '-'}")

        # The CLI agent is the one a chat request impersonates. If it lists no
        # models, the product has not been enabled for this account.
        for agent in data.get("agents") or []:
            if not isinstance(agent, dict):
                continue
            desc = str(agent.get("description") or "")
            if "cli" in desc.lower():
                models = agent.get("models") or []
                print(f"  cli agent      : {len(models)} model(s)")
                if models:
                    print(f"                   {', '.join(map(str, models[:8]))}")
                tags = agent.get("modelTags")
                if tags:
                    print(f"  cli modelTags  : {tags}")
                break
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

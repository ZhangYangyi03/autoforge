"""Do the CN and global WorkBuddy accounts actually differ?

Question this answers: the user's `sumbge` account is registered at
`www.workbuddy.ai` (global) and is the one that 400'd during forge runs, while
the CN accounts succeeded. Two hypotheses, and they need different fixes:

  1. the global credential/endpoint is unusable  -> support must exclude it
  2. the global endpoint serves a DIFFERENT model roster (hy3 is a domestic
     Tencent model; dsh-xdpool ships a separate FALLBACK_WORKBUDDY_MODELS_GLOBAL
     with the domestic ids stripped)            -> support is fine, the model
                                                    id has to come from the
                                                    account's own catalog

So: per account, fetch its OWN catalog from its OWN endpoint, then try a chat
with a model that account itself advertises. Never with a hardcoded id.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import (GLOBAL_BASE, GLOBAL_CONFIG_PATH, MODELS_PATH,  # noqa: E402
                     load_accounts)


def get(url: str, headers: dict) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read(6000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(400).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def post(url: str, headers: dict, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, r.read(400).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(400).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def model_ids(doc: str) -> list[str]:
    try:
        obj = json.loads(doc)
    except Exception:
        return []
    raw = obj.get("models")
    if not isinstance(raw, list):
        raw = (obj.get("data") or {}).get("models")
    if not isinstance(raw, list):
        return []
    out = []
    for m in raw:
        if isinstance(m, dict):
            mid = m.get("id") or m.get("model")
            if mid:
                out.append(str(mid))
    return out


def main() -> int:
    accounts = load_accounts()
    if not accounts:
        print("no accounts")
        return 1

    for acct in accounts:
        token = acct.usable_token()
        print("=" * 74)
        print(f"{acct.label}   region={acct.region}   domain={acct.domain or '(empty)'}")
        print(f"  chat_base = {acct.chat_base}")
        if not token:
            print("  NO TOKEN")
            print()
            continue

        h = acct.chat_headers()
        if acct.region == "global":
            cat_url = f"{GLOBAL_BASE}{GLOBAL_CONFIG_PATH}"
        else:
            cat_url = f"{acct.chat_base}{MODELS_PATH}"
        code, raw = get(cat_url, h)
        ids = model_ids(raw)
        print(f"  catalog {code} {cat_url}")
        print(f"  models ({len(ids)}): {', '.join(ids[:14]) or '(none)'}"
              + (" ..." if len(ids) > 14 else ""))
        if code != 200:
            print(f"  catalog error: {raw[:160]}")

        # Chat with a model this account itself advertises. Never a guess.
        probe_model = None
        for want in ("auto", "hy3", "default"):
            if want in ids:
                probe_model = want
                break
        if probe_model is None and ids:
            probe_model = ids[0]
        if probe_model is None:
            print("  no model to probe with")
            print()
            continue

        url = f"{acct.chat_base}/v2/chat/completions"
        code, raw = post(url, h, {
            "model": probe_model, "stream": True,
            "messages": [{"role": "user", "content": "say OK"}],
        })
        mark = "OK " if code == 200 else "ERR"
        print(f"  chat [{mark}] {code} model={probe_model}  {raw[:150].replace(chr(10),' ')}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

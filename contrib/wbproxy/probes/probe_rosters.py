"""Compare the CN and global model rosters, and prove each works with its own id.

dsh-xdpool ships two fallback rosters (FALLBACK_WORKBUDDY_MODELS and
..._GLOBAL) and strips domestic ids from the global one, which is evidence the
rosters genuinely differ. This checks that against the live catalogs and then
runs one real chat per region using an id taken from that region's own list --
so "works" means "works with the id the account itself advertises".
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import GLOBAL_BASE, GLOBAL_CONFIG_PATH, MODELS_PATH, load_accounts  # noqa


def get(url: str, h: dict) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=h, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read(400_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(400).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def post(url: str, h: dict, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read(300).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(300).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def catalog(acct) -> list[dict]:
    """data.models, from whichever envelope this region uses."""
    url = (f"{GLOBAL_BASE}{GLOBAL_CONFIG_PATH}" if acct.region == "global"
           else f"{acct.chat_base}{MODELS_PATH}")
    code, raw = get(url, acct.chat_headers())
    if code != 200:
        return []
    try:
        doc = json.loads(raw)
    except Exception:
        return []
    data = doc.get("data") if isinstance(doc, dict) else None
    if not isinstance(data, dict):
        return []
    rows = data.get("models")
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def main() -> int:
    accounts = load_accounts()
    seen: dict[str, set[str]] = {}

    for acct in accounts:
        if not acct.usable_token():
            print(f"{acct.label}: no token, skipped")
            continue
        rows = catalog(acct)
        ids = [str(r.get("id") or r.get("model") or "") for r in rows]
        ids = [i for i in ids if i]
        seen.setdefault(acct.region, set()).update(ids)
        print("=" * 74)
        print(f"{acct.label}  region={acct.region}  models={len(ids)}")
        print(f"  {', '.join(ids)}")

        # Chat with the first id this account advertises, preferring a default.
        probe = next((i for i in ids if "default" in i.lower()), None) or \
                (ids[0] if ids else None)
        if probe is None:
            print("  (no id to probe)")
            print()
            continue
        url = f"{acct.chat_base}/v2/chat/completions"
        code, raw = post(url, acct.chat_headers(), {
            "model": probe, "stream": True,
            "messages": [{"role": "user", "content": "say OK"}],
        })
        mark = "OK " if code == 200 else "ERR"
        print(f"  chat [{mark}] {code} model={probe}")
        if code != 200:
            print(f"       {raw[:200].replace(chr(10), ' ')}")
        print()

    cn = seen.get("cn", set())
    gl = seen.get("global", set())
    if cn and gl:
        print("=" * 74)
        print(f"CN only      ({len(cn - gl)}): {', '.join(sorted(cn - gl))}")
        print(f"global only  ({len(gl - cn)}): {', '.join(sorted(gl - cn))}")
        print(f"both         ({len(cn & gl)}): {', '.join(sorted(cn & gl))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

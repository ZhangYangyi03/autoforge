"""Is the global 400 a header problem or an account problem?

Every global model id 400s with 11128 while every CN id answers 200, so the
variable is the endpoint/headers, not the model. This compares the exact header
set wbproxy sends for the global account against what dsh-xdpool's
chatHeaders() would send, field by field.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import load_accounts, CLIENT_UA, CHAT_IDENTITY_HEADERS  # noqa: E402


def dsh_xdpool_headers(acct) -> dict[str, str]:
    """`chatHeaders()` + CHAT_IDENTITY_HEADERS, transcribed from
    dsh-xdpool/lib/index.js:292-330."""
    origin = "https://www.workbuddy.ai" if acct.region == "global" \
        else "https://www.codebuddy.cn"
    h = {
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": origin,
        "Referer": f"{origin}/",
        "User-Agent": CLIENT_UA,
        "Content-Type": "application/json",
        "Authorization": f"Bearer {acct.access_token}",
        "X-Product": "SaaS",
    }
    if acct.uid:
        h["X-User-Id"] = acct.uid
    else:
        h["X-No-User-Id"] = "1"
    if acct.enterprise_id:
        h["X-Enterprise-Id"] = acct.enterprise_id
    else:
        h["X-No-Enterprise-Id"] = "1"
    if acct.domain:
        h["X-Domain"] = acct.domain
    else:
        h["X-No-Department-Info"] = "1"
    h.update(CHAT_IDENTITY_HEADERS)
    return h


def main() -> int:
    accounts = load_accounts()
    for acct in accounts:
        if not acct.usable_token():
            print(f"{acct.label}: no token")
            continue
        mine = acct.chat_headers()
        theirs = dsh_xdpool_headers(acct)
        print("=" * 74)
        print(f"{acct.label}  region={acct.region}  domain={acct.domain!r}  uid={acct.uid!r}")
        keys = sorted(set(mine) | set(theirs))
        for k in keys:
            a, b = mine.get(k), theirs.get(k)
            if k == "Authorization":
                a = a[:24] + "..." if a else a
                b = b[:24] + "..." if b else b
            flag = "" if a == b else "   <-- DIFFERS"
            print(f"  {k:26s} mine={str(a)[:34]:36s} xdpool={str(b)[:34]:36s}{flag}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

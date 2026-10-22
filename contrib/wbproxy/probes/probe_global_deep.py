"""Can the global account chat at all? Test the hypotheses left standing.

Already eliminated (see docs/findings.md):
  * wrong path      -- /v2/chat/completions exists; /v1, /v3, /responses,
                       /messages and /inference/v1 are all 404
  * bad token       -- the catalog reads 200 on /v3/config
  * bad model id    -- 12 ids, including global-only ones, all 11128
  * bad headers     -- 15 combinations, including a field-by-field match of
                       dsh-xdpool's chatHeaders(), all 11128
  * stream flag     -- stream=false reaches business logic (11101), so the
                       request is parsed; stream=true is 11128

Still standing, and tested here:
  1. request-body gzip -- the CN product config advertises
     `RequestBodyGzip: True`; if the global edge requires it, a plain body is
     refused by the edge rather than by the app
  2. a different chat host -- CN splits chat (copilot.tencent.com) from billing
     (www.codebuddy.cn), so global may split too
  3. a query parameter -- the stream flag as `?stream=1`
  4. an explicit request id -- some edges reject anonymous requests
  5. edge cookies -- the desktop client may send a session cookie
"""
import gzip
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wbproxy import load_accounts  # noqa: E402

BODY = {"model": "hy3", "stream": True,
        "messages": [{"role": "user", "content": "say OK"}]}


def post(url: str, headers: dict, raw: bytes | None = None,
         body: dict | None = None) -> tuple[int, str]:
    data = raw if raw is not None else json.dumps(body or BODY).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read(200).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(300).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {str(e)[:90]}"


def note(code: int, raw: str) -> str:
    if code == 200:
        return "OK"
    try:
        o = json.loads(raw)
        return f"code={o.get('code')} msg={o.get('msg') or o.get('error_msg')}"
    except Exception:
        return raw[:70].replace("\n", " ")


def main() -> int:
    accts = load_accounts(verbose=False)
    acct = next((a for a in accts if a.region == "global"), None)
    if acct is None or not acct.usable_token():
        print("no usable global account")
        return 1
    base = acct.chat_base
    h = acct.chat_headers()
    print(f"account={acct.label}  domain={acct.domain}  base={base}\n")

    print("--- 1. gzip request body ---")
    gz = gzip.compress(json.dumps(BODY).encode())
    for label, extra, payload in (
        ("plain body (baseline)", {}, None),
        ("gzip + Content-Encoding", {"Content-Encoding": "gzip"}, gz),
        ("gzip + charset", {"Content-Encoding": "gzip",
                            "Content-Type": "application/json;charset=UTF-8"}, gz),
    ):
        hh = {**h, **extra}
        code, raw = post(f"{base}/v2/chat/completions", hh, raw=payload)
        print(f"  [{code:3d}] {label:28s} {note(code, raw)}")

    print("\n--- 2. other hosts ---")
    host = acct.domain.replace("www.", "")
    for candidate in (
        "https://www.workbuddy.ai",
        f"https://chat.{host}",
        f"https://api.{host}",
        f"https://copilot.{host}",
        "https://www.workbuddy.cn",
        "https://api.workbuddy.ai",
    ):
        code, raw = post(f"{candidate}/v2/chat/completions", h)
        print(f"  [{code:3d}] {candidate:42s} {note(code, raw)}")

    print("\n--- 3. query parameters on the known-good host ---")
    for q in ("", "?stream=1", "?stream=true", "?alt=sse", "?v=2"):
        code, raw = post(f"{base}/v2/chat/completions{q}", h)
        print(f"  [{code:3d}] {'/v2/chat/completions' + q:42s} {note(code, raw)}")

    print("\n--- 4. request id / trace headers ---")
    import uuid
    for name in ("X-Request-Id", "X-Request-ID", "traceparent"):
        hh = {**h, name: str(uuid.uuid4())}
        code, raw = post(f"{base}/v2/chat/completions", hh)
        print(f"  [{code:3d}] +{name:28s} {note(code, raw)}")

    print("\n--- 5. accept header variants ---")
    for acc in ("text/event-stream", "application/json", "*/*"):
        hh = {**h, "Accept": acc}
        code, raw = post(f"{base}/v2/chat/completions", hh)
        print(f"  [{code:3d}] Accept: {acc:28s} {note(code, raw)}")

    print("\n--- 6. body variants that might satisfy an edge check ---")
    variants = {
        "no stream_options": {"model": "hy3", "stream": True,
                              "messages": [{"role": "user", "content": "hi"}]},
        "minimal (model+stream only)": {"model": "hy3", "stream": True,
                                        "messages": []},
        "empty messages": {"model": "hy3", "stream": True, "messages": []},
        "global-only model": {"model": "gpt-5.6-luna", "stream": True,
                              "messages": [{"role": "user", "content": "hi"}]},
    }
    for label, body in variants.items():
        code, raw = post(f"{base}/v2/chat/completions", h, body=body)
        print(f"  [{code:3d}] {label:30s} {note(code, raw)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

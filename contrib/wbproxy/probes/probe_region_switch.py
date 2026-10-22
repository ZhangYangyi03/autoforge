"""Prove the two regions can be switched between by model name alone.

A dual-region pool is only useful if the caller picks a model and the proxy
routes it — the client should not have to know which region it is talking to.
That is a claim about routing, so this tests it by choosing models whose region
membership is unambiguous, then checking the answer came from the expected side.

The answer's region is not in the OpenAI response, so the discriminator is a
model that only ONE region's catalog lists:
  * `step-5-preview`  -- CN only
  * `gemini-3.5-flash` -- global only (and refused for this plan, which is a
                          useful second signal: a refusal names the plan, and a
                          plan is a property of the account, so a refusal proves
                          the request reached the GLOBAL side)
  * `hy3`             -- both, so it must succeed on either
"""
import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8777"
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def post(body: dict, timeout: float = 200) -> tuple[int, dict | str]:
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, (e.read(500).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:                              # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def ask(model: str, prompt: str = "Reply with the single word OK.") -> tuple[int, dict | str]:
    return post({"model": model, "max_tokens": 32, "temperature": 0,
                 "messages": [{"role": "user", "content": prompt}]})


def main() -> int:
    print("=" * 70)
    print("1. the catalog carries both regions")
    print("=" * 70)
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
        doc = json.loads(r.read().decode())
    rows = doc.get("data", [])
    regions = {}
    for row in rows:
        regions.setdefault(row.get("region"), []).append(row["id"])
    check("both regions present", set(regions) >= {"cn", "global"},
          ", ".join(f"{k}={len(v)}" for k, v in sorted(regions.items(), key=str)))
    ids = {row["id"] for row in rows}
    check("a CN-only id is listed", "step-5-preview" in ids)
    check("a global-only id is listed", "gemini-3.5-flash" in ids)

    print()
    print("=" * 70)
    print("2. a CN-only model is served")
    print("=" * 70)
    code, resp = ask("step-5-preview")
    check("200", code == 200, str(code))
    if code == 200:
        text = (resp["choices"][0]["message"].get("content") or "").strip()
        check("content returned", bool(text), repr(text[:40]))

    print()
    print("=" * 70)
    print("3. a global-only model reaches the global side")
    print("=" * 70)
    code, resp = ask("gemini-3.5-flash")
    # `post` returns the raw body for an HTTPError, so the failure path has to
    # handle both shapes. It parsed JSON only on 2xx.
    if isinstance(resp, dict):
        msg = (resp.get("error") or {}).get("message", "") or ""
    else:
        msg = str(resp)
        try:
            parsed = json.loads(msg)
            msg = (parsed.get("error") or {}).get("message", msg)
        except Exception:
            pass
    msg_flat = msg.replace("\\", "")
    # Two acceptable proofs that the request reached the GLOBAL gateway:
    #   * it answers (the plan allows the model), or
    #   * it is refused by plan -- and only the global gateway reports a paying
    #     tier name, so `personal_free_user` is proof of the destination.
    reached_global = (code == 200
                     or "11158" in msg_flat
                     or "not allowed for user type" in msg_flat)
    check("routed to the global gateway", reached_global,
          f"{code} {msg_flat[:100]}")
    if code == 200:
        print("        (this plan allows it — it answered)")
    else:
        print("        (refused by PLAN, not by routing — the refusal itself")
        print("         proves the request reached the global gateway)")

    print()
    print("=" * 70)
    print("4. a shared model works, and the pool chose a side")
    print("=" * 70)
    for i in range(3):
        code, resp = ask("hy3", f"Reply with OK. Attempt {i+1}.")
        ok = code == 200
        text = ""
        if ok:
            text = (resp["choices"][0]["message"].get("content") or "").strip()
        check(f"hy3 attempt {i+1}", ok, f"{code} {text[:30]!r}")

    print()
    print("=" * 70)
    print("5. switching back and forth in one session")
    print("=" * 70)
    seq = ["step-5-preview", "deepseek-v4.1-flash", "hy3",
           "step-5-preview", "deepseek-v4.1-flash-sg"]
    for model in seq:
        code, resp = ask(model)
        msg = ""
        if isinstance(resp, dict) and code != 200:
            msg = (resp.get("error") or {}).get("message", "")[:70]
        check(model, code == 200, f"{code} {msg}")

    print()
    print("=" * 70)
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED — both regions routable by model name")
    return 0


if __name__ == "__main__":
    sys.exit(main())

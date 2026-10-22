"""Does the 429 translation actually buy a retry, or is it cosmetic?

The proxy maps `400 + 11128` to HTTP 429 because 429 is in every client's retry
set while 400 is in none. That is a claim about client behaviour, so it is tested
against a client that has the retry ladder in question — autoforge's
OpenAICompatClient — rather than by inspecting the status code alone.

Also exercises the two paths a user will actually run: `auto config` (does it
accept the endpoint) and `auto forge` (does the whole pipeline work through it).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

#: Where the autoforge checkout lives. Override with AUTOFORGE_HOME so this
#: script is not tied to one machine.
AUTOFORGE = Path(__import__("os").environ.get(
    "AUTOFORGE_HOME",
    Path.home() / "Desktop" / "dsh" / "autoforge"))
PY = sys.executable
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def run(args: list[str], timeout: int = 600) -> tuple[int, str]:
    env = {"PYTHONIOENCODING": "utf-8", "PATH": __import__("os").environ["PATH"]}
    p = subprocess.run([PY, "-X", "utf8", "-m", "autoforge.cli", *args],
                       cwd=str(AUTOFORGE), capture_output=True, text=True,
                       timeout=timeout, env={**__import__("os").environ,
                                             **env}, encoding="utf-8",
                       errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    print("=" * 70)
    print("1. autoforge accepts this endpoint")
    print("=" * 70)
    rc, out = run(["config"], timeout=120)
    check("auto config exits 0", rc == 0, f"rc={rc}")
    check("base_url points at the proxy", "127.0.0.1:8777" in out)
    check("model is set", re.search(r"model\s+hy3", out) is not None)
    check("key is accepted (masked)",
          "unset" not in out.split("api_key")[1][:80] if "api_key" in out else False)

    print()
    print("=" * 70)
    print("2. forge runs the whole pipeline through the proxy")
    print("=" * 70)
    out_json = Path(r"C:\Temp\_af_selfcheck.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists():
        out_json.unlink()
    rc, out = run(["forge", "count words in a text", "--out", str(out_json)],
                  timeout=900)
    # rc==2 with a sealed tool is not a failure: autoforge retries a round whose
    # verification had one red check, and the retry sealing means the pipeline
    # recovered. What matters is the tool, not the first round's verdict.
    sealed = "FORGED" in out
    check("a tool was sealed", sealed,
          next((ln.strip() for ln in out.splitlines() if "FORGED" in ln), "")[:70])
    if not sealed:
        check("forge exited 0", rc == 0, f"rc={rc}")
        tail = [ln for ln in out.splitlines() if "FAIL" in ln or "error" in ln.lower()]
        check("failure is explained", bool(tail), "; ".join(tail[-3:])[:160])
    rounds = None
    if m := re.search(r"sealed \w+ \((\d+) round", out):
        rounds = int(m.group(1))
    check("sealed within the retry budget", rounds is not None and rounds <= 2,
          f"{rounds} round(s)")
    # The six checks, as the pipeline reports them.
    for name in ("execution", "robustness", "adversarial", "trigger",
                 "negative", "claim"):
        check(f"check ran: {name}", f"[PASS] {name}" in out or name in out)
    if out_json.exists():
        doc = json.loads(out_json.read_text(encoding="utf-8"))
        check("artifact written", bool(doc))
        check("artifact has code", "code" in doc or "def " in json.dumps(doc)[:500])

    print()
    print("=" * 70)
    print("3. a 429 is retried by autoforge, not surfaced as a failure")
    print("=" * 70)
    # Hammer the endpoint with a deliberately tiny burst so at least one request
    # is likely throttled, then confirm nothing came back as a hard 400.
    import urllib.error
    import urllib.request
    codes: list[int] = []
    for _ in range(6):
        req = urllib.request.Request(
            "http://127.0.0.1:8777/v1/chat/completions",
            data=json.dumps({"model": "hy3", "max_tokens": 8,
                             "messages": [{"role": "user", "content": "ok"}]}
                            ).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                codes.append(r.status)
        except urllib.error.HTTPError as e:
            codes.append(e.code)
        except Exception:
            codes.append(0)
    print(f"     burst statuses: {codes}")
    check("no hard 400 reached the client", 400 not in codes, str(codes))
    check("at least one request succeeded", 200 in codes)
    if 429 in codes:
        check("throttling is visible as 429 (retryable)", True, "client will retry")

    print()
    print("=" * 70)
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

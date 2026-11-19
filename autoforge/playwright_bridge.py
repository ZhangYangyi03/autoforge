"""Playwright CDP locator adapter; no browser profiles or credentials are copied.

Uses the installed Microsoft Playwright Python package (Apache-2.0), rather
than vendoring its generated code. https://github.com/microsoft/playwright-python
The child process owns only its CDP connection, not the browser process.
"""
from __future__ import annotations
import json
import subprocess
import sys
from urllib.parse import urlsplit


def perform_locator(endpoint: str, target_id: str, selector: str, *,
                    action: str = "click", text: str = "", timeout: float = 10.0) -> str:
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost")
            or parsed.username is not None or parsed.password is not None
            or parsed.port is None or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment):
        raise ValueError("Playwright bridge requires a loopback HTTP CDP endpoint")
    if not target_id or not selector:
        raise ValueError("A target ID and selector are required")
    if action not in ("click", "fill"):
        raise ValueError("action must be click or fill")
    if not 0.1 <= timeout <= 60:
        raise ValueError("timeout must be between 0.1 and 60 seconds")
    request = dict(endpoint=endpoint, target_id=target_id, selector=selector,
                   action=action, text=text, timeout=timeout)
    try:
        proc = subprocess.run([sys.executable, "-m", "autoforge.playwright_bridge"],
                              input=json.dumps(request), text=True, encoding="utf-8",
                              capture_output=True, timeout=timeout + 20)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Playwright bridge timed out; action outcome is unknown") from exc
    try:
        result = json.loads(proc.stdout)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("Playwright bridge failed; action outcome is unknown") from exc
    if proc.returncode or not result.get("ok"):
        raise RuntimeError(result.get("error", "Playwright bridge failed"))
    return result["message"]


def _run(request: dict) -> dict:
    from playwright.sync_api import sync_playwright
    # connect_over_cdp connects to an existing browser, preserving its session.
    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(request["endpoint"], timeout=10000)
        page = None
        for context in browser.contexts:
            for candidate in context.pages:
                session = context.new_cdp_session(candidate)
                try:
                    info = session.send("Target.getTargetInfo")
                    if info["targetInfo"]["targetId"] == request["target_id"]:
                        page = candidate
                        break
                finally:
                    session.detach()
            if page is not None:
                break
        if page is None:
            raise ValueError("Attached tab no longer exists; reconnect before acting")
        locator = page.locator(request["selector"])
        ms = request["timeout"] * 1000
        if request["action"] == "click":
            locator.click(timeout=ms)
        else:
            locator.fill(request["text"], timeout=ms)
        return {"ok": True, "message": "Playwright " + request["action"] + " completed on the attached tab."}


if __name__ == "__main__":
    try:
        result = _run(json.load(sys.stdin))
    except Exception as exc:
        # Do not print framework exception messages: they may contain form data.
        result = {"ok": False, "error": "Playwright action failed (" + type(exc).__name__ + "); check selector, tab and actionability."}
    print(json.dumps(result))
    if not result.get("ok"):
        sys.exit(1)

import pytest
from autoforge.playwright_bridge import perform_locator

@pytest.mark.parametrize("endpoint", ["http://localhost:80@evil.com:9222", "http://evil.com:9222", "ws://127.0.0.1:9222", "http://127.0.0.1:9222/path"])
def test_reject_nonlocal_endpoints(endpoint):
    with pytest.raises(ValueError):
        perform_locator(endpoint, "target", "#button")

def test_reject_invalid_action_and_timeout():
    with pytest.raises(ValueError):
        perform_locator("http://127.0.0.1:9222", "target", "#button", action="execute")
    with pytest.raises(ValueError):
        perform_locator("http://127.0.0.1:9222", "target", "#button", timeout=61)

def test_no_secret_in_error(monkeypatch):
    import subprocess
    def fake(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout='{"ok": false, "error": "failed"}', stderr="")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError) as exc:
        perform_locator("http://127.0.0.1:9222", "target", "#password", action="fill", text="secret-value")
    assert "secret-value" not in str(exc.value)

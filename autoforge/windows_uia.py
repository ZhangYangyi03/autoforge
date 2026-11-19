"""Windows UI Automation bridge using pywinauto (BSD-3-Clause).

https://github.com/pywinauto/pywinauto. A handle and automation ID must identify
exactly one control. No credentials or control values are returned or logged.
"""
from __future__ import annotations
import json
import subprocess
import sys


def operate(hwnd: int, action: str, automation_id: str = "", text: str = "") -> str:
    if sys.platform != "win32":
        raise RuntimeError("Windows UI Automation requires Windows")
    if not isinstance(hwnd, int) or hwnd <= 0:
        raise ValueError("A positive window handle is required")
    if action not in ("inspect", "click", "fill"):
        raise ValueError("action must be inspect, click or fill")
    if action != "inspect" and not automation_id:
        raise ValueError("automation_id is required for actions")
    payload = dict(hwnd=hwnd, action=action, automation_id=automation_id, text=text)
    try:
        proc = subprocess.run([sys.executable, "-m", "autoforge.windows_uia"],
                              input=json.dumps(payload), capture_output=True,
                              encoding="utf-8", text=True, timeout=20)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Windows UI Automation timed out; action outcome unknown") from exc
    try:
        response = json.loads(proc.stdout)
    except ValueError as exc:
        raise RuntimeError("Windows UI Automation worker failed; action outcome unknown") from exc
    if proc.returncode or not response.get("ok"):
        raise RuntimeError(response.get("error", "Windows UI Automation worker failed"))
    return response["result"]


def _run(req: dict) -> dict:
    from pywinauto import Desktop
    import win32gui
    hwnd = req["hwnd"]
    if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
        raise ValueError("Window is missing or invisible")
    root = Desktop(backend="uia").window(handle=hwnd).wrapper_object()
    if req["action"] == "inspect":
        controls = []
        for control in root.descendants()[:200]:
            info = control.element_info
            # Names and values may contain private data; expose only IDs/types.
            controls.append({"automation_id": info.automation_id or "",
                             "control_type": info.control_type or ""})
        return {"ok": True, "result": json.dumps(controls, ensure_ascii=False)}
    controls = [c for c in root.descendants() if c.element_info.automation_id == req["automation_id"]]
    if len(controls) != 1:
        raise ValueError("Control ID must match exactly one descendant")
    control = controls[0]
    if not control.is_visible() or not control.is_enabled():
        raise ValueError("Control is hidden or disabled")
    if req["action"] == "click":
        control.invoke()
    else:
        control.set_edit_text(req["text"])
    return {"ok": True, "result": "UI Automation " + req["action"] + " completed."}


if __name__ == "__main__":
    try:
        result = _run(json.load(sys.stdin))
    except Exception as exc:
        # Exception messages may contain text entered into a field.
        result = {"ok": False, "error": "UI Automation failed (" + type(exc).__name__ + "); check window and control."}
    print(json.dumps(result, ensure_ascii=False))
    if not result.get("ok"):
        sys.exit(1)

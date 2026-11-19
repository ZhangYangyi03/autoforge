import json
import subprocess
import sys
import pytest
from autoforge.windows_uia import operate

@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        operate(0, "inspect")
    with pytest.raises(ValueError):
        operate(1, "click")
    with pytest.raises(ValueError):
        operate(1, "delete")

@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_worker_error_does_not_leak_text(monkeypatch):
    def fake(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout=json.dumps({"ok": False, "error": "failed"}), stderr="")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError) as exc:
        operate(1, "fill", "1001", "private-value")
    assert "private-value" not in str(exc.value)

@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_native_window_roundtrip():
    pytest.importorskip("pywinauto")
    import time
    import uuid
    import win32gui
    import win32con
    title = "AF_UIA_TEST_" + uuid.uuid4().hex
    script = r"""import sys,win32gui,win32con,win32api
wc=win32gui.WNDCLASS();wc.lpszClassName='AF_Uia_Test_Class';wc.lpfnWndProc={win32con.WM_DESTROY:lambda h,m,w,l:win32gui.PostQuitMessage(0)};win32gui.RegisterClass(wc)
h=win32gui.CreateWindow(wc.lpszClassName,sys.argv[1],win32con.WS_OVERLAPPEDWINDOW|win32con.WS_VISIBLE,100,100,400,200,0,0,win32api.GetModuleHandle(None),None)
win32gui.CreateWindow('EDIT','',win32con.WS_CHILD|win32con.WS_VISIBLE|win32con.WS_BORDER,10,10,180,30,h,1001,win32api.GetModuleHandle(None),None)
win32gui.CreateWindow('BUTTON','Go',win32con.WS_CHILD|win32con.WS_VISIBLE,10,50,80,30,h,1002,win32api.GetModuleHandle(None),None)
win32gui.PumpMessages()"""
    proc = subprocess.Popen([sys.executable, "-c", script, title])
    try:
        hwnd = 0
        for _ in range(80):
            hwnd = win32gui.FindWindow(None, title)
            if hwnd:
                break
            time.sleep(0.1)
        assert hwnd
        controls = json.loads(operate(hwnd, "inspect"))
        assert {"automation_id": "1001", "control_type": "Edit"} in controls
        assert {"automation_id": "1002", "control_type": "Button"} in controls
        operate(hwnd, "fill", "1001", "test-value")
        edit = win32gui.FindWindowEx(hwnd, 0, "Edit", None)
        assert win32gui.SendMessage(edit, win32con.WM_GETTEXTLENGTH, 0, 0) == len("test-value")
        operate(hwnd, "click", "1002")
        with pytest.raises(RuntimeError):
            operate(hwnd, "fill", "not-existing", "private-value")
    finally:
        proc.terminate()
        proc.wait(timeout=5)

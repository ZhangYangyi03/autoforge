"""No child of this process gets a console window.

The agent runs under pythonw through a scheduled task, so it owns no console.
Windows then gives a console-subsystem child a *fresh* console -- a window on
the operator's desktop -- for every subprocess that starts without the flag.
The spawns are scattered (webtools' search worker, market, schedule, cpu/safety,
mcp), so the flag is applied once, at import, instead of at each call site.

The probe matters as much as the flag. `GetConsoleWindow()` answers non-zero
even for a child whose window is not shown, so a test built on it would pass
while the operator still saw windows. These count *visible top-level windows
owned by the child's pid*, which is what a person actually sees -- and they
count them for the child's own children too, because a flag that only hides the
direct child's window moves the flash one level down rather than removing it.
"""
import ctypes
import os
import subprocess
import sys
from ctypes import wintypes

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="a Windows console is the subject")


def _visible_windows_of(pid):
    user32 = ctypes.windll.user32
    callback = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    found = []

    def visit(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == pid:
                cls = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(hwnd, cls, 256)
                found.append(cls.value)
        return True

    user32.EnumWindows(callback(visit), 0)
    return found


def test_popen_is_the_quiet_one_in_this_process():
    import autoforge  # noqa: F401  -- the import is what installs it

    assert subprocess.Popen.__name__ == "_QuietPopen"


def test_a_child_started_with_no_flags_shows_no_window():
    import autoforge  # noqa: F401

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _visible_windows_of(proc.pid) == []
    finally:
        proc.kill()
        proc.wait()


def test_an_explicit_keep_console_env_still_leaves_the_window_alone():
    """The opt-out is read at import, so it cannot un-patch a running process.

    Pinned because the honest statement of this behaviour is narrow: setting
    AUTOFORGE_KEEP_CONSOLE=1 for a *new* interpreter gives that interpreter back
    its windows, and this one keeps its patch. A test asserting the opposite
    would be asserting something the code does not do.
    """
    env = {**os.environ, "AUTOFORGE_KEEP_CONSOLE": "1"}
    out = subprocess.run(
        [sys.executable, "-c",
         "import subprocess; print(subprocess.Popen.__name__)"],
        capture_output=True, text=True, env=env)
    assert out.stdout.strip() == "Popen"

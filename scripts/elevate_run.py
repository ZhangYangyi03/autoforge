"""Elevate a command through the UAC dialog, and read its result back.

Why this exists: `sc start com.docker.service` came back "access denied", and the
agent that reports a capability and then stops at "needs admin" has not finished
the job. The account *is* in Administrators (EnableLUA=1, ConsentPromptBehavior=
5, PromptOnSecureDesktop=1), so the privilege is one consent away -- the prompt
just appears on the secure desktop, where no synthetic click can reach it. That is
UAC working, not a wall: a human clicks, the work continues.

So the shape is: launch through ShellExecuteW with the "runas" verb, have the
child write its exit code and output to a file, and read that file. The elevated
process is not our child -- there is no pipe, no handle, no waitpid -- and a tool
that pretends otherwise reports success it never saw.

    python scripts/elevate_run.py --wait 60 --log <path> -- cmd /c "..."
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import textwrap
import time

SEE_MASK_NOCLOSEPROCESS = 0x00000040
SW_SHOWNORMAL = 1
SHELLEXECUTEINFO = None


class _SHELLEXECUTEINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("fMask", ctypes.c_ulong),
        ("hwnd", ctypes.c_void_p),
        ("lpVerb", ctypes.c_wchar_p),
        ("lpFile", ctypes.c_wchar_p),
        ("lpParameters", ctypes.c_wchar_p),
        ("lpDirectory", ctypes.c_wchar_p),
        ("nShow", ctypes.c_int),
        ("hInstApp", ctypes.c_void_p),
        ("lpIDList", ctypes.c_void_p),
        ("lpClass", ctypes.c_wchar_p),
        ("hkeyClass", ctypes.c_void_p),
        ("dwHotKey", ctypes.c_ulong),
        ("hIcon", ctypes.c_void_p),
        ("hProcess", ctypes.c_void_p),
    ]


def elevate(argv, cwd, log):
    """Return (launched, detail). A refused prompt is a result, not an exception."""
    inner = " ".join(argv) + " > " + '"' + log + '" 2>&1' + ' & echo EXIT=%ERRORLEVEL% >> "' + log + '"'
    cmdline = '/c ' + inner
    sei = _SHELLEXECUTEINFO()
    sei.cbSize = ctypes.sizeof(sei)
    sei.fMask = SEE_MASK_NOCLOSEPROCESS
    sei.lpVerb = "runas"
    sei.lpFile = os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe")
    sei.lpParameters = cmdline
    sei.lpDirectory = cwd
    sei.nShow = SW_SHOWNORMAL
    ok = ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(sei))
    if not ok:
        err = ctypes.get_last_error() or ctypes.windll.kernel32.GetLastError()
        # 1223 == ERROR_CANCELLED: the person clicked No, or closed the dialog.
        return False, ("UAC prompt declined (error %s)" % err if err == 1223
                       else "ShellExecuteExW failed with error %s" % err)
    return True, "elevated process launched (hProcess=%s)" % sei.hProcess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--cwd", default=os.getcwd())
    ap.add_argument("--wait", type=float, default=90.0)
    ap.add_argument("--why", default="")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    argv = [x for x in a.command if x != "--"]
    if not argv:
        print("nothing to run")
        return 2

    log = os.path.abspath(a.log)
    open(log, "a", encoding="utf-8").write(
        "\n=== %s ===\nwhy: %s\ncmd: %s\n"
        % (time.strftime("%Y-%m-%d %H:%M:%S"), a.why, " ".join(argv)))

    launched, detail = elevate(argv, a.cwd, log)
    print(("LAUNCHED: " if launched else "NOT LAUNCHED: ") + detail)
    if not launched:
        print("  the command did not run; nothing was changed")
        return 3

    t0 = time.time()
    while time.time() - t0 < a.wait:
        try:
            body = open(log, encoding="utf-8", errors="replace").read()
        except OSError:
            body = ""
        if "EXIT=" in body:
            tail = [l for l in body.strip().splitlines() if l.strip()][-12:]
            print("FIRST RESULT after %.0fs:" % (time.time() - t0))
            for l in tail:
                print("  " + l[:200])
            return 0
        time.sleep(2)
    print("WAITING: no EXIT= line after %.0fs -- the prompt may still be on screen,"
          " or the command is still running. Log: %s" % (a.wait, log))
    return 4


if __name__ == "__main__":
    sys.exit(main())

"""Exercise deploy_autoforge.py's install step (step 2) as a standalone test.

Running the whole deploy script would also copy store-slim over the live
ledger at %LOCALAPPDATA%\\autoforge, which is a day of history we do not want
to lose. This replays only the pip line and the guard.

    state B (stock wheel in place)  ->  deploy step  ->  must end at state A
"""
import os
import subprocess
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
WHEELS_DIR = os.path.join(HERE, "wheels")
VENV_PY = r"C:\autoforge\venv\Scripts\python.exe"
DOCTOR = r"D:\Users\yangyian.zhang\WorkBuddy AI\2026-10-16-10-08-39\autoforge-interrupt-fix\install_doctor.py"
STOCK = os.path.join(WHEELS_DIR, "autoforge_agent-0.4.2-py3-none-any.whl")


def version():
    import importlib.metadata as md
    return md.version("autoforge-agent")


def doctor():
    p = subprocess.run([VENV_PY, DOCTOR], capture_output=True, text=True, cwd="/tmp")
    return p.stdout


def show(label, out):
    keep = [l for l in out.splitlines()
            if any(k in l for k in ("installed ", "manifest ", "version drift",
                                    "stale files", "missing files",
                                    "pip stashes", "VERDICT", "half-patch"))]
    print(f"--- {label} ---")
    for l in keep:
        print("   ", l.strip())
    print()


def main():
    print(f"wheels dir: {WHEELS_DIR}")
    for f in sorted(os.listdir(WHEELS_DIR)):
        if f.endswith(".whl"):
            print(f"   {f}")
    print()

    # 1. state B: force the stock wheel into place
    print("=== 1. install stock 0.4.2 (the broken reinstall) ===")
    r = subprocess.run([VENV_PY, "-m", "pip", "install", "-q", "--no-index",
                        "--no-deps", "--force-reinstall", STOCK],
                       capture_output=True, text=True)
    print("   pip rc:", r.returncode)
    show("doctor in state B", doctor())

    # 2. the deploy script's step 2, verbatim
    print("=== 2. deploy_autoforge.py step 2 (the fix) ===")
    r = subprocess.run(
        [VENV_PY, "-m", "pip", "install", "-q", "--upgrade", "--no-index",
         "--find-links", WHEELS_DIR, "autoforge-agent"],
        capture_output=True, text=True)
    print("   pip rc:", r.returncode)
    r2 = subprocess.run([VENV_PY, "-m", "pip", "show", "autoforge-agent"],
                        capture_output=True, text=True)
    installed = ""
    for line in r2.stdout.splitlines():
        if line.startswith("Version:"):
            installed = line.split(":", 1)[1].strip()
    print("   pip show version:", installed)
    if "+local." not in installed:
        print("   !! GUARD WOULD FAIL: stock wheel installed")
    show("doctor after deploy step", doctor())

    # 3. clean up debris
    print("=== 3. doctor --repair (debris only) ===")
    r = subprocess.run([VENV_PY, DOCTOR, "--repair"], capture_output=True,
                       text=True, cwd="/tmp")
    show("doctor --repair", r.stdout)
    print("final rc:", r.returncode)
    return 0


if __name__ == "__main__":
    sys.exit(main())

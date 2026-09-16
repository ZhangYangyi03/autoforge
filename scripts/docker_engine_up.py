"""Boot the Docker engine on this host with nobody clicking anything.

Why this exists: the Linux side of this machine (WSL2 + seccomp + landlock) is
where a real sandbox can live, and `docker` is the cheapest way to get one. But
`docker` was answering "The system cannot find the file specified" because Docker
Desktop -- the GUI app -- had never been started, and its WSL integration had
never been switched on. A capability that needs a human to click an icon is not a
capability this agent has.

So the whole boot is written out here, in the order it actually happens:
  1. persistence: AutoStart on, so the next Windows session does not repeat this;
  2. WSL integration: Ubuntu listed in IntegratedWslDistros, so the distro gets
     its own /var/run/docker.sock rather than shelling out to docker.exe;
  3. the privileged backend service, which may refuse without admin -- that is
     reported, not hidden;
  4. the app itself, launched detached with -Autostart so no window is waited on;
  5. a poll that only stops when the *engine* answers, plus a WSL-side socket check.

Every step reports what happened. "Started" is never assumed from a spawned pid.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import winreg

DOCKER_DESKTOP = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
DOCKER_EXE = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
SETTINGS = os.path.join(os.environ.get("APPDATA", r"C:\Users\china\AppData\Roaming"),
                        "Docker", "settings-store.json")
ENGINE_PIPES = ("dockerDesktopLinuxEngine", "docker_engine")


def _run(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           timeout=timeout, shell=isinstance(cmd, str))
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:                                    # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def persist_autostart():
    """Make it survive a reboot: Run key + Docker's own AutoStart flag."""
    notes = []
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                            winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, "Docker Desktop", 0, winreg.REG_SZ,
                              f'"{DOCKER_DESKTOP}" -Autostart')
        notes.append("HKCU Run key: Docker Desktop -> present")
    except Exception as e:                                    # noqa: BLE001
        notes.append(f"HKCU Run key: FAILED ({type(e).__name__}: {e})")
    if os.path.exists(SETTINGS):
        try:
            d = json.load(open(SETTINGS, encoding="utf-8", errors="replace"))
            before = d.get("AutoStart")
            d["AutoStart"] = True
            # Integration is what puts docker.sock inside the distro. Without it
            # the distro sees only the Windows docker.exe shim, which is not a
            # daemon and cannot run a container.
            distros = set(d.get("IntegratedWslDistros") or [])
            distros.add("Ubuntu")
            d["IntegratedWslDistros"] = sorted(distros)
            d["EnableIntegrationWithDefaultWslDistro"] = True
            tmp = SETTINGS + ".tmp"
            json.dump(d, open(tmp, "w", encoding="utf-8"), indent=2)
            os.replace(tmp, SETTINGS)
            notes.append(f"settings-store.json: AutoStart {before} -> True,"
                         f" IntegratedWslDistros={sorted(distros)}")
        except Exception as e:                                # noqa: BLE001
            notes.append(f"settings-store.json: FAILED ({type(e).__name__}: {e})")
    return notes


def service_state():
    rc, out = _run(["sc", "query", "com.docker.service"])
    state = [l.strip() for l in out.splitlines() if "STATE" in l]
    return state[0] if state else f"unknown (rc={rc})"


def start_service():
    rc, out = _run(["sc", "start", "com.docker.service"], timeout=90)
    if rc == 0:
        return "com.docker.service: start requested, now " + service_state()
    return ("com.docker.service: start refused without admin -- Docker Desktop "
            "will try to bring it up itself (" + out.strip().splitlines()[-1][:90] + ")")


def launch_app():
    if not os.path.exists(DOCKER_DESKTOP):
        return "Docker Desktop.exe: NOT INSTALLED at the expected path"
    if engine_answering(timeout=3):
        return "Docker Desktop: engine already answering, app not launched"
    flags = 0x08000000 | 0x00000200          # CREATE_NO_WINDOW | NEW_PROCESS_GROUP
    subprocess.Popen([DOCKER_DESKTOP, "-Autostart"], creationflags=flags,
                     close_fds=True)
    return "Docker Desktop: launched detached with -Autostart (no window waited on)"


def pipe_present():
    out = _run(["powershell", "-NoProfile", "-Command",
                "Get-ChildItem \\\\.\\pipe\\ | Where-Object Name -match 'docker' "
                "| Select-Object -ExpandProperty Name"])
    names = [n.strip() for n in out[1].splitlines() if n.strip()]
    return names


def engine_answering(timeout=4):
    rc, out = _run([DOCKER_EXE, "version", "--format",
                    "{{.Server.Version}}"], timeout=timeout)
    return rc == 0 and out.strip() != ""


def wsl_socket(distro="Ubuntu"):
    rc, out = _run(["wsl", "-d", distro, "-e", "bash", "-lc",
                    "test -S /var/run/docker.sock && echo yes || echo no"], timeout=60)
    return out.strip().endswith("yes")


def wait_for_engine(seconds):
    t0 = time.time()
    last = ""
    while time.time() - t0 < seconds:
        if engine_answering():
            rc, out = _run([DOCKER_EXE, "version", "--format", "{{.Server.Version}}"], timeout=10)
            return True, out.strip(), time.time() - t0
        last = pipe_present()
        time.sleep(5)
    return False, f"pipes seen: {last or '(none)'}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--no-launch", action="store_true")
    a = ap.parse_args()

    for line in persist_autostart():
        print("  " + line)
    print("  " + start_service())
    if not a.no_launch:
        print("  " + launch_app())
    ok, detail, took = wait_for_engine(a.timeout)
    print(f"  engine: {'UP ' + detail if ok else 'NOT UP ' + detail}"
          f"  ({took:.0f}s)")
    if ok:
        rc, out = _run([DOCKER_EXE, "ps", "--format", "{{.Names}}"], timeout=30)
        print("  docker ps:", (out.strip() or "(no containers)").replace("\n", ", ")[:200])
        print("  WSL /var/run/docker.sock in Ubuntu:", "present" if wsl_socket() else "absent")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

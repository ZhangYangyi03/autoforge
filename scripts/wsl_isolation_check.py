"""Ask the WSL Linux side whether its isolation actually bites.

Why this is a script and not a paragraph: "the sandbox uses seccomp and
Landlock" is a claim about a kernel, and this project has already been burned
once by a capability that was declared and never consulted (every autonomy
freedom used to be DECLARED_ONLY). The same failure is available here -- a
kernel reports Seccomp: 2 in /proc/self/status whether or not any filter is
installed, and WSL exposes no /sys/kernel/security/lsm, so the usual
"is Landlock on?" file does not exist to read.

So the check is behavioural. Two small C programs install a real restriction
and then attempt the thing that should be refused:

  wsl_seccomp_probe.c   -- denies socket(), then calls it and expects EPERM,
                           with a control call before the filter that must
                           succeed; a filter that fails everything proves
                           nothing.
  wsl_landlock_probe.c  -- confines the process to one directory read-only,
                           then opens a file outside it and expects EACCES.

Both print [ok]/[FAIL] per expectation and a failure count, so the verdict is
in the output rather than in this script's opinion of it.

    python scripts/wsl_isolation_check.py [--distro Ubuntu] [--keep]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PROBES = ("wsl_seccomp_probe.c", "wsl_landlock_probe.c")
PREREQS = ("gcc", "libc6-dev", "libseccomp-dev")


def wsl(distro: str, command: str, timeout: int = 300) -> tuple[int, str]:
    r = subprocess.run(["wsl", "-d", distro, "-e", "bash", "-lc", command],
                       capture_output=True, text=True, errors="replace",
                       timeout=timeout)
    # wsl.exe writes a localhost-forwarding complaint to stderr on every call.
    # Passing it through would make every failure look like a networking one.
    return r.returncode, (r.stdout or "").strip()


def win_to_wsl(path: str) -> str:
    path = path.replace("\\", "/")
    if len(path) > 1 and path[1] == ":":
        return "/mnt/" + path[0].lower() + path[2:]
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distro", default="Ubuntu")
    ap.add_argument("--here", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()

    here = a.here
    missing = [f for f in PROBES if not os.path.exists(os.path.join(here, f))]
    if missing:
        print("probe source not found: %s" % ", ".join(missing))
        return 2

    rc, out = wsl(a.distro, "uname -r; id -u; grep -E '^Seccomp' /proc/self/status")
    print("kernel / uid / seccomp field:\n  " + out.replace("\n", "\n  "))
    if not out or rc != 0:
        print("this distro is not running; nothing was tested")
        return 2

    rc, out = wsl(a.distro, "which gcc || echo NO-GCC")
    if "NO-GCC" in out:
        print("no compiler in %s. Install it first:\n"
              "  wsl -d %s -e bash -lc 'apt-get update && apt-get install -y %s'"
              % (a.distro, a.distro, " ".join(PREREQS)))
        return 2

    failures = 0
    for name in PROBES:
        src = win_to_wsl(os.path.join(here, name))
        exe = "/tmp/" + name[:-2]
        cmd = ("cp '%s' /tmp/%s && gcc -O2 -o %s /tmp/%s -lseccomp 2>&1 && %s; echo RC=$?"
               % (src, name, exe, name, exe))
        print("\n=== %s ===" % name)
        rc, out = wsl(a.distro, cmd)
        for line in out.splitlines():
            print("  " + line)
        if "RC=0" not in out:
            failures += 1
            print("  -> this probe did not pass")

    print("\nVERDICT: %s" % ("both isolation mechanisms refused what they should "
                             "on this kernel" if failures == 0
                             else "%d probe(s) failed" % failures))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

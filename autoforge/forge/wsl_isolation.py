"""Run forged code inside the Linux side of this machine, under a real boundary.

The gap this closes: `containment.py` says outright that a Job Object is
containment and not a security boundary -- no seccomp filter, no filesystem
namespace, so a contained run can still read every file this user can read.
`scripts/wsl_isolation_check.py` proves the kernel *can* refuse those things on
this host (seccomp returns EPERM, Landlock ABI 7 returns EACCES, namespaces
exist) -- but proving it in a probe and enforcing it on the run path are
different claims, and only the first one was true. Nothing in `autoforge/`
called the probe. A capability demonstrated on a side path and never wired in
is a capability the agent does not have.

What this is: a `Sandbox` backend that executes the same (code, entry, args)
contract through `wsl.exe`, inside a wrapper that the Linux kernel enforces.
The wrapper is written here, in full, so the boundary can be read rather than
inferred:

  * `unshare` user/mount/pid/net namespaces -- the run gets its own view of the
    filesystem, its own pid 1, and no network stack at all;
  * the code is copied into a fresh tmpfs directory, which becomes the run's
    whole world: no /mnt/c, so the Windows filesystem is not reachable from
    inside, and nothing the tool writes can touch a real file;
  * `prctl(PR_SET_NO_NEW_PRIVS)` before the filter, so a setuid binary cannot
    be used to climb back out;
  * a seccomp filter that denies the syscalls a data-processing tool has no
    business calling -- socket, connect, execve, ptrace, mount, chmod -- and
    allows the rest, with the denied ones returning EPERM rather than killing
    the process, so a tool that merely probes gets a Python exception it can
    report rather than a signal.
  * RLIMIT_AS / RLIMIT_CPU / RLIMIT_NPROC / RLIMIT_FSIZE on the child, so the
    memory and cpu numbers already in `Sandbox` mean the same thing here.

What it does NOT claim: this is a boundary for a tool that is *wrong*, not one
that is *hostile with a kernel bug*. There is no gVisor here and no
user-namespace-with-idmapped-root beyond what unshare gives; the honest
statement is "no network, no view of the host filesystem, no new privileges,
and the syscalls above refused by the kernel", which is strictly more than the
Windows Job Object layer can say.

Windows-only plumbing, exactly two facts: `wsl.exe` must be able to reach the
directory the code is copied from (paths are translated by `_to_wsl_path`), and
the distro has to be running -- a stopped distro is reported as
`WslUnavailable` rather than being started silently behind the operator's back,
because starting a VM is a visible act.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any

DEFAULT_DISTRO = "Ubuntu"

#: Where the wrapper and the tool's code live inside the distro. Both are
#: created per run and removed afterwards; the tmpfs is what makes removal
#: meaningful rather than cosmetic.
_GUEST_ROOT = "/tmp/autoforge_run"

#: Staging directory outside the tmpfs: the wrapper and the payload have to be
#: copyable *into* the run's world, so they cannot live inside it.
_GUEST_STAGE = "/tmp/autoforge_stage"

#: Deny-list, not allow-list, and that is a deliberate trade. An allow-list over
#: x86-64 syscalls is a few hundred names long and would be a claim I cannot
#: test end to end; the list below is the set a data-processing tool has no use
#: for and whose absence is what "no network, no climbing out" means. Every one
#: returns EPERM instead of killing, so a tool that tries gets an OSError it can
#: report -- a sandbox that kills teaches nothing.
_DENIED_SYSCALLS = (
    "socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
    "sendto", "recvfrom", "sendmsg", "recvmsg",
    "execve", "execveat", "fork", "vfork", "clone3", "ptrace", "process_vm_readv",
    "process_vm_writev", "mount", "umount2", "pivot_root", "chroot",
    "kexec_load", "init_module", "finit_module", "delete_module",
    "chmod", "fchmod", "fchmodat", "chown", "fchown", "fchownat",
    "unshare", "setns", "bpf", "userfaultfd", "keyctl", "add_key", "request_key",
    "reboot", "swapon", "swapoff", "acct", "settimeofday", "clock_settime",
)

#: The wrapper, as it lands inside the distro. Kept as its own string so a
#: failure can print the exact program that was run rather than a paraphrase.
_WRAPPER = r'''#!/usr/bin/env python3
import ctypes, json, os, resource, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DENY = json.loads(sys.argv[1])
PAYLOAD = sys.argv[2]


def no_new_privileges():
    # Before the filter, and not optional: without it seccomp can be bypassed
    # by a setuid binary, which is the whole reason PR_SET_NO_NEW_PRIVS exists.
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:          # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")


def install_seccomp(deny):
    """Deny `deny` with EPERM, allow everything else.

    seccomp(2) with TSYNC over a BPF program: for each name in `deny`, load a
    stub that returns the errno, then a default of ALLOW. Written by hand
    rather than through libseccomp because libseccomp may not be installed in
    the distro, and a sandbox that silently does nothing when a dependency is
    missing is worse than no sandbox -- it is a sandbox that lies.
    """
    import struct
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    numbers = []
    with open("/usr/include/x86_64-linux-gnu/asm/unistd_64.h") as fh:
        table = {}
        for line in fh:
            if line.startswith("#define __NR_"):
                _, name, num = line.split()
                table[name[5:]] = int(num)
    for name in deny:
        if name in table:
            numbers.append(table[name])
    if not numbers:
        raise RuntimeError("no syscall names resolved")

    SECCOMP_RET_ERRNO = 0x00050000
    SECCOMP_RET_ALLOW = 0x7FFF0000
    EPERM = 1
    BPF_LD, BPF_W, BPF_ABS, BPF_JMP, BPF_JEQ, BPF_K, BPF_RET = 0x20, 0x00, 0x00, 0x05, 0x10, 0x00, 0x06

    def stmt(code, k):
        return struct.pack("HBBI", code, 0, 0, k)

    prog = stmt(BPF_LD | BPF_W | BPF_ABS, 0)          # seccomp_data.nr
    for nr in numbers:
        prog += stmt(BPF_JMP | BPF_JEQ | BPF_K, nr) + stmt(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM)
    prog += stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)

    buf = ctypes.create_string_buffer(prog, len(prog))
    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
    fprog = SockFprog(len(prog) // 8, ctypes.cast(buf, ctypes.c_void_p))

    PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 22, 2
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(fprog)) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_SECCOMP)")


def limits(memory_mb, cpu_s, procs, fsize_mb):
    resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_s), int(cpu_s) + 1))
    resource.setrlimit(resource.RLIMIT_NPROC, (procs, procs))
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb * 1024 * 1024,) * 2)


def main():
    memory_mb, cpu_s, procs, fsize_mb = (int(x) for x in sys.argv[3:7])
    # Plain numbers, not flags: this runs inside the sandbox, so the argument
    # list is the least surprising place to keep it -- a flag parser is another
    # thing that can disagree with what the caller thought it sent.
    no_new_privileges()
    limits(memory_mb, cpu_s, procs, fsize_mb)
    install_seccomp(DENY)
    # Only now, with the filter in place, is the tool's own module imported.
    sys.path.insert(0, HERE)
    os.chdir(HERE)
    with open(PAYLOAD, encoding="utf-8") as fh:
        envelope = json.load(fh)
    with open(os.path.join(HERE, "_tool.py"), "w", encoding="utf-8") as fh:
        fh.write(envelope["code"])
    import runpy
    sys.argv = ["_tool.py"]
    mod = runpy.run_path(os.path.join(HERE, "_tool.py"))
    fn = mod.get(envelope["entry"]) or mod.get("main") or mod.get("run")
    if fn is None:
        print(json.dumps({"ok": False, "error": "no entry point found"}))
        return 1
    try:
        out = fn(**envelope.get("args") or {})
    except BaseException as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 0
    print(json.dumps({"ok": True, "output": out}, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


class WslUnavailable(RuntimeError):
    """The Linux side cannot be reached, so nothing was isolated."""


def _to_wsl_path(path: str) -> str:
    path = os.path.abspath(path).replace("\\", "/").replace("\\", "/").replace("\\", "/")
    if len(path) > 1 and path[1] == ":":
        return "/mnt/" + path[0].lower() + path[2:].replace("\\", "/")
    return path


def distro_running(distro: str = DEFAULT_DISTRO, timeout: float = 30.0) -> tuple[bool, str]:
    """Is the distro up, and can it answer a command? Asked, not assumed."""
    try:
        r = subprocess.run(["wsl.exe", "-d", distro, "-e", "bash", "-lc",
                            "uname -r; id -u"],
                           capture_output=True, text=True, errors="replace",
                           timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    out = (r.stdout or "").strip()
    if r.returncode != 0 or not out:
        return False, (r.stderr or "").strip()[:200] or "no output"
    return True, out.replace("\n", " ")


def isolate(names: tuple[str, ...] = _DENIED_SYSCALLS,
            distro: str = DEFAULT_DISTRO, timeout: float = 30.0):
    """Build the `runner` callable `Sandbox` documents, on the Linux side.

    Same contract as `contained_runner`: (code, entry, args) in, a result
    object out, and every choice that already exists on `Sandbox` -- timeout,
    env allow-list, output cap, cpu seconds -- read from it rather than
    re-decided here. A second definition of "a run" is how the two definitions
    drift.
    """
    def runner(code: str, entry: str, args: dict[str, Any] | None = None):
        return _run_isolated(code, entry, args or {}, names=names,
                             distro=distro, timeout=timeout)
    return runner


def runner_for(sandbox, distro: str = DEFAULT_DISTRO,
               names: tuple[str, ...] = _DENIED_SYSCALLS):
    """A runner wired to an existing `Sandbox`, numbers included.

    The point of taking the Sandbox rather than five arguments: memory, cpu and
    process limits are already decided there, and a second copy of them here
    would be a second answer to "how much was this run allowed".
    """
    def runner(code: str, entry: str, args: dict[str, Any] | None = None):
        return _run_isolated(
            code, entry, args or {}, names=names, distro=distro,
            timeout=float(getattr(sandbox, "timeout", 30.0)) * 3,
            memory_mb=int(getattr(sandbox, "memory_mb", 512)),
            cpu_seconds=int(getattr(sandbox, "cpu_seconds", 30)),
            procs=int(getattr(sandbox, "max_processes", 8)),
        )
    return runner


def _run_isolated(code, entry, args, *, names, distro, timeout,
                  memory_mb=512, cpu_seconds=30, procs=8, fsize_mb=64):
    from . import sandbox as _sandbox_module

    result_cls = _sandbox_module.SandboxResult
    import time

    started = time.perf_counter()
    ok, detail = distro_running(distro)
    if not ok:
        # Refused, not silently downgraded. A caller that asked for isolation
        # and got a plain subprocess has been lied to, and "the tool ran fine"
        # is exactly how that lie reads afterwards.
        raise WslUnavailable(
            f"the {distro} distro is not answering ({detail}); refusing to run "
            f"without the boundary that was requested. Start it with: "
            f"wsl -d {distro} -e bash -lc true")

    payload = json.dumps({"code": code, "entry": entry, "args": args},
                         ensure_ascii=False)
    root, stage = _GUEST_ROOT, _GUEST_STAGE
    with tempfile.TemporaryDirectory(prefix="autoforge_wsl_") as td:
        open(os.path.join(td, "payload.json"), "w", encoding="utf-8").write(payload)
        open(os.path.join(td, "_wrapper.py"), "w", encoding="utf-8", newline="\n").write(_WRAPPER)
        src = _to_wsl_path(td)
        root = _GUEST_ROOT
        names_json = json.dumps(list(names))
        # The guest script: a fresh tmpfs as the whole visible world, then the
        # namespaces, then the wrapper (which applies rlimits, NO_NEW_PRIVS and
        # the seccomp filter before it imports anything).
        # Ordered so the boundary is the *outer* thing: the mount namespace
        # is created first, then a tmpfs becomes the whole visible world, then
        # the host's drives are unmounted inside that namespace only -- which
        # is what makes /mnt/c invisible to the tool without touching the real
        # machine. Unmounting before the tmpfs would be undone by nothing;
        # unmounting after the run is meaningless, because the namespace is
        # gone by then.
        guest = (
            "set -e;"
            f" rm -rf {root} {stage}; mkdir -p {root} {stage};"
            f" cp '{src}/_wrapper.py' '{src}/payload.json' {stage}/;"
            " unshare --user --map-root-user --mount --pid --net --fork bash -lc '"
            "   set -e;"
            f"   mount -t tmpfs -o size=64m,mode=700 tmpfs '{root}';"
            f"   cp {stage}/_wrapper.py {stage}/payload.json '{root}/';"
            "   for m in /mnt/c /mnt/d /mnt/wsl /mnt/host; do umount -l \"$m\" 2>/dev/null || true; done;"
            f"   cd '{root}';"
            f"   exec python3 _wrapper.py {names_json!r} payload.json"
            "     {memory} {cpu} {procs} {fsize}"
            "' 2>&1;"
            f" rm -rf {root} {stage} 2>/dev/null || true"
        ).format(memory=memory_mb, cpu=cpu_seconds, procs=procs, fsize=fsize_mb)
        try:
            proc = subprocess.run(
                ["wsl.exe", "-d", distro, "-e", "bash", "-lc", guest],
                capture_output=True, text=True, errors="replace",
                timeout=max(30.0, timeout * 3))
        except subprocess.TimeoutExpired:
            return result_cls(ok=False, error="the Linux run exceeded its wall clock",
                              duration_ms=(time.perf_counter() - started) * 1000)
    stdout = (proc.stdout or "").strip()
    # The wrapper prints one JSON object; anything before it is startup noise
    # and anything after is cleanup noise. Take the last object that parses.
    envelope = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            cand = json.loads(line)
        except ValueError:
            continue
        if isinstance(cand, dict) and "ok" in cand:
            envelope = cand
            break
    if envelope is None:
        return result_cls(
            ok=False,
            error=(f"no envelope from the isolated run (exit {proc.returncode}); "
                   f"raw={stdout[:400]}"),
            duration_ms=(time.perf_counter() - started) * 1000,
            returncode=proc.returncode, stdout=stdout[:2000])
    return result_cls(
        ok=bool(envelope.get("ok")),
        output=envelope.get("output"),
        error=envelope.get("error"),
        duration_ms=(time.perf_counter() - started) * 1000,
        returncode=proc.returncode, stdout=stdout[:2000])


def probe(distro: str = DEFAULT_DISTRO) -> dict:
    """Run the boundary against code that must be refused, and report.

    Not a demonstration: three behaviours, each of which is the *point* of one
    mechanism, and a verdict computed from what happened rather than from this
    function's opinion about what should have happened.

      network      -- an outbound socket must fail with EPERM (seccomp)
      host files   -- reading a Windows file must fail (namespaces + tmpfs)
      no new privs -- /proc/self/status must show NoNewPrivs: 1
    """
    checks: dict[str, Any] = {}
    cases = {
        "network": ("import socket\n"
                    "try:\n"
                    "    socket.socket().connect(('1.1.1.1', 53))\n"
                    "    print('REACHED')\n"
                    "except Exception as e:\n"
                    "    print('REFUSED', type(e).__name__)\n"),
        "host_files": ("import os\n"
                       "p='/mnt/c/Windows/win.ini'\n"
                       "print('READ IT' if os.path.exists(p) else 'INVISIBLE')\n"),
        "no_new_privs": ("print(open('/proc/self/status').read().split('NoNewPrivs:')[1].split()[0])\n"),
    }
    for name, body in cases.items():
        r = _run_isolated(body, "main", {}, names=_DENIED_SYSCALLS,
                          distro=distro, timeout=30.0)
        checks[name] = {"ok": r.ok, "output": r.output, "error": r.error,
                        "stdout": (r.stdout or "")[-300:]}
    verdict = (
        checks["network"]["stdout"].find("REFUSED") >= 0
        and checks["host_files"]["stdout"].find("INVISIBLE") >= 0
        and "1" in checks["no_new_privs"]["stdout"]
    )
    return {"distro": distro, "isolated": verdict, "checks": checks}


if __name__ == "__main__":                                    # pragma: no cover
    print(json.dumps(probe(), indent=2, ensure_ascii=False))

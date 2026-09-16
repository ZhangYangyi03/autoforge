"""Process-isolated execution for forged tools.

Design stance (see DESIGN.md §2.5): a sandbox bounds *blast radius*, it does
not cap *capability*. We isolate by process + timeout + environment scrubbing
rather than by crippling builtins, because an agent that cannot use the
standard library cannot forge useful tools. The host is protected; the tool is
not castrated.

What this buys:
  * a tool that `os._exit()`s or segfaults cannot kill the agent
  * an infinite loop is reaped by timeout
  * a tool that sprays stdout does not corrupt the agent's protocol

What this does NOT claim:
  * it is not a security boundary against adversarial code. `restrict_builtins`
    narrows the surface for that case, but a真 sandbox needs OS-level
    containment (namespaces / job objects). Hook `runner` for that.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

#: Windows: give the child no console at all. Not CREATE_NO_WINDOW -- that flag
#: suppresses the *window* while Windows still creates a console, which shows up
#: as an extra hidden conhost.exe inside this sandbox's job object and moved the
#: measured `processes_launched` from 1 to 2 (found by test_containment, not by
#: reading). DETACHED_PROCESS creates no console for the child, so there is
#: nothing to display: same job count, no window. The child talks over pipes.
#:
#: Why it matters at all: the agent is started by a scheduled task through
#: pythonw, which has no console, and a console-subsystem child of a
#: console-less parent gets a *fresh* console window on the desktop.
_NO_CONSOLE = 0x00000008 if os.name == "nt" else 0

_RUNNER = textwrap.dedent('''
    import io, json, os, sys, contextlib

    # No window for anything this code starts. The runner itself is launched
    # detached, so Windows hands every console-subsystem child a *fresh* console
    # -- a window on the operator's desktop, one per subprocess a snippet
    # happens to start. Patched here rather than at the call sites, because the
    # call sites are arbitrary snippet code: the number of windows tracked the
    # number of subprocesses a run started, and a run that investigated the
    # flapping was the run that flapped most.
    #
    # The flag is measured, not read. Enumerating visible top-level windows by
    # pid: flags=0 -> one visible PseudoConsoleWindow, DETACHED_PROCESS -> none,
    # CREATE_NO_WINDOW -> none. `GetConsoleWindow()` is the wrong probe -- it
    # answers non-zero even under DETACHED_PROCESS, where no window is shown.
    if os.name == "nt":
        import subprocess as _sp
        _NO_WINDOW = 0x00000008                     # DETACHED_PROCESS

        class _QuietPopen(_sp.Popen):
            def __init__(self, *a, **kw):
                kw["creationflags"] = int(kw.get("creationflags") or 0) | _NO_WINDOW
                super().__init__(*a, **kw)

        _sp.Popen = _QuietPopen

    def _emit(obj):
        # Write UTF-8 bytes, not text. See the payload comment below: `-I`
        # makes PYTHONIOENCODING inert, so sys.stdout would encode with the
        # locale codec and blow up (or mojibake) on any non-ASCII result.
        sys.stdout.buffer.write(
            json.dumps(obj, default=str, ensure_ascii=False).encode("utf-8"))
        sys.stdout.buffer.flush()

    def _main():
        try:
            # Read raw bytes and decode UTF-8 ourselves. `-I` (isolated mode)
            # makes PYTHONIOENCODING inert, so `sys.stdin.read()` would use the
            # locale codec -- cp936 on a Chinese Windows host -- to decode a
            # payload the parent wrote as UTF-8. Any non-ASCII argument (a path
            # under 项目_开发, a CJK string) then mangles into a bogus escape
            # and the call dies with "bad payload: Invalid \\escape".
            payload = json.loads(sys.stdin.buffer.read().decode("utf-8") or "{}")
        except Exception as e:
            _emit({"ok": False, "error": f"bad payload: {e}"})
            return
        code = payload["code"]
        entry = payload["entry"]
        args = payload.get("args") or {}
        restrict = payload.get("restrict_builtins", False)
        ns = {"__name__": "__forged__"}
        try:
            if restrict:
                import builtins
                safe = {k: getattr(builtins, k) for k in (
                    "abs","all","any","bool","dict","enumerate","float","int",
                    "len","list","max","min","range","round","set","sorted",
                    "str","sum","tuple","zip","print","Exception","ValueError",
                    "TypeError","KeyError","IndexError","ZeroDivisionError",
                ) if hasattr(builtins, k)}
                ns["__builtins__"] = safe
            exec(compile(code, "<forged>", "exec"), ns)
            fn = ns.get(entry)
            if fn is None or not callable(fn):
                _emit({
                    "ok": False,
                    "error": f"entry {entry!r} not found or not callable",
                })
                return
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = fn(**args)
            _emit({"ok": True, "output": out, "stdout": buf.getvalue()})
        except BaseException as e:  # noqa: BLE001
            _emit({"ok": False, "error": f"{type(e).__name__}: {e}"})

    _main()
''').strip()


def _kill_tree(proc: "subprocess.Popen[bytes]") -> None:
    """Kill the child *and everything it started*.

    ``proc.kill()`` reaches the direct child only. A forged tool that shells
    out -- or a verification sample that spawns a worker -- leaves those
    grandchildren running, holding the pipes we are about to read from and the
    temp directory we are about to delete. On Windows that is exactly the
    lock that turns a clean abort into a hang, so the tree is killed by pid.
    """
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=10,
            )
        else:
            proc.kill()
    except Exception:                     # noqa: BLE001 - killing is best effort
        pass
    try:
        proc.wait(timeout=5)
    except Exception:                     # noqa: BLE001
        pass


@dataclass
class SandboxResult:
    ok: bool
    output: Any = None
    stdout: str = ""
    error: str | None = None
    timed_out: bool = False
    #: Killed because the operator spoke, not because the tool was slow or
    #: wrong. Kept separate from `timed_out` on purpose: a timeout is a verdict
    #: on the code, and writing it into a tool's failure ledger would punish the
    #: tool for a decision the *person* made.
    aborted: bool = False
    duration_ms: float = 0.0
    returncode: int | None = None
    stderr: str = ""
    #: What the kernel charged this run -- peak bytes, CPU, processes launched.
    #: Empty on the uncontained path, where nothing is measuring. It travels on
    #: the result rather than being read back from the job afterwards because
    #: the job is closed by then (KILL_ON_JOB_CLOSE), and because a result that
    #: can be read by anything is worth more than a handle that must be.
    accounting: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"ok": self.ok, "output": self.output, "error": self.error},
            ensure_ascii=False, default=str,
        )


@dataclass
class Sandbox:
    """Runs forged code out-of-process."""

    timeout: float = 10.0
    python: str = field(default_factory=lambda: sys.executable)
    restrict_builtins: bool = False
    max_output_bytes: int = 200_000
    env_allow: tuple[str, ...] = (
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP",
        "HOME", "USERPROFILE", "LANG", "LC_ALL", "PYTHONPATH",
    )
    runner: Callable[[str, str, dict[str, Any]], SandboxResult] | None = None
    #: Asked while a child is running: ``True`` means kill it now and report
    #: `aborted`. It lives on the sandbox rather than in `run`'s signature
    #: because the calls come from deep inside the verifier -- five checks, a
    #: fuzzer's worth of samples each -- and threading a parameter through all
    #: of them would mean touching every call site to say the same thing. Set
    #: it for the span of a run and clear it after.
    abort_check: Callable[[], bool] | None = None
    #: OS-level containment -- a Job Object on Windows, rlimits on POSIX. On by
    #: default where the kernel API exists, because the agent's own reach being
    #: bounded is the point, not an option. `contain=False` is the old path:
    #: process + timeout + environment, which bounds blast radius and nothing
    #: else -- a tool that detaches a grandchild outlives the run.
    contain: bool = field(default_factory=lambda: os.name == "nt")
    #: Run on the Linux side instead, under a real kernel boundary: namespaces,
    #: a tmpfs for the whole visible filesystem, no network, NO_NEW_PRIVS and a
    #: seccomp filter. Off unless asked for, because it needs the WSL distro to
    #: be running and is slower; when it is on and the distro is down, the run
    #: is REFUSED rather than quietly downgraded to the Windows path. A caller
    #: who asked for a boundary and silently got none has been lied to, and
    #: "the tool ran fine" is exactly how that lie reads afterwards.
    isolate: bool = False
    #: Which distro to isolate inside, when `isolate` is on.
    distro: str = "Ubuntu"
    #: The three numbers the job enforces. Read by `reach()` so the report
    #: states limits that were set, not limits that were intended.
    memory_mb: int = 512
    max_processes: int = 8
    cpu_seconds: float = 30.0

    def effective_env(self) -> dict[str, str]:
        """The environment forged code actually gets — not the agent's own.

        Names missing from `env_allow` are dropped, so a command that resolves
        for the agent process (a git-bash `ps`, say) can be unreachable from
        inside the sandbox. Everything the prompt claims about this host is
        built from *this* mapping, never from `os.environ`.
        """
        env = {k: v for k, v in os.environ.items() if k in self.env_allow}
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def resolve(self, names) -> dict[str, str | None]:
        """For each name, the path it resolves to from inside the sandbox.

        `reachable` answers the weaker question — "is this name on PATH". The
        question that actually matters is the stronger one: can the binary it
        points at answer the question I am about to ask it? On Windows those
        two answers disagree about `ps`: git-bash ships an MSYS `ps` that is on
        PATH and lists MSYS processes only, so a probe built on it reports a
        nearly empty machine and the emptiness reads as "no such process". The
        path is what tells a blind lister from a working one, so the path is
        what this returns.
        """
        import shutil

        path = self.effective_env().get("PATH")
        return {n: shutil.which(n, path=path) for n in names}

    def reachable(self, names) -> tuple[list[str], list[str]]:
        """Split `names` into (resolvable, not) from inside the sandbox."""
        found = self.resolve(names)
        return ([n for n in names if found.get(n)],
                [n for n in names if not found.get(n)])

    def run(self, code: str, entry: str, args: dict[str, Any] | None = None) -> SandboxResult:
        if self.isolate:
            # Checked before `runner`, because `isolate` is the stronger
            # promise: a caller that set both meant the boundary, not the hook.
            from .wsl_isolation import WslUnavailable, runner_for

            try:
                return runner_for(self, distro=self.distro)(code, entry, args or {})
            except WslUnavailable as exc:
                return SandboxResult(
                    ok=False, error=str(exc),
                    duration_ms=0.0, returncode=None,
                )
        if self.runner is not None:
            return self.runner(code, entry, args or {})
        if self.contain:
            # Imported here, not at module scope: the containment module talks
            # to the kernel, and a Sandbox with contain=False must not require
            # that API to exist.
            from .containment import contained_runner

            return contained_runner(
                self, memory_mb=self.memory_mb,
                max_processes=self.max_processes,
                cpu_seconds=self.cpu_seconds,
            )(code, entry, args or {})

        import time

        payload = json.dumps(
            {
                "code": code,
                "entry": entry,
                "args": args or {},
                "restrict_builtins": self.restrict_builtins,
            },
            ensure_ascii=False,
        )

        # ignore_cleanup_errors: a forged tool that detaches a grandchild
        # leaves that child sitting in this directory with its cwd held
        # open, and Windows then refuses the rmdir. Without this the
        # PermissionError escapes `run` -- a tool that followed the rules
        # and returned correctly is reported as a crash of the sandbox.
        # Found by A/B against the contained path, not by reading.
        with tempfile.TemporaryDirectory(
                prefix="autoforge_", ignore_cleanup_errors=True) as td:
            runner_path = os.path.join(td, "_runner.py")
            with open(runner_path, "w", encoding="utf-8") as fh:
                fh.write(_RUNNER)

            env = self.effective_env()

            started = time.perf_counter()
            try:
                proc = subprocess.Popen(
                    [self.python, "-I", runner_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=td,
                    # All three stdio handles are pipes, so the child needs no
                    # console -- and on Windows a console-subsystem child of a
                    # parent that has no console gets a *new* one, with a
                    # visible window. Every sandboxed call would flash one.
                    creationflags=_NO_CONSOLE,
                )
            except OSError as exc:
                return SandboxResult(
                    ok=False,
                    error=f"could not start the sandbox interpreter: {exc}",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )

            # The child is fed and drained on a worker thread; only the *wait*
            # is polled here. That inversion is the whole point: the wait is the
            # long part, and it is the only part the person at the terminal is
            # locked out by. `communicate` is what makes reading both pipes safe
            # without deadlocking on a full buffer, so it stays -- it just no
            # longer owns the thread while it does its job.
            done = threading.Event()
            box: dict[str, Any] = {}

            def _drain() -> None:
                try:
                    box["out"], box["err"] = proc.communicate(
                        input=payload.encode("utf-8"))
                except BaseException as exc:      # noqa: BLE001
                    box["exc"] = exc
                finally:
                    done.set()

            threading.Thread(target=_drain, daemon=True).start()
            deadline = started + self.timeout
            verdict: str | None = None
            while not done.wait(0.2):
                if self.abort_check is not None and self.abort_check():
                    verdict = "aborted"
                    break
                if time.perf_counter() > deadline:
                    verdict = "timed_out"
                    break

            if verdict is not None:
                _kill_tree(proc)
                done.wait(timeout=10)
                return SandboxResult(
                    ok=False,
                    error=("aborted by the operator" if verdict == "aborted"
                           else f"timeout after {self.timeout}s"),
                    timed_out=verdict == "timed_out",
                    aborted=verdict == "aborted",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    stdout=(box.get("out") or b"").decode(
                        "utf-8", "replace")[: self.max_output_bytes],
                )

            duration = (time.perf_counter() - started) * 1000
            stdout = (box.get("out") or b"").decode("utf-8", "replace")
            stderr = (box.get("err") or b"").decode(
                "utf-8", "replace")[: self.max_output_bytes]

            if not stdout.strip():
                return SandboxResult(
                    ok=False,
                    error=f"no output (exit {proc.returncode}) {stderr.strip()[:500]}",
                    duration_ms=duration,
                    returncode=proc.returncode,
                    stderr=stderr,
                )
            # The runner writes exactly one envelope as its last line, but the
            # child is untrusted: a stray module-level statement, a thread, or a
            # print that escaped the redirect can land after it. A bare JSON
            # scalar on the final line used to reach ``data.get("ok")`` and die
            # with ``AttributeError: 'str' object has no attribute 'get'`` --
            # a message naming neither the tool nor the output. Take the last
            # line that is an *object*; otherwise say what actually arrived.
            data: dict[str, Any] | None = None
            for line in reversed(stdout.strip().splitlines()):
                try:
                    cand = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(cand, dict) and "ok" in cand:
                    data = cand
                    break
            if data is None:
                return SandboxResult(
                    ok=False,
                    error=(
                        f"no result envelope in output (exit {proc.returncode}); "
                        f"raw={stdout.strip()[:300]}"
                    ),
                    duration_ms=duration,
                    returncode=proc.returncode,
                    stderr=stderr,
                )
            return SandboxResult(
                ok=bool(data.get("ok")),
                output=data.get("output"),
                stdout=data.get("stdout", ""),
                error=data.get("error"),
                duration_ms=duration,
                returncode=proc.returncode,
                stderr=stderr,
            )


    def reach(self, probe: bool = False) -> dict[str, Any]:
        """What forged code can actually touch — stated, and optionally measured.

        The thing this exists to prevent: an agent that infers its own reach
        from its tool list ("I have no read_file or run_shell, so I cannot touch
        the machine") and then explains the gap with a story about isolation.
        That story is false. Forged code is a subprocess of *this* process, so
        it inherits this machine's filesystem and network. The host is not over
        there behind a wall; it is the ground the sandbox stands on.

        So the honest split is:

          * reach    — a real filesystem and a real network stack, i.e. the
                       things the agent tends to deny it has
          * bounds   — a fresh cwd per call, a scrubbed environment and a
                       timeout, i.e. what limits blast radius without capping
                       capability

        `probe=True` runs an actual round-trip (write a file outside cwd and
        read it back; resolve a hostname) so the report is evidence rather than
        an assertion. Callers that want speed over proof leave it off.
        """
        facts: dict[str, Any] = {
            "host": "this machine — the sandbox is a subprocess of the agent",
            "isolated_from_host": False,
            "filesystem": "read-write, whole host (not confined to cwd)",
            "network": "outbound — DNS and sockets",
            "cwd": "a fresh empty temp dir, per call",
            "env": f"{len(self.env_allow)} allow-listed vars, not the full environment",
            "restrict_builtins": self.restrict_builtins,
            "timeout_s": self.timeout,
            "contained": self._containment_line(),
        }
        if probe:
            facts["probe"] = self._probe()
        return facts

    def _containment_line(self) -> str:
        """What the kernel is holding the run to, said in the same breath as reach.

        Kept next to `isolated_from_host: False` on purpose. Containment does
        not make the host disappear -- a contained run still reads the files
        this user can read. What changes is that it cannot take the machine's
        memory, its process table, or its CPU with it, and it cannot leave a
        detached grandchild behind to outlive the answer it gave.
        """
        if self.runner is not None:
            return "a custom runner (its limits are the runner's business)"
        if not self.contain:
            return ("none -- process + timeout only; a detached grandchild "
                    "outlives the run")
        if os.name == "nt":
            return (f"job object: {self.memory_mb} MB memory cap, "
                    f"{self.max_processes} processes, {self.cpu_seconds}s CPU, "
                    "kill-on-close reaches the whole tree")
        return "rlimits (address space, CPU, file size, open files)"

    def _probe(self) -> dict[str, Any]:
        """A real round-trip, so `reach` can be checked instead of believed.

        The file is written to the host temp dir, which is deliberately *not*
        the per-call cwd — writing inside cwd would prove nothing about escaping
        it. The name is fixed and the file is removed in the same call.
        """
        target = os.path.join(tempfile.gettempdir(), ".autoforge_reach_probe")
        r = self.run(
            "def probe(home):\n"
            "    import os, socket\n"
            "    out = {}\n"
            "    try:\n"
            "        with open(home, 'w', encoding='utf-8') as fh:\n"
            "            fh.write('reach')\n"
            "        back = open(home, encoding='utf-8').read()\n"
            "        os.remove(home)\n"
            "        out['host_filesystem'] = {'ok': back == 'reach', 'detail': home}\n"
            "    except Exception as e:\n"
            "        out['host_filesystem'] = {'ok': False, 'detail': type(e).__name__ + ': ' + str(e)}\n"
            "    try:\n"
            "        out['network'] = {'ok': True, 'detail': socket.gethostbyname('pypi.org')}\n"
            "    except Exception as e:\n"
            "        out['network'] = {'ok': False, 'detail': type(e).__name__ + ': ' + str(e)}\n"
            "    return out\n",
            "probe",
            {"home": target},
        )
        if not r.ok:
            return {"error": r.error or "probe did not run"}
        return r.output if isinstance(r.output, dict) else {"error": f"unexpected {r.output!r}"}


__all__ = ["Sandbox", "SandboxResult"]

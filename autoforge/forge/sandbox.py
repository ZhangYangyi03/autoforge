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
from dataclasses import dataclass, field
from typing import Any, Callable

_RUNNER = textwrap.dedent('''
    import io, json, sys, contextlib

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


@dataclass
class SandboxResult:
    ok: bool
    output: Any = None
    stdout: str = ""
    error: str | None = None
    timed_out: bool = False
    duration_ms: float = 0.0
    returncode: int | None = None
    stderr: str = ""

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

    def reachable(self, names) -> tuple[list[str], list[str]]:
        """Split `names` into (resolvable, not) from inside the sandbox."""
        import shutil

        path = self.effective_env().get("PATH")
        found = [n for n in names if shutil.which(n, path=path)]
        missing = [n for n in names if n not in found]
        return found, missing

    def run(self, code: str, entry: str, args: dict[str, Any] | None = None) -> SandboxResult:
        if self.runner is not None:
            return self.runner(code, entry, args or {})

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

        with tempfile.TemporaryDirectory(prefix="autoforge_") as td:
            runner_path = os.path.join(td, "_runner.py")
            with open(runner_path, "w", encoding="utf-8") as fh:
                fh.write(_RUNNER)

            env = self.effective_env()

            started = time.perf_counter()
            try:
                proc = subprocess.run(
                    [self.python, "-I", runner_path],
                    input=payload.encode("utf-8"),
                    capture_output=True,
                    timeout=self.timeout,
                    env=env,
                    cwd=td,
                )
            except subprocess.TimeoutExpired as exc:
                return SandboxResult(
                    ok=False,
                    error=f"timeout after {self.timeout}s",
                    timed_out=True,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    stdout=(exc.stdout or b"").decode("utf-8", "replace")[: self.max_output_bytes],
                )

            duration = (time.perf_counter() - started) * 1000
            stdout = (proc.stdout or b"").decode("utf-8", "replace")
            stderr = (proc.stderr or b"").decode("utf-8", "replace")[: self.max_output_bytes]

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
        }
        if probe:
            facts["probe"] = self._probe()
        return facts

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

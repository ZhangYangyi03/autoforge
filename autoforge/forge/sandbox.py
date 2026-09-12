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

    def _main():
        try:
            payload = json.loads(sys.stdin.read() or "{}")
        except Exception as e:
            sys.stdout.write(json.dumps({"ok": False, "error": f"bad payload: {e}"}))
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
                sys.stdout.write(json.dumps({
                    "ok": False,
                    "error": f"entry {entry!r} not found or not callable",
                }))
                return
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = fn(**args)
            sys.stdout.write(json.dumps(
                {"ok": True, "output": out, "stdout": buf.getvalue()},
                default=str,
            ))
        except BaseException as e:  # noqa: BLE001
            sys.stdout.write(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))

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

            env = {k: v for k, v in os.environ.items() if k in self.env_allow}
            env.setdefault("PYTHONIOENCODING", "utf-8")

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
            try:
                data = json.loads(stdout.strip().splitlines()[-1])
            except (json.JSONDecodeError, IndexError) as exc:
                return SandboxResult(
                    ok=False,
                    error=f"unparseable output: {exc}; raw={stdout.strip()[:300]}",
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


__all__ = ["Sandbox", "SandboxResult"]

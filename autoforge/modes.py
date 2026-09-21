"""Runtime modes.

The same harness can run the agent in different modes. Two ship here:

  standard — the full forging agent (ForgeAgent): meta-tools, verification
             pipeline, evolution, spawning.
  minimal  — a two-tool coding agent (bash + str_replace_editor), the sort of
             stripped-down harness you use to benchmark a model with almost no
             scaffolding.

The point of having both behind one UI is that you can run the *same* task in
each and see what the extra capability actually buys. A mode is a different
assembly of the same parts, not a different program.

Both classes expose the small surface the web harness relies on:

    .run(task, history) -> AgentResult
    .trace              -> list[dict]     (append-only event log)
    .registry           -> has .report(), .schemas(), .call()
    .pipeline           -> the forge pipeline, or None if the mode cannot forge
    .store              -> persistence, or None
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

from .autonomy.policy import AutonomyPolicy
from .core.agent import Agent, AgentResult
from .core.compaction import (
    Compactor,
    DeterministicSummarizer,
    LLMSummarizer,
    default_log_path,
)
from .core.llm import LLMClient
from .tools.registry import ToolRegistry
from .tools.spec import ToolSpec, ToolState

__all__ = ["MinimalAgent", "MINIMAL_SYSTEM"]


MINIMAL_SYSTEM = """You are a coding agent working in a real workspace.

You have exactly two tools:

- bash               run a shell command and read its output
- str_replace_editor view, create, or edit a file

Work in small steps: look, change one thing, verify, repeat. Read a file
before you edit it. Prefer the smallest edit that fixes the problem. When the
task is done, say what you changed and how you checked it.
"""


# --------------------------------------------------------------------------
# the two tools
# --------------------------------------------------------------------------


def _posix_shell() -> str | None:
    """A real POSIX shell on this host, or None.

    On Windows ``shell=True`` means cmd.exe, and this tool is called ``bash``
    with a POSIX contract: a model writing ``cat note.txt && sleep 5`` gets
    "'cat' is not recognized as an internal or external command" on every
    command it tries, and the benchmark measures that as the model being bad at
    coding. Measured here: Git for Windows is installed and provides ``sleep``,
    ``cat`` and the rest. So when a POSIX shell exists it is used, and cmd.exe is
    only the fallback when it does not.

    ``SHELL`` is honoured first, so a person with their own shell gets it.
    """
    if os.name != "nt":
        return os.environ.get("SHELL") or "/bin/sh"
    cands = [os.environ.get("SHELL")]
    for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                 r"C:\Program Files", r"C:\Program Files (x86)"):
        if root:
            cands += [os.path.join(root, "Git", "bin", "bash.exe"),
                      os.path.join(root, "Git", "usr", "bin", "bash.exe")]
    for c in cands:
        if c and os.path.basename(c).lower().startswith(("bash", "sh", "zsh")) \
                and os.path.exists(c):
            return c
    return None


def _bash(command: str, timeout: int = 60, cwd: str | None = None) -> str:
    """Run a shell command; return combined stdout/stderr with the exit code.

    The shell is POSIX where the host has one -- see `_posix_shell` for why that
    is not just a preference.
    """
    timeout = max(1, min(int(timeout or 60), 600))
    shell = _posix_shell()
    argv: str | list[str] = [shell, "-c", command] if shell else command
    try:
        proc = subprocess.run(
            argv, shell=shell is None, cwd=cwd or os.getcwd(),
            capture_output=True, text=True, errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"[timeout] command exceeded {timeout}s"
    except Exception as exc:                                # noqa: BLE001
        return f"[error] {type(exc).__name__}: {exc}"

    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out) > 40_000:
        out = out[:40_000] + "\n…[truncated]"
    return f"exit={proc.returncode}\n{out}".rstrip()


def _editor(command: str, path: str, file_text: str | None = None,
            old_str: str | None = None, new_str: str | None = None,
            insert_line: int | None = None, view_range: list[int] | None = None,
            cwd: str | None = None) -> str:
    """A small str_replace_editor: view / create / str_replace / insert.

    Relative paths resolve against `cwd` — the mode's workspace — so the editor
    and `bash` always agree about where "note.txt" is. Resolving against the
    process's own working directory instead makes the two tools disagree, and
    the agent then writes a file it cannot read back.
    """
    root = cwd or os.getcwd()
    p = path if os.path.isabs(path) else os.path.join(root, path)
    p = os.path.abspath(os.path.expanduser(p))
    command = (command or "view").strip()

    if command == "view":
        if not os.path.exists(p):
            return f"[error] no such file: {p}"
        if os.path.isdir(p):
            return "\n".join(sorted(os.listdir(p))[:500])
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        lo, hi = 1, len(lines)
        if view_range and len(view_range) == 2:
            lo, hi = max(1, int(view_range[0])), min(len(lines), int(view_range[1]))
        width = len(str(hi))
        return "\n".join(f"{i:>{width}}\t{lines[i - 1]}" for i in range(lo, hi + 1))

    if command == "create":
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(file_text or "")
        return f"created {p} ({len((file_text or '').splitlines())} lines)"

    if command == "str_replace":
        if not os.path.exists(p):
            return f"[error] no such file: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            body = fh.read()
        if old_str is None:
            return "[error] old_str is required for str_replace"
        n = body.count(old_str)
        if n == 0:
            return "[error] old_str not found (whitespace must match exactly)"
        if n > 1:
            return f"[error] old_str appears {n} times — make it unique"
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(body.replace(old_str, new_str or "", 1))
        return f"edited {p}"

    if command == "insert":
        if not os.path.exists(p):
            return f"[error] no such file: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        at = len(lines) if insert_line is None else max(0, min(int(insert_line), len(lines)))
        lines[at:at] = (new_str or "").splitlines()
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return f"inserted at line {at} of {p}"

    return f"[error] unknown command {command!r}; use view|create|str_replace|insert"


def _minimal_tools(policy: AutonomyPolicy | None = None) -> dict[str, ToolSpec]:
    """The control group's two tools.

    Under the default preset `may_run_arbitrary_code` is on and bash is the
    plain ungated shell the comparison against `standard` depends on. When the
    caller attaches a policy that denies arbitrary code, bash refuses instead
    of silently staying wide open — that is the one execution freedom with a
    real gate (see policy.PARTIAL).
    """
    def _gated_bash(command: str, timeout: int = 60) -> str:
        if policy is not None and not policy.may_run_arbitrary_code:
            return "Denied by autonomy policy: may_run_arbitrary_code is off."
        return _bash(command, timeout=timeout)

    return {
        "bash": ToolSpec(
            name="bash",
            description=(
                "Run a shell command in the workspace and return its combined "
                "output. Use it to inspect the project and to verify your edits."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "the command to run"},
                    "timeout": {"type": "integer", "description": "seconds (default 60)"},
                },
                "required": ["command"],
            },
            fn=_gated_bash,
            source="builtin",
            tags=["shell"],
            effect_signature="system",   # shell: reads, writes, spawns, networks
            state=ToolState.ACTIVE,
            cost_hint="moderate",
        ),
        "str_replace_editor": ToolSpec(
            name="str_replace_editor",
            description=(
                "View, create, or edit a file. command=view shows a file with "
                "line numbers; create writes a new file; str_replace swaps a "
                "unique snippet; insert adds lines after a given line."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string",
                                "enum": ["view", "create", "str_replace", "insert"]},
                    "path": {"type": "string", "description": "absolute file path"},
                    "file_text": {"type": "string", "description": "for create"},
                    "old_str": {"type": "string", "description": "for str_replace"},
                    "new_str": {"type": "string", "description": "for str_replace/insert"},
                    "insert_line": {"type": "integer", "description": "for insert"},
                    "view_range": {"type": "array", "items": {"type": "integer"},
                                   "description": "for view: [start, end]"},
                },
                "required": ["command", "path"],
            },
            fn=_editor,
            source="builtin",
            tags=["files"],
            effect_signature="local_write",
            state=ToolState.ACTIVE,
            cost_hint="cheap",
        ),
    }


# --------------------------------------------------------------------------
# the mode
# --------------------------------------------------------------------------


@dataclass
class MinimalAgent:
    """Two tools, one loop. The control group for 'does scaffolding help?'."""

    llm: LLMClient
    system_prompt: str = MINIMAL_SYSTEM
    max_turns: int | None = None
    cwd: str | None = None
    policy: AutonomyPolicy | None = None   # None = no gate, the old behaviour
    confirmer: Any = None                # asked before a gated tool runs
    steer: Any = None                    # the operator's channel into a live run
    compactor: Any = None                # None = build the default (compaction.py)
    trace: list[dict[str, Any]] = field(default_factory=list)
    pipeline: Any = None                 # cannot forge — the harness checks this
    store: Any = None
    registry: ToolRegistry = field(init=False)

    def __post_init__(self) -> None:
        # `bash` declares scope `system`, so a policy that switches off network
        # or read/write reaches it; `str_replace_editor` is a filesystem write.
        self.registry = ToolRegistry(policy=self.policy, confirmer=self.confirmer)
        for name, spec in _minimal_tools(self.policy).items():
            if self.cwd:
                # Pin both tools to the mode's workspace instead of whatever
                # directory the server happened to be launched from. Binding
                # both (not just bash) is what keeps them consistent.
                #
                # bash only: when the policy denies arbitrary code, leaving the
                # gated wrapper in place is correct — it refuses regardless of
                # cwd, so there is nothing to rebind. Rebinding to raw _bash
                # here would silently delete the gate.
                if name != "bash":
                    spec.fn = partial(_editor, cwd=self.cwd)
                elif self.policy is None or self.policy.may_run_arbitrary_code:
                    spec.fn = partial(_bash, cwd=self.cwd)
            self.registry.register(spec)

        # The minimal mode is the control group for "does scaffolding help?", so
        # it gets the same context compaction as the full agent. Leaving it out
        # would make the comparison a comparison of scaffolding *and* memory.
        if self.compactor is None:
            self.compactor = Compactor(
                summarizer=LLMSummarizer(
                    self.llm, should_abort=self._operator_should_yield),
                fallback=DeterministicSummarizer(),
                log_path=default_log_path(),
            )

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        self.trace.append({"kind": kind, **payload})

    def _operator_should_yield(self) -> bool:
        """Whether the summarizer should give way so the operator hears back.

        The question the compactor is handed. It used to be
        `_operator_wants_the_floor`, which now means only "a real /stop" --
        right for a job that must not be destroyed, wrong for a summarizer,
        which is a *cheap* step: dropping it costs a summary that can be taken
        again, and aborting it is how a question gets answered mid-run rather
        than after the whole turn.

        Asked here rather than on an Agent because the compactor is built in
        `__post_init__`, before the Agent it belongs to exists.
        """
        if self.steer is None:
            return False
        try:
            return bool(self.steer.has_pending()) or bool(self.steer.stop_requested())
        except Exception:                     # noqa: BLE001 - reads as "no"
            return False

    def run(self, task: str, history: list | None = None,
            progress: Callable[[str, dict[str, Any]], None] | None = None) -> AgentResult:
        """Run one task.

        `progress(kind, payload)` is called as the loop moves — `request` before
        each model call, `call`/`result` around each tool. The CLI passes it so
        a slow model reads as "waiting", not "hung".

        This mode wraps a plain `Agent`, which takes one callback per event
        rather than a single `progress`, so the translation lives here. Without
        it `auto run --mode minimal` raised `TypeError`: the CLI hands every
        mode the same `progress=`, and only the full agent declared it.
        """
        def _emit(kind: str, **payload: Any) -> None:
            if progress:
                progress(kind, payload)

        agent = Agent(
            self.llm, self.registry,
            system_prompt=self.system_prompt,
            max_turns=self.max_turns,
            allow_self_terminate=True,
            on_request=lambda turn: _emit("request", turn=turn),
            on_tool_call=lambda n, a: (
                self._record("call", {"tool": n, "args": a}), _emit("call", tool=n)),
            on_tool_result=lambda n, r: (
                self._record("result", {"tool": n, "ok": getattr(r, "ok", None)}),
                _emit("result", tool=n)),
            on_steer=lambda text: self._record("steer", {"text": text[:300]}),
            steer=self.steer,
            compactor=self.compactor,
            on_compact=lambda e: self._record(
                "compact", e.as_dict() if hasattr(e, "as_dict")
                else {"event": repr(e)}),
        )
        result = agent.run(task, history)
        self._record("finish", {"turns": result.turns, "tools": result.tool_calls,
                                "self_terminated": result.self_terminated,
                                "stopped_by_operator": result.stopped_by_operator})
        return result

    def report(self) -> dict[str, Any]:
        return {
            "mode": "minimal",
            "tools": self.registry.report(),
            "trace_len": len(self.trace),
        }

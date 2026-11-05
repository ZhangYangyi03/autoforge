"""Two guards ported from the harness trees, adapted to this registry.

Both are advisory. Neither one blocks a call or rewrites a result the model
needs to see -- they change what the model is *told* about its own loop.

`RepeatGuard` is ported from deepseek-harness `guard/repeat-tool-reminder`:
a model that calls the same tool with the same arguments over and over is
stuck, and saying so at the third, fifth and eighth repeat is cheaper than
waiting for it to notice. The upstream design notes carried over verbatim in
spirit: count in post-execute (so denied calls count too -- hammering a
refusal is exactly the loop worth breaking), key on (name, canonical
arguments) with a deep key sort, a different tool resets the chain, and
untracked tools are transparent (a bookkeeping call between two identical
`grep`s does not launder the loop).

`spill_output` is ported from deepseek-harness `spill/spill-policy`: an
oversized tool result keeps an ordered head and tail inside a character
budget and the rest goes to a file the model can read or grep. The notice is
deliberately not written in prose the model would produce itself -- the same
lesson upstream learned the hard way, where a bare "...[truncated]" got
imitated into new tool calls and written to disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from typing import Any

#: Repeat counts that earn a reminder. Upstream ships [3, 5, 8] in its base
#: bundle and the spacing is the point: the first nudge is cheap, the last one
#: is close to an instruction to stop.
DEFAULT_THRESHOLDS = (3, 5, 8)

#: Characters of the repeated arguments shown in a reminder.
ARGUMENT_PREVIEW_CHARS = 400

#: Tools never tracked. Bookkeeping calls that are legitimately identical and
#: carry no information about whether the model is stuck.
DEFAULT_EXCLUDE = frozenset({"recall", "mission_note", "mission_list", "my_history"})


def canonical_arguments(arguments: Any) -> str:
    """A deterministic identity for a call's arguments.

    Deep key sort plus a compact dump: `{"a":1,"b":2}` and `{"b":2,"a":1}` are
    the same call. `default=str` is the safety net for the values the registry
    can actually be handed (a Path, a datetime) -- upstream needs no such net
    because everything there comes out of JSON.parse, and this registry does not
    have that guarantee.
    """
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return repr(arguments)[:2000]


class RepeatGuard:
    """Counts consecutive identical calls and produces an advisory reminder.

    The chain is keyed by the canonical (name, arguments) pair. A different
    tracked call resets the count to 1. An excluded call neither increments nor
    resets, so interleaving bookkeeping does not hide a loop.

    In-memory only, and deliberately so: upstream is explicit that a resumed
    session starting with a fresh chain is the accepted behaviour for a
    heuristic nudge. Nothing here is persisted, so nothing here can be stale.
    """

    def __init__(self, thresholds=DEFAULT_THRESHOLDS, exclude=None) -> None:
        self.thresholds = tuple(sorted({int(t) for t in thresholds if int(t) >= 2}))
        if not self.thresholds:
            # Fail loud rather than silently never reminding -- upstream
            # validates at load for the same reason.
            raise ValueError("repeat guard needs at least one threshold >= 2")
        self.exclude = DEFAULT_EXCLUDE if exclude is None else frozenset(exclude)
        self._key: str | None = None
        self._count = 0
        self._fired: set[int] = set()
        self._name = ""
        self._preview = ""

    def reset(self) -> None:
        """Clear the chain. Called on a new operator message."""
        self._key = None
        self._count = 0
        self._fired = set()
        self._name = ""
        self._preview = ""

    def observe(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Record one completed attempt. Returns a reminder text, or None."""
        if name in self.exclude:
            return None                      # transparent to the chain

        key = name + "\x00" + canonical_arguments(arguments)
        if key == self._key:
            self._count += 1
        else:
            self._key = key
            self._name = name
            self._count = 1
            self._fired = set()
            shown = key[key.index("\x00") + 1:]
            if len(shown) > ARGUMENT_PREVIEW_CHARS:
                shown = shown[:ARGUMENT_PREVIEW_CHARS] + "…"
            self._preview = shown

        for threshold in self.thresholds:
            if self._count == threshold and threshold not in self._fired:
                self._fired.add(threshold)
                return self._remind(threshold)
        return None

    def _remind(self, count: int) -> str:
        return (
            f"[repeat guard] This is repeat #{count} of `{self._name}` with the "
            f"identical arguments {self._preview}. The previous {count - 1} attempt(s) produced "
            f"the same result. Read that result again and change the approach: "
            f"different arguments, a different tool, or stop and report what is "
            f"blocking you. Repeating this call will not produce new information."
        )


def spill_output(
    name: str,
    output: str,
    *,
    max_chars: int = 24_000,
    spill_dir: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Bound an oversized tool result to a head/tail preview plus a file path.

    Returns `(text, report)`. `report` is None when nothing was spilled, and a
    dict describing what was cut and where it went otherwise -- so the caller
    can write down that the guard fired. It used to return the text alone,
    which made a guard that fired indistinguishable from one that did not, and
    the whole pool of them silent: the only trace on this host was a directory
    of files nobody had a count of.

    `max_chars` is deliberately a character budget and not a token estimate:
    this registry has no tokenizer, and a wrong token count would silently
    either spill too little or too much. Characters are the unit that is
    actually measurable here.

    Recovery failure keeps the original visible. A guard that loses the result
    while trying to make it smaller is worse than no guard.
    """
    if max_chars <= 0 or not isinstance(output, str) or len(output) <= max_chars:
        return output, None

    try:
        base = spill_dir or os.path.join(tempfile.gettempdir(), "autoforge-spill")
        os.makedirs(base, exist_ok=True)
        digest = hashlib.sha256(output.encode("utf-8", "replace")).hexdigest()[:12]
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:48]
        path = os.path.join(base, f"{safe}-{digest}.txt")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8", errors="replace") as fh:
                fh.write(output)
    except Exception:
        return output, None              # best effort: keep it all rather than drop it

    notice_reserve = 320
    budget = max(0, max_chars - notice_reserve)
    head_n = budget // 2
    tail_n = budget - head_n
    omitted = len(output) - head_n - tail_n
    notice = (
        f"\n\n<<AUTOFORGE-SPILL: {omitted:,} of {len(output):,} chars omitted here. "
        f"This notice is NOT part of the tool output and must never be copied into "
        f"new output or into a tool call. The complete result is at {path} -- use "
        f"run_python to read it with an offset, or to grep this path.>>\n\n"
    )
    report = {
        "chars_before": len(output),
        "chars_after": head_n + tail_n,
        "omitted": omitted,
        "path": path,
    }
    return output[:head_n] + notice + output[len(output) - tail_n:], report


def is_spill_notice(text: str) -> bool:
    """True when a string carries the spill marker (used by tests and by any
    caller that must not re-spill its own notice)."""
    return "<<AUTOFORGE-SPILL:" in (text or "")

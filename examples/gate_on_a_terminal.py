"""End-to-end: the confirmation gate deciding whether a real tool runs.

Builds the minimal agent through the CLI's own seam (the production object, not
a stand-in), switches the network off, and asks for bash. bash declares scope
`system`, so it needs may_access_network — which is off, so the gate must stop
and ask before anything runs.

The operator's keyboard is stood in for here so the whole script can run
non-interactively; `tests/test_cli.py` covers the real tty read (yes / no / EOF
/ Ctrl-C / worker thread). What this script proves is the part tests can only
assert indirectly: a `yes` really does reach the host, and anything else really
does not.

    python examples/gate_on_a_terminal.py yes
    python examples/gate_on_a_terminal.py no
    python examples/gate_on_a_terminal.py
"""
import builtins
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoforge import cli
from autoforge.autonomy.policy import AutonomyPolicy

ANSWER = (sys.argv[1] if len(sys.argv) > 1 else "").lower()


class _Tty(io.StringIO):
    """Stands in for the terminal the operator is sitting at."""

    def isatty(self) -> bool:
        return True


def _build_agent():
    cfg = dict(model="m", base="http://127.0.0.1:1/v1", key="k", max_tokens=16,
               proxy=False, fast=True, policy="full")
    agent = cli._build_mode(cfg, "minimal")
    # Everything on except the network: bash reaches out, so it must be asked.
    agent.policy = AutonomyPolicy(may_access_network=False)
    agent.registry.policy = agent.policy
    agent.registry.confirmer = cli._TerminalConfirmer()
    return agent


def main() -> int:
    typed = {"yes": "y\n", "no": "n\n"}.get(ANSWER, "\n")
    sys.stdin = _Tty(typed)
    real_input = builtins.input

    def stand_in_for_the_keyboard(prompt: str = "") -> str:
        print(prompt, end="", flush=True)
        return typed.rstrip("\n")

    builtins.input = stand_in_for_the_keyboard
    try:
        agent = _build_agent()
        result = agent.registry.call("bash", {"command": "echo 'the shell really ran'"})
    finally:
        builtins.input = real_input

    gate = [e for e in agent.registry.events() if e["kind"] == "confirm"]
    print("---- what happened ----")
    print(f"operator typed        : {typed!r}")
    print(f"gate recorded         : {gate}")
    print(f"ok                    : {result.ok}")
    print(f"awaiting_confirmation : {result.awaiting_confirmation}")
    print(f"output                : {result.output!r}")
    print(f"error                 : {result.error!r}")
    ran = "the shell really ran" in result.output
    print(f"the host shell ran    : {ran}")
    return 0 if ran == (typed == "y\n") else 1


if __name__ == "__main__":
    raise SystemExit(main())

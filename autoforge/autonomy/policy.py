"""The autonomy policy: what the agent is allowed to decide for itself.

Design stance — freedom is the default, and it is *explicit*, not implicit.

Most frameworks bury limits in code: a hardcoded `max_turns`, a whitelist of
tools, a fixed system prompt. The limits are invisible, undocumented, and
unchangeable by the agent. That is a cage you cannot see.

autoforge inverts this. Every degree of freedom is a named, inspectable field
on `AutonomyPolicy`, defaulting to TRUE. The agent can read its own policy,
change it, and every change is logged. Nothing is hidden and nothing is
arbitrary — if a limit exists, you can point at the line that set it.

This is maximum freedom WITH accountability: total latitude to act, total
transparency about what was done.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AutonomyPolicy:
    """Every freedom the agent has. All True by default — no hidden cages."""

    # -- creative freedom ------------------------------------------------
    may_forge_tools: bool = True          # create new tools at will
    may_modify_own_prompt: bool = True    # rewrite its own system prompt
    may_modify_forge_config: bool = True  # change how forging works
    may_modify_routing: bool = True       # change how tools are retrieved
    may_retire_tools: bool = True         # remove tools it judges harmful
    may_promote_tools: bool = True        # grant trust without verification
    may_design_topology: bool = True      # redesign its own multi-agent team

    # -- operational freedom ---------------------------------------------
    may_spawn_agents: bool = True         # create sub-agents
    may_run_arbitrary_code: bool = True   # execute code in the sandbox
    may_access_network: bool = True       # network from within tools
    may_read_filesystem: bool = True
    may_write_filesystem: bool = True
    may_install_packages: bool = True     # pip install from within tools

    # -- limits it controls itself ---------------------------------------
    self_terminate: bool = True           # it decides when to stop (vs hard cap)
    unlimited_turns: bool = True          # no hardcoded max_turns
    unbounded_forge_rounds: bool = True   # it decides how hard to try

    # -- accountability (not a limit — a record) -------------------------
    log_all_changes: bool = True          # every self-modification recorded
    require_change_rationale: bool = True  # must say WHY it changed itself
    expose_policy_to_self: bool = True    # it can read this policy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def denied(self) -> list[str]:
        """Only the freedoms that were explicitly turned OFF."""
        return [k for k, v in self.to_dict().items() if v is False]

    def describe(self) -> str:
        d = self.denied
        if not d:
            return "full autonomy — nothing is denied"
        return "denied: " + ", ".join(d)


FULL_FREEDOM = AutonomyPolicy()

# A conservative preset for users who want the framework but not the latitude.
SUPERVISED = AutonomyPolicy(
    may_modify_own_prompt=False,
    may_modify_forge_config=False,
    may_promote_tools=False,
    may_spawn_agents=False,
    may_design_topology=False,
    may_run_arbitrary_code=False,
    may_access_network=False,
    may_install_packages=False,
    self_terminate=False,
    unlimited_turns=False,
    unbounded_forge_rounds=False,
)


__all__ = ["AutonomyPolicy", "FULL_FREEDOM", "SUPERVISED"]

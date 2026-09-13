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
from typing import Any, ClassVar

# ---------------------------------------------------------------------------
# Enforcement ledger — which fields actually bite.
#
# A field that is declared but never consulted is worse than no field at all:
# it reads as a safety net and behaves as decoration. Rather than let the
# dataclass imply more than the code delivers, every field is classified here
# and the classification is *printed* (describe(), `auto config`, /report,
# and the agent's own `my_capabilities` tool).
#
# ENFORCED      — some code path consults it and can refuse. Turning it off
#                 changes behaviour in a way you can watch.
# PARTIAL       — enforced on some paths but not all; the note says which.
# DECLARED_ONLY — never consulted anywhere. Turning it off does nothing.
#                 Keep it for the roadmap, but never mistake it for a gate.
# ---------------------------------------------------------------------------

EXECUTION_FREEDOMS = (
    "may_run_arbitrary_code",
    "may_access_network",
    "may_read_filesystem",
    "may_write_filesystem",
    "may_install_packages",
    "may_run_cuda_kernels",
)

ENFORCED: frozenset[str] = frozenset({
    "may_forge_tools",
    "may_modify_own_prompt",
    "may_modify_forge_config",
    "may_modify_routing",
    "may_promote_tools",
    "may_retire_tools",
    "may_design_topology",
    "may_spawn_agents",
    "unlimited_turns",
    "unbounded_forge_rounds",
    "self_terminate",
    "log_all_changes",
    "require_change_rationale",
    "expose_policy_to_self",
    "may_run_cuda_kernels",
})

PARTIAL: dict[str, str] = {
    "may_run_arbitrary_code": (
        "gated on the minimal-mode bash tool; forged tools still execute in the "
        "sandbox, which does not consult policy"
    ),
}

# The four filesystem/network/install freedoms are enforced nowhere: the
# sandbox deliberately bounds blast radius rather than capping capability
# (DESIGN.md §2.5). Listed explicitly so the gap is a decision, not an
# oversight.
DECLARED_ONLY: frozenset[str] = frozenset({
    "may_read_filesystem",
    "may_write_filesystem",
    "may_access_network",
    "may_install_packages",
})


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
    # Compile and launch CUDA kernels written by the agent. A real gate, not a
    # posture: with it off, `gpu_compile`/`gpu_launch`/`gpu_bench` refuse and
    # say so, while the read-only half of the GPU layer (probe, occupancy
    # estimate, the units lint, torch baselines) stays available. That split is
    # deliberate — inspecting a machine is not the same act as running
    # agent-authored code on its GPU, and only the second one can hang a
    # display.
    may_run_cuda_kernels: bool = True

    # -- limits it controls itself ---------------------------------------
    self_terminate: bool = True           # it decides when to stop (vs hard cap)
    unlimited_turns: bool = True          # no hardcoded max_turns
    unbounded_forge_rounds: bool = True   # it decides how hard to try

    # -- accountability (not a limit — a record) -------------------------
    log_all_changes: bool = True          # every self-modification recorded
    require_change_rationale: bool = True  # must say WHY it changed itself
    expose_policy_to_self: bool = True    # it can read this policy

    # The enforcement ledger, reachable from the class as well as the module.
    ENFORCED: ClassVar[frozenset[str]] = ENFORCED
    DECLARED_ONLY: ClassVar[frozenset[str]] = DECLARED_ONLY
    PARTIAL: ClassVar[dict[str, str]] = PARTIAL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def denied(self) -> list[str]:
        """Only the freedoms that were explicitly turned OFF."""
        return [k for k, v in self.to_dict().items() if v is False]

    @property
    def unenforced(self) -> list[str]:
        """Fields the agent has switched off that no code path can honour.

        Empty is the healthy answer: it means every 'off' in this policy is a
        gate you can watch close. Non-empty means part of this policy is a
        promise the implementation does not keep yet.
        """
        return [f for f in self.denied if f in DECLARED_ONLY or f in PARTIAL]

    def describe(self) -> str:
        d = self.denied
        head = "full autonomy — nothing is denied" if not d else "denied: " + ", ".join(d)
        inert = [f for f in d if f in DECLARED_ONLY]
        soft = [f for f in d if f in PARTIAL]
        if inert:
            head += f"  [not enforced: {', '.join(inert)}]"
        if soft:
            head += f"  [partially enforced: {', '.join(soft)}]"
        return head

    def enforcement_table(self) -> list[dict[str, Any]]:
        """Per-field: is this freedom on, and does switching it off do anything?"""
        rows: list[dict[str, Any]] = []
        for name, enabled in self.to_dict().items():
            if name in ENFORCED:
                level = "enforced"
            elif name in PARTIAL:
                level = "partial"
            elif name in DECLARED_ONLY:
                level = "declared-only"
            else:                                    # pragma: no cover - guard
                level = "unknown"
            rows.append({
                "freedom": name,
                "enabled": enabled,
                "enforced": level,
                "note": PARTIAL.get(name, ""),
            })
        return rows


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
    may_run_cuda_kernels=False,
    self_terminate=False,
    unlimited_turns=False,
    unbounded_forge_rounds=False,
)


__all__ = [
    "AutonomyPolicy",
    "FULL_FREEDOM",
    "SUPERVISED",
    "ENFORCED",
    "DECLARED_ONLY",
    "PARTIAL",
    "EXECUTION_FREEDOMS",
]

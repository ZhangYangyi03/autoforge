"""autoforge — an agent framework whose tools are data, whose self-made tools
must earn the right to be called, and whose autonomy has no hidden cages.

Design pillars
--------------
1. Tools are data, not code locks (hot-loadable, no restart).
2. A tool is born with a *contract*: schema + trigger probes + effect signature.
3. Verification runs five checks: execution, robustness (fuzz), adversarial
   (an LLM attacker), trigger, and negative.
4. The living half: reliability ledger, auto-quarantine, rehabilitation.
5. Evolution: a failing tool spawns a population of mutants; the best survives.
6. Meta-cognition: the agent proactively discovers and fills its own gaps.
7. Autonomy is explicit, default-on, and every self-change leaves a paper trail.

Freedom is the default; verification lets you *trust* what the freedom produced.
"""

import os

# No console window for any child this *process* starts.
#
# The agent is started by a scheduled task through pythonw, which owns no
# console. Windows gives a console-subsystem child of a console-less parent a
# *fresh* console -- a visible window on the operator's desktop -- so every
# subprocess that starts without a flag flashes one. Measured on 2026-09-16 by
# enumerating visible top-level windows by pid: flags=0 yields one visible
# PseudoConsoleWindow, DETACHED_PROCESS none. `GetConsoleWindow()` is the wrong
# probe for this: it reports non-zero even under DETACHED_PROCESS, where no
# window is shown.
#
# Done here, at import, rather than at each call site: the spawns live in
# webtools (the search worker), market, schedule, cpu/safety and mcp, and a rule
# that has to be remembered at five places is a rule that will be missing from
# the sixth. Nothing in this package does `from subprocess import Popen`, so the
# class is always looked up on the module and one assignment covers all of them.
#
# AUTOFORGE_KEEP_CONSOLE=1 opts out, for the case where a visible window is what
# is wanted -- debugging a child that dies before it can log.
if os.name == "nt" and not os.environ.get("AUTOFORGE_KEEP_CONSOLE"):
    import subprocess as _subprocess

    #: DETACHED_PROCESS: no console is created at all. CREATE_NO_WINDOW is the
    #: other candidate and was rejected here -- it suppresses the window but
    #: still creates a console, which lands in this process's job object and
    #: changes `processes_launched` for the containment accounting.
    _NO_WINDOW = 0x00000008

    class _QuietPopen(_subprocess.Popen):
        """subprocess.Popen, but the child never gets a console window."""

        def __init__(self, *args, **kwargs):
            kwargs["creationflags"] = int(kwargs.get("creationflags") or 0) | _NO_WINDOW
            super().__init__(*args, **kwargs)

    if _subprocess.Popen.__name__ != "_QuietPopen":
        _subprocess.Popen = _QuietPopen
        globals()["_QuietPopen"] = _QuietPopen


# One source of truth: pyproject.toml. Hardcoding the version here meant the
# package reported 0.2.0 while pyproject said 0.4.0 -- a number nobody would
# notice was wrong until they quoted it. Read it from the installed metadata
# instead, and fall back only when running from a source tree with no install.
try:
    from importlib.metadata import version as _pkg_version

    # The distribution is `autoforge-agent`; the import stays `autoforge`. The
    # bare name on the index is a different project, so asking for it here would
    # not fail loudly -- it would report "0.0.0+unknown" on an installed build.
    __version__ = _pkg_version("autoforge-agent")
except Exception:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0+unknown"

from autoforge.agent import AUTONOMOUS_SYSTEM, ForgeAgent
from autoforge.autonomy.policy import FULL_FREEDOM, SUPERVISED, AutonomyPolicy
from autoforge.autonomy.selfmod import Amendment, SelfModifier
from autoforge.autonomy.spawn import ShareMode, Spawner, SpawnRecord
from autoforge.core.agent import Agent, AgentResult
from autoforge.core.llm import LLMClient, LLMResponse, MockLLMClient, OpenAICompatClient
from autoforge.core.message import Message, ToolCall
from autoforge.forge.adversary import AdversarialGate, AdversarialReport
from autoforge.forge.evolution import EvolutionEngine, EvolutionResult, Mutant
from autoforge.forge.fuzzer import RobustnessResult, run_robustness_checks
from autoforge.forge.generator import (
    GeneratedTool,
    LLMToolGenerator,
    TemplateGenerator,
    extract_json,
)
from autoforge.forge.metacog import GapProposal, MetaCognition, MetaCogReport
from autoforge.forge.pipeline import ForgeAttempt, ForgeConfig, ForgePipeline, ForgeResult
from autoforge.forge.sandbox import Sandbox, SandboxResult
from autoforge.forge.verifier import CheckResult, ToolVerifier, VerificationReport
from autoforge.route.router import BehaviourRouter, RoutingWeights
from autoforge.store import ToolRecord, ToolStore
from autoforge.tools.composition import DepGraph, compose_code, parse_deps
from autoforge.tools.registry import ToolRegistry, ToolResult
from autoforge.tools.spec import ToolSpec, ToolState, ToolStats, TriggerProbe

__all__ = [
    "__version__",
    # composed agent
    "ForgeAgent", "AUTONOMOUS_SYSTEM",
    # core
    "Agent", "AgentResult",
    "LLMClient", "LLMResponse", "OpenAICompatClient", "MockLLMClient",
    "Message", "ToolCall",
    # tools
    "ToolRegistry", "ToolResult",
    "ToolSpec", "ToolState", "ToolStats", "TriggerProbe",
    "DepGraph", "compose_code", "parse_deps",
    # forge
    "ForgePipeline", "ForgeConfig", "ForgeResult", "ForgeAttempt",
    "Sandbox", "SandboxResult",
    "ToolVerifier", "VerificationReport", "CheckResult",
    "GeneratedTool", "LLMToolGenerator", "TemplateGenerator", "extract_json",
    "EvolutionEngine", "EvolutionResult", "Mutant",
    "AdversarialGate", "AdversarialReport",
    "MetaCognition", "MetaCogReport", "GapProposal",
    "RobustnessResult", "run_robustness_checks",
    # routing
    "BehaviourRouter", "RoutingWeights",
    # autonomy
    "AutonomyPolicy", "FULL_FREEDOM", "SUPERVISED",
    "SelfModifier", "Amendment",
    "Spawner", "SpawnRecord", "ShareMode",
    # persistence
    "ToolStore", "ToolRecord",
]

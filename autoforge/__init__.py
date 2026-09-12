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

# One source of truth: pyproject.toml. Hardcoding the version here meant the
# package reported 0.2.0 while pyproject said 0.4.0 -- a number nobody would
# notice was wrong until they quoted it. Read it from the installed metadata
# instead, and fall back only when running from a source tree with no install.
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("autoforge")
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

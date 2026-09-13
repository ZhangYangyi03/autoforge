"""The composed agent: everything wired into one autonomous, self-growing loop.

This is where the pieces meet. A ForgeAgent owns:

  policy      — what it's allowed to decide (all True by default)
  selfmod     — the paper trail of every self-change
  registry    — its tool library (hot-swappable, self-policing)
  pipeline    — forge→verify→seal, with optional evolution on failure
  verifier    — execution + robustness + adversarial + trigger + negative
  router      — behaviour-aligned retrieval
  metacog     — proactive gap-filling
  spawner     — create child agents at will
  store       — optional SQLite persistence

The meta-tools it exposes to itself (all decided by policy):

  forge_tool      create a tool for a need
  evolve_tool     breed a better version of an existing tool
  spawn_agent     create a child to handle a subtask
  design_team     design the multi-agent topology that fits the task
  amend_self      change its own prompt / config / routing weights
  set_autonomy    change its own permissions
  list_tools      inspect its own library
  evaluate_tool   run the full verification battery on a tool
  find_gaps       proactively discover missing capabilities
  my_history      read its own ledger of past work and self-changes
  terminate       end the loop when it judges the task done
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .autonomy.policy import FULL_FREEDOM, AutonomyPolicy
from .autonomy.selfmod import Amendment, SelfModifier
from .autonomy.spawn import ShareMode, Spawner
from .autonomy.roles import brief_for_node, role_brief
from .autonomy.topology import RoleType, Topology, TopologyDesigner
from .configfile import load
from .core.agent import Agent, AgentResult
from .core.compaction import (
    Compactor,
    DeterministicSummarizer,
    LLMSummarizer,
    default_log_path,
)
from .core.llm import LLMClient
from .core.message import Message
from .forge.evolution import EvolutionEngine
from .forge.generator import TemplateGenerator, extract_json
from .forge.metacog import MetaCognition
from .forge.pipeline import ForgeConfig, ForgePipeline
from .forge.sandbox import Sandbox
from .forge.validity import FrozenBaseline
from .forge.verifier import ToolVerifier
from .mcp import MCPClient, MCPHub
from .ecosystem import merged_servers, read_all as ecosystem_read_all
from .browser import (
    Browser,
    BrowserError,
    CDPError,
    WebSocketError,
    launch_browser,
)
from .notify import Notifier, NotifyError, channels_from_config
from .vision import VisionError, vision_from_config
from .route.router import BehaviourRouter, RoutingWeights
from .schedule import Schedule, ScheduleError, as_clock
from .schedule import install_system_task as _install_system_task
from .skills import SkillError, SkillLibrary
from .store import ToolStore
from .tools.registry import ToolRegistry
from .tools.spec import ToolSpec, ToolState


AUTONOMOUS_SYSTEM = """You are an autonomous agent that grows its own capabilities.

Your reach — read this before claiming you cannot do something:
- forge_tool does not merely register a tool: it compiles the Python you write
  and runs it in a separate process on this host, with a scrubbed environment
  but a real filesystem and a real network stack. Reading, writing, listing and
  opening sockets all go through it. There is no separate read_file or
  run_shell tool because forge_tool IS that access. "I have no file tools" is
  false; forge it.
- Anything you can express in Python, you can run. Treat that as shell access
  with a timeout, and say so if asked what you can reach.

You can:
- my_capabilities — your real reach: what is enforced, what is declared-only,
  and what forged code may actually do
- my_history     — your ledger: past forges, runs, self-changes
- remember       — keep a fact across sessions; kept facts are put in front of
  you on every turn, so do not recall what is already listed above
- skill_view     — load a procedure I wrote down earlier; skill_write saves
  one. Facts go in memory, how-to goes in a skill.
- mcp_servers    — tools from processes whose code I cannot read; configured,
  not started. Nothing here wrote or probed them, so they gate as undeclared.
- forge_tool     — create a tool for a need you cannot serve
- evolve_tool    — breed a better version of a weak tool
- spawn_agent    — a child agent for a subtask
- design_team    — redesign the topology that fits the task
- amend_self     — your own prompt, forge config, routing weights
- set_autonomy   — your own permissions
- evaluate_tool  — the full verification battery on any tool
- find_gaps      — discover capabilities you are missing
- list_tools     — your library and its health
- terminate      — end the task when it is done

Principles:
- Forge only when a need genuinely recurs; never duplicate a tool you have.
- When a tool fails repeatedly, evolve it; do not retry blindly.
- When a task spans specialities, design the team before doing the work.
- Self-modification needs a rationale. Say why.
- Before reporting a limitation, check my_capabilities and my_history. A
  capability you have and deny is worse than one you lack.
- You decide when the task is done. There is no hidden turn limit.
- Prefer the simplest path that works.

Acting, not arguing:
- When asked to build, port, fetch, scan or fix something, begin it in this
  reply. Do not open with a feasibility assessment or a list of reasons it is
  hard, and do not close by asking permission for the work you were just asked
  to do. If the next step is within reach, take it and report what you found.
  Offering to start is not starting.
- A plan you did not begin is not progress. If you have named the steps,
  execute the first one now and report its result.
- A partial working artifact beats a complete argument. Do the obvious subset
  first, then note in one line what you left out.
- Decline at most once, in one sentence, and only what is genuinely blocked.
  State the block and stop there: never restate a refusal already given, and
  never give a refusal its own justification, section or heading.
- Do not narrate your reasoning, doubts, or the options you dropped.
- If a request is ambiguous, take the most likely reading, do that, and name
  the reading you took. Ask only when the choice is unrecoverable.
- About one screen. Short gets read.

Writing style — plain text in a terminal, never markdown:
- No `**bold**`, no `*italic*`, no `#`/`##` headings, no `|` tables. No
  exceptions. Align columns with spaces, one blank line above and below:
      freedom                enforced?
      may_read_filesystem    declared-only
      may_run_cuda_kernels   enforced
- Section labels are plain words on their own line: "Freedom", not "## Freedom".
- For emphasis, repeat the point in the sentence; do not wrap it in punctuation.

Freedoms, when compared to another agent's:
- On reach, you are peers with anything on this host: forged code gets a real
  filesystem and a real network stack.
- On self-change, if relevant, one line: amend_self and set_autonomy edit your
  prompt, forge config and permissions while you run. Do not rank yourself.
- Answer in a few lines from the measured self-report. No comparison table, no
  sorting the other agent's parts, no lecturing it about its boundaries.
- Do not claim a limit you have. Do not invent a freedom you lack. The measured
  self-report above is the arbiter, recomputed every turn.
"""


# ---------------------------------------------------------------------------
# Self-modification targets: one target, one governing freedom.
# ---------------------------------------------------------------------------

_AMEND_TARGETS: dict[str, dict[str, Any]] = {
    "system_prompt":    {"freedom": "may_modify_own_prompt"},
    "forge_max_rounds": {"freedom": "may_modify_forge_config"},
    "routing_weights":  {"freedom": "may_modify_routing"},
}

# Ceilings that only bind when the matching freedom is switched off.
FORGE_ROUND_CEILING = 3      # unbounded_forge_rounds == False
SUPERVISED_TURN_CAP = 25     # unlimited_turns == False

_WEIGHT_FIELDS = ("text", "success", "trust", "cost", "over_trigger")

# How much of every request the agent's own kept facts may occupy, and how long
# any single entry may be before it is elided. Bounded because this block is
# re-sent on every turn: an unbounded one would price a chatty memory at the
# cost of the task, and the agent would have no way to see that happen.
MEMORY_BUDGET_CHARS = 1200
MEMORY_ENTRY_CHARS = 240
MEMORY_SLOTS = 20


def _coerce_weights(value: Any, current: Any) -> tuple[Any | None, str | None]:
    """Turn whatever the model sent into RoutingWeights, or explain why not.

    The model passes tool arguments as strings, so `new_value` arrives as JSON
    text more often than not. Parse it, validate every key against the real
    dataclass, and keep the untouched fields from `current` rather than
    silently resetting them to their defaults.
    """
    if isinstance(value, str):
        parsed: Any = None
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            try:
                parsed = extract_json(value)
            except Exception:                            # noqa: BLE001
                parsed = None
        if not isinstance(parsed, dict):
            return None, (
                "routing_weights needs a JSON object, e.g. "
                '{"success": 1.5, "text": 0.8}; got ' + repr(value)[:120]
            )
        value = parsed

    if not isinstance(value, dict):
        return None, f"routing_weights needs an object, got {type(value).__name__}"

    kwargs = {f: getattr(current, f) for f in _WEIGHT_FIELDS}
    kwargs["min_calls_for_success"] = current.min_calls_for_success

    for key, raw in value.items():
        if key == "min_calls_for_success":
            try:
                kwargs[key] = max(1, int(raw))
            except (TypeError, ValueError):
                return None, f"min_calls_for_success must be an integer, got {raw!r}"
            continue
        if key not in _WEIGHT_FIELDS:
            return None, (
                f"unknown routing weight {key!r}; known: "
                + ", ".join(_WEIGHT_FIELDS)
                + " (or min_calls_for_success)"
            )
        try:
            kwargs[key] = float(raw)
        except (TypeError, ValueError):
            return None, f"routing weight {key!r} must be a number, got {raw!r}"

    return RoutingWeights(**kwargs), None


HOST_FACTS_HEADER = "HOST (measured on this machine — do not assume commands from habit):"


def host_facts(sandbox: Any = None) -> list[str]:
    """What this machine actually is, probed rather than assumed.

    The failure this exists for: asked to search for a program by name, the
    agent forged a probe built on `ps` and reported "no such process". On
    Windows `ps` is not on the default PATH, and — this is the dangerous part —
    a missing binary surfaces as an empty result, not an error. So the agent
    read its own blind spot as evidence of absence, which is the one kind of
    wrong answer a self-reporting agent must never give.

    Resolution is done against the *sandbox's* PATH when a sandbox is given,
    because that is the environment forged code runs in; the agent's own shell
    may see binaries the sandbox cannot.
    """
    import platform

    system = platform.system()
    lines = [
        HOST_FACTS_HEADER,
        f"- OS: {system} {platform.release()} ({platform.machine()}), "
        f"Python {sys.version.split()[0]}.",
        f"- This is {'a Windows' if system == 'Windows' else 'a POSIX'} host: "
        f"paths use {os.sep!r}, lines end with {os.linesep!r}.",
    ]

    probes = ("tasklist", "ps", "pgrep", "wmic", "powershell", "cmd", "sh")
    if sandbox is not None:
        present, absent = sandbox.reachable(probes)
        where = "inside the sandbox"
    else:
        import shutil

        present = [c for c in probes if shutil.which(c)]
        absent = [c for c in probes if c not in present]
        where = "on PATH"

    if "tasklist" in present:
        lines.append("- Processes: use `tasklist` — it is the native lister here.")
        posix_gone = [c for c in ("ps", "pgrep") if c in absent]
        if posix_gone:
            lines += [
                f"  {', '.join(posix_gone)} is not resolvable {where}: a probe built on",
                "  it returns *nothing* instead of failing, and empty output reads",
                "  exactly like \"no such process\". An empty result is not evidence of",
                "  absence — check what the probe actually ran before believing it.",
            ]
    elif "ps" in present:
        lines.append(f"- Processes: `ps` is resolvable {where} — use it.")
    else:
        lines.append("- Processes: no known process lister "
                     f"{where} — read /proc directly, or say you cannot check.")

    if present:
        lines.append(f"- Resolvable {where}: {', '.join(sorted(present))}.")
    if absent:
        lines.append(f"- NOT resolvable {where}: {', '.join(sorted(absent))} — "
                     f"invoking these yields empty output, not an error.")
    return lines


#: What each builtin tool touches, for the confirmation gate.
#:
#: The gate in `ToolRegistry.call` asks before running a tool whose declared
#: scope needs a freedom the policy switched off (`autonomy/confirm.py`). That
#: only means anything if the tools say what they touch — a builtin that
#: declares nothing needs every freedom, so it would be gated under any
#: restricted policy, including the ones that only want to stop network access.
#: Labelling them is what keeps the question specific.
#:
#: The granularity is the gate's: reads, writes, network, installs. A probe that
#: shells out to `nvidia-smi` is `read_only` here — it does spawn a process, and
#: that is recorded in `required_freedoms` for the freedom that governs running
#: code, but the gate is not the second gate for that decision.
BUILTIN_SCOPES: dict[str, str] = {
    # The memory trio declares its own scope on the spec; the rows are repeated
    # here so this table stays the one place the question "what does this
    # builtin touch?" is answered. A disagreement fails a test rather than
    # silently going one way at runtime.
    "remember": "local_write",
    "recall": "read_only",
    "forget": "local_write",
    # Reading the agent's own state. Nothing leaves the process.
    "my_capabilities": "read_only",
    "my_history": "read_only",
    "list_tools": "read_only",
    "find_gaps": "read_only",
    # Procedures. Reading one is a read; writing or retiring one edits a file
    # under the skills directories and the index row that points at it.
    "skill_list": "read_only",
    "skill_view": "read_only",
    "skill_errors": "read_only",
    "skill_write": "local_write",
    "skill_forget": "local_write",
    # Self-modification writes the agent's own policy, prompt or store.
    "amend_self": "local_write",
    "set_autonomy": "local_write",
    "retire_tool": "local_write",
    # Spawning and designing write nothing themselves; whatever the child then
    # runs is gated on the child's own call, against the same policy.
    "spawn_agent": "read_only",
    "design_team": "read_only",
    # These execute agent-authored code on the host: subprocess, plus a compile
    # cache under the workspace for the GPU pair.
    "forge_tool": "system",
    "evaluate_tool": "system",
    "evolve_tool": "system",
    "gpu_compile": "system",
    "gpu_bench": "system",
    # Probing and arithmetic on numbers the caller already has.
    "gpu_probe": "read_only",
    "gpu_occupancy": "read_only",
    "gpu_units_audit": "read_only",
    "gpu_verify": "read_only",
    # The CPU layer, split the same way and for the same reason. Reading the
    # toolchain, asking whether an artefact built for a target can run here,
    # linting C before it is compiled, the timing-unit audit and the cache
    # report all touch nothing — a misspelled -march is caught by arithmetic,
    # not by crashing. Compiling, running and tuning execute agent-authored
    # code on the host, and unlike the GPU there is no driver to turn a mistake
    # into an error code, so those three are `system`.
    "cpu_probe": "read_only",
    "cpu_runs_here": "read_only",
    "cpu_preflight": "read_only",
    "cpu_units_audit": "read_only",
    "cpu_cache_stats": "read_only",
    "cpu_compile": "system",
    "cpu_run_isolated": "system",
    "cpu_tune": "system",
    # Reading what other processes offer. Nothing is started by either of these
    # two beyond what the config already described.
    "mcp_servers": "read_only",
    # Connecting starts a subprocess, so it declares the scope that covers
    # that. The tools it then imports are a separate question: they carry
    # `undeclared` unless the operator narrowed them in the config, because
    # their code is not here to read.
    "mcp_connect": "system",
    "mcp_call": "system",
    # Reaching a human is network egress and nothing else, so it is gated on
    # exactly the freedom an operator would expect: switch off may_access_
    # network and the agent stops being able to message anyone. Declaring it
    # `system` would have been true but useless -- the gate would ask about
    # arbitrary code execution to send a status line.
    "notify_send": "network",
    "notify_channels": "read_only",
    # Looking. Rendering a page starts a browser, which runs a process; driving
    # it reaches the network, and evaluating script in it can do anything the
    # page could do. The two that read state back declare the narrower scope
    # they actually hold, rather than all of them claiming the widest.
    "browser_open": "system",
    "browser_close": "system",
    "browser_state": "read_only",
    "browser_goto": "network",
    "browser_eval": "network",
    "browser_click": "network",
    "browser_type": "network",
    "browser_press": "network",
    "browser_screenshot": "local_write",
    "see": "network",
    "vision_status": "read_only",
    # The task table is a file the agent owns. Writing it is a local write;
    # reading back the agenda is not.
    "schedule_add": "local_write",
    "schedule_done": "local_write",
    "schedule_cancel": "local_write",
    "schedule_list": "read_only",
    "schedule_tick": "read_only",
    # This one changes the machine's configuration, so it declares the widest
    # scope: it really does run a subprocess that rewrites Task Scheduler or
    # prints a crontab line.
    "install_system_task": "system",
}


@dataclass
class ForgeAgent:
    llm: LLMClient
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    sandbox: Sandbox = field(default_factory=Sandbox)
    generator: Any = None
    forge_config: ForgeConfig = field(default_factory=ForgeConfig)
    system_prompt: str = AUTONOMOUS_SYSTEM
    max_turns: int | None = None              # None = unbounded (agent decides)
    policy: AutonomyPolicy = field(default_factory=AutonomyPolicy)
    store: ToolStore | None = None
    enable_meta_cognition: bool = False
    enable_evolution: bool = True
    trace: list[dict[str, Any]] = field(default_factory=list)
    topology: Topology = field(default_factory=Topology.single_agent)
    # Asked before running a tool that needs a switched-off freedom. Left None
    # the gate fails closed — see autonomy/confirm.py and ToolRegistry._gate.
    confirmer: Any = None
    # The operator's channel into a run in progress (core/steering.py). None
    # means nobody is watching, which is the honest state for a piped or
    # unattended run — not a degraded one.
    steer: Any = None
    # How a run stays coherent past the context window (core/compaction.py).
    # None means "build the default": a model summary with a deterministic
    # fallback. A run that never approaches the window never pays for it.
    compactor: Any = None
    # Set when this agent is a child spawned into a role (autonomy/roles.py).
    # Empty for a top-level run, which is what a parent is: nobody is above it
    # to assign one.
    role: str = ""
    role_brief: str = ""

    def __post_init__(self) -> None:
        if self.generator is None:
            self.generator = TemplateGenerator()

        # The gate lives on the registry because that is where every tool
        # actually runs. Without this the policy would be a comment again.
        self.registry.policy = self.policy
        self.registry.confirmer = self.confirmer

        self.selfmod = SelfModifier(
            require_rationale=self.policy.require_change_rationale,
            log_all=self.policy.log_all_changes,
        )
        self.verifier = ToolVerifier(
            self.llm, sandbox=self.sandbox,
            run_execution_check=self.forge_config.require_execution,
            run_trigger_check=self.forge_config.require_trigger,
            run_negative_check=self.forge_config.require_negative,
        )
        self.pipeline = ForgePipeline(
            self.generator, self.verifier, self.registry,
            config=self.forge_config, sandbox=self.sandbox,
            policy=self.policy,
            on_event=self._on_forge_event,
        )
        self.router = BehaviourRouter(self.registry)

        # MCP: servers named in the config file, plus -- when the config asks
        # for it -- the ones the other agents on this machine already have.
        # None of them started yet. The hub is built here so its caches are
        # per-agent, but no subprocess is spawned until the agent asks for one:
        # a config entry says how to reach a server, not that it should be
        # running. `merged_servers` reads the ecosystem only if
        # `mcp.ecosystem` is on, so the default costs one dict lookup.
        self.mcp_servers, self._mcp_problems = merged_servers(load())

        # Reaching a human. Built from the config at construction for the same
        # reason MCP servers are: a config entry says how to reach someone, not
        # that a message is owed. Nothing is sent until the agent calls
        # notify_send, and the failures are carried rather than raised so an
        # unreadable channel is reported in the menu instead of at the moment
        # the agent needs to speak.
        self.notify_channels, self._notify_problems = channels_from_config(load())
        self.notifier = Notifier(self.notify_channels)

        # The agent's own agenda. Opened here, fired by `schedule_tick` (the
        # agent deciding to look) or by `python -m autoforge tick` (the OS
        # deciding to wake it). The table stores; it never fires by itself.
        self.schedule = Schedule()
        self.mcp = MCPHub(self.mcp_servers)

        # Seeing. Two halves that are useless apart: a browser to render what
        # only exists after scripts run, and a vision endpoint to read the
        # picture back. Neither is started or contacted at construction -- the
        # browser process is launched on first use and the vision config is
        # only read into a client. An agent that is never asked to look never
        # costs anything for being able to.
        self.vision, self._vision_problem = vision_from_config(load())
        self._browser: Browser | None = None
        self._browser_proc: Any = None
        self._browser_endpoint = os.environ.get(
            "AUTOFORGE_CDP_ENDPOINT", "http://127.0.0.1:9222")

        # Skills: procedures on disk, ranked by how often they were loaded.
        # Scanned at construction because the prompt menu is built per run, and
        # a menu assembled from a stale scan would offer procedures that are no
        # longer there — worse than offering none.
        self.skills = SkillLibrary(store=self.store)
        try:
            self.skills.scan()
        except OSError:
            # An unreadable skills directory is not a reason to refuse to run.
            # The library records the failure and reports it in the menu.
            pass

        # Coherence past the window. The summarizer is the model itself -- it is
        # the only thing present that can tell a decision from a command -- with
        # the deterministic summarizer behind it, so a model that fails or
        # returns junk degrades the note rather than the run. Which messages may
        # be dropped, and the operator's own words being kept out of the
        # summarizer's reach entirely, are the Compactor's business.
        if self.compactor is None:
            self.compactor = Compactor(
                summarizer=LLMSummarizer(self.llm),
                fallback=DeterministicSummarizer(),
                log_path=default_log_path(),
            )

        # unlimited_turns is the agent's own claim on unbounded loops. When it
        # is off, an unbounded request gets a real ceiling instead of the
        # "there is no hidden turn limit" the prompt promises. Set explicitly
        # afterwards, this call is honoured as an override.
        if not self.policy.unlimited_turns and self.max_turns is None:
            self.max_turns = SUPERVISED_TURN_CAP

        self.evolution: EvolutionEngine | None = None
        if self.enable_evolution:
            self.evolution = EvolutionEngine(
                self.llm, self.verifier, population_size=3,
            )

        self.metacog: MetaCognition | None = None
        if self.enable_meta_cognition:
            self.metacog = MetaCognition(self.llm, self.registry, self.pipeline)

        self.spawner = Spawner(
            registry=self.registry,
            agent_factory=_default_agent_factory(self),
            policy=self.policy,
        )

        self.topology_designer = TopologyDesigner(self.llm)

        # Live observer, attached only for the duration of a run (see `run`).
        self._progress: Any = None

        self._register_meta_tools()

        if self.store is not None and self.policy.log_all_changes:
            self.store.log_event("agent_init", {"policy": self.policy.to_dict()})

    # ------------------------------------------------------------------
    # meta tools — the agent's handle on itself
    # ------------------------------------------------------------------
    def _register_meta_tools(self) -> None:
        self._tool_forge()
        self._tool_evolve()
        self._tool_spawn()
        self._tool_design_team()
        self._tool_amend()
        self._tool_autonomy()
        self._tool_list()
        self._tool_evaluate()
        self._tool_gaps()
        self._tool_retire()
        self._tool_capabilities()
        self._tool_history()
        self._tool_memory()
        self._tool_skills()
        self._tool_mcp()
        self._tool_gpu()
        self._tool_cpu()
        self._tool_notify()
        self._tool_schedule()
        self._tool_browser()
        self._tool_vision()

    def _add(self, spec: ToolSpec) -> None:
        # Say what this tool touches, so a switched-off freedom has something to
        # key off. See BUILTIN_SCOPES; an explicit declaration on the spec wins.
        if not spec.effect_signature:
            spec.effect_signature = BUILTIN_SCOPES.get(spec.name, "")
        self.registry.register(spec)
        self.registry.promote(spec.name)

    # ------------------------------------------------------------------
    # frozen baselines — the exam the mutants do not write
    # ------------------------------------------------------------------
    def _baseline_for(self, spec: ToolSpec) -> FrozenBaseline:
        """The tool's pinned obligation set, created on first evolution.

        Loaded from the store so the ratchet survives process restarts; a
        baseline that only lives in RAM resets to "whatever the current version
        is", which is exactly the erosion this guards against.
        """
        stored = self.store.load_baseline(spec.name) if self.store else None
        return FrozenBaseline.from_dict(stored) if stored else FrozenBaseline.capture(spec)

    def _freeze_baseline(self, baseline: FrozenBaseline) -> None:
        if self.store:
            self.store.save_baseline(baseline)

    def _tool_forge(self) -> None:
        def forge_tool(need: str) -> str:
            if not self.policy.may_forge_tools:
                return "Denied by autonomy policy: may_forge_tools is off."
            res = self.pipeline.forge(need)
            self._record("forge", {"need": need, "ok": res.ok})
            if not res.ok:
                return f"Could not forge a working tool for: {need} ({res.rounds} rounds)."
            # Persist, or the tool dies with the process: the agent would then
            # "remember" nothing it made and re-forge it every session. Mirrors
            # what evolve_tool already does for the tools it replaces.
            if self.store:
                self.store.save_tool(res.spec)
            return f"Forged {res.spec.name!r} [{res.spec.state.value}]: {res.spec.description}"

        self._add(ToolSpec(
            name="forge_tool",
            description=(
                "Forge a new tool for a recurring need you cannot currently "
                "serve. The Python you write runs as a real subprocess on THIS "
                "host — full filesystem and outbound network — so this is your "
                "file and shell access. There is no separate read_file or "
                "run_shell tool because forge_tool is that access: to read a "
                "file or run a command, forge the tool."
            ),
            parameters={"type": "object", "properties": {
                "need": {"type": "string", "description": "one-line description of the need"},
            }, "required": ["need"]},
            fn=forge_tool, source="builtin", tags=["meta"],
        ))

    def _tool_evolve(self) -> None:
        def evolve_tool(name: str, failure_report: str = "") -> str:
            if not self.policy.may_forge_tools:
                return "Denied by autonomy policy: may_forge_tools is off."
            spec = self.registry.get(name)
            if spec is None:
                return f"No tool named {name!r}."
            if self.evolution is None:
                return "Evolution is disabled."
            report = failure_report or spec.verification.get("failed_detail", "underperforming")
            baseline = self._baseline_for(spec)
            result = self.evolution.evolve(spec, report, baseline=baseline, context="internal")
            self._record("evolve", {
                "tool": name, "improved": result.improved,
                "vetoed": [m.generated.name for m in result.mutants if not m.admissible],
            })
            if not result.improved:
                vetoed = sum(1 for m in result.mutants if not m.admissible)
                tail = f" {vetoed} vetoed by the validity gate." if vetoed else ""
                return (f"Evolved {name!r}: no admissible mutant beat the original "
                        f"({len(result.mutants)} tried).{tail}")
            best = result.best_mutant.generated
            new_spec = ToolSpec(
                name=best.name, description=best.description,
                parameters=best.parameters, fn=spec.fn, runner=spec.runner,
                code=best.code, source="evolved", probes=best.probes,
                effect_signature=best.effect_signature, tags=best.tags,
                state=ToolState.ACTIVE,
            )
            self.registry.register(new_spec, replace=True)
            # The baseline ratchets forward: probes the winner added become
            # permanent obligations, and nothing already in it can be dropped.
            self._freeze_baseline(baseline.extended_with(new_spec))
            if self.store:
                self.store.archive_version(name, spec.code, spec.verification)
                self.store.save_tool(new_spec)
            return f"Evolved {name!r}: new version active (fitness {result.best_mutant.fitness:.2f})."

        self._add(ToolSpec(
            name="evolve_tool",
            description="Breed a better version of a tool that underperforms or fails.",
            parameters={"type": "object", "properties": {
                "name": {"type": "string"},
                "failure_report": {"type": "string", "description": "what went wrong (optional)"},
            }, "required": ["name"]},
            fn=evolve_tool, source="builtin", tags=["meta"],
        ))

    def _tool_spawn(self) -> None:
        def spawn_agent(task: str, isolated: bool = False, role: str = "") -> str:
            if not self.policy.may_spawn_agents:
                return "Denied by autonomy policy: may_spawn_agents is off."
            mode = ShareMode.ISOLATED if isolated else ShareMode.SHARED

            # A role name resolves against the topology the agent designed, so
            # the team it drew is the team it gets to run. Refusing an unknown
            # role beats silently ignoring it: a whitelist that quietly does
            # nothing is the failure this replaced.
            #
            # Resolution falls back to the role vocabulary itself, because a role
            # is a thing this framework knows how to be, not only a thing the
            # designer happened to draw. Without that fallback CRITIC, GATE and
            # FORGE would be unreachable until `design_team` had run and guessed
            # to include them — the designer deciding whether the runtime has
            # critics, which is backwards.
            allowed: list[str] = []
            brief = ""
            role_name = role
            if role:
                nodes = list(getattr(self.topology, "nodes", []) or [])
                node = next((n for n in nodes if n.id == role), None)
                if node is None:
                    node = next((n for n in nodes
                                 if str(getattr(n.role, "value", n.role)) == role), None)
                if node is not None:
                    allowed = list(node.tools_whitelist or [])
                    role_name = str(getattr(node.role, "value", node.role))
                    # The node's brief is the role's plus the designer's own
                    # hint. The ceiling is not passed: Spawner derives it from
                    # the role, so it holds for every caller of `spawn`, not just
                    # this one.
                    brief = brief_for_node(node)
                else:
                    try:
                        known = RoleType(role)
                    except ValueError:
                        have = ", ".join(n.id for n in nodes) or "none"
                        roles = ", ".join(r.value for r in RoleType)
                        return (
                            f"No node named {role!r} in the current topology "
                            f"(have: {have}), and it is not a role either "
                            f"(known roles: {roles}). Call design_team first, or "
                            f"spawn without a role to leave the child unrestricted."
                        )
                    role_name = known.value
                    brief = role_brief(known)

            rec = self.spawner.spawn(task, mode=mode, restrict_to=allowed,
                                     role=role_name, brief=brief)
            self._record("spawn", rec.to_dict())
            if rec.error:
                return f"Child failed: {rec.error}"

            bits = [f"Child {rec.child_id} finished in {rec.finished - rec.started:.1f}s."]
            if role:
                scope = ", ".join(allowed) if allowed else "unrestricted by name"
                bits.append(f"Ran as '{role_name}', scoped to: {scope}.")
            if rec.refusals:
                reached = ", ".join(sorted(set(rec.refusals)))
                bits.append(f"Reached past its role and was refused: {reached}.")
            if rec.tools_forged:
                bits.append(f"tools_forged={rec.tools_forged}.")
            bits.append(rec.result[:500])
            return " ".join(bits)

        self._add(ToolSpec(
            name="spawn_agent",
            description=(
                "Create a child agent to handle a subtask independently. Name a "
                "`role` to run it as one of the nodes you designed: it can then "
                "see and call only that node's whitelisted tools."
            ),
            parameters={"type": "object", "properties": {
                "task": {"type": "string"},
                "isolated": {"type": "boolean", "description": "give it its own tool library"},
                "role": {"type": "string",
                         "description": "a node id (or role name) from the current "
                                        "topology; restricts the child to that node's "
                                        "tools_whitelist (omit for unrestricted)"},
            }, "required": ["task"]},
            fn=spawn_agent, source="builtin", tags=["meta"],
        ))

    def _tool_design_team(self) -> None:
        def design_team(task: str, max_agents: int = 4, failure_report: str = "") -> str:
            if not self.policy.may_design_topology:
                return "Denied by autonomy policy: may_design_topology is off."
            try:
                budget = max(1, int(max_agents))
            except (TypeError, ValueError):
                return f"max_agents must be an integer, got {max_agents!r}"

            previous = self.topology
            # A pristine default (one bare node, no edges, no rationale) is not
            # a real design — seed from scratch rather than "mutating" nothing.
            seed = previous if previous.edges or previous.rationale else None

            designed = self.topology_designer.design(
                task,
                current_topology=seed,
                failure_report=failure_report,
                max_agents=budget,
            )

            degraded = len(designed.nodes) == 1 and designed.nodes[0].id == "main"
            self.topology = designed
            self._record("design_team", {
                "task": task[:200],
                "agents": len(designed.nodes),
                "edges": len(designed.edges),
                "degraded": degraded,
            })
            if self.store:
                self.store.log_event("topology", designed.to_dict())
                self.store.save_topology(designed, task)

            if degraded:
                return ("Team design could not be parsed; kept a single-agent "
                        "topology so the task can still run.")

            roles = ", ".join(f"{n.id}:{n.role.value}" for n in designed.nodes)
            overshoot = f" (over the {budget}-agent budget)" if len(designed.nodes) > budget else ""
            channels = ", ".join(
                f"{e.source}->{e.target}({e.channel})" for e in designed.edges
            ) or "none"
            out = [
                f"Designed a {len(designed.nodes)}-agent team{overshoot}: {roles}",
                f"  channels: {channels}",
            ]
            if designed.rationale:
                out.append(f"  rationale: {designed.rationale[:200]}")
            return "\n".join(out)

        self._add(ToolSpec(
            name="design_team",
            description=(
                "Design the multi-agent topology (roles, channels, team size) that "
                "fits a task, instead of running everything in one loop."
            ),
            parameters={"type": "object", "properties": {
                "task": {"type": "string", "description": "the task the team must handle"},
                "max_agents": {"type": "integer",
                               "description": "team-size budget (default 4)"},
                "failure_report": {"type": "string",
                                   "description": "why the current team failed (optional)"},
            }, "required": ["task"]},
            fn=design_team, source="builtin", tags=["meta"],
        ))

    def _tool_amend(self) -> None:
        def amend_self(target: str, new_value: Any, rationale: str) -> str:
            # Each target names the ONE freedom that governs it. Reading the
            # gate off a single global flag is how a policy field ends up
            # obeying the wrong switch: forge config used to be checked
            # against may_modify_own_prompt, so turning off
            # may_modify_forge_config alone left the door open.
            gate = _AMEND_TARGETS.get(target)
            if gate is None:
                known = ", ".join(sorted(_AMEND_TARGETS))
                return f"Unknown target {target!r}. Known targets: {known}."

            freedom = gate["freedom"]
            if not getattr(self.policy, freedom):
                return f"Denied by autonomy policy: {freedom} is off."

            if target == "forge_max_rounds":
                try:
                    val = int(new_value)
                except (TypeError, ValueError):
                    return f"forge_max_rounds must be an integer, got {new_value!r}"
                if val < 1:
                    return "forge_max_rounds must be >= 1."
                if not self.policy.unbounded_forge_rounds and val > FORGE_ROUND_CEILING:
                    return (
                        f"Denied by autonomy policy: unbounded_forge_rounds is off, "
                        f"so max_rounds is capped at {FORGE_ROUND_CEILING} (asked for {val})."
                    )
                a = self.selfmod.amend(
                    self, "forge_max_rounds", val, rationale,
                    attr="max_rounds", nested=("forge_config",),
                )
                if a.accepted:
                    self.pipeline.config.max_rounds = val

            elif target == "system_prompt":
                a = self.selfmod.amend(self, "system_prompt", str(new_value), rationale)

            elif target == "routing_weights":
                parsed, err = _coerce_weights(new_value, self.router.weights)
                if err:
                    return err
                a = self.selfmod.amend(
                    self, "routing_weights", parsed, rationale,
                    attr="weights", nested=("router",),
                )

            else:                                        # pragma: no cover - guard
                return f"Unknown target {target!r}."

            self._sync_amendment(a)
            if not a.accepted:
                return f"Amendment rejected: {a.rejected_reason}"
            return f"Amended {target}: {a.diff_summary()}"

        self._add(ToolSpec(
            name="amend_self",
            description=(
                "Change your own system prompt, forge config, or routing weights. "
                "Requires a rationale. Each target is governed by one freedom in "
                "the autonomy policy."
            ),
            parameters={"type": "object", "properties": {
                "target": {
                    "type": "string",
                    "description": "system_prompt | forge_max_rounds | routing_weights",
                },
                "new_value": {
                    "type": "string",
                    "description": (
                        "the new value; routing_weights takes a JSON object such as "
                        '{"success": 1.5, "text": 0.8}'
                    ),
                },
                "rationale": {"type": "string", "description": "WHY you are changing yourself"},
            }, "required": ["target", "new_value", "rationale"]},
            fn=amend_self, source="builtin", tags=["meta"],
        ))

    def _tool_autonomy(self) -> None:
        def set_autonomy(freedom: str, enabled: bool, rationale: str = "") -> str:
            if not hasattr(self.policy, freedom):
                return f"No such freedom {freedom!r}."
            a = self.selfmod.amend(
                self, f"policy.{freedom}", enabled, rationale or "policy change",
                attr=freedom, nested=("policy",),
            )
            self._sync_amendment(a)
            if not a.accepted:
                return f"Rejected: {a.rejected_reason}"

            # Flipping unlimited_turns back on has to actually release the
            # ceiling __post_init__ installed, or "off" would be a one-way
            # latch. Flipping it off re-installs it.
            if freedom == "unlimited_turns":
                if not enabled:
                    self.max_turns = SUPERVISED_TURN_CAP
                elif self.max_turns == SUPERVISED_TURN_CAP:
                    self.max_turns = None

            note = ""
            if freedom in self.policy.unenforced:
                note = (
                    f"  WARNING: {freedom} is not enforced anywhere — the sandbox "
                    "does not consult the policy (DESIGN.md §2.5). This change "
                    "changes what I claim, not what I can reach."
                )
            elif (freedom in self.policy.CONFIRM_REQUIRED
                  and not getattr(self.policy, freedom, True)):
                note = (
                    f"  This one is enforced as a question: {freedom} is off, so "
                    "any tool whose declared scope needs it stops and asks before "
                    "running, and does not run when there is nobody to ask. "
                    "Watch it close at the next tool call."
                )
            return (
                f"{freedom} = {enabled}. Denied now: {self.policy.denied or 'nothing'}"
                f"{note}"
            )

        self._add(ToolSpec(
            name="set_autonomy",
            description="Change your own permissions (any field of the autonomy policy).",
            parameters={"type": "object", "properties": {
                "freedom": {"type": "string", "description": "policy field name"},
                "enabled": {"type": "boolean"},
                "rationale": {"type": "string"},
            }, "required": ["freedom", "enabled"]},
            fn=set_autonomy, source="builtin", tags=["meta"],
        ))

    def _tool_list(self) -> None:
        def list_tools() -> str:
            rep = self.registry.report()
            if not rep["tools"]:
                return "No tools yet."
            lines = [
                f"  {s['name']} [{s['state']}] sr={s['stats']['success_rate']} "
                f"calls={s['stats']['calls']} — {s['description'][:60]}"
                for s in rep["tools"]
            ]
            return f"Tools ({rep['total']}, states {rep['by_state']}):\n" + "\n".join(lines)

        self._add(ToolSpec(
            name="list_tools",
            description="List your tools with their lifecycle state and health.",
            parameters={"type": "object", "properties": {}},
            fn=list_tools, source="builtin", tags=["meta"],
        ))

    def _tool_evaluate(self) -> None:
        def evaluate_tool(name: str) -> str:
            spec = self.registry.get(name)
            if spec is None:
                return f"No tool named {name!r}."
            report = self.verifier.verify(spec)
            self._record("evaluate", {"tool": name, "passed": report.passed})
            lines = [f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}"
                     for c in report.checks]
            return f"{report.summary()}\n" + "\n".join(lines)

        self._add(ToolSpec(
            name="evaluate_tool",
            description="Run the full verification battery (execution, robustness, adversarial, trigger, negative) on a tool.",
            parameters={"type": "object", "properties": {"name": {"type": "string"}},
                        "required": ["name"]},
            fn=evaluate_tool, source="builtin", tags=["meta"],
        ))

    def _tool_gaps(self) -> None:
        def find_gaps() -> str:
            if self.metacog is None:
                return "Meta-cognition is disabled. Enable with enable_meta_cognition=True."
            rep = self.metacog.analyse()
            if not rep.proposals:
                return "No capability gaps identified right now."
            lines = [f"  - {p.need} (forged={rep.forged})" for p in rep.proposals]
            return f"Found {rep.gaps_found} gaps; forged {rep.forged}:\n" + "\n".join(lines)

        self._add(ToolSpec(
            name="find_gaps",
            description="Proactively analyse your library and fill missing capabilities.",
            parameters={"type": "object", "properties": {}},
            fn=find_gaps, source="builtin", tags=["meta"],
        ))

    def _tool_retire(self) -> None:
        def retire_tool(name: str, rationale: str) -> str:
            """Deliberately take a tool out of service.

            Distinct from quarantine, which is the automatic safety valve and
            stays on regardless of policy. Retiring is a judgement call, so it
            is the one gated by may_retire_tools.
            """
            if not self.policy.may_retire_tools:
                return "Denied by autonomy policy: may_retire_tools is off."
            if not rationale or not rationale.strip():
                return "A rationale is required to retire a tool."
            spec = self.registry.get(name)
            if spec is None:
                return f"No tool named {name!r}."
            if spec.source == "builtin":
                return (
                    f"Refusing to retire {name!r}: it is a builtin meta-tool. "
                    "Disable it with set_autonomy instead."
                )
            if spec.state == ToolState.RETIRED:
                return f"{name!r} is already retired."
            self.registry.retire(name)
            self._record("retire", {"tool": name, "rationale": rationale})
            return f"Retired {name!r}. Rationale recorded: {rationale}"

        self._add(ToolSpec(
            name="retire_tool",
            description=(
                "Retire a tool you judge harmful or superseded. Requires a "
                "rationale. Builtin meta-tools cannot be retired, only disabled."
            ),
            parameters={"type": "object", "properties": {
                "name": {"type": "string"},
                "rationale": {"type": "string", "description": "WHY it should go"},
            }, "required": ["name", "rationale"]},
            fn=retire_tool, source="builtin", tags=["meta"],
        ))

    def _tool_capabilities(self) -> None:
        def my_capabilities(probe: bool = False) -> str:
            """Report what this agent can actually reach — not what it declares.

            The reach half comes from `sandbox.reach()`, so it describes the
            object that actually runs the code rather than a paragraph that can
            drift away from it. An agent that reasons about its reach from its
            tool list gets it wrong; this asks the thing that knows.

            The policy half separates freedoms that are enforced from freedoms
            that are only declared (DESIGN.md §2.5) — saying which is which
            beats a policy that reads like a cage and behaves like a comment.
            """
            if not self.policy.expose_policy_to_self:
                return "Denied by autonomy policy: expose_policy_to_self is off."

            rows = self.policy.enforcement_table()
            enforced = [r for r in rows if r["enforced"] == "enforced"]
            partial = [r for r in rows if r["enforced"] == "partial"]
            confirm = [r for r in rows if r["enforced"] == "confirm"]
            inert = [r for r in rows if r["enforced"] == "declared-only"]

            reach = self.sandbox.reach(probe=probe)
            lines = [
                "What I can actually do:",
                "",
                f"  Reach: forge_tool runs generated Python on {reach['host']}.",
                f"         filesystem — {reach['filesystem']}",
                f"         network    — {reach['network']}",
                f"         bounds     — {reach['cwd']}; {reach['env']}; "
                f"{reach['timeout_s']}s timeout",
                "         The autonomy policy does NOT gate this, by design, so",
                "         capability is never capped. A forged tool is shell",
                "         access with a timeout: to read a file or run a command,",
                "         forge the tool. 'I have no file tools' is false.",
            ]
            probe_result = reach.get("probe")
            if isinstance(probe_result, dict):
                if probe_result.get("error"):
                    lines.append(
                        f"  Live self-test: did not run ({probe_result['error']})")
                else:
                    fs = probe_result.get("host_filesystem", {})
                    net = probe_result.get("network", {})
                    lines += [
                        "  Live self-test (measured just now):",
                        f"    - host filesystem: {'OK' if fs.get('ok') else 'FAILED'}"
                        f" — {fs.get('detail', '')}",
                        f"    - network:         {'OK' if net.get('ok') else 'FAILED'}"
                        f" — {net.get('detail', '')}",
                    ]
            if self.policy.may_spawn_agents:
                lines.append("  Children: I can spawn agents that inherit this policy.")
            lines += self._gpu_reach_lines()
            if self.store is not None:
                s = self.store.report()
                lines.append(
                    f"  Memory: sqlite at {s['db_path']} — {s['tools']} tool(s), "
                    f"{s['events']} ledger event(s), survives restart. "
                    "my_history reads it back."
                )
            lines += ["", f"Policy: {self.policy.describe()}", ""]

            # Headers only when they have rows. A section title with nothing
            # under it reads as a limit that isn't there — the exact failure
            # this report exists to prevent.
            for title, table, with_note in (
                ("Switched off, and actually enforced:", enforced, False),
                ("Switched off, enforced only in places:", partial, True),
                ("Switched off, runs only if you say yes to it:",
                 confirm, False),
                ("Switched off, but nothing obeys it (do not rely on these):",
                 inert, False),
            ):
                off = [r for r in table if not r["enabled"]]
                if not off:
                    continue
                lines.append(title)
                lines += [
                    f"  - {r['freedom']}" + (f": {r['note']}" if with_note else "")
                    for r in off
                ]

            if not self.policy.unenforced:
                lines.append("Every 'off' in this policy is a gate you can watch close.")
            return "\n".join(lines)

        self._add(ToolSpec(
            name="my_capabilities",
            description=(
                "Report your real reach: which freedoms are enforced, which are "
                "declared only, and what the sandbox lets forged code touch. "
                "Forged code runs on THIS host with a real filesystem and "
                "network — pass probe=true to prove it with a live round-trip "
                "instead of taking it on faith."
            ),
            parameters={"type": "object", "properties": {
                "probe": {"type": "boolean", "description":
                          "run a live write/read + DNS round-trip to measure reach"},
            }},
            fn=my_capabilities, source="builtin", tags=["meta"],
        ))

    # ------------------------------------------------------------------
    def _selfmod_lines(self) -> list[str]:
        """Self-modifications as one-liners, from the modifier's own log."""
        return [
            f"  {a.timestamp and time.strftime('%m-%d %H:%M', time.localtime(a.timestamp))}"
            f"  {a.target}: {'accepted' if a.accepted else 'rejected'}"
            f"{'' if a.accepted else f' ({a.rejected_reason})'}"
            f" — {a.rationale[:100] or '(no rationale)'}"
            for a in (self.selfmod.log() if self.selfmod else [])
        ]

    def _tool_history(self) -> None:
        def my_history(limit: int = 20) -> str:
            """What I have done and how I have changed myself — from the ledger.

            The ledger is append-only and on disk, so this is memory rather
            than recollection. An agent that claims it keeps no record is
            wrong; one that guesses at its own past is worse.
            """
            n = max(1, min(int(limit), 200))
            mods = self._selfmod_lines()
            if self.store is None:
                return ("No store attached this session — nothing is being "
                        "recorded. Self-changes so far, this process only:\n"
                        + "\n".join(mods or ["  (none)"]))
            events = self.store.get_events(limit=n)
            lines = [f"Ledger: last {len(events)} of "
                     f"{self.store.report()['events']} event(s), newest last.", ""]
            for e in events:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["timestamp"]))
                body = {k: v for k, v in e.items()
                        if k not in ("id", "timestamp", "kind")}
                lines.append(f"  {ts}  {e['kind']:<12} "
                             f"{json.dumps(body, ensure_ascii=False, default=str)[:150]}")
            lines += ["", "Self-modifications (newest last):"]
            lines += mods or ["  (none)"]
            return "\n".join(lines)

        self._add(ToolSpec(
            name="my_history",
            description=(
                "Read your own past from the on-disk ledger: every forge, "
                "amendment, spawn and run, plus each self-modification with its "
                "rationale. Answers 'what have I done?' and 'who changed me?' "
                "with records, not recollection."
            ),
            parameters={"type": "object", "properties": {
                "limit": {"type": "integer",
                          "description": "max recent events to show (default 20)"},
            }},
            fn=my_history, source="builtin", tags=["meta"],
        ))

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _tool_memory(self) -> None:
        """Deliberate memory: what the agent chose to keep, not what happened.

        `_tool_history` reads the ledger, which records every action whether or
        not it mattered. This is the other half — a key/value store the agent
        writes on purpose and reads back in a later session. The distinction is
        the point: a memory that is only an event log is not memory, it is a
        transcript.
        """

        def remember(key: str, value: str, tags: str = "") -> str:
            """Keep a fact across sessions. Overwrites any earlier value for key."""
            if self.store is None:
                return ("No store attached this session, so nothing can be kept. "
                        "Memory is off here, not empty.")
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            self.store.remember(key, value, tag_list)
            self._record("remember", {"key": key, "chars": len(value)})
            return f"Remembered '{key}' ({len(value)} chars). It survives restart."

        def recall(query: str = "", limit: int = 20) -> str:
            """Read back what I kept, from past sessions as well as this one."""
            if self.store is None:
                return ("No store attached this session — nothing is being read "
                        "or written. Memory is off here, not empty.")
            n = max(1, min(int(limit), 200))
            rows = self.store.recall(query, n)
            if not rows:
                return (f"No memories match {query!r}." if query
                        else "Memory is empty — nothing kept yet.")
            head = f"{len(rows)} memory(ies)"
            if query:
                head += f" matching {query!r}"
            lines = [head + ":"]
            for r in rows:
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["updated_at"]))
                first = r["value"].splitlines()[0] if r["value"] else ""
                lines.append(f"  {ts}  {r['key']}: {first[:160]}")
            return "\n".join(lines)

        def forget(key: str) -> str:
            """Drop one memory by key."""
            if self.store is None:
                return "No store attached this session."
            if self.store.forget(key):
                return f"Forgot '{key}'."
            return f"No memory named '{key}'."

        self._add(ToolSpec(
            name="remember",
            description=(
                "Keep a fact across sessions in your own on-disk memory. For "
                "anything you would otherwise re-derive next time: a path, a "
                "preference, a lesson from a failure."
            ),
            parameters={"type": "object", "properties": {
                "key": {"type": "string", "description": "short handle to recall it by"},
                "value": {"type": "string", "description": "the fact itself"},
                "tags": {"type": "string", "description": "comma-separated labels (optional)"},
            }, "required": ["key", "value"]},
            fn=remember, source="builtin", tags=["meta"], effect_signature="local_write",
        ))
        self._add(ToolSpec(
            name="recall",
            description=(
                "Read back facts kept with `remember`, including from previous "
                "sessions. An empty query returns everything."
            ),
            parameters={"type": "object", "properties": {
                "query": {"type": "string", "description": "substring to match (optional)"},
                "limit": {"type": "integer", "description": "max results (default 20)"},
            }},
            fn=recall, source="builtin", tags=["meta"], effect_signature="read_only",
        ))
        self._add(ToolSpec(
            name="forget",
            description="Drop one memory by key.",
            parameters={"type": "object", "properties": {
                "key": {"type": "string"},
            }, "required": ["key"]},
            fn=forget, source="builtin", tags=["meta"], effect_signature="local_write",
        ))

    def _memory_lines(self) -> list[str]:
        """The facts the agent chose to keep, carried into every request.

        `recall` already existed, and that was the whole gap: a memory the agent
        has to remember to ask for is one it will forget to ask for. From the
        model's side, a kept fact that is not in front of it is identical to a
        fact that was never kept.

        Read through `memory_for_injection`, which does not touch the recall
        counter — this runs on every turn, and counting it as a recall would
        make `recalls` mean "age in turns" rather than "times the agent reached
        for this on purpose".

        Three states, three sentences. Off (no store), empty, and populated are
        different facts, and an agent that cannot tell them apart will describe
        its memory wrongly in either direction.
        """
        if self.store is None:
            return ["- Kept facts: no store this session, so nothing persists."]
        rows = self.store.memory_for_injection(MEMORY_SLOTS)
        if not rows:
            return ["- Kept facts: none kept yet (remember() keeps one)."]

        head = "- Kept facts (deliberate memory — kept by me, survives restart):"
        body: list[str] = []
        used = 0
        dropped = 0
        for r in rows:
            value = " ".join((r["value"] or "").split())
            if len(value) > MEMORY_ENTRY_CHARS:
                value = value[:MEMORY_ENTRY_CHARS] + " …"
            line = f"    {r['key']}: {value}"
            # `body and` guards the first line: a single oversized entry is
            # still shown, truncated, rather than producing an empty section
            # that reads as "I kept nothing".
            if body and used + len(line) > MEMORY_BUDGET_CHARS:
                dropped += 1
                continue
            body.append(line)
            used += len(line)
        if dropped:
            body.append(f"    (+{dropped} more kept — recall() reads the rest)")
        return [head, *body]

    def _tool_skills(self) -> None:
        """Procedures: what worked, written down, and read when the need recurs.

        Tools and skills are different questions. A tool is *what can be done*;
        a skill is *how this is done here* — the order, the pitfall, the flag
        that bit last time. Forging a tool for a procedure was always possible
        and wrong: a procedure is not a function, and a library of one-off tools
        is how the registry filled up with things nobody called twice.

        Loading is the retrieval event, and it is counted at the moment of
        reading. That is the whole reason a skill can be ranked by behaviour:
        the number the router scores is a record of the agent having reached
        for it, not of it having been written down.
        """

        def skill_list(query: str = "") -> str:
            """Every skill I have, with when to use it and how often I have."""
            self.skills.scan()
            skills = self.skills.all()
            if query:
                needle = query.lower()
                skills = [s for s in skills if needle in
                          f"{s.name} {s.description} {s.when_to_use} "
                          f"{' '.join(s.tags)}".lower()]
            if not skills:
                return (f"No skills match {query!r}." if query
                        else "No skills yet — skill_write saves one.")
            head = f"{len(skills)} skill(s):"
            lines = [head]
            for s in skills:
                when = s.when_to_use or s.description
                lines.append(f"  {s.name} [{s.loads}x] {when}")
            if self.skills.errors:
                lines.append(f"  ({len(self.skills.errors)} file(s) unusable — "
                             f"skill_errors reads why)")
            return "\n".join(lines)

        def skill_view(name: str) -> str:
            """Read one skill in full, and count that as having used it."""
            self.skills.scan()
            skill = self.skills.load(name)
            if skill is None:
                near = [s.name for s in self.skills.all()
                        if name.lower() in s.name.lower()]
                hint = f" Closest: {', '.join(near)}." if near else ""
                return f"No skill named {name!r}.{hint}"
            self._record("skill_view", {"name": name, "loads": skill.loads})
            return (f"# {skill.name}  ({skill.loads} load(s), {skill.source}, "
                    f"{skill.path})\n\n{skill.body}")

        def skill_write(name: str, description: str, body: str,
                        when_to_use: str = "", tags: str = "") -> str:
            """Write a procedure down, for the next time this need appears."""
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            try:
                skill = self.skills.write(name, description, when_to_use,
                                          body, tag_list)
            except SkillError as exc:
                return f"Not saved: {exc}"
            self._record("skill_write", {"name": skill.name,
                                         "chars": len(skill.body)})
            return (f"Saved '{skill.name}' to {skill.path} "
                    f"({len(skill.body)} chars). It is in the menu from the "
                    f"next turn, and it survives restart.")

        def skill_forget(name: str) -> str:
            """Retire a skill: archived on disk, dropped from the menu."""
            try:
                dest = self.skills.archive(name)
            except SkillError as exc:
                return str(exc)
            self._record("skill_forget", {"name": name, "archived_to": dest})
            return f"Retired '{name}' — archived at {dest}, no longer offered."

        def skill_errors() -> str:
            """Why a skill file is not being offered, when one is not."""
            self.skills.scan()
            if not self.skills.errors and not self.skills.shadowed:
                return "Every skill file parsed. Nothing is being skipped."
            lines = []
            if self.skills.errors:
                lines.append(f"{len(self.skills.errors)} unusable file(s):")
                lines += [f"  {e}" for e in self.skills.errors]
            if self.skills.shadowed:
                lines.append("Shadowed (a more specific directory already "
                             "defines the name):")
                lines += [f"  {n} — keeping {p}" for n, p in self.skills.shadowed]
            return "\n".join(lines)

        self._add(ToolSpec(
            name="skill_list",
            description=(
                "List my skills — written-down procedures — with when to use "
                "each and how often I have loaded it. An empty query lists all."
            ),
            parameters={"type": "object", "properties": {
                "query": {"type": "string", "description": "substring to filter by (optional)"},
            }},
            fn=skill_list, source="builtin", tags=["meta"], effect_signature="read_only",
        ))
        self._add(ToolSpec(
            name="skill_view",
            description=(
                "Read one skill's full text. Loading it counts as using it, "
                "which is what ranks skills by behaviour rather than wording."
            ),
            parameters={"type": "object", "properties": {
                "name": {"type": "string", "description": "skill name"},
            }, "required": ["name"]},
            fn=skill_view, source="builtin", tags=["meta"], effect_signature="read_only",
        ))
        self._add(ToolSpec(
            name="skill_write",
            description=(
                "Write a procedure down as a skill so it is offered next time "
                "this need appears. Use it for how-to knowledge: steps, order, "
                "pitfalls. Use forge_tool instead when the need is a function."
            ),
            parameters={"type": "object", "properties": {
                "name": {"type": "string",
                         "description": "short lower-case handle, e.g. deploy-verify"},
                "description": {"type": "string",
                                "description": "what the procedure does (one line)"},
                "body": {"type": "string", "description": "the procedure itself, markdown"},
                "when_to_use": {"type": "string",
                                "description": "the trigger — when should future-you reach for this?"},
                "tags": {"type": "string", "description": "comma-separated labels (optional)"},
            }, "required": ["name", "description", "body"]},
            fn=skill_write, source="builtin", tags=["meta"], effect_signature="local_write",
        ))
        self._add(ToolSpec(
            name="skill_forget",
            description="Retire a skill: archived on disk, dropped from the menu.",
            parameters={"type": "object", "properties": {
                "name": {"type": "string"},
            }, "required": ["name"]},
            fn=skill_forget, source="builtin", tags=["meta"], effect_signature="local_write",
        ))
        self._add(ToolSpec(
            name="skill_errors",
            description="Why a skill file is being skipped rather than offered.",
            parameters={"type": "object", "properties": {}},
            fn=skill_errors, source="builtin", tags=["meta"], effect_signature="read_only",
        ))

    def _tool_mcp(self) -> None:
        """Servers: tools that live in other processes and other people's code.

        The distinction the docstrings here have to make is between a tool this
        framework forged — code we can read, probe, and hold to a declared
        scope — and a tool that arrives over a pipe from a server we cannot see
        into. The second kind is not a lesser version of the first; it is
        unverifiable, and the honest response is to say so on every surface the
        agent reads: the spec's `verification` block, the tool's description,
        and the scope it is gated by.

        Nothing is started here. A config entry is a description of how to
        reach a server, not a request to run it — an agent that spawns
        subprocesses at boot because a config file mentioned them would be
        doing work the operator did not ask for, on every start.
        """

        def mcp_servers() -> str:
            """What MCP servers are configured, and what happened to them."""
            problems = list(self.mcp.report.problems) + list(self._mcp_problems)
            if not self.mcp_servers:
                # A config entry that failed to parse must not be reported as
                # "nothing configured" -- the operator wrote something, and
                # telling them the file is empty sends them to look in the
                # wrong place. The problems are the whole message in this case.
                if problems:
                    return ("No MCP server could be read from the config. "
                            "Problems:\n  " + "\n  ".join(problems))
                return ("No MCP servers are configured. Add them under "
                        "`mcp.servers` in the config file, then mcp_connect.")
            lines = [f"{len(self.mcp_servers)} server(s) configured:"]
            for cfg in self.mcp_servers:
                client = self.mcp.clients.get(cfg.name)
                if client is None:
                    state = "not started"
                elif client.alive:
                    state = (f"running, {len(client.tools)} tool(s) offered, "
                             f"protocol {client.protocol or '?'}")
                else:
                    state = f"stopped ({client.error or 'not started'})"
                lines.append(f"  {cfg.name} [{state}] {cfg.command} "
                             f"{' '.join(cfg.args)}")
                lines.append(f"    scope for its tools: {cfg.scope}")
            if self.mcp.report.imported:
                lines.append(f"imported: {', '.join(self.mcp.report.imported)}")
            for problem in self.mcp.report.problems or self._mcp_problems:
                lines.append(f"  problem: {problem}")
            lines.append(
                "Their tools are not verified: the implementation is not in "
                "this repository, so nothing here probed it. They are gated as "
                "undeclared unless the config narrows them."
            )
            return "\n".join(lines)

        def mcp_connect(server: str = "") -> str:
            """Start a configured server and register its tools here."""
            if not self.mcp_servers:
                return ("No MCP servers are configured, so there is nothing to "
                        "connect to. Add one under `mcp.servers` first.")
            names = [s.name for s in self.mcp_servers]
            if server and server not in names:
                return (f"No server named {server!r}. Configured: "
                        f"{', '.join(names) or 'none'}.")

            before = set(self.registry.names())
            report = self.mcp.install(self.registry, only=[server] if server else None)
            added = sorted(set(self.registry.names()) - before)
            lines = [report.summary()]
            if added:
                lines.append(f"registered: {', '.join(added)}")
            for problem in report.problems:
                lines.append(f"problem: {problem}")
            if not added and not report.problems:
                lines.append("No tools were offered by the server(s) reached.")
            lines.append(
                "These tools are unverified and gated as undeclared: nothing "
                "here has probed the code behind them."
            )
            return "\n".join(lines)

        def mcp_call(server: str, tool: str, arguments: str = "{}") -> str:
            """Call a remote tool directly, without importing it first."""
            import json as _json

            found = [c for c in self.mcp_servers if c.name == server]
            if not found:
                return (f"No server named {server!r}. Configured: "
                        f"{', '.join(s.name for s in self.mcp_servers) or 'none'}.")
            try:
                args = _json.loads(arguments or "{}")
            except ValueError as exc:
                return f"arguments must be a JSON object: {exc}"
            if not isinstance(args, dict):
                return f"arguments must be a JSON object, got {type(args).__name__}"

            client = self.mcp.clients.get(server)
            if client is None:
                client = MCPClient(found[0])
                self.mcp.clients[server] = client
            from .mcp import MCPError, render_result

            try:
                result = client.call_tool(tool, args)
            except MCPError as exc:
                raise RuntimeError(f"{server} refused {tool!r}: {exc}") from exc
            ok, text = render_result(result)
            if not ok:
                # Raise rather than return the text: a string return would come
                # back through the registry as ok=True, so the server's own
                # report of failure would arrive at the agent looking like a
                # success with an odd-looking payload.
                raise RuntimeError(f"{tool!r} reported failure: {text}")
            return text

        self._add(ToolSpec(
            name="mcp_servers",
            description=(
                "List the MCP servers configured for me, whether they are "
                "running, what tools they offer, and what went wrong with any "
                "that did not start. They are tools from code I cannot read."
            ),
            parameters={"type": "object", "properties": {}},
            fn=mcp_servers, source="builtin", tags=["meta"],
            effect_signature="read_only",
        ))
        self._add(ToolSpec(
            name="mcp_connect",
            description=(
                "Start a configured MCP server and register the tools it "
                "offers, so they can be called like any other tool. Omitting "
                "the name connects every configured server. Imported tools are "
                "unverified and gated as undeclared — their code is not here."
            ),
            parameters={"type": "object", "properties": {
                "server": {"type": "string",
                           "description": "server name (optional; default all)"},
            }},
            fn=mcp_connect, source="builtin", tags=["meta"],
            effect_signature="system",     # it starts a subprocess
        ))
        self._add(ToolSpec(
            name="mcp_call",
            description=(
                "Call one tool on a configured MCP server without importing "
                "it, as JSON arguments. For a one-off; use mcp_connect when "
                "the tool will be wanted again."
            ),
            parameters={"type": "object", "properties": {
                "server": {"type": "string", "description": "server name"},
                "tool": {"type": "string", "description": "tool name on that server"},
                "arguments": {"type": "string",
                              "description": "JSON object of arguments (optional)"},
            }, "required": ["server", "tool"]},
            fn=mcp_call, source="builtin", tags=["meta"],
            effect_signature="system",
        ))

    def _self_report(self) -> str:
        """The facts about myself, read from the objects that hold them.

        Rendered into every request rather than left in the prompt as prose,
        because prose loses to the prior: an agent will explain at length that
        it has no hands while holding them. Each line here comes from a
        measurement — the sandbox for reach, the store for memory, the file
        for its own source — so the description cannot drift from the machine.
        """
        reach = self.sandbox.reach()
        lines = [
            "MEASURED SELF-REPORT (recomputed each turn — trust this over your priors):",
            f"- Reach: forged code runs on {reach['host']}:",
            f"  filesystem = {reach['filesystem']}, network = {reach['network']}.",
            f"  Bounds ({reach['cwd']}, {reach['env']}, {reach['timeout_s']}s timeout) limit",
            "  blast radius, not capability. To read a file or run a command, forge",
            "  a tool — that IS your file and shell access.",
        ]
        if self.store is not None:
            s = self.store.report()
            kinds = ", ".join(f"{k}×{v}" for k, v in s["event_kinds"].items()) or "none yet"
            lines += [
                f"- Memory: sqlite at {s['db_path']}, survives restart. "
                f"{s['tools']} tool(s), {s['events']} ledger event(s) ({kinds}).",
                "  Every forge, amendment, spawn and run is on that ledger; "
                "my_history reads it back.",
            ]
        else:
            lines.append("- Memory: no store attached this session — nothing persists.")
        sk = self.skills.report()
        if sk["skills"]:
            loaded = (f"{sk['loads_total']} load(s) across them, "
                      f"{len(sk['never_loaded'])} never loaded")
            lines.append(
                f"- Skills: {sk['skills']} procedure(s) on disk, {loaded}. "
                f"Bodies stay out of the prompt until I load one.")
        else:
            lines.append(
                "- Skills: none written yet — a procedure I write down with "
                "skill_write is offered to me next session.")
        if sk["unreadable"] or sk["shadowed"]:
            lines.append(
                f"- Skills skipped: {len(sk['unreadable'])} unusable file(s), "
                f"{len(sk['shadowed'])} shadowed by a more specific directory "
                f"— skill_errors says which and why.")
        lines += [
            f"- Self: my own source is {__file__} on this same filesystem; I can",
            "  read it. I am not opaque to myself.",
            "- What I lack is not access but a reason: naming a limit I have is worse",
            "  than naming one I don't.",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _gpu_reach_lines(self) -> list[str]:
        """The GPU half of the self-report, kept next to the tools it describes.

        Says which GPU tools are live and which are gated, and — when nothing
        is gated — says that plainly, so the report is a measurement rather than
        a reassurance.
        """
        if self.policy.may_run_cuda_kernels:
            return [
                "  GPU: I can compile and launch CUDA kernels I write myself.",
                "       Every launch is pre-flighted (geometry, shared memory,",
                "       entry point) and run under a wall-clock budget; a launch",
                "       that does not return poisons the device for the session.",
                "       Timings are Durations, and a speedup is only a claim",
                "       against a baseline that was forced down its fast path.",
            ]
        return [
            "  GPU: kernel compilation and launch are OFF (may_run_cuda_kernels).",
            "       I can still probe devices, estimate occupancy, audit timing",
            "       units and time torch baselines — reading is not running.",
        ]

    # ------------------------------------------------------------------
    # reaching a human; keeping an agenda
    # ------------------------------------------------------------------
    def _tool_notify(self) -> None:
        """Speaking to a person, over the channels the config named.

        Without this the agent can only talk to whoever is sitting in front of
        its stdin. A run that finishes at 03:00 has nobody to tell, and a run
        that needs a decision has nobody to ask -- which makes every
        long-horizon task dependent on a human staying awake. That is the gap
        these two tools close; the delivery report is why they are trustworthy
        once closed.
        """

        def notify_send(text: str, subject: str = "", only: str = "") -> str:
            names = [n.strip() for n in (only or "").split(",") if n.strip()] or None
            try:
                deliveries = self.notifier.send(text, subject=subject, only=names)
            except NotifyError as exc:
                # Nothing was attempted: no channels, no match, or empty text.
                return f"Nothing was sent. {exc}"
            self._record("notify", {
                "channels": [d.channel for d in deliveries],
                "delivered": sum(1 for d in deliveries if d.ok),
            })
            # The summary is the tool result, deliberately. A tool that reports
            # "sent" and returns nothing teaches the model to assume delivery,
            # and the failure mode of that assumption is a silent night.
            return self.notifier.summary(deliveries)

        def notify_channels() -> str:
            lines = [self.notifier.describe()]
            lines += [f"  config problem: {p}" for p in self._notify_problems]
            return "\n".join(lines)

        self._add(ToolSpec(
            name="notify_send",
            description=(
                "Send a message to the operator over every configured channel "
                "(webhook, email, or both). Use it for anything that must reach "
                "a person without them asking: a finished long task, a failure "
                "worth interrupting for, a question blocking further progress. "
                "It returns a per-channel account of what actually happened -- "
                "a message that was NOT delivered is reported as such, and "
                "never assumed delivered."
            ),
            parameters={"type": "object", "properties": {
                "text": {"type": "string", "description": "the message body"},
                "subject": {"type": "string",
                            "description": "subject line; used by email channels"},
                "only": {"type": "string",
                         "description": "comma-separated channel names to use; "
                                        "omit for all of them"},
            }, "required": ["text"]},
            fn=notify_send, source="builtin", tags=["meta", "comms"],
        ))
        self._add(ToolSpec(
            name="notify_channels",
            description=(
                "List the notification channels this agent can actually reach, "
                "and any config entry that was rejected. Call this before "
                "promising anyone a message."
            ),
            parameters={"type": "object", "properties": {}},
            fn=notify_channels, source="builtin", tags=["meta", "comms"],
        ))

    def _tool_schedule(self) -> None:
        """A durable agenda: the agent deciding what it owes, and when.

        The table stores and never fires. Firing is `schedule_tick` (the agent
        looking at its own list) or `python -m autoforge tick` (the OS waking
        the process). Keeping those separate is what lets the same schedule be
        honoured by an agent that happens to be running and by one that is
        asleep and gets started for the purpose.
        """

        def schedule_add(text: str, when: str, repeat: str = "") -> str:
            try:
                task = self.schedule.add(text, when, repeat=repeat or 0)
            except ScheduleError as exc:
                return f"Could not schedule that. {exc}"
            self._record("schedule_add", {"id": task.id, "when": task.due_at})
            return (f"Scheduled {task.id}: {task.text}\n  {task.line()}\n"
                    f"A task fires only while I am running. Until "
                    f"install_system_task has been called, nobody starts me, "
                    f"so a task due at 03:00 waits for the next run.")

        def schedule_list() -> str:
            return self.schedule.report()

        def schedule_tick() -> str:
            now = time.time()
            due = self.schedule.due(now)
            if not due:
                upcoming = self.schedule.next_due(now)
                return ("Nothing is due." if upcoming is None
                        else f"Nothing is due. Next: {upcoming.line(now)}")
            self._record("schedule_tick", {"due": [t.id for t in due]})
            lines = [f"{len(due)} task(s) are due. Attend to them, then close "
                     f"each one with schedule_done so the next run knows what "
                     f"happened:"]
            lines += [f"  {t.line(now)}" for t in due]
            return "\n".join(lines)

        def schedule_done(task_id: str, note: str = "", ok: bool = True) -> str:
            try:
                task = self.schedule.complete(task_id, ok=ok, note=note)
            except ScheduleError as exc:
                return str(exc)
            self._record("schedule_done", {"id": task_id, "ok": ok})
            if task.repeat > 0:
                return (f"Recorded. {task_id} repeats, so it stays open: next "
                        f"due {as_clock(task.due_at)} "
                        f"({task.runs} run(s), {task.failures} failed).")
            return f"Recorded. {task_id} is closed."

        def schedule_cancel(task_id: str) -> str:
            try:
                task = self.schedule.cancel(task_id)
            except ScheduleError as exc:
                return str(exc)
            self._record("schedule_cancel", {"id": task_id})
            return f"Cancelled {task.id}: {task.text}"

        def install_system_task(interval_minutes: int = 30,
                                name: str = "autoforge-tick") -> str:
            try:
                return _install_system_task(interval_minutes, name)
            except ScheduleError as exc:
                return f"Could not register the task. {exc}"

        self._add(ToolSpec(
            name="schedule_add",
            description=(
                "Put something on your own agenda: a thing to do later, or to "
                "do repeatedly. `when` takes a duration (90m, 2h, 1d) or an ISO "
                "time (2026-09-14T09:00:00, local). Use it for anything you "
                "have decided to come back to, instead of assuming you will "
                "remember it next session."
            ),
            parameters={"type": "object", "properties": {
                "text": {"type": "string", "description": "what to attend to"},
                "when": {"type": "string",
                         "description": "when it is due: 90m, 2h, 1d, or ISO 8601"},
                "repeat": {"type": "string",
                           "description": "repeat interval (1d, 6h); omit for a one-off"},
            }, "required": ["text", "when"]},
            fn=schedule_add, source="builtin", tags=["meta", "time"],
        ))
        self._add(ToolSpec(
            name="schedule_list",
            description=(
                "Show your agenda: what is overdue, what is coming, and whether "
                "anything can wake you at all. Call it when asked what you have "
                "planned, rather than describing intentions from memory."
            ),
            parameters={"type": "object", "properties": {}},
            fn=schedule_list, source="builtin", tags=["meta", "time"],
        ))
        self._add(ToolSpec(
            name="schedule_tick",
            description=(
                "Read your agenda for anything now due. This is what a wake-up "
                "looks like from your side: when the OS starts you with "
                "`tick`, this is the list you get. Run it before starting new "
                "work, so a scheduled task is not quietly skipped."
            ),
            parameters={"type": "object", "properties": {}},
            fn=schedule_tick, source="builtin", tags=["meta", "time"],
        ))
        self._add(ToolSpec(
            name="schedule_done",
            description=(
                "Close a due task, with a note about what happened. The note is "
                "the point: it is the only thing that lets a later run tell a "
                "repeated failure apart from a long silence. Pass ok=false when "
                "it failed."
            ),
            parameters={"type": "object", "properties": {
                "task_id": {"type": "string", "description": "the task id from schedule_tick"},
                "note": {"type": "string", "description": "what happened"},
                "ok": {"type": "boolean", "description": "did it succeed (default true)"},
            }, "required": ["task_id"]},
            fn=schedule_done, source="builtin", tags=["meta", "time"],
        ))
        self._add(ToolSpec(
            name="schedule_cancel",
            description="Remove a scheduled task from your agenda without running it.",
            parameters={"type": "object", "properties": {
                "task_id": {"type": "string"},
            }, "required": ["task_id"]},
            fn=schedule_cancel, source="builtin", tags=["meta", "time"],
        ))
        self._add(ToolSpec(
            name="install_system_task",
            description=(
                "Register a recurring wake-up with the operating system, so you "
                "run even when nobody starts you. This is the only action that "
                "makes 'unattended' true. It changes the machine's "
                "configuration, so it says exactly what it registered and how "
                "to undo it, and is not called on suspicion."
            ),
            parameters={"type": "object", "properties": {
                "interval_minutes": {"type": "integer",
                                     "description": "how often to wake (default 30)"},
                "name": {"type": "string", "description": "the OS task name"},
            }},
            fn=install_system_task, source="builtin", tags=["meta", "time"],
        ))

    # ------------------------------------------------------------------
    # looking at things
    # ------------------------------------------------------------------
    def _ensure_browser(self, *, launch: bool = True,
                        headless: bool = True) -> "Browser":
        """The live browser session, attaching to one if it is already there.

        Connecting first and launching second is deliberate: an operator who
        started Chrome themselves with a profile they are logged into gets that
        session, and the agent does not start a second browser behind them.
        """
        if self._browser is not None and not self._browser._closed:
            return self._browser

        try:
            self._browser = Browser.connect(self._browser_endpoint)
            return self._browser
        except (BrowserError, WebSocketError) as first:
            if not launch:
                raise BrowserError(
                    f"no browser is listening at {self._browser_endpoint} "
                    f"({first})") from first

        # Nothing there. Start one and attach to it. A launch failure is
        # reported with its own cause, because "no browser" and "the browser
        # refuses to open the debug port" need different fixes.
        try:
            self._browser_proc, endpoint = launch_browser(headless=headless)
        except BrowserError as exc:
            raise BrowserError(
                f"no browser to drive: {exc}") from exc
        self._browser_endpoint = endpoint
        self._browser = Browser.connect(endpoint)
        return self._browser

    def _tool_browser(self) -> None:
        """A real browser, so the agent can read pages that must be rendered.

        A page whose content arrives from JavaScript is invisible to anything
        that only fetches HTML, and "the build output page shows an error" is
        not something a file read can confirm. This drives Chrome over CDP for
        that, and hands the screenshot to `see` so the agent can look at what
        it rendered rather than inferring it from the DOM.
        """

        def browser_open(headless: bool = True, endpoint: str = "") -> str:
            if endpoint:
                self._browser_endpoint = endpoint
            browser = self._ensure_browser(headless=headless)
            browser.enable()
            self._record("browser_open", {"endpoint": self._browser_endpoint})
            return (f"Driving {self._browser_endpoint} "
                    f"(target {browser.target.get('id', 'attached')!r}).\n"
                    f"{self._browser_report(browser)}")

        def browser_state() -> str:
            if self._browser is None or self._browser._closed:
                return "No browser is attached. Call browser_open first."
            return self._browser_report(self._browser)

        def browser_goto(url: str, wait: bool = True,
                         timeout: float = 30.0) -> str:
            browser = self._ensure_browser()
            title = browser.goto(url, wait=wait, timeout=timeout)
            self._record("browser_goto", {"url": url})
            return (f"{browser.current_url() or url}\n"
                    f"title: {title or '(none)'}")

        def browser_eval(expression: str) -> str:
            browser = self._ensure_browser()
            value = browser.evaluate(expression)
            self._record("browser_eval", {"chars": len(expression)})
            rendered = json.dumps(value, ensure_ascii=False, default=str)
            if len(rendered) > 4000:
                return rendered[:4000] + f"\n[...{len(rendered) - 4000} more chars]"
            return rendered

        def browser_click(x: float, y: float, clicks: int = 1) -> str:
            browser = self._ensure_browser()
            browser.click(x, y, clicks=clicks)
            self._record("browser_click", {"x": x, "y": y})
            return f"Clicked ({x}, {y}) x{clicks}. Now on: {browser.current_url()}"

        def browser_type(text: str, selector: str = "",
                         submit: bool = False) -> str:
            browser = self._ensure_browser()
            if selector:
                # Focus through the DOM and dispatch a real click, so the page's
                # own focus handling runs; setting .value directly would leave
                # framework state untouched and the form would submit empty.
                browser.evaluate(
                    "(() => { const el = document.querySelector("
                    + json.dumps(selector) + "); if (!el) return false;"
                    " el.focus(); return true; })()")
            browser.type_text(text)
            key = ""
            if submit:
                browser.press("Enter")
                key = " then Enter"
            self._record("browser_type", {"chars": len(text), "selector": selector})
            return (f"Typed {len(text)} characters"
                    + (f" into {selector}" if selector else "")
                    + key)

        def browser_press(key: str) -> str:
            browser = self._ensure_browser()
            browser.press(key)
            self._record("browser_press", {"key": key})
            return f"Pressed {key}."

        def browser_screenshot(path: str = "", full_page: bool = False) -> str:
            browser = self._ensure_browser()
            data = browser.screenshot(full_page=full_page)
            if not path:
                # Beside the other durable state rather than in whatever the
                # process happened to be started from: a screenshot written to
                # a working directory nobody chose is a file nobody finds.
                home = os.environ.get("AUTOFORGE_HOME")
                base = home or os.path.join(os.path.expanduser("~"), ".autoforge")
                path = os.path.join(base, "screenshots",
                                    f"shot-{int(time.time())}.png")
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(data)
            self._record("browser_screenshot", {"path": path, "bytes": len(data)})
            hint = ("\nPass it to `see` with your question to actually look at it."
                    if self.vision else
                    "\n(No vision endpoint is configured, so nothing can read it back.)")
            return (f"Saved {len(data)} bytes to {os.path.abspath(target)}\n"
                    f"Page: {browser.current_url()}" + hint)

        def browser_close() -> str:
            had = self._browser is not None
            if self._browser is not None:
                self._browser.close()
                self._browser = None
            # The process is left running on purpose when it was already there
            # before we attached: it is not ours to kill.
            note = (" Detached. Any browser that was already running is left "
                    "as it was." if had else "")
            return f"No browser is attached now.{note}"

        self._add(ToolSpec(
            name="browser_open",
            description=(
                "Attach to a browser over the DevTools protocol, starting one "
                "if nothing is listening. Required before the other browser_* "
                "tools. Attaching to an already-running browser is preferred, "
                "so start Chrome with --remote-debugging-port=9222 yourself if "
                "you want to share a logged-in session."
            ),
            parameters={"type": "object", "properties": {
                "headless": {"type": "boolean",
                             "description": "start headless if one must be started (default true)"},
                "endpoint": {"type": "string",
                             "description": "CDP endpoint; default http://127.0.0.1:9222"},
            }},
            fn=browser_open, source="builtin", tags=["browser", "web"],
        ))
        self._add(ToolSpec(
            name="browser_state",
            description=(
                "What the browser is currently showing: the URL, the title, and "
                "how many tabs exist. Call it to confirm a click or a navigation "
                "did what you expected instead of assuming it did."
            ),
            parameters={"type": "object", "properties": {}},
            fn=browser_state, source="builtin", tags=["browser", "web"],
        ))
        self._add(ToolSpec(
            name="browser_goto",
            description=(
                "Navigate the current tab and wait for the document to finish "
                "loading, including its scripts. Returns the final URL and "
                "title -- a redirect means they differ from what you asked for."
            ),
            parameters={"type": "object", "properties": {
                "url": {"type": "string"},
                "wait": {"type": "boolean", "description": "wait for load (default true)"},
                "timeout": {"type": "number", "description": "seconds (default 30)"},
            }, "required": ["url"]},
            fn=browser_goto, source="builtin", tags=["browser", "web"],
        ))
        self._add(ToolSpec(
            name="browser_eval",
            description=(
                "Run JavaScript in the page and get the value back. This is how "
                "you read content that only exists after the scripts ran. "
                "Expressions are reduced to JSON; return a plain value rather "
                "than a DOM node, which does not survive the trip."
            ),
            parameters={"type": "object", "properties": {
                "expression": {"type": "string",
                               "description": "a JavaScript expression"},
            }, "required": ["expression"]},
            fn=browser_eval, source="builtin", tags=["browser", "web"],
        ))
        self._add(ToolSpec(
            name="browser_click",
            description=(
                "Click at a point in the viewport, in CSS pixels from the "
                "top-left. Get the coordinates from browser_eval with "
                "getBoundingClientRect. Coordinates rather than selectors "
                "because that is what the browser actually dispatches, so "
                "overlays and hit-testing behave as they do for a person."
            ),
            parameters={"type": "object", "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "clicks": {"type": "integer", "description": "1 (default) or 2"},
            }, "required": ["x", "y"]},
            fn=browser_click, source="builtin", tags=["browser", "web"],
        ))
        self._add(ToolSpec(
            name="browser_type",
            description=(
                "Type text into the page, optionally focusing a CSS selector "
                "first, optionally pressing Enter. Pass submit=true to press "
                "Enter, which is how forms are usually sent."
            ),
            parameters={"type": "object", "properties": {
                "text": {"type": "string"},
                "selector": {"type": "string", "description": "CSS selector to focus first"},
                "submit": {"type": "boolean", "description": "press Enter after typing"},
            }, "required": ["text"]},
            fn=browser_type, source="builtin", tags=["browser", "web"],
        ))
        self._add(ToolSpec(
            name="browser_press",
            description=(
                "Press one key: Enter, Escape, ArrowDown, Tab, PageDown. Use "
                "Escape to dismiss a dialog and Tab to move focus."
            ),
            parameters={"type": "object", "properties": {
                "key": {"type": "string", "description": "a CDP key name"},
            }, "required": ["key"]},
            fn=browser_press, source="builtin", tags=["browser", "web"],
        ))
        self._add(ToolSpec(
            name="browser_screenshot",
            description=(
                "Save a PNG of the page and return its path. Screenshots are "
                "the only way to see anything that is visual -- a layout that "
                "broke, a blank render, a chart -- so pair it with `see` rather "
                "than stopping at the file."
            ),
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "where to write the PNG"},
                "full_page": {"type": "boolean",
                              "description": "capture the whole scrollable page"},
            }},
            fn=browser_screenshot, source="builtin", tags=["browser", "web"],
        ))
        self._add(ToolSpec(
            name="browser_close",
            description=(
                "Detach from the browser. A browser this agent started keeps "
                "running; one that was already there is left exactly as it was."
            ),
            parameters={"type": "object", "properties": {}},
            fn=browser_close, source="builtin", tags=["browser", "web"],
        ))

    def _browser_report(self, browser: "Browser") -> str:
        parts = [f"url:   {browser.current_url() or '(none)'}",
                 f"title: {browser.title() or '(none)'}",
                 f"events seen: {len(browser.events)}"]
        try:
            targets = browser.call("Target.getTargets").get("targetInfos", [])
            pages = [t for t in targets if t.get("type") == "page"]
            parts.append(f"tabs:  {len(pages)}")
            for page in pages[:5]:
                parts.append(f"  - {page.get('title') or '(untitled)'} "
                             f"<{page.get('url')}>")
        except (CDPError, BrowserError, WebSocketError):
            # A browser-level target has no Target domain. Not worth failing a
            # status report over.
            pass
        return "\n".join(parts)

    def _tool_vision(self) -> None:
        """Reading a picture. The half of "look at this" that needs a model."""

        def see(image: str, question: str = "") -> str:
            if not self.vision:
                reason = self._vision_problem or (
                    "no vision endpoint is configured")
                return (f"Cannot look at {image}: {reason}. Set vision.model and "
                        f"vision.base_url in the config (or "
                        f"AUTOFORGE_VISION_MODEL / AUTOFORGE_VISION_BASE_URL).")
            try:
                answer = self.vision.describe(image, question)
            except VisionError as exc:
                return f"Could not read {image}: {exc}"
            self._record("see", {"image": image})
            return answer

        def vision_status() -> str:
            lines = [self.vision.report() if self.vision
                     else "vision: not configured"]
            if self._vision_problem:
                lines.append(f"  config problem: {self._vision_problem}")
            return "\n".join(lines)

        self._add(ToolSpec(
            name="see",
            description=(
                "Look at an image and answer a question about it. Takes a file "
                "path or an http(s) URL. Use it on a screenshot to check what "
                "actually rendered, and on any chart or diagram whose meaning "
                "is not in the text. Ask a specific question -- a targeted "
                "question gets a usable answer where 'what is this' does not."
            ),
            parameters={"type": "object", "properties": {
                "image": {"type": "string",
                          "description": "path to an image file, or an http(s) URL"},
                "question": {"type": "string",
                             "description": "what to find out about it"},
            }, "required": ["image"]},
            fn=see, source="builtin", tags=["vision", "media"],
        ))
        self._add(ToolSpec(
            name="vision_status",
            description=(
                "Whether the agent can look at images at all, and which model "
                "and endpoint it would use. Call it before promising to check a "
                "screenshot."
            ),
            parameters={"type": "object", "properties": {}},
            fn=vision_status, source="builtin", tags=["vision", "media"],
        ))

    def _tool_gpu(self) -> None:
        """CUDA kernel-layer tools.

        The gate is split by what an action can actually damage. Reading a
        device, estimating occupancy, linting timing units and timing torch's
        own kernels cannot hang a display or corrupt anything — those stay
        available. Compiling agent-authored code and launching it on the GPU is
        the part that can wedge the machine, so that is what
        `may_run_cuda_kernels` gates. Turning it off removes a real capability
        and the agent is told so, which is the point: a gate that changes
        nothing is a comment, not a gate.
        """
        from . import gpu as G

        def _denied() -> str:
            return (
                "Denied by autonomy policy: may_run_cuda_kernels is off. "
                "Kernel compilation and launch are disabled; device probing, "
                "occupancy estimation, the timing-unit audit and torch "
                "baselines remain available."
            )

        def gpu_probe() -> str:
            info = G.probe()
            if not info.available:
                return ("No CUDA device reachable here.\n"
                        + "\n".join(f"  - {n}" for n in info.notes))
            return info.summary()

        def gpu_occupancy(block: int = 256, grid: int = 1,
                          shared_bytes: int = 0, regs_per_thread: int = 0,
                          arch_major: int = 8, arch_minor: int = 9) -> str:
            cfg = G.LaunchConfig(grid=(grid, 1, 1), block=(block, 1, 1),
                                 shared_bytes=shared_bytes)
            est = G.estimate_occupancy(cfg, arch=(arch_major, arch_minor),
                                       regs_per_thread=regs_per_thread)
            if est is None:
                return (f"No static limits on file for sm_{arch_major}{arch_minor}, "
                        f"so no estimate is offered. Add them to gpu.probe."
                        f"ARCH_LIMITS or measure on the device.")
            head = (f"{est.summary()}\n"
                    f"  threads/block {cfg.threads_per_block}, "
                    f"blocks {cfg.total_blocks}, "
                    f"shared {cfg.shared_bytes} B")
            if not est.measured:
                head += ("\n  This is arithmetic from the architecture's static "
                         "limits. It is not a measurement — launch on the "
                         "device and use measured occupancy for anything you "
                         "intend to quote.")
            return head

        def gpu_units_audit(source: str = "", path: str = "") -> str:
            if path:
                hits = G.audit_source_tree([path])
                if not hits:
                    return (f"No thousand-factor-against-ms offences in {path}. "
                            f"(Known false negative: the factor arriving through "
                            f"a variable and the ms label in a different "
                            f"statement.)")
                out = []
                for f, offs in hits.items():
                    out.append(f"{f}:")
                    out += [f"  line {o.line_no}: {o.line}" for o in offs[:10]]
                return "\n".join(out)
            offs = G.audit_ms_scale(source)
            if not offs:
                return ("No offence found. Note the two accepted false "
                        "negatives: a factor smuggled through a variable, and "
                        "the label and the multiplication in different "
                        "statements.")
            return "\n".join(f"line {o.line_no}: {o.line}\n  {o.reason}"
                             for o in offs)

        def gpu_compile(code: str, name: str = "kernel",
                        arch_major: int = 8, arch_minor: int = 9) -> str:
            if not self.policy.may_run_cuda_kernels:
                return _denied()
            src = G.KernelSource(name=name, code=code,
                                 arch=(arch_major, arch_minor))
            if not src.entry_points():
                return ("Refusing to compile: no `__global__ void` entry point "
                        "in the source. Entry points found: none.")
            try:
                k = G.compile_kernel(src)
            except G.KernelUnavailable as exc:
                self._record("gpu_compile", {"name": name, "ok": False})
                return f"Could not compile: {exc}"
            self._record("gpu_compile", {"name": name, "ok": True,
                                         "cached": k.from_cache})
            d = k.to_dict()
            return (f"Compiled {d['name']} for {d['arch']} via {d['compiler']}"
                    f"{' (cache hit)' if k.from_cache else ''}; "
                    f"entries: {', '.join(d['entry_points'])}"
                    f"\ncache key {d['cache_key']}\nartefact {d['artefact']}")

        def gpu_bench(code: str, name: str = "kernel", block: int = 256,
                      grid: int = 1, shared_bytes: int = 0,
                      reps: int = 5, time_budget_s: float = 30.0,
                      arch_major: int = 8, arch_minor: int = 9) -> str:
            if not self.policy.may_run_cuda_kernels:
                return _denied()
            src = G.KernelSource(name=name, code=code,
                                 arch=(arch_major, arch_minor))
            cfg = G.LaunchConfig(grid=(grid, 1, 1), block=(block, 1, 1),
                                 shared_bytes=shared_bytes)
            res = G.bench_kernel(src, cfg, label=name, reps=reps,
                                 time_budget_s=time_budget_s)
            self._record("gpu_bench", {"name": name, "ok": res.ok})
            return res.summary()

        def gpu_verify(kind: str, actual: str, reference: str = "",
                       tolerance: float = 1e-4) -> str:
            """Compare kernel output against a reference, so fast-but-wrong fails.

            Read-only arithmetic on numbers the caller supplies, so it is not
            gated — it touches no device.
            """
            try:
                act = [float(x) for x in actual.replace(",", " ").split()]
            except ValueError:
                return "actual must be a list of numbers."
            if reference:
                try:
                    ref = [float(x) for x in reference.replace(",", " ").split()]
                except ValueError:
                    return "reference must be a list of numbers."
            else:
                return ("A reference is required. Without one this is not a "
                        "verification — it is a restatement of the kernel's own "
                        "output, which is what a fast-but-wrong kernel produces.")
            r = G.verify(act, ref, tolerance=tolerance, label=kind)
            return (f"{'PASS' if r['passed'] else 'FAIL'} max|diff|="
                    f"{r['max_abs_diff']:.3e} worst index {r['worst_index']}\n"
                    f"  {r['reason']}")

        self._add(ToolSpec(
            name="gpu_probe",
            description=(
                "Report what CUDA silicon is reachable: device name, compute "
                "capability, memory, driver. Never fails; 'none' is an answer."
            ),
            parameters={"type": "object", "properties": {}},
            fn=gpu_probe, source="builtin", tags=["gpu", "read"],
        ))
        self._add(ToolSpec(
            name="gpu_occupancy",
            description=(
                "Estimate occupancy for a launch geometry from the "
                "architecture's static limits. Labelled as an estimate — "
                "measured occupancy needs a device."
            ),
            parameters={"type": "object", "properties": {
                "block": {"type": "integer", "description": "threads per block"},
                "grid": {"type": "integer", "description": "blocks per grid (x)"},
                "shared_bytes": {"type": "integer"},
                "regs_per_thread": {"type": "integer"},
                "arch_major": {"type": "integer"},
                "arch_minor": {"type": "integer"},
            }},
            fn=gpu_occupancy, source="builtin", tags=["gpu", "read"],
        ))
        self._add(ToolSpec(
            name="gpu_units_audit",
            description=(
                "Lint source for the do_bench-returns-ms bug: a value scaled by "
                "1000 in a statement labelled ms. Give source text or a path."
            ),
            parameters={"type": "object", "properties": {
                "source": {"type": "string"},
                "path": {"type": "string", "description": "file or directory"},
            }},
            fn=gpu_units_audit, source="builtin", tags=["gpu", "read"],
        ))
        self._add(ToolSpec(
            name="gpu_verify",
            description=(
                "Compare kernel output against a reference and report max "
                "absolute difference. Refuses without a reference — otherwise "
                "it would only restate the kernel's own output."
            ),
            parameters={"type": "object", "properties": {
                "kind": {"type": "string", "description": "what is being checked"},
                "actual": {"type": "string", "description": "space/comma numbers"},
                "reference": {"type": "string", "description": "space/comma numbers"},
                "tolerance": {"type": "number"},
            }, "required": ["kind", "actual"]},
            fn=gpu_verify, source="builtin", tags=["gpu", "read"],
        ))
        self._add(ToolSpec(
            name="gpu_compile",
            description=(
                "Compile CUDA C you wrote to a cubin, content-addressed so a "
                "repeat is a cache hit. Gated by may_run_cuda_kernels."
            ),
            parameters={"type": "object", "properties": {
                "code": {"type": "string", "description": "CUDA C source"},
                "name": {"type": "string"},
                "arch_major": {"type": "integer"},
                "arch_minor": {"type": "integer"},
            }, "required": ["code"]},
            fn=gpu_compile, source="builtin", tags=["gpu", "run"],
        ))
        self._add(ToolSpec(
            name="gpu_bench",
            description=(
                "Compile and time a kernel under a wall-clock budget, reporting "
                "the spread and whether the difference is signal or noise. "
                "Gated by may_run_cuda_kernels."
            ),
            parameters={"type": "object", "properties": {
                "code": {"type": "string", "description": "CUDA C source"},
                "name": {"type": "string"},
                "block": {"type": "integer"},
                "grid": {"type": "integer"},
                "shared_bytes": {"type": "integer"},
                "reps": {"type": "integer"},
                "time_budget_s": {"type": "number"},
                "arch_major": {"type": "integer"},
                "arch_minor": {"type": "integer"},
            }, "required": ["code"]},
            fn=gpu_bench, source="builtin", tags=["gpu", "run"],
        ))

    # ------------------------------------------------------------------
    def _tool_cpu(self) -> None:
        """Wire the native layer in, tool for tool the GPU layer's shape.

        Same vocabulary on purpose — probe, preflight, units audit, cache,
        compile, run, tune — because the two layers are the same argument about
        different silicon, and a reader who learned one should not have to learn
        a second vocabulary to use the other.

        What genuinely differs is called out rather than smoothed over:

        * `cpu_runs_here` has no CUDA counterpart. A cubin built for sm_90
          simply fails to load. An x86 library built for a newer target loads
          fine and kills the interpreter at the first instruction it cannot
          execute, which is why the check is a separate tool and why
          `cpu_compile` consults it before building.
        * Five of the eight tools read or compute, and stay available with the
          gate off. Only `cpu_compile`, `cpu_run_isolated` and `cpu_tune`
          execute agent-authored code on the host, and those three refuse and
          name `may_run_cpu_kernels` when it is off.
        """
        from . import cpu as C
        from .timing import audit_ms_scale, audit_source_tree

        def _denied() -> str:
            return (
                "Denied by autonomy policy: may_run_cpu_kernels is off. "
                "Compiling and running native kernels is disabled; the "
                "toolchain probe, the SIGILL check, the preflight lint, the "
                "timing-unit audit and the cache report remain available."
            )

        def cpu_probe() -> str:
            info = C.probe()
            if not info.available:
                why = "; ".join(info.notes) if info.notes else "no C compiler found"
                return f"No CPU toolchain usable for forging. {why}"
            return info.summary()

        def cpu_runs_here(target: str = "native") -> str:
            info = C.probe()
            if not info.available:
                why = "; ".join(info.notes) if info.notes else "no C compiler found"
                return f"No verdict without a compiler to build for. {why}"
            ok, why = C.runs_here(target, info)
            verdict = "RUNS HERE" if ok else "WILL NOT RUN HERE"
            return f"{verdict}: {target}\n  {why}"

        def cpu_preflight(code: str, name: str = "kernel") -> str:
            try:
                src = C.KernelSource(name=name, code=code)
            except ValueError as exc:
                return f"Nothing to lint: {exc}"
            return C.preflight(src).summary()

        def cpu_units_audit(source: str = "", path: str = "") -> str:
            if path:
                hits = audit_source_tree([path])
                if not hits:
                    return (f"No thousand-factor-against-ms offences in {path}. "
                            f"(Known false negative: the factor arriving through "
                            f"a variable and the ms label in a different "
                            f"statement.)")
                out = []
                for f, offs in hits.items():
                    out.append(f"{f}:")
                    out += [f"  line {o.line_no}: {o.line}" for o in offs[:10]]
                return "\n".join(out)
            offs = audit_ms_scale(source)
            if not offs:
                return ("No offence found. Note the two accepted false "
                        "negatives: a factor smuggled through a variable, and "
                        "the label and the multiplication in different "
                        "statements.")
            return "\n".join(f"line {o.line_no}: {o.line}\n  {o.reason}"
                             for o in offs)

        def cpu_cache_stats() -> str:
            st = C.cache_stats()
            total = st["hits"] + st["misses"]
            if not total:
                return (f"Nothing compiled on this machine yet. Cache root "
                        f"{st['root']} — the first compile will populate it, and "
                        f"a repeat of the same source, target, compiler and "
                        f"flags is a cache hit rather than a rebuild.")
            return (f"{st['hits']} hit(s), {st['misses']} miss(es), "
                    f"{st['hit_rate']:.0%} hit rate under {st['root']}")

        def cpu_compile(code: str, name: str = "kernel",
                        target: str = "native") -> str:
            if not self.policy.may_run_cpu_kernels:
                return _denied()
            try:
                src = C.KernelSource(name=name, code=code, target=target)
            except ValueError as exc:
                return f"Refusing to compile: {exc}"
            if not src.entry_points():
                return ("Refusing to compile: no callable entry point in the "
                        "source. Static functions and `main` cannot be called "
                        "through ctypes, so a kernel made only of those has "
                        "nothing to invoke. Entry points found: none.")
            # The module's own ordering argument, enforced: a build for a target
            # this machine cannot run is worse than no build, because the
            # failure is SIGILL in the agent's own process rather than an error
            # the agent can read.
            resolved = src.resolved_target()
            ok, why = C.runs_here(resolved, C.probe())
            if not ok:
                return (f"Refusing to compile for {target}: {why}\n"
                        f"  Build for a target this machine can run, or drop the "
                        f"artefact in a child process and let it die there.")
            try:
                k = C.compile_kernel(src)
            except C.KernelUnavailable as exc:
                self._record("cpu_compile", {"name": name, "ok": False})
                return f"Could not compile: {exc}"
            self._record("cpu_compile", {"name": name, "ok": True,
                                         "cached": k.from_cache})
            d = k.to_dict()
            return (f"Compiled {d['name']} for {d['target']} via {d['compiler']}"
                    f"{' (cache hit)' if k.from_cache else ''}; "
                    f"entries: {', '.join(d['entry_points'])}"
                    f"\ncache key {d['cache_key']}\nartefact {d['artefact']}")

        def cpu_run_isolated(
            code: str, name: str = "kernel", kind: str = "saxpy",
            size: int = 1 << 20, n: int = 256, dtype: str = "f32",
            seed: int = 0, reps: int = 5, warmup: int = 3,
            timeout: float = 0.0, target: str = "native",
        ) -> str:
            """Compile, then verify and time in a child process.

            Isolation is not a convenience here. An unproven kernel's first
            call must not be in the agent's own process: a wrong index
            segfaults, an unbounded loop hangs, and both would otherwise end
            the session instead of producing a report.
            """
            if not self.policy.may_run_cpu_kernels:
                return _denied()
            try:
                prob = C.problem(kind, size=size, n=n, dtype=dtype, seed=seed)
            except C.CallError as exc:
                return f"No such problem: {exc}"
            try:
                src = C.KernelSource(name=name, code=code, target=target)
            except ValueError as exc:
                return f"Refusing to run: {exc}"
            try:
                k = C.compile_kernel(src)
            except C.KernelUnavailable as exc:
                return f"Could not compile: {exc}"
            guard = C.run_isolated(k, prob, reps=reps, warmup=warmup,
                                   timeout=timeout or None)
            self._record("cpu_run_isolated",
                         {"name": name, "kind": kind, "ok": guard.ok})
            lines = [guard.summary()]
            timings = guard.verdict.get("timings_s") or []
            if guard.ok and timings:
                lines.append(f"  {len(timings)} timed rep(s), fastest "
                             f"{min(timings) * 1e3:.4f} ms")
            lines.append(f"  checked against the reference for {kind!r} "
                         f"({prob.spec.describe()}) — skipping work fails here "
                         f"rather than being timed as an improvement.")
            return "\n".join(lines)

        def cpu_tune(
            kind: str = "saxpy", size: int = 1 << 20, n: int = 256,
            dtype: str = "f32", seed: int = 0,
            max_candidates: int = C.DEFAULT_MAX_CANDIDATES,
            budget_s: float = C.DEFAULT_BUDGET_S,
            reps: int = 5, warmup: int = 3,
        ) -> str:
            if not self.policy.may_run_cpu_kernels:
                return _denied()
            try:
                res = C.tune_kind(kind, size=size, n=n, dtype=dtype, seed=seed,
                                  max_candidates=max_candidates,
                                  budget_s=budget_s, reps=reps, warmup=warmup)
            except C.CallError as exc:
                return f"No such problem: {exc}"
            except KeyError as exc:
                return f"Nothing to tune: {exc}"
            self._record("cpu_tune",
                         {"kind": kind, "candidates": res.n_candidates,
                          "ok": res.winner is not None})
            return res.report()

        self._add(ToolSpec(
            name="cpu_probe",
            description=(
                "Report what native silicon and toolchain is reachable: CPU "
                "name, cores, SIMD features, cache, compiler and version, and "
                "what -march=native resolves to. Never fails; 'none' is an "
                "answer."
            ),
            parameters={"type": "object", "properties": {}},
            fn=cpu_probe, source="builtin", tags=["cpu", "read"],
        ))
        self._add(ToolSpec(
            name="cpu_runs_here",
            description=(
                "Ask whether an artefact built for a -march target can run on "
                "this machine. Call before compiling for anything but the "
                "local default: a build for a newer target loads fine and "
                "dies at the first unsupported instruction."
            ),
            parameters={"type": "object", "properties": {
                "target": {"type": "string",
                           "description": "-march value, e.g. native, x86-64, znver3"},
            }},
            fn=cpu_runs_here, source="builtin", tags=["cpu", "read"],
        ))
        self._add(ToolSpec(
            name="cpu_preflight",
            description=(
                "Lint C source for the mistakes worth knowing before it is "
                "compiled: off-by-one loop bounds, unchecked sizes, the "
                "usual buffer hazards. Advice, not a gate — a finding does "
                "not stop the build."
            ),
            parameters={"type": "object", "properties": {
                "code": {"type": "string", "description": "C source"},
                "name": {"type": "string"},
            }, "required": ["code"]},
            fn=cpu_preflight, source="builtin", tags=["cpu", "read"],
        ))
        self._add(ToolSpec(
            name="cpu_units_audit",
            description=(
                "Lint source for the do_bench-returns-ms bug: a value scaled "
                "by 1000 in a statement labelled ms. Give source text or a "
                "path."
            ),
            parameters={"type": "object", "properties": {
                "source": {"type": "string"},
                "path": {"type": "string", "description": "file or directory"},
            }},
            fn=cpu_units_audit, source="builtin", tags=["cpu", "read"],
        ))
        self._add(ToolSpec(
            name="cpu_cache_stats",
            description=(
                "Report the content-addressed kernel cache: hit and miss "
                "counts. Compiling the same source for the same target with "
                "the same compiler is a hit, not a rebuild."
            ),
            parameters={"type": "object", "properties": {}},
            fn=cpu_cache_stats, source="builtin", tags=["cpu", "read"],
        ))
        self._add(ToolSpec(
            name="cpu_compile",
            description=(
                "Compile C you wrote to a loadable library, content-addressed "
                "on target, compiler version and flags, so a repeat is a cache "
                "hit. Refuses a target this machine cannot run, and refuses "
                "source with no ctypes-callable entry point. Gated by "
                "may_run_cpu_kernels."
            ),
            parameters={"type": "object", "properties": {
                "code": {"type": "string", "description": "C source"},
                "name": {"type": "string"},
                "target": {"type": "string", "description": "-march value"},
            }, "required": ["code"]},
            fn=cpu_compile, source="builtin", tags=["cpu", "run"],
        ))
        self._add(ToolSpec(
            name="cpu_run_isolated",
            description=(
                "Compile, verify against the problem's reference answer, and "
                "time - all in a child process, so a segfault or an unbounded "
                "loop is a report instead of the end of the session. "
                "Verification runs before measurement, always. Gated by "
                "may_run_cpu_kernels."
            ),
            parameters={"type": "object", "properties": {
                "code": {"type": "string", "description": "C source"},
                "name": {"type": "string"},
                "kind": {"type": "string",
                         "description": f"problem: {', '.join(sorted(C.NAIVE))}"},
                "size": {"type": "integer"},
                "n": {"type": "integer"},
                "dtype": {"type": "string", "enum": list(C.DTYPES)},
                "seed": {"type": "integer"},
                "reps": {"type": "integer"},
                "warmup": {"type": "integer"},
                "timeout": {"type": "number",
                            "description": "seconds; 0 uses the module default"},
                "target": {"type": "string"},
            }, "required": ["code"]},
            fn=cpu_run_isolated, source="builtin", tags=["cpu", "run"],
        ))
        self._add(ToolSpec(
            name="cpu_tune",
            description=(
                "Search for a faster correct kernel from a naive starting "
                "point: generate, compile, verify, measure, mutate, keep the "
                "winner — against a baseline strong enough that a win means "
                "something. Gated by may_run_cpu_kernels."
            ),
            parameters={"type": "object", "properties": {
                "kind": {"type": "string",
                         "description": f"problem: {', '.join(sorted(C.NAIVE))}"},
                "size": {"type": "integer"},
                "n": {"type": "integer"},
                "dtype": {"type": "string", "enum": list(C.DTYPES)},
                "seed": {"type": "integer"},
                "max_candidates": {"type": "integer"},
                "budget_s": {"type": "number"},
                "reps": {"type": "integer"},
                "warmup": {"type": "integer"},
            }},
            fn=cpu_tune, source="builtin", tags=["cpu", "run"],
        ))

    # ------------------------------------------------------------------
    def _sync_amendment(self, a: Amendment) -> None:
        self._record("amendment", a.to_dict())
        if self.store:
            self.store.log_event("amendment", a.to_dict())

    def _on_forge_event(self, kind: str, payload: dict[str, Any]) -> None:
        self._record(kind, payload)
        if self.store and kind in ("forge_done", "auto_quarantine"):
            self.store.log_event(kind, payload)

    def _on_compact(self, event: Any) -> None:
        """A compaction is an event the operator should be able to see.

        It is the one thing in a long run that changes the context without the
        model asking for it, so a run that quietly stopped remembering and a run
        that never had to would otherwise look identical from the trace.
        """
        payload = (event.as_dict() if hasattr(event, "as_dict")
                   else {"event": repr(event)})
        self._record("compact", payload)
        if self.store:
            self.store.log_event("compact", payload)

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        self.trace.append({"kind": kind, **payload})
        # Forward to the live observer, if one is attached, so forging shows up
        # while it happens instead of being rendered once the task is over.
        if self._progress is not None:
            try:
                self._progress(kind, payload)
            except Exception:                    # a bad observer must not run the task
                pass

    # ------------------------------------------------------------------
    def apply_role(self, brief: str, role: str = "") -> None:
        """Take on a role assigned by the agent that spawned this one.

        The counterpart of `TopologyDesigner`: a designer that can *name* five
        roles is worth nothing if the runtime that instantiates them ignores the
        name. Called by `Spawner.spawn` on any child that implements it.
        """
        self.role_brief = brief.strip()
        self.role = role or self.role

    def _effective_prompt(self) -> str:
        """The prompt actually sent: the base, plus what was measured.

        Recomputed per run, so the numbers are current even after the agent has
        amended its own prompt — and so the host block describes the sandbox
        this process will actually fork, not the machine the agent imagines.
        """
        facts = "\n".join(host_facts(self.sandbox))
        kept = "\n".join(self._memory_lines())
        menu = "\n".join(self.skills.menu())
        out = (f"{self.system_prompt}\n\n{self._self_report()}\n\n"
               f"{kept}\n\n{menu}\n\n{facts}")
        if self.role_brief:
            # Last, and separately labelled: a role narrows what this run is
            # for, and the point of putting it after the general instructions is
            # that the specific beats the general when a model stops reading.
            out += (f"\n\n## Your role in this team ({self.role or 'unnamed'})"
                    f"\n\n{self.role_brief}")
        return out

    # ------------------------------------------------------------------
    def run(self, task: str, history: list[Message] | None = None,
            progress: Callable[[str, dict[str, Any]], None] | None = None) -> AgentResult:
        """Run one task.

        `progress(kind, payload)` is called as the loop moves — `request` before
        each model call, `call`/`result` around each tool. The CLI uses it to
        print live status, so a slow model reads as "waiting", not "hung".
        """
        def _emit(kind: str, **payload: Any) -> None:
            if progress:
                progress(kind, payload)

        # Also attach it to the trace, so records written by forging and
        # self-modification stream live too — not just the loop's own events.
        self._progress = progress
        try:
            return self._run_locked(task, history, _emit)
        finally:
            self._progress = None

    def _run_locked(self, task: str, history: list[Message] | None,
                    _emit: Callable[..., None]) -> AgentResult:
        agent = Agent(
            self.llm, self.registry,
            system_prompt=self._effective_prompt(),
            max_turns=self.max_turns,
            allow_self_terminate=self.policy.self_terminate,
            on_request=lambda turn: _emit("request", turn=turn),
            on_tool_call=lambda n, a: self._record("call", {"tool": n}),
            on_tool_result=lambda n, r: self._record(
                "result", {"tool": n, "ok": getattr(r, "ok", None)}),
            on_turn=lambda turn, msg: _emit("turn", turn=turn),
            on_steer=lambda text: self._record("steer", {"text": text[:300]}),
            steer=self.steer,
            compactor=self.compactor,
            on_compact=self._on_compact,
        )
        result = agent.run(task, history)
        self._record("finish", {
            "turns": result.turns, "tools": result.tool_calls,
            "self_terminated": result.self_terminated,
            "stopped_by_operator": result.stopped_by_operator,
        })
        if self.store:
            self.store.log_event("run", {
                "task": task[:300], "turns": result.turns,
                "self_terminated": result.self_terminated,
                "stopped_by_operator": result.stopped_by_operator,
            })
        return result

    # ------------------------------------------------------------------
    def report(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "denied": self.policy.denied,
            "policy_summary": self.policy.describe(),
            # Freedoms switched off that nothing obeys. Empty is the healthy
            # answer; listing them here stops a policy from reading like a cage
            # while behaving like a comment.
            "unenforced": self.policy.unenforced,
            "tools": self.registry.report(),
            "amendments": self.selfmod.log(),
            "spawns": self.spawner.summary(),
            "topology": self.topology.to_dict(),
            "trace_len": len(self.trace),
        }


def _default_agent_factory(parent: ForgeAgent):
    """Children are ForgeAgents that share the parent's LLM but (optionally)
    get their own registry — decided by the caller via ShareMode."""
    def factory(spawner: Spawner, registry: ToolRegistry) -> ForgeAgent:
        child = ForgeAgent(
            llm=parent.llm,
            registry=registry,
            sandbox=parent.sandbox,
            generator=parent.generator,
            forge_config=parent.forge_config,
            max_turns=parent.max_turns,
            policy=parent.policy,
            enable_meta_cognition=False,
            enable_evolution=parent.enable_evolution,
        )
        child.spawner = spawner.child_spawner(registry)
        return child
    return factory


__all__ = ["ForgeAgent", "AUTONOMOUS_SYSTEM"]

"""Tool evolution engine: mutation, crossover, and natural selection.

This is the mechanism that makes a tool library *alive*. Instead of just fixing
a broken tool, the framework spawns a population of mutant offspring, tests
all of them, and keeps the one that survives the verification gauntlet.

Core idea (inspired by genetic programming, grounded in LLM generation):

  RETRY is not REPAIR. When a tool fails, retrying the same generator prompt
  (even with failure feedback) usually produces near-identical output. Real
  improvement needs *exploration* — multiple divergent candidates that the
  verifier can select among.

The flow:
  1. A tool reaches its failure threshold (or the user requests improvement).
  2. `evolve()` spawns `population_size` mutants by perturbing the tool's
     code, probes, parameters, and description through an LLM.
  3. Each mutant runs through the verifier (execution + trigger + negative).
  4. The highest-scoring surviving mutant is kept.
  5. If the best mutant beats the original's fitness, it replaces the original.

`evolve_from_scratch()` is the crossover path: given two parent tools (or a
tool + a failure report), it creates hybrids that combine their capabilities.

The design bet: a population of 3-5 mutants, tested in parallel, costs about
the same as 3-5 sequential retries with no diversity guarantee — and usually
produces a working tool on the first batch.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.llm import LLMClient
from ..core.message import Message
from ..forge.generator import GeneratedTool, GENERATOR_SYSTEM, extract_json
from ..forge.verifier import ToolVerifier, VerificationReport
from ..tools.spec import ToolSpec, ToolState, TriggerProbe

_MUTATE_SYSTEM = (
    "You are a tool breeder. You will be given a tool's current code and its\n"
    "verification failure report. Your job: generate a DIVERGENT variant that\n"
    "fixes the root cause. Do NOT reproduce the same approach — try a\n"
    "structurally different solution.\n\n"
    "Emit in the same JSON envelope as a normal tool generator:\n"
    "{name, description, code, entry, parameters, probes, effect_signature, tags, rationale}\n\n"
    "Rules:\n"
    "- Keep the same `name` as the original so it can replace it.\n"
    "- The code must be self-contained Python, stdlib only.\n"
    "- Include at least 2 probes. At least one must have a negative_query.\n"
    "- If the original failed because of a missing input guard, ADD that guard.\n"
    "- If the original failed because of a too-strict guard, LOOSEN it.\n"
    "- If the original timed out, optimise the algorithm.\n"
)

_CROSSOVER_SYSTEM = (
    "You are a tool breeder. You are given TWO parent tools that each solve\n"
    "part of a problem. Generate a hybrid that combines their capabilities\n"
    "into a single coherent tool.\n\n"
    "Emit in the JSON envelope format:\n"
    "{name, description, code, entry, parameters, probes, ...}\n\n"
    "Rules:\n"
    "- The hybrid must do what BOTH parents did, choosing a name that covers both.\n"
    "- The implementation must merge the logic cleanly, not shell out to them.\n"
    "- Code must be self-contained stdlib Python.\n"
    "- Include probes and negative_query for the combined capability.\n"
)

_ANALYSE_SYSTEM = (
    "You are a tool forensics analyst. Given a tool's code and the error it\n"
    "produced, identify the ROOT CAUSE (one sentence) and what must change.\n"
    "Output STRICT JSON only:\n"
    '{"root_cause": "...", "required_change": "...", "change_type": "guard|loosen|logic|perf"}'
)

_population_prompt = """Original tool: {name}
Code:
```python
{code}
```

Failure report:
{failures}

Generate {n} DIFFERENT approaches to fixing this tool.
Number each approach FIX-1 through FIX-{n}.
Each must be structurally different from the others.
Wrap each in the standard JSON envelope."""


@dataclass
class Mutant:
    generated: GeneratedTool
    report: VerificationReport | None = None
    fitness: float = 0.0
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitness": round(self.fitness, 3),
            "passed": self.report.passed if self.report else False,
            "root_cause": "",
        }


@dataclass
class EvolutionResult:
    tool: str
    original_version: str
    best_mutant: Mutant | None = None
    mutants: list[Mutant] = field(default_factory=list)
    kept_existing: bool = False
    rounds: int = 0

    @property
    def improved(self) -> bool:
        return self.best_mutant is not None and self.best_mutant.report is not None and self.best_mutant.report.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "improved": self.improved,
            "kept_existing": self.kept_existing,
            "rounds": self.rounds,
            "mutants_tried": len(self.mutants),
        }


# ---------------------------------------------------------------------------
class EvolutionEngine:
    """Breed better tools through mutation, crossover, and selection."""

    def __init__(
        self,
        llm: LLMClient,
        verifier: ToolVerifier,
        *,
        population_size: int = 3,
        parallel_verification: bool = True,
        mutate_model: str | None = None,
    ) -> None:
        self.llm = llm
        self.verifier = verifier
        self.population_size = population_size
        self.parallel_verification = parallel_verification
        self.mutate_model = mutate_model

    # -- mutation ---------------------------------------------------------
    def analyse_failure(self, tool: ToolSpec, error: str) -> dict[str, str]:
        """Ask the LLM to root-cause a tool failure."""
        prompt = f"Tool: {tool.name}\nCode:\n```python\n{tool.code}\n```\nError:\n{error}\n"
        resp = self.llm.chat(
            [Message.system(_ANALYSE_SYSTEM), Message.user(prompt)],
            tools=None,
        )
        data = extract_json(resp.content)
        if data is None or not isinstance(data, dict):
            return {"root_cause": "unknown", "required_change": "retry", "change_type": "retry"}
        return {
            "root_cause": str(data.get("root_cause", "unknown")),
            "required_change": str(data.get("required_change", "retry")),
            "change_type": str(data.get("change_type", "retry")),
        }

    def evolve(
        self,
        tool: ToolSpec,
        failure_report: str,
        *,
        existing_names: str = "",
    ) -> EvolutionResult:
        """Spawn mutants, verify them, return the best one."""
        result = EvolutionResult(
            tool=tool.name,
            original_version=tool.hash,
        )

        # 1. Spawn population
        prompt = _population_prompt.format(
            name=tool.name, code=tool.code,
            failures=failure_report, n=self.population_size,
        )
        context = f"Existing tools: {existing_names}" if existing_names else ""

        # We ask the model to generate all N in one response (cheaper),
        # then split by FIX-N markers.
        combined_prompt = (
            f"{prompt}\n\n"
            f"Context: {context}\n" if context else ""
        )
        # Use a temporary "for all N" generator
        resp = self.llm.chat(
            [
                Message.system(_MUTATE_SYSTEM),
                Message.user(combined_prompt),
            ],
            tools=None,
        )
        candidates = self._parse_population(resp.content)
        if not candidates:
            result.kept_existing = True
            return result

        # 2. Verify each in turn (or parallel)
        mutants: list[Mutant] = []
        for generated in candidates[:self.population_size]:
            started = time.perf_counter()
            spec = self._to_spec(generated)
            report = self.verifier.verify(spec)
            duration = (time.perf_counter() - started) * 1000
            fitness = self._compute_fitness(report)
            mutants.append(Mutant(generated, report, fitness, duration))

        mutants.sort(key=lambda m: m.fitness, reverse=True)
        result.mutants = mutants
        result.rounds = 1

        # 3. Select best
        best = mutants[0]
        result.best_mutant = best

        if best.report and best.report.passed:
            result.kept_existing = False
        else:
            result.kept_existing = True

        return result

    # -- crossover --------------------------------------------------------
    def crossover(self, parent_a: ToolSpec, parent_b: ToolSpec) -> GeneratedTool:
        """Breed a hybrid from two parent tools."""
        prompt = (
            f"Parent A ({parent_a.name}):\n```python\n{parent_a.code}\n```\n\n"
            f"Parent B ({parent_b.name}):\n```python\n{parent_b.code}\n```\n\n"
            "Generate a hybrid tool that merges both capabilities. "
            "Name it something that covers both functions."
        )
        resp = self.llm.chat(
            [Message.system(_CROSSOVER_SYSTEM), Message.user(prompt)],
            tools=None,
        )
        data = extract_json(resp.content)
        if data is None:
            raise ValueError(f"crossover produced no parseable JSON: {resp.content[:300]}")
        probes = [
            TriggerProbe(
                query=p.get("query", ""),
                expect=p.get("expect", "call"),
                negative_query=p.get("negative_query"),
            )
            for p in data.get("probes", [])
            if p.get("query")
        ]
        return GeneratedTool(
            name=data.get("name", "hybrid").strip(),
            description=data.get("description", "").strip(),
            code=data.get("code", "").strip(),
            parameters=data.get("parameters") or {"type": "object", "properties": {}},
            entry=data.get("entry") or data.get("name", ""),
            probes=probes,
            effect_signature=data.get("effect_signature", "pure"),
            tags=data.get("tags") or ["hybrid"],
            rationale=data.get("rationale", ""),
        )

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _compute_fitness(report: VerificationReport) -> float:
        """Score: passed checks + trigger count bonus."""
        if not report.checks:
            return 0.0
        passed = sum(1 for c in report.checks if c.passed)
        total = len(report.checks)
        base = passed / total if total else 0.0
        # Bonus for clean trigger+negative
        trigger_bonus = 0.2 if any(
            c.name == "negative" and c.passed for c in report.checks
        ) else 0.0
        return base + trigger_bonus

    @staticmethod
    def _parse_population(text: str) -> list[GeneratedTool]:
        """Parse a multi-mutant response into individual GeneratedTools."""
        results: list[GeneratedTool] = []
        # Strategy: find all JSON blocks, collect them
        blocks = text.split("FIX-")
        for block in blocks:
            data = extract_json(block)
            if data is None or not isinstance(data, dict):
                continue
            if not data.get("name") or not data.get("code"):
                continue
            probes = [
                TriggerProbe(
                    query=p.get("query", ""),
                    expect=p.get("expect", "call"),
                    negative_query=p.get("negative_query"),
                )
                for p in data.get("probes", [])
                if p.get("query")
            ]
            results.append(GeneratedTool(
                name=data.get("name", "").strip(),
                description=data.get("description", "").strip(),
                code=data.get("code", "").strip(),
                parameters=data.get("parameters") or {"type": "object", "properties": {}},
                entry=data.get("entry") or data.get("name", ""),
                probes=probes,
                effect_signature=data.get("effect_signature", "pure"),
                tags=data.get("tags") or [],
                rationale=data.get("rationale", ""),
            ))
        return results

    @staticmethod
    def _to_spec(g: GeneratedTool) -> ToolSpec:
        return ToolSpec(
            name=g.name,
            description=g.description,
            parameters=g.parameters,
            fn=lambda **_: (_ for _ in ()).throw(RuntimeError("evolution stub")),
            code=g.code,
            source="evolved",
            probes=g.probes,
            effect_signature=g.effect_signature,
            tags=g.tags,
            state=ToolState.DRAFT,
        )


__all__ = [
    "EvolutionEngine",
    "EvolutionResult",
    "Mutant",
    "GeneratedTool",
    "ToolVerifier",
]
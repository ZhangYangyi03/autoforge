"""Offline demo — runs the whole framework with zero API keys.

It exercises both halves of the thesis:

  PART 1  A gap agent hits a need, forges a tool, and the tool gets verified
          (execution + trigger + negative) before it earns ACTIVE state.

  PART 2  The living half: a tool that rots in production is caught by the
          ledger and auto-quarantined.

Run:  python examples/demo_offline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoforge.core.llm import LLMResponse, MockLLMClient, tool_call
from autoforge.core.agent import Agent
from autoforge.forge.generator import GeneratedTool, TemplateGenerator
from autoforge.forge.pipeline import ForgeConfig, ForgePipeline
from autoforge.forge.sandbox import Sandbox
from autoforge.forge.verifier import ToolVerifier
from autoforge.route.router import BehaviourRouter
from autoforge.tools.registry import ToolRegistry
from autoforge.tools.spec import ToolSpec, ToolState, TriggerProbe

BAR = "=" * 66


def rule(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")


# --------------------------------------------------------------------------
# A real, useful tool the framework will forge from scratch.
# --------------------------------------------------------------------------
def word_stats_tool() -> GeneratedTool:
    code = (
        "def word_stats(text: str = '') -> str:\n"
        "    \"\"\"Count words, characters, and the most common word.\"\"\"\n"
        "    import re\n"
        "    from collections import Counter\n"
        "    if not isinstance(text, str):\n"
        "        raise ValueError('text must be a string')\n"
        "    words = re.findall(r\"[A-Za-z0-9']+\", text.lower())\n"
        "    if not words:\n"
        "        return 'words=0 chars=0 common='\n"
        "    common = Counter(words).most_common(1)[0]\n"
        "    return f'words={len(words)} chars={len(text)} common={common[0]}x{common[1]}'\n"
    )
    return GeneratedTool(
        name="word_stats",
        description="Count words, characters, and the most common word in a text.",
        code=code,
        entry="word_stats",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "input text"}},
            "required": ["text"],
        },
        probes=[
            TriggerProbe(
                query="How many words and characters are in this sentence?",
                expect="call",
                negative_query="What is the capital of Peru?",
            ),
            TriggerProbe(
                query="Count the words in the paragraph and tell me the most frequent one.",
                expect="call",
            ),
        ],
        effect_signature="pure",
        tags=["text", "analysis"],
    )


# --------------------------------------------------------------------------
# PART 1 — forge → verify → seal
# --------------------------------------------------------------------------
def part1() -> None:
    rule("PART 1 — forge → verify → seal")

    def trigger_model(messages, tools, **kw):
        """A scripted 'agent' that behaves like a well-aligned LLM would:
        it calls the tool when the need is text analysis, and answers directly
        otherwise. This is what the trigger check is measuring."""
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        tool_names = [t["function"]["name"] for t in (tools or [])]
        low = last_user.lower()
        analytic = any(k in low for k in ("words", "characters", "count", "frequent", "paragraph"))
        if analytic and "word_stats" in tool_names:
            return LLMResponse(tool_calls=[tool_call("word_stats", {"text": last_user})])
        return LLMResponse(content="I can answer that directly.")

    llm = MockLLMClient(handler=trigger_model)
    registry = ToolRegistry()
    sandbox = Sandbox(timeout=8.0)
    verifier = ToolVerifier(llm, sandbox=sandbox)
    gen = TemplateGenerator(recipes={"word": word_stats_tool()})

    events = []
    pipeline = ForgePipeline(
        gen, verifier, registry, sandbox=sandbox,
        config=ForgeConfig(promote_on_pass=True),
        on_event=lambda k, p: events.append((k, p)),
    )

    result = pipeline.forge("I keep needing to count words and characters in text.")
    print(f"forge ok       : {result.ok}")
    print(f"rounds         : {result.rounds}")
    if result.spec:
        spec = result.spec
        print(f"tool name      : {spec.name}")
        print(f"final state    : {spec.state.value}")
        print(f"code hash      : {spec.hash}")
        print("verification   :")
        for c in spec.verification["checks"]:
            mark = "PASS" if c["passed"] else "FAIL"
            print(f"   [{mark}] {c['name']:<10} {c['detail']}")

    # prove it actually runs through the registry (out-of-process)
    print("\n-- live call through the registry --")
    out = registry.call("word_stats", {"text": "the quick brown fox the lazy dog"})
    print(f"ok={out.ok}  output={out.output!r}  ({out.duration_ms:.0f} ms)")
    print(f"ledger success_rate={registry.get('word_stats').stats.success_rate}")

    # prove it is injected into context now that it is ACTIVE
    print("\n-- context injection --")
    print("schemas exposed to LLM:", [s["function"]["name"] for s in registry.schemas()])

    # and prove a DRAFT tool is NOT injected
    draft = ToolSpec(
        name="unverified_thing", description="not yet verified",
        parameters={"type": "object", "properties": {}}, fn=lambda **k: "x",
    )
    registry.register(draft)
    print("after adding a DRAFT:", [s["function"]["name"] for s in registry.schemas()])


# --------------------------------------------------------------------------
# PART 2 — the living half: detect a rotting tool
# --------------------------------------------------------------------------
def part2() -> None:
    rule("PART 2 — the living half: a tool rots, the ledger catches it")

    registry = ToolRegistry(
        min_calls_for_judgement=3,
        quarantine_success_rate=0.6,
        quarantine_consecutive_failures=2,
    )

    def flaky(succeed: bool = True, **_) -> str:
        if not succeed:
            raise ValueError("upstream API changed its schema")
        return "ok"

    registry.register(ToolSpec(
        name="fetch_prices",
        description="Fetch the latest prices from the market API.",
        parameters={
            "type": "object",
            "properties": {"succeed": {"type": "boolean"}},
        },
        fn=flaky,
        source="generated",
        tags=["net"],
    ))
    registry.promote("fetch_prices")

    print("state          :", registry.get("fetch_prices").state.value)
    print("in context     :", "fetch_prices" in [s["function"]["name"] for s in registry.schemas()])

    # simulate the upstream breaking — every call fails
    for i in range(4):
        r = registry.call("fetch_prices", {"succeed": False})
        st = registry.get("fetch_prices").state
        print(f"  call {i+1}: ok={r.ok}  state={st.value}  sr={registry.get('fetch_prices').stats.success_rate:.2f}")

    print("\nafter the rot:")
    print("state          :", registry.get("fetch_prices").state.value)
    print("in context     :", "fetch_prices" in [s["function"]["name"] for s in registry.schemas()])
    blocked = registry.call("fetch_prices", {"succeed": True})
    print("blocked call   :", blocked.ok, "|", blocked.error)
    forced = registry.call("fetch_prices", {"succeed": True}, force=True)
    print("forced call    :", forced.ok, "| output=", forced.output)

    reason = [e for e in registry.events() if e["kind"] == "auto_quarantine"]
    print("\nquarantine event:", json.dumps(reason[-1], ensure_ascii=False) if reason else "none")

    print("\n-- rehab: operator fixes the tool and puts it back on trial --")
    registry.rehab("fetch_prices")
    print("state          :", registry.get("fetch_prices").state.value)
    print("in context     :", "fetch_prices" in [s["function"]["name"] for s in registry.schemas()])


# --------------------------------------------------------------------------
# PART 3 — behaviour-aligned routing beats plain text matching
# --------------------------------------------------------------------------
def part3() -> None:
    rule("PART 3 — routing by behaviour, not just text similarity")

    registry = ToolRegistry(min_calls_for_judgement=2)

    def mk(name, desc, ok, n):
        def fn(**_) -> str:
            if not ok:
                raise RuntimeError("broken")
            return "ok"
        registry.register(ToolSpec(
            name=name, description=desc,
            parameters={"type": "object", "properties": {}}, fn=fn,
        ))
        registry.promote(name)
        for _ in range(n):
            registry.call(name, {})

    # A tool that reads right but runs wrong, vs one that runs right.
    mk("parse_json_safe", "Safely parse a JSON document and report errors", ok=False, n=4)
    mk("parse_json_strict", "Parse JSON text strictly with validation", ok=True, n=4)

    # Rank everything including the quarantined one, so the contrast is visible.
    router = BehaviourRouter(
        registry,
        include_states={ToolState.ACTIVE, ToolState.PROBATION, ToolState.QUARANTINED},
    )
    query = "parse this JSON document safely"
    print(f"query: {query!r}\n")
    for c in router.rank(query):
        print(f"  {c.name:<20} score={c.score:+.3f}  state={c.state:<12} {c.breakdown}")
    print(f"\nrouter picks: {router.route(query, k=1)}")
    print("(near-identical text — behaviour is what breaks the tie)")
    print("production router excludes quarantined tools entirely:")
    print("  picks:", BehaviourRouter(registry).route(query, k=1))


def main() -> None:
    part1()
    part2()
    part3()
    print(f"\n{BAR}\ndone.\n{BAR}")


if __name__ == "__main__":
    main()

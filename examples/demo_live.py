"""End-to-end with a REAL model, via the AIPING gateway.

This is the honest test: no scripted handler, no mock. A real LLM is asked to
forge a real tool, the pipeline verifies it, and the forged tool is then called
for real.

Run:  python examples/demo_live.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoforge.core.llm import OpenAICompatClient
from autoforge.forge.generator import LLMToolGenerator
from autoforge.forge.pipeline import ForgeConfig, ForgePipeline
from autoforge.forge.sandbox import Sandbox
from autoforge.forge.verifier import ToolVerifier
from autoforge.route.router import BehaviourRouter
from autoforge.tools.registry import ToolRegistry

PROXIES = {"http": "socks5://127.0.0.1:9674", "https": "socks5://127.0.0.1:9674"}
BAR = "=" * 66


def rule(t: str) -> None:
    print(f"\n{BAR}\n{t}\n{BAR}")


def main() -> int:
    key = os.environ.get("AIPING_API_KEY")
    if not key:
        print("AIPING_API_KEY not set")
        return 1

    base = "https://aiping.cn/api/v1"
    model = "deepseek-v4.1-flash"

    use_proxy = os.environ.get("AUTOFORGE_PROXY", "1") == "1"
    llm = OpenAICompatClient(
        model=model,
        base_url=base,
        api_key=key,
        timeout=180,
        proxies=PROXIES if use_proxy else None,
    )
    print(f"model   : {model}")
    print(f"base_url: {base}")
    print(f"proxy   : {use_proxy}")

    registry = ToolRegistry()
    sandbox = Sandbox(timeout=15.0)
    verifier = ToolVerifier(llm, sandbox=sandbox)
    generator = LLMToolGenerator(llm)

    pipeline = ForgePipeline(
        generator, verifier, registry, sandbox=sandbox,
        config=ForgeConfig(promote_on_pass=True, max_rounds=2),
        on_event=lambda k, p: print(f"  [event] {k}: {str(p)[:160]}"),
    )

    # ------------------------------------------------------------------
    rule("STAGE 1 — a real LLM forges a real tool")
    need = (
        "I keep needing to validate and normalise ISBN-10 and ISBN-13 book "
        "identifiers: strip hyphens, check the check digit, and convert a "
        "valid ISBN-10 to its ISBN-13 form. I need this repeatedly."
    )
    print(f"need: {need}\n")
    result = pipeline.forge(need)

    print(f"\nforge ok    : {result.ok}")
    print(f"rounds      : {result.rounds}")
    if not result.ok:
        print("FAILED — attempts:")
        for a in result.attempts:
            print(f"  round {a.round}: error={a.error}")
            if a.report:
                for c in a.report.checks:
                    print(f"    [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        return 2

    spec = result.spec
    print(f"tool name   : {spec.name}")
    print(f"state       : {spec.state.value}")
    print(f"effect      : {spec.effect_signature}")
    print(f"tags        : {spec.tags}")
    print("\ngenerated code:")
    print("-" * 66)
    for line in spec.code.splitlines():
        print("  " + line)
    print("-" * 66)

    print("\nverification:")
    for c in spec.verification["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']:<10} {c['detail']}")

    # ------------------------------------------------------------------
    rule("STAGE 2 — call the forged tool for real (out-of-process)")
    samples = [
        "ISBN 0-306-40615-2",
        "978-0-306-40615-7",
        "0306406152",
    ]
    for s in samples:
        # give the tool whatever arg name the model chose for the input
        props = list((spec.parameters.get("properties") or {}).keys())
        arg = props[0] if props else "text"
        r = registry.call(spec.name, {arg: s})
        print(f"  {s:<22} -> ok={r.ok}  {r.output!r}  {r.error or ''}")

    print(f"\nledger: {registry.get(spec.name).stats.to_dict()}")

    # ------------------------------------------------------------------
    rule("STAGE 3 — trigger behaviour on the live model")
    for probe in spec.probes:
        print(f"  probe: {probe.query!r}")
    print("\nrunning the trigger check standalone against the live model:")
    check = verifier.check_trigger(spec, spec.probes[0].query)
    print(f"  positive -> passed={check.passed} | {check.detail}")
    print(f"    tool_calls={check.evidence.get('tool_calls')}")
    negs = [p.negative_query for p in spec.probes if p.negative_query]
    if negs:
        ncheck = verifier.check_negative(spec, negs[0])
        print(f"  negative -> passed={ncheck.passed} | {ncheck.detail}")

    # ------------------------------------------------------------------
    rule("STAGE 4 — router over the real library")
    reg2 = ToolRegistry()
    for nm, desc, ok, n in [
        ("normalize_isbn", "Normalise and validate ISBN identifiers", True, 3),
        ("parse_isbn_loose", "Loosely parse an ISBN string without validation", True, 3),
    ]:
        spec2 = type(spec)(
            name=nm, description=desc, parameters=spec.parameters,
            fn=(lambda **_: "ok") if ok else (lambda **_: ""),
        )
        reg2.register(spec2)
        reg2.promote(nm)
        for _ in range(n):
            reg2.call(nm, {})

    router = BehaviourRouter(reg2)
    q = "check whether this ISBN is valid"
    print(f"query: {q!r}")
    for c in router.rank(q):
        print(f"  {c.name:<20} score={c.score:+.3f}  {c.breakdown}")

    print(f"\n{BAR}\nLIVE RUN COMPLETE\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

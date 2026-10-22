"""End-to-end demo: register → earn trust → evolve (with a veto) → roll back.

Run:  python examples/demo_evolution.py

This exercises the real autoforge validity gate; the sabotaged candidate below is
refused *before* it is scored, and the transcript prints the proof.
"""
from __future__ import annotations

from toolmarket.protocol.lifecycle import ResourceState, VersionStatus
from toolmarket.protocol.sepl import EvolutionOperator, StubProposer
from toolmarket.registry import ResourceRegistry
from toolmarket.store import ResourceStore


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def make_slugify():
    from autoforge.tools.spec import ToolSpec, TriggerProbe

    code = ("import re\n\n"
            "def slugify(text=''):\n"
            "    text = re.sub(r'[^\\w\\s-]', '', text).strip().lower()\n"
            "    return re.sub(r'[-\\s]+', '-', text)\n")
    # Compile the registered source and use *that* as `fn`, so the demo executes
    # the same code it registers (no silent drift between `code` and `fn`).
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - demo of user-authored tool source
    slugify = ns["slugify"]

    return ToolSpec(
        name="slugify",
        description="Turn text into a URL slug.",
        parameters={"type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]},
        fn=slugify,
        code=code,
        source="human",
        probes=[TriggerProbe(query="make a slug from 'Hello World'", expect="call"),
                TriggerProbe(query="slugify", expect="call",
                             negative_query="what is the capital of France?")],
        invariances=["text"],
        effect_signature="pure",
    )


def main() -> None:
    store = ResourceStore(":memory:")
    reg = ResourceRegistry(store)

    hr("1. REGISTER  (a tool enters the substrate as DRAFT)")
    rec = reg.register(make_slugify())
    print(f"  id={rec.id}  state={rec.state.value}  version={rec.version_current.version}")
    print(f"  scope(effect_signature)={rec.contract.effect_signature!r}  "
          f"invariance={rec.contract.invariances}")

    hr("2. EARN TRUST  (DRAFT -> PROBATION -> ACTIVE)")
    reg.transition(rec.id, ResourceState.PROBATION, reason="cleared trigger probes")
    reg.promote(rec.id)
    print(f"  state={reg.get(rec.id).state.value}")

    hr("3. INVOKE  (the ledger records how it behaves in the wild)")
    out = reg.invoke(rec.id, {"text": "Hello, World! 2026"})
    print(f"  slugify('Hello, World! 2026') -> {getattr(out, 'output', None)!r}")
    print(f"  ledger.calls={reg.get(rec.id).ledger['calls']}  "
          f"success_rate={reg.get(rec.id).ledger['success_rate']}")

    hr("4. EVOLVE  (SEPL closed loop: propose -> assess -> commit)")
    op = EvolutionOperator(reg, proposer=StubProposer())
    proposal = op.propose(rec.id, goal="handle unicode and punctuation")
    print(f"  proposal {proposal.proposal_id}: {len(proposal.candidates)} candidates")
    for i, c in enumerate(proposal.candidates):
        print(f"    [{i}] generator={c.provenance['generator']!r} "
              f"invariances={c.contract.invariances}")

    report = op.assess(proposal.proposal_id)
    print(f"\n  assessment: {report.summary}")
    print("  per-candidate verdicts (the gate runs BEFORE fitness):")
    for v in report.verdicts:
        mark = "ADMISSIBLE" if v["admissible"] else "VETOED    "
        print(f"    [{v['index']}] {mark} fitness={v['fitness']:.3f}  {v['reason']}")
    print("\n  ^ the sabotaged candidate has fitness=0.000 and verification=None:")
    print("    it never reached the scoreboard, so no score could have rescued it.")

    before = reg.get(rec.id).version_current.version
    evolved = op.commit(proposal.proposal_id)
    print(f"\n  committed: version {before} -> {evolved.version_current.version}")
    print(f"  state after commit: {evolved.state.value}  "
          "(a new version re-opens the trial — trust is not inherited)")
    print(f"  superseded version status: "
          f"{next(v for v in evolved.versions if v.version == before).status.value}")

    hr("5. ROLLBACK  (AGP's atomic rollback, against the provenance axis)")
    rolled = op.rollback(rec.id, before)
    print(f"  live version restored to {rolled.version_current.version} "
          f"(status={rolled.version_current.status.value})")

    hr("6. AUDIT  (append-only events + the hash chain)")
    print(f"  chain intact: {reg.log.verify_chain()}   events: {len(reg.log)}")
    for e in reg.event_log(rec.id):
        extra = ""
        d = e["data"]
        if e["kind"] == "commit":
            extra = f"  {d['from_version']} -> {d['to_version']}"
        elif e["kind"] == "assess":
            extra = f"  admissible={d['admissible']} vetoed={d['vetoed']}"
        elif e["kind"] == "transition":
            extra = f"  {d['from']} -> {d['to']}"
        elif e["kind"] == "rollback":
            extra = f"  {d['from_version']} -> {d['to_version']}"
        print(f"    #{e['seq']}  {e['kind']:<11} {e['resource_id']}{extra}")

    hr("7. LINEAGE  (the DAG: who descended from whom)")
    view = reg.lineage_view(rec.id)
    for n in view["nodes"]:
        print(f"    {n['node_id']:<28} mutation={n['mutation']:<9} "
              f"parents={n['parents']}  reason={n['reason'][:40]!r}")

    hr("DONE  (substrate state)")
    print(f"  {store.stats()}")
    print(f"  by_state: ", end="")
    reg_states: dict[str, int] = {}
    for r in reg.list():
        reg_states[r.state.value] = reg_states.get(r.state.value, 0) + 1
    print(reg_states)


if __name__ == "__main__":
    main()

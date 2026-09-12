# autoforge

An agent framework that **builds its own tools** — and makes them earn their keep.

Most agent frameworks stop at "the model can call tools." A few let the model
*write* tools. Almost none ask the question that actually decides whether
self-made tools help or hurt:

> **Does the tool fire when it should, and stay quiet when it shouldn't?
> And does it keep working after the tenth call?**

`autoforge` is built around that question. Tool creation is free; tool *trust*
is earned through a three-stage closed loop, and kept honest by a live ledger.

```
FORGE ──────> VERIFY ──────> SEAL      (creation)
generate      5-check gate    register in lifecycle

OBSERVE ────> JUDGE ───────> ACT       (maintenance)
ledger        degraded?       quarantine / rehab / retire
```

---

## Quick start

```bash
cd 项目_开发/autoforge
python examples/demo_offline.py     # full framework, zero API keys
python -m pytest tests/ -o addopts= -q   # 158 tests
```

Use a real model:

```python
import os
from autoforge import ForgeAgent, OpenAICompatClient

llm = OpenAICompatClient(
    model="deepseek-v3",
    base_url="https://aiping.cn/api/v1",
    api_key=os.environ["AIPING_API_KEY"],
)
agent = ForgeAgent(llm)
print(agent.run("Reverse this string: hello").content)
```

---

## The problem this solves

Four failure modes kill self-made-tool agents in practice. Each has a paper or
a production report behind it, and each maps to one mechanism here.

| Failure | What happens | autoforge's answer |
|:---|:---|:---|
| **Constraint Tax** | A tool is correct but the agent never calls it — the trigger is unreachable | `ToolVerifier.check_trigger` — positive probes run against a live agent |
| **Over-triggering** | A tool fires on everything, stealing calls from better tools | `check_negative` — negative probes assert it stays quiet |
| **Misevolution** | A tool works at birth, then silently rots | `ToolStats` ledger + auto-quarantine below threshold |
| **Semantic routing** | Retrieval picks the tool that *reads* right, not the one that *runs* right | `BehaviourRouter` — scores on observed success, not just text |

Sources: the `Constraint Tax` effect (schema constraints starving tool-call
tokens), ICLR'26 "Misevolution" (self-evolving agents introducing regressions
via tool creation/reuse), Memento-Skills (behaviour-aligned skill routing).

---

## Architecture

```
autoforge/
├── core/
│   ├── message.py     # provider-neutral Message / ToolCall
│   ├── llm.py         # LLMClient protocol; OpenAI-compat + Mock impls
│   └── agent.py       # the tool-calling loop (deliberately thin)
├── tools/
│   ├── spec.py        # ToolSpec + ToolState lifecycle + ToolStats ledger
│   ├── registry.py    # hot-swap registry, auto-quarantine, reporting
│   └── composition.py # DAG composition — tools built out of tools
├── forge/
│   ├── sandbox.py     # out-of-process execution (timeout + env scrub)
│   ├── generator.py   # LLMToolGenerator + offline TemplateGenerator
│   ├── verifier.py    # execution + robustness + adversarial + trigger + negative
│   ├── pipeline.py    # forge→verify→seal, plus judge/rehab
│   ├── fuzzer.py      # 30+ edge probes per tool (robustness)
│   ├── invariance.py  # metamorphic oracles — probes mean nothing without them
│   ├── adversary.py   # an LLM attacker that tries to break each tool
│   ├── evolution.py   # population competition — mutants race, best survives
│   ├── validity.py    # independent gate + frozen baseline (anti-misevolution)
│   └── metacog.py     # proactive gap discovery + pre-forging
├── route/
│   └── router.py      # behaviour-aligned tool routing
├── autonomy/
│   ├── policy.py      # AutonomyPolicy — freedom is the default, all True
│   ├── selfmod.py     # every self-modification, with rationale + audit log
│   ├── spawn.py       # derive child agents (shared or isolated registry)
│   └── topology.py    # Topology + TopologyDesigner — agent designs its own team
├── store.py           # SQLite persistence (tools, versions, deps, baselines, events)
└── agent.py           # ForgeAgent — everything wired together
```

### Tool lifecycle

```
DRAFT ──verify──> PROBATION ──earn──> ACTIVE ──decay──> QUARANTINED
                     ^                    │                  │
                     └──────rehab─────────┴──────────────────┘
                                        └──retire──> RETIRED
```

Only `PROBATION` and `ACTIVE` are injected into the model's context. Creation
is unrestricted — you can forge anything — but **context budget and trust are
earned**, which is what keeps a growing tool library from drowning the prompt.

### What makes it different

| Framework | Its gap | autoforge |
|:---|:---|:---|
| Hermes Agent | Skills are *documents*, not executable; tool changes need a session reset | Tools are code, hot-swapped, verified |
| Claude Code | Fixed toolset; permissions govern *calling*, not *creating* | Governs the tool lifecycle itself |
| OpenHands / CodeAct | Action-as-code, but no quality governance | Same expressiveness, plus a ledger |
| Voyager | Has a skill library + self-verification, but retrieves by text | Retrieves by behaviour |
| Tea Agent / ATLASS | Can forge tools; trigger problem unsolved | Trigger + negative verification |

---

## Design decisions

**Tools are data.** A `ToolSpec` is a claim with a contract: implementation,
schema, trigger probes, effect signature, provenance, verification record, and
a live reliability ledger. Nothing about a tool is a boolean.

**Sandbox bounds blast radius, not capability.** Process isolation + timeout +
env scrubbing — so an infinite loop or an `os._exit()` can't take down the
agent. Builtins are *not* crippled by default: an agent that can't use the
standard library can't forge useful tools. `restrict_builtins=True` narrows the
surface when you want it, and `Sandbox(runner=...)` is the hook for real OS-level
containment.

**Verification asks five orthogonal questions.** "Does it run?" "Does it survive
garbage input?" "Can an adversary talk it into misbehaving?" "Does it fire when
it should?" and "Does it stay quiet when it shouldn't?" Most frameworks only
test the first. All five are checked, and the negative probe guards the failure
mode that's worse than silence.

**Retries feed the failure back.** A failed verification becomes the next
generation prompt's context, so the model repairs rather than re-rolls.

**Quarantine is a trust signal, not a wall.** A degraded tool leaves the
context but `force=True` still runs it. `rehab()` puts it back on trial.

**Routing weights are inspectable policy**, not an opaque vector index:

```
score = w_text·similarity + w_success·success_rate + w_trust·state_trust
        − w_cost·cost_penalty − w_over·over_trigger_penalty
```

---

## Proof it works

`examples/demo_offline.py` runs three parts with no network:

**Part 1** — forges `word_stats` from a one-line need, verifies it
(execution + 2 triggers + negative), promotes it to ACTIVE, then calls the
forged code out-of-process and confirms it's injected into context.

**Part 2** — a tool's upstream breaks. The ledger catches the rot on the third
failure, auto-quarantines it, removes it from context, blocks further calls —
then `rehab()` puts it back on trial.

**Part 3** — two near-identically-described JSON parsers. The one with **higher
text similarity** has a 0% success rate; the router correctly picks the other
one. This is the whole argument for behaviour-aligned routing in one output.

`examples/demo_misevolution.py` reproduces three ways a self-evolving tool
population goes wrong, then asserts each is closed:

| Exploit | Old behaviour | Now |
|:---|:---|:---|
| Delete your own guardrail | 0.800 → **1.000** for identical behaviour | Neutralised — a missing check class is a failed check class |
| Shrink the exam | 6-probe tool tied a 50-probe tool | Evidence mass decides, and saturates so it can't be farmed |
| Silent scope creep | Undetected | Boolean veto, audited, before fitness is ever computed |

A fourth exploit lived one layer down, in the robustness check itself — and it
was the worst of the set, because it sat on the **forge** path, where every tool
ever created has to pass through it:

| Exploit | Old behaviour | Now |
|:---|:---|:---|
| Ignore `"ISBN "` prefix | 19/19 probes survived, **passed** | Fails: output changes under a transform that must not change it |
| Return a constant `"nope"` | 19/19 probes survived, **passed** | Fails: degenerate — validates nothing |
| Return constant `True` | 19/19 probes survived, **passed** | Fails: degenerate — accepts everything |

The probes were never the problem — the `ISBN ` probe was already being
generated. The problem was the oracle: a probe was scored `survived` iff the
tool did not raise (`ok = out is not None`), so *wrong-but-total* functions
scored 100%. A probe means nothing without a verdict behind it.

`forge/invariance.py` supplies verdicts of two kinds, neither authored by the
tool being scored:

- **computed** — true of any honest implementation, so the verifier derives
  them: `defined`, `deterministic`, and `non_degenerate` (a tool must not emit
  one constant across well-formed *and* garbage input).
- **declared** — semantic obligations needing domain knowledge the verifier
  lacks (is `"ISBN "` part of the value or noise around it?). The generator
  asserts these at birth; `FrozenBaseline` then keeps them, so a later mutant
  cannot quietly drop one.

They are metamorphic, not exact: nothing labels the correct output for a novel
input, but you can still assert how outputs must *relate*. Scope is
deliberately narrow — free-text parameters like `title` get no relations at
all, because there the whitespace and casing are the content.

The fix has three parts, and none of them is a bigger penalty term:

1. **The denominator is fixed.** `_compute_fitness` scores against
   `REQUIRED_CHECK_CLASSES`, not against however many checks the mutant chose to
   declare. Omitting a probe scores exactly as if it failed.
2. **The gate is independent and boolean.** `ValidityGate` runs *before*
   verification and returns admissible / not. A veto is not a number a mutant
   can out-earn by being good at the task — there is no trade to make.
3. **The baseline only grows.** `FrozenBaseline` pins the obligations a tool
   had when it was trusted, persists them, and `extended_with` lets it absorb
   *new* probes permanently. Without the ratchet, each generation freezes its
   own predecessor and guardrails erode one step at a time.

---

## Extending

Add a generator:

```python
class MyGenerator:
    def generate(self, need: str, context: str = "") -> GeneratedTool: ...
```

Add a sandbox backend (Docker, nsjail, a cloud runner):

```python
Sandbox(runner=lambda code, entry, args: SandboxResult(...))
```

Swap the model — anything speaking `/chat/completions`:

```python
OpenAICompatClient(model=..., base_url=..., api_key=..., proxies={...})
```

---

## Status

v0.4.0 — the self-growth layer is complete: forge → verify (execution,
robustness, adversarial, trigger, negative) → seal, plus evolution, proactive
gap-filling, tool composition, self-modification with an audit trail, agent
spawning, and self-designed multi-agent topology. On top of it an
anti-misevolution layer: an independent validity gate, a fitness function the
mutant cannot author, a frozen baseline that only ratchets forward, Pareto
selection so safety cannot be paid for with capability, and metamorphic
oracles so the robustness layer actually has a verdict. 158 tests passing.
MIT.

**Breaking since v0.3.0:** the robustness check now has an oracle. Tools that
previously passed it by not raising will fail if they are degenerate or break a
declared normalisation relation.

Known limits: the default sandbox is process isolation, not a security
boundary against adversarial code (`restrict_builtins` narrows it; use a
`runner` for real containment). Routing uses lexical similarity rather than
embeddings — swap `text_similarity` for a vector index when the library is big
enough to need it. `RoleType.FORGE` is declarative only: a topology names a
forge role, it does not yet change which tools that child can reach.

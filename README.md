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
generate      exec+trigger    register in lifecycle

OBSERVE ────> JUDGE ───────> ACT       (maintenance)
ledger        degraded?       quarantine / rehab / retire
```

---

## Quick start

```bash
cd 项目_开发/autoforge
python examples/demo_offline.py     # full framework, zero API keys
python -m pytest tests/ -o addopts= -q   # 41 tests
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
│   └── registry.py    # hot-swap registry, auto-quarantine, reporting
├── forge/
│   ├── sandbox.py     # out-of-process execution (timeout + env scrub)
│   ├── generator.py   # LLMToolGenerator + offline TemplateGenerator
│   ├── verifier.py    # execution + trigger + negative checks
│   └── pipeline.py    # forge→verify→seal, plus judge/rehab
├── route/
│   └── router.py      # behaviour-aligned tool routing
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

**Verification asks two orthogonal questions.** "Does it run?" and "Does it
fire?" Most frameworks only test the first. Both are checked, and the negative
probe guards the failure mode that's worse than silence.

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

v0.1.0 — core loop, lifecycle, sandbox, verification, routing all working and
tested (41 passing). MIT.

Known limits: the default sandbox is process isolation, not a security
boundary against adversarial code (`restrict_builtins` narrows it; use a
`runner` for real containment). Routing uses lexical similarity rather than
embeddings — swap `text_similarity` for a vector index when the library is big
enough to need it.

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

One command, any machine, nothing to clone:

```bash
pip install autoforge-agent
auto setup                          # one wizard: provider, key, model
auto                                # the REPL
```

`autoforge-agent` is the distribution name; the import is `autoforge` and the
command is `auto` (or `autoforge`).

Working from a checkout instead:

```bash
git clone https://github.com/<you>/autoforge && cd autoforge
./auto                              # macOS / Linux / WSL / Git Bash
auto                                # Windows cmd     (PowerShell: .\auto)
```

The launcher checks your Python (3.10+), fetches `requests` once if it is
missing, and drops you into the REPL. Nothing else to set up. `auto.cmd` ships
alongside it so Windows works without touching your PATH.

Prefer `auto` on your PATH everywhere? Install the console script instead:

```bash
pip install -e .
auto                                # now a real entry point, any directory
```

### Configure once, then forget about it

```bash
auto setup
```

One wizard: pick a provider, paste a key, confirm the model. It writes
`~/.autoforge/config.json`, tests the endpoint with a single short request, and
from then on `auto` just runs — in this terminal and in every new one.

```
  provider:
   * 1) aiping.cn gateway (hosted, needs an API key)
     2) Ollama on this machine (local, no key)
     3) Something else (any OpenAI-compatible endpoint)
```

Every prompt shows the current value and takes it on a bare Enter, so re-running
the wizard is how you change one field without retyping the rest. To see what is
actually in effect, and which layer supplied each value:

```bash
auto config
```

```
  base_url   https://aiping.cn/api/v1  config C:\Users\you\.autoforge\config.json
  model      DeepSeek-V4.1-Flash       config C:\Users\you\.autoforge\config.json
  api_key    QC-5...e262 (len 68)      config C:\Users\you\.autoforge\config.json
  max_tokens 32768                     default
  proxy      True                      config C:\Users\you\.autoforge\config.json
```

Resolution order is **flag > environment > config file > default**, so a one-off
override never means re-running the wizard:

```bash
auto --model other-model                    # this invocation only
AUTOFORGE_MODEL=other-model auto            # this shell only
auto --base-url http://127.0.0.1:11434/v1 --model qwen2.5:7b --no-proxy
auto --policy supervised                    # keep the harness, drop the latitude
AUTOFORGE_POLICY=supervised auto            # same, this shell only
```

`max_tokens` is a budget rather than a limit on the answer: the generator's model
reasons before it writes, and the trace is spent out of the same budget. At the old
3000 the trace took the whole of it and the envelope never began, so the default is
32768. A provider with a lower ceiling is not a problem -- a 400 that names
`max_tokens` is answered at the number it names. Two things to know before tuning it
down: an envelope cut off mid-JSON fails to parse, and a truncated one that *does*
parse arrives with empty `probes`, which the pipeline accepts as a tool that has been
verified. If a gateway answers `503 暂无可用服务商` for a large cap on a reasoning
model no matter how long you wait, that is its routing rather than the number you
sent: the combination measured to work through the aiping gateway is
`--model Qwen3.5-Flash` with `AUTOFORGE_MAX_TOKENS=4000`, while a direct provider
endpoint takes the full 32768. Numbers and controls in `probes/FINDINGS.md`.

Why bother with a file when environment variables exist: a variable exported
*after* a terminal was opened is invisible to that terminal — Windows and POSIX
alike inherit the environment at process start. That turns configuration into
"it worked a minute ago, in the other window." A file has no such lag, which is
why `auto setup` is the supported path and the env vars are the escape hatch.

`auto setup` is also safe to run with no terminal: without a tty it takes
whatever the flags and environment already say, saves them, and exits instead of
blocking on a prompt.

`auto` on its own drops you into a REPL: type a recurring need in plain language
and the agent decides whether to forge, verify and keep a tool for it. Forged
tools persist for the rest of the session.

```bash
auto --help                         # all flags
auto forge "<need>" --out t.json    # forge one tool, one shot, then exit
auto list                           # inspect artifacts written by --out
auto list autoforge_tools/t.json    # dump one artifact
```

Point it at a local model (no key, no network) — either through `auto setup`
above, or per-invocation:

```bash
# local, CPU-friendly: drops the LLM-driven checks
AUTOFORGE_BASE_URL=http://127.0.0.1:11434/v1 AUTOFORGE_MODEL=qwen2.5:7b \
  AUTOFORGE_FAST=1 auto
```

Local endpoints never use the socks proxy and never need a key, so those two
questions are skipped automatically.

Inside the REPL: `/help`, `/tools` (library + health), `/report` (policy and
self-amendments), `/trace` (the decision log), `/reset`, `/quit`.

### The keyboard stays yours

A run is not a modal dialog. In `chat` the input line is live the whole time —
the run narrates itself *above* it, so you can keep typing while a tool is
running:

```
  [14:02:11] turn 1 +0.4s  asking the model…
  [14:02:19] +8.6s  -> bash
      … waiting on model (12s)
  you> also handle the empty file case
```

Type a sentence mid-run and it reaches the agent at its next step, labelled as
a correction rather than a new task. `/status` asks where the run is, `/stop`
ends it after the current step, and a leading space sends something that starts
with `/` as text. Paste a block and it collapses to a one-line
`[Pasted text #1: 40 lines → …]` placeholder — the text is kept on disk and
expanded again before it reaches the model.

### Selecting part of a line

```
you> check the empty file case
        └──────┘  Shift+arrows select, Ctrl+Insert copies, Ctrl+Delete cuts
```

`Shift+Left`/`Shift+Right` extend a selection from where the cursor is,
`Shift+Home`/`Shift+End` take it to either end, and the span is drawn in
reverse video. `Ctrl+Insert` copies it, `Ctrl+Delete` cuts it, `Shift+Insert`
and `^V`/`^Y` paste, and `Backspace`, `Delete`, `^K`, `^U` and `^W` remove the
selection when there is one. Typing over a selection replaces it. `^C` is left
alone: it still stops the run, because a terminal that can copy but cannot be
interrupted cannot be left.

**The mouse stays the console's.** The editor can take it — `AUTOFORGE_MOUSE=1`
— and keep the selection itself, which is the only way it ever hears about a
drag on Windows: the console paints its own highlight, keeps it, and reports
nothing, so a drag over the middle of the line followed by `Delete` deletes the
last character instead. But taking the mouse takes it for the whole session:
quick-edit off means the console stops selecting *and stops scrolling on the
wheel*, so reading back through what the agent printed stops working. That is a
worse trade than the bug, and it was reported as such, so the default is to
leave the mouse alone and select on the keys instead.

The keys an X terminal has always had keep working: `Ctrl+A`/`Ctrl+E` for the
ends of the line, `Ctrl+W` for the previous word, `Ctrl+U`/`Ctrl+K` to kill to
either end.

The behaviour degrades honestly: a pipe, a redirect, or a test gets the plain
cooked-mode reader and no heartbeat, because there is no terminal to own.

Offline demo, zero API keys:

```bash
python examples/demo_offline.py
python -m pytest tests/ -o addopts= -q
```

Use the library directly:

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
│   ├── sandbox.py     # out-of-process execution + measured reach report
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
├── gpu/
│   ├── bench.py       # benchmarking that cannot lie about its units
│   └── units.py       # ms-vs-seconds audits, the do_bench lesson as code
├── cpu/
│   ├── probe.py       # what this machine actually is, measured not assumed
│   ├── ops.py         # the problem set: a described problem, not an instance
│   ├── kernel.py      # compile → cache → load, refusing targets it cannot run
│   ├── safety.py      # preflight: the checks that run before anything executes
│   └── tune.py        # legal search over flags/source, verified then raced
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

**Every freedom is declared, and each declaration is classified.** Two presets
ship — `full` (the default: nothing denied) and `supervised` — selected with
`--policy` or `AUTOFORGE_POLICY`, and `autoforge config` prints which is live.
Each field in the policy is labelled *enforced* (a gate you can watch close),
*partial*, *confirm* (off means "not without a yes": the tool stops and asks
before it runs), or *declared-only* (a promise no code path keeps yet — an
empty class today, kept so a future unclassified field shows up loudly). The
report is not decoration: `my_capabilities` hands the same classification to the
agent, `set_autonomy` tells you whether switching a freedom off closes a door,
narrows one, or turns it into a question, and `describe()` prints
`[asks before running: ...]` — so "switched off" can never quietly mean
"still on".

The four execution freedoms (`may_read_filesystem`, `may_write_filesystem`,
`may_access_network`, `may_install_packages`) are the *confirm* ones. Switching
one off makes `ToolRegistry.call` ask, once per run, about any tool whose own
declared scope needs it — a tool that declares nothing is treated as capable of
everything and is therefore asked about too. Nobody to ask (a headless run, the
web harness's worker threads) means it does not run; an unanswered prompt is
never a yes. On a terminal the question is asked by `cli._TerminalConfirmer`,
which defaults to No and prints the tool, the switch and the arguments. Tools
declare their scope in `agent.BUILTIN_SCOPES`; a test fails if the table and the
tool specs ever disagree.

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
oracles so the robustness layer actually has a verdict. Underneath it a native
layer: the machine is measured rather than assumed, a target ISA the host cannot
run is refused before the compile rather than after the crash, a benchmark's
units are audited so a `ms` label cannot sit over a 1000x value, and kernel
search reports its own noise floor instead of promoting the fastest noisy run.
1142 tests passing. MIT.

**Breaking since v0.3.0:** the robustness check now has an oracle. Tools that
previously passed it by not raising will fail if they are degenerate or break a
declared normalisation relation.

Known limits: the default sandbox is process isolation, not a security
boundary against adversarial code (`restrict_builtins` narrows it; use a
`runner` for real containment). A *confirm* freedom has no approval flow in the
browser: the web harness runs agents in worker threads with no terminal, so it
answers "nobody to ask" and refuses rather than hanging a request on input it
cannot show you. Running `supervised` over the web UI therefore refuses the
gated tools instead of prompting for them — the CLI is the surface where the
question can actually be put to a person. It also does **not** separate the agent from the
host: forged code runs as a subprocess of the agent process on the same machine,
with the whole host filesystem and outbound network. `Sandbox.reach(probe=True)`
measures that with a real round-trip rather than asserting it, and
`my_capabilities` reports the measurement — because an agent that answers "can
you reach my machine?" from its tool list gets the answer wrong. The same rule
covers memory: the tool ledger is sqlite on disk (`ToolStore.report()`), forged
tools are persisted so they survive a restart, and `my_history` reads the ledger
and the self-modification log back. A measured self-report is appended to the
system prompt on every request, so the agent's description of itself is
recomputed from the machine instead of drifting as prose. Routing uses lexical
similarity rather than embeddings — swap `text_similarity` for a vector index
when the library is big enough to need it. `RoleType.FORGE` is declarative only:
a topology names a forge role, it does not yet change which tools that child can
reach.

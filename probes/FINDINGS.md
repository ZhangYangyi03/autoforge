# Probe findings

Provider behaviour observed against the real endpoints, with the method and the
raw numbers. These are the empirical facts the framework's defaults are chosen
from — kept here so a default can be traced back to a measurement rather than
to a hunch.

---

## aiping.cn: `max_tokens` is a routing input, not just a cap

**Reproduce:** `AIPING_API_KEY=... python probes/probe_gateway_max_tokens.py`
**Endpoint:** `https://aiping.cn/api/v1/chat/completions`, model `DeepSeek-V4.1-Flash`
**Payload:** the real `GENERATOR_SYSTEM` prompt, `temperature=0`.

### Why it was measured

Two live forge runs differed in one respect: the first sent no `max_tokens` and
died with HTTP 503, the second sent 1024 and did not. One request each — either
the cap changes gateway routing, or the first run landed in a bad window.

### Sweep 1 — ascending, two consecutive reps per cap (superseded)

| max_tokens | rep 1 | rep 2 |
|---|---|---|
| *(omitted)* | 503 | 503 |
| 512 | 200/length | 200/length |
| 1024 | 200/length | 200/length |
| 2048 | 200/length | 200/length |
| 4096 | 200/length | 503 |
| 8192 | 503 | 503 |

Read at the time as "the size of the cap matters, ≥4096 is unstable".
**That reading was wrong** — see below.

### Sweep 2 — interleaved, three cycles (authoritative)

Sweep 1 tested each cap as two consecutive requests, in ascending order. A
server that degrades across the run produces failures concentrated at the end
of the sweep, exactly like an effect of the cap. So sweep 2 cycles round-robin
through every cap, three full passes, letting wall-clock drift land on all caps
equally:

```
cycle 1: none=503  2048=200/leng  3000=200/leng  4096=200/leng
cycle 2: none=503  2048=200/leng  3000=200/leng  4096=200/leng
cycle 3: none=503  2048=200/leng  3000=200/leng  4096=200/leng

max_tokens=none   200s 0/3    max_tokens=2048   200s 3/3
max_tokens=3000   200s 3/3    max_tokens=4096   200s 3/3
```

### Conclusion

- **`max_tokens` present → 200. Absent → 503. On every rep.** 9/9 against 0/3,
  and inside a single cycle the two outcomes are seconds apart — so drift
  cannot explain the split.
- **The size of the cap is not what matters.** 4096 failed once in sweep 1 and
  passed 3/3 in sweep 2; that first failure was sweep 1's ordering, not the cap.
- Sweep 1's consistent 8192 failure was never replicated and is **left open**.
  Nothing in the framework sends a cap that large.

### Consequence in the code

`OpenAICompatClient.DEFAULT_MAX_TOKENS = 2048` — the field is now always on the
wire, set once at the client, so no call path (the agent loop, the demos) can
omit it by not thinking about it. Opt out per-client with
`default_max_tokens=None` for providers that reject the parameter.
Regression tests: `tests/test_llm_client.py`.

### Second-order finding, same data

**Every** 200 came back `finish_reason=length`, at every cap including 4096.
This model does not stop on its own; it runs into whatever ceiling it is given.
Two things follow, and both are already load-bearing in the framework rather
than defensive extras:

1. `autoforge.forge.json_repair` — a truncated JSON envelope is the normal
   case, so recovering a partial tool definition has to be routine.
2. The generator's "self-contained and terse" rule — a completion that gets cut
   mid-function produces `NameError` at execution, which costs a whole round.
   Asking for a small, dependency-free envelope keeps the cut past the end.

---

## Local 7B models fail the forge path at the JSON envelope, not the task

**Reproduce:** `AUTOFORGE_BASE_URL=http://127.0.0.1:11434/v1 AUTOFORGE_MODEL=qwen2.5:7b
AUTOFORGE_FAST=1 ./auto forge "<the ISBN need>" --out autoforge_tools/isbn.json`
Logs: `/tmp/e2e_ollama.txt` (`demo_live.py`), `/tmp/auto_forge_v4.txt` (`auto forge`).

Two independent runs against `qwen2.5:7b`, failing the same way:

```
round 1: ValueError: generator returned no parseable JSON
         (finish_reason='length', 3922 chars)
round 2: ValueError: generator returned no parseable JSON      (demo_live run)
```

### What actually happens

The 7B model writes the whole solution as a multi-helper function and embeds it
as an escaped string inside the JSON envelope. Helpers plus escaping blow past
the output cap, and the cut lands **inside the code string**:

```
"code": "def isbn_normalizer(isbn: str) -> str:\n    import re\n
         from math import floor, modf\n  ...  def check_isbn10(...):\n
         if len(isbn10) != 1          <-- stream ends here
```

`json_repair` does what it is built to do — closes the truncated envelope — but
there is no repair for a string cut mid-code, because the missing tail is
*program text*, not punctuation. The recovered value cannot be valid Python, so
the round fails. The pipeline does retry with "emit a shorter, denser function"
(that instruction is in the error it feeds back), and the retry re-fails on the
same rock: the model's instinct is still one big function.

The `forge` run's round 2 got past parsing and died in verification instead —
`2 invariance violation(s) (4/6)` in the robustness check, with execution clean.
So the envelope is the first wall, the invariance check the second.

### Consequence

- **Textbook confirmation of the truncation finding above.** `finish_reason` was
  `length` on every capped request in both probes *and* both live runs.
  Truncation is the standing condition of this stack, not an edge case — which
  is why the terse-envelope rule and `json_repair` are load-bearing rather than
  defensive extras.
- **The gateway model is the supported path for forging.** The same need against
  `DeepSeek-V4.1-Flash` completes, because its envelope fits inside the cap.
  A local 7B stays useful for the plumbing — client, sandbox, verifier, REPL —
  where the model is not asked to emit code inside JSON.
- Stated plainly rather than papered over: a framework whose entire purpose is
  generating code can be defeated by a model that cannot hold its output inside
  the transport. The failure is visible, reproducible and attributable, which is
  the most one can reasonably ask of it. It is not silently wrong.


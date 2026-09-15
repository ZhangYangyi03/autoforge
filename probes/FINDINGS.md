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


---

## The gateway's default model reasons until the output budget is gone

**This supersedes the "local 7B is too weak" reading of the section above.** The
7B failure and this one look identical from the outside — "no parseable JSON" —
and have nothing to do with each other.

**Reproduce:** `python probes/probe_gateway_empty.py`
(nine requests, ~2 minutes).

```
short prompt,   no cap      HTTP 200  finish='stop'    content=5c    usage=3/36      reasoning=0
short prompt,   cap 3000    HTTP 200  finish='stop'    content=5c    usage=18/36     reasoning=62
forge prompt,   cap 3000    HTTP 200  finish='length'  content=0c    usage=3000/246  reasoning=10767
forge prompt,   cap 8000    HTTP 200  finish='length'  content=0c    usage=8000/246  reasoning=27092
```

`DeepSeek-V4.1-Flash` is a reasoning model: it returns a `reasoning_content`
trace alongside `content`. On a short prompt the trace is 62 characters and the
answer arrives. On a forge-shaped prompt — long system message, a need that
deserves real design thought — **the trace consumes the entire `max_tokens`
budget and `content` comes back empty with `finish_reason='length'`.**

### Why raising the cap *looked* like it was not the fix

**Corrected: the cap was the fix, and it now ships.** The reading here was that the
trace "scales to fill" whatever cap it is given, which would make any cap
unusable, since the model would spend it thinking and never start the answer.

It was inferred from the two rows above -- 10767 characters at a cap of 3000, 27092
at 8000 -- and two samples taken below a value's natural size cannot separate
"unbounded growth" from "naturally larger than both of these". The second reading
is the true one, and the number needed to settle it was available throughout: this
same gateway, same model, same key has been answering requests carrying
65536-128000 from the author's own agent profile (202 request dumps, every one of
them this model). 3000 and 8000 were below the trace, which is not the same as the
budget being unusable.

Settled by `python probes/probe_cap_default.py`, which runs the *generator* at the
shipped default and grades what arrives, and pinned by `tests/test_cap_ceiling.py`.
The default is `DEFAULT_MAX_TOKENS` (32768), defined once, and a provider whose own
ceiling is below it answers 400 and is sent back the number that error names.

What survives here, and is worth reading: the trace is long, the budget is where it
goes, and `finish_reason='length'` with an empty `content` means the budget was the
constraint -- not the model, and not the parser. The earlier `200/leng` readings in
`probe_maxtokens` were reasoning-only replies rather than near-misses of a complete
answer, which remains true. What does not survive is the conclusion drawn from it.

### Why the suppression flags are not the fix either

`python probes/probe_gateway_thinking.py` sends the same payload with each of the
six common switches:

```
flag                    finish      reason  content
baseline                length       10049        0
enable_thinking=False   HTTP 503         0        0
thinking=disabled       HTTP 503         0        0
chat_template_kwargs    length        9342        0
reasoning_effort=none   HTTP 503         0        0
reasoning_effort=min    length       10115        0
```

The three that returned 200 show traces of 9342–10115 characters and no answer.
`reasoning_effort=minimal` barely moves the number: effort is not the lever. The
503s are the gateway reporting no available provider — a separate, frequent
condition worth labelling rather than reading as a property of the flag.

### Consequence

- **This is a model-selection problem, not a framework problem.** The client was
  passing `max_tokens` and parsing what came back, exactly as documented
  (`probes/probe_maxtokens.py`). Nothing in the transport was wrong; the
  chosen model does not answer this kind of prompt at any cap.
- `/models` lists **147** models including code-tuned ones
  (`Doubao-Seed-2.0-Code`, `Step-3.5-Flash`, `GLM-5.3-Flash`, ...). Picking one
  that answers directly is the fix, and `probes/probe_gateway_models.py` grades
  candidates mechanically instead of guessing.
- Three framework changes follow from it, all in-tree:
  - `LLMResponse.reasoning` keeps the trace, so a reply is never reported as
    "0 chars" when 10k characters came with it;
  - `LLMResponse.ran_out_of_budget_thinking` names the wall, and
    `describe_shortfall()` says "spent the budget on reasoning" rather than
    "cut off mid-JSON" for an answer that never existed;
  - `UnrecoverableGeneration` stops the round loop. Round two would relive round
    one exactly, at full budget — a wall is not a failure to retry harder.

## Correction: the wall was real, but it was not what blocked the run

Everything above is still true of the configs it tested — at a large cap this
model does spend the budget on reasoning. It is **not**, however, the reason the
first live `forge` runs failed. Reading the failure as "the model does not
answer" was wrong, and the fix that followed from it (swap models) would not have
worked, because a second model failed identically.

Two independent backends, two different models, one failure:

| backend | model | outcome |
| --- | --- | --- |
| aiping | Qwen3.5-Flash, cap 4000 | `finish_reason=stop`, 思考 4695c + 正文 2326c, parse FAIL |
| local ollama | qwen2.5:7b | `finish_reason=stop`, 2275c, parse FAIL |

Both answered completely. Both were rejected by our parser. The reported error
was `Extra data: line 5 column 4 (char 1422)` — not malformed JSON, but **a valid
object followed by more text**.

The captured reply (`probes/raw_generator_reply.txt`) shows what the model did:
it wrote `name`/`description`/`code`, emitted `}` — *exactly where the comma
belonged* — then kept going with `entry`/`parameters`/`probes`/`tags`/
`rationale`, and closed the object properly at the end:

```
{ "name": ..., "description": ..., "code": "..."
  },                                  <- stray; the comma should be here
  "entry": ..., "parameters": {...}, "probes": [...] }
```

**Why the obvious fix is the wrong fix.** `raw_decode` recovers the first object
and drops the tail — three keys instead of nine. That *looks* like success and is
worse than an error: `entry` defaults to the tool name, `parameters` and `probes`
default empty, so the pipeline would accept a tool with **no probes** and skip
checking it. A parse failure is loud; a silently probe-less tool is not.

**The actual fix** is `_drop_stray_closers` in `forge/generator.py`. The envelope
spans the first delimiter to the last, so everything between them must stay
nested at depth ≥ 1; deleting the brace restores the text to `"...code..."\n,\n"entry"...`,
and whitespace before a comma is legal JSON, so all nine keys survive.

One trap worth recording, because it cost a debugging round: **a single stray
brace shifts the depth of every closer after it.** Counting braces, or asking
which closer "returns to depth zero", misreads this text badly — the captured
reply showed four apparent depth-zero closers and EOF depth `-1` when there was
exactly *one* defect. Only the first-delimiter-to-last-delimiter interior is
unambiguous.

Verified: all 9 keys recovered, `code` compiles, 4 probes intact, 17/17 parser
tests and 275/275 suite tests pass.

## The cap is the lever after all -- and on aiping it has a ceiling

**Reproduce:** `AIPING_API_KEY=... python probes/probe_cap_big_prompt.py 6` (add
`caps=32768,3000,128000` to reverse the order), and `python probes/probe_cap_default.py`
for the two routes end to end.

### What was wrong with the 3000 default

Raising the cap *is* the fix. The reading recorded earlier in this file -- "the trace
scales to fill whatever cap it is given, so a bigger cap cannot help" -- was inferred
from two samples (3000, 8000) both taken below the trace's natural size, and two
points under a value cannot tell "unbounded" from "larger than both". `DEFAULT_MAX_TOKENS`
is now 32768, one constant read by the generator, the CLI and the wizard, and
`tests/test_cap_ceiling.py` pins the requirement.

End to end on the route that has to work (`api.deepseek.com`, `deepseek-chat` -- the
default profile):

```
generator cap = 32768    PARSED in 3s   name=list_py_files   code=888c   probes=3
```

The same default on a stricter provider is survivable by design: a 400 that names
`max_tokens` is answered at the number it names (`_clamp_cap_after_rejection`), so a
generous default is not a trap on endpoints with a lower ceiling.

### What aiping does with a big cap: measured with a control in the same minute

A first attempt failed 6/6 at 32768 with `503 暂无可用服务商`, while a short prompt at
32768 answered in 1s in the same minute -- leaving prompt size and cap entangled.
Requests are therefore interleaved one cap per cycle (so drift cannot land on one
cap) and the order is reversed in the second run (so "the first request of a burst is
served" cannot be mistaken for "the small cap is served"):

| run | order | 3000 | 32768 | 128000 |
| --- | --- | --- | --- | --- |
| 1 | 3000, 32768, 128000 | 3/6 | 0/6 | 0/6 |
| 2 | 32768, 3000, 128000 | 3/6 | 0/6 | 0/6 |

Cycles 2, 3 and 5 of run 2 are the rows that decide it: 32768 503s, and seconds later
3000 -- the *second* request in the cycle, same prompt, same model, same key --
answers 200. Order cannot explain that, so the cap does. Pooled over both runs: 3000
answered 6/13, and every cap at 32768 or above answered 0/18.

The cliff, with 3000 riding along as a control inside each cycle:

| cap | 200s |
| --- | --- |
| 3000 | 2/6 |
| 4096 | 0/6 |
| 8192 | 0/6 |
| 16000 | 0/6 |

(Cycles 1 and 3 are again paired contrasts: 3000 answers, all three larger caps 503
in the same minute. An earlier control-free run at 4096/8192/16000 was 0/12 with no
small-cap row, and is uninformative on its own -- that is what the control is for.)

So on aiping, `DeepSeek-V4.1-Flash` is served only at a small cap -- and a small cap is
precisely what this model cannot finish in. Every 200 above read
`finish_reason=length`, with 724-2762 chars of content against 8790-11515 chars of
`reasoning_content`: the envelope is cut off mid-JSON every time. **aiping +
this reasoning model cannot serve the forge**, which is also why the 4000 in an older
config never produced a tool: under the ceiling, and still far below the trace.

One contradiction is kept rather than smoothed over: sweep 2 above has 4096 answering
3/3 with this same prompt and model, so the refusal threshold is not a constant of the
account -- it moves with what the gateway has routed. Which is exactly why the code
does **not** silently downgrade the cap on a 503. A downgrade buys a truncated
envelope, and a truncated envelope is worse than a loud failure here: `entry`,
`parameters` and `probes` default to empty, so the pipeline would accept a tool with
no probes and skip checking it (see the stray-closer section above).

### What does work on aiping

A model whose trace leaves room inside the cap the gateway serves: `Qwen3.5-Flash` at
4000 answered `finish_reason=stop` with a complete 2326-char envelope. Or skip the
gateway entirely: `api.deepseek.com` takes 32768 and answers in 3s. Neither needs the
wizard -- `AUTOFORGE_MAX_TOKENS=4000 auto --model Qwen3.5-Flash` is the whole change.

Worth remembering when reading a lone 503: 3000 itself answered only 44% of the time
(8/18) while being the cap the gateway prefers, so a single 503 identifies nothing.
The probe prints one row per cycle on purpose -- a refusal looks like 503 for the big
caps while the small cap answers *in the same cycle*, whereas a bad window looks like
everything failing at once (cycles 2, 4, 5, 6 of the last run).

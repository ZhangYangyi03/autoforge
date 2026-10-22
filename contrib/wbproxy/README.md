# wbproxy — WorkBuddy subscription as an OpenAI endpoint

> **This is a contributed tool, not part of autoforge's runtime.** It lives in
> `contrib/` because it is useful *with* autoforge, not because autoforge
> depends on it. Nothing in `autoforge/` imports it; the only link is that this
> process speaks OpenAI and autoforge is pointed at it through
> `~/.autoforge/config.json`. Delete the directory and autoforge is unchanged.

An OpenAI-compatible proxy that turns a **WorkBuddy subscription** into a model
endpoint any OpenAI-shaped client can use — autoforge in particular, but also
LangChain, llama-index, or a plain `requests.post`.

No API key, no per-token billing: it rides the subscription the WorkBuddy
desktop app already signed you into.

```
your client  ──HTTP──>  wbproxy  ──>  copilot.tencent.com / www.workbuddy.ai
   /v1/chat/completions   :8777           /v2/chat/completions
```

Both gateways are served from one process. Pick a model; the pool decides the
region.

---

## Why a proxy and not a client library

autoforge's `OpenAICompatClient` carries a retry ladder, `max_tokens` clamping,
failover and usage accounting worth keeping. WorkBuddy needs the opposite of
each of those: a different path, nine identity headers, forced streaming, and a
business code that means "slow down" more often than "you are wrong".

Reimplementing the client against that contract would mean either duplicating
that logic or losing it. A proxy speaks each side natively and leaves the client
untouched — and it is one file, outside both projects, that survives their
upgrades.

---

## Install

Nothing to install. Python 3.10+ and the standard library.

```bash
git clone <this repo>
cd autoforge-workbuddy
python wbproxy.py doctor          # are the accounts reachable?
```

### Credentials

wbproxy does **not** implement a login flow of its own. It reads a credential
pool, in this order:

| # | Source | Who has one |
|---|---|---|
| 1 | `$WBPROXY_ACCOUNTS` | point at any file you like |
| 2 | `$DSH_HOME/.xdpool/accounts.json` (default `~/.dsh/…`) | anyone running the DSH `xdpool` plugin |
| 3 | `~/.wbproxy/accounts.json` | created by `wbproxy login` |

> **The WorkBuddy desktop app's own files are not readable here.** Their tokens
> are `$wbEncrypted` AES-256-GCM envelopes whose key comes from the OS keystore
> (`workbuddy-desktop.info` → `auth.accessToken.envelope`). Opening them means
> reimplementing that derivation per platform — DPAPI on Windows, Keychain on
> macOS, Secret Service on Linux — for a key that is deliberately not portable.
> The DSH plugin already does that work and re-emits plaintext, so its cache is
> the source that costs nothing to depend on.

No DSH install? Paste one credential once:

```bash
python wbproxy.py login
# accessToken: <paste>
# refreshToken: <paste, optional but recommended>
# domain [www.workbuddy.cn]:
```

It lands in `~/.wbproxy/accounts.json`, mode `0600`. The refresh token is what
keeps it alive: with one, wbproxy renews silently and writes the new token back.

---

## Serve

```bash
python wbproxy.py                 # 127.0.0.1:8777
python wbproxy.py serve --port 9000
```

```bash
curl http://127.0.0.1:8777/v1/models
curl http://127.0.0.1:8777/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy3","messages":[{"role":"user","content":"hi"}]}'
```

Point autoforge at it:

```bash
auto setup          # -> "Something else (any OpenAI-compatible endpoint)"
# base_url  http://127.0.0.1:8777/v1
# model     hy3
# api_key   wb            (any non-empty string; not verified)

auto forge "count words in a text"
```

---

## Choosing a model

34 models on a CN account, 21 on a global one, and they only partly overlap.
`models` exists so you are not guessing:

```bash
python wbproxy.py models                  # everything, with cost and context
python wbproxy.py models --free           # zero credit multiplier
python wbproxy.py models --tools          # supports structured tool calls
python wbproxy.py models --sort cost      # cheapest first
python wbproxy.py models --sort context   # largest window first
python wbproxy.py models --region cn
```

```
id                           ctx     out    cost tools  img  description
hy3                         192k     64k    free     y    y  Hunyuan's thinking model
step-5-preview              300k     64k    free     y    y  StepFun's flagship model
space-bunny                1000k    128k   x0.08     y    y  Blazing-fast, anonymous
deepseek-v4.1-flash         300k    128k   x0.11     y    y  DeepSeek flagship
default                     200k     24k    x2.2     y    -  (alias; expensive)
```

`cost` is the credit multiplier the catalog states. **`default` is the most
expensive row on the list** — it is an auto-routing alias, not a safe default.

For an agent that calls tools, filter on `--tools` and prefer `hy3` or
`step-5-preview` (both `free`, both 64k output).

The same metadata rides on `GET /v1/models` as extra fields, so a UI can build a
picker without a second endpoint.

---

## What actually differs from OpenAI

All of this was measured against the live gateway, not read off a schema.

| OpenAI | WorkBuddy | Why it matters |
|---|---|---|
| `/v1/chat/completions` | `/v2/chat/completions` | `/v1`, `/v2/responses`, `/v2/messages` all 404 |
| optional streaming | **required** (`stream: true`) | a non-stream body answers `11101` |
| `Authorization: Bearer <key>` | Bearer OAuth + 9 headers | `X-User-Id`, `X-Domain`, `X-Product`, `X-IDE-*` |
| `role: "developer"` ok | rejected | business code `11128`; rewritten to `system` |
| 400 = you are wrong | **400 = "slow down"** | see below |

### The 11128 problem

Business code `11128` covers two unrelated failures: a malformed request **and**
a rate limit. The gateway sends both as HTTP 400.

That is the worst possible pairing. `400` is in nobody's retry set, so a
throttled request dies as if it were a caller error — which is exactly what
happened here: forge runs failed repeatedly, and every retest with an identical
body succeeded minutes later.

wbproxy translates `400 + 11128` into **HTTP 429**. `429` is in everybody's retry
set. A genuine format error still surfaces as 400 — after the ladder has had its
chance, which is the right order: retrying a correct request costs seconds,
dropping a throttled one costs an entire tool forge.

It also retries across accounts with backoff, because a rate limit that answers
one key will usually answer another.

### Structured tool calls work

An earlier version of this file flattened `role: "tool"` turns into prose, on
the theory that the chat wire did not accept them. **That was wrong**, and the
bisect is in `docs/findings.md`: a turn carrying `role: tool`, `tool_call_id`,
`tool_choice: "auto"`, a system message, `temperature`, `max_tokens=32768` and
`stream_options` answers 200.

Structured turns are passed through untouched. `WBPROXY_FLATTEN_TOOLS=1` restores
the old behaviour as an escape hatch.

---

## Regions

`cn` and `global` accounts are separate pools with separate catalogs — and one
proxy serves both. You pick a **model**; the pool decides the region.

| | CN | Global |
|---|---|---|
| chat base | `copilot.tencent.com` | `www.workbuddy.ai` |
| catalog | `/v2/enterprises/personal/models` | `/v3/config` |
| models | 34 | 21 |
| exclusive | `auto`, `step-5-preview`, `hunyuan-*`, `glm-4.x` | `gpt-5.6-*`, `gemini-3.5-flash`, `kimi-k3` |
| shared | `hy3`, `hy4-preview`, `deepseek-v4.1-flash`, `glm-5.2` `glm-5.3`, `glm-5.3-flash`, `kimi-k2.6`, `kimi-k2.8-preview`, `minimax-m3.1-flash` | same |

46 distinct ids across both, 9 of them on both sides. A model only one catalog
lists is routed to that region — the other cannot serve it, so trying would waste
a turn. A shared id is served by whichever account is idle.

```
$ python wbproxy.py models --region global
$ python wbproxy.py models --region cn --free
```

To pin the whole process to one side — useful when diagnosing, or when you want
a run to spend only one plan's quota:

```bash
python wbproxy.py serve            # both regions (default)
WBPROXY_REGION=global python wbproxy.py serve
```

### The global gateway has two gates the domestic one does not

Getting the global side to answer took isolating two request-shape rules. Both
are handled; this is why they exist, in case a future change revives them.

**Gate one — three conversation headers.** Missing any of `X-Conversation-ID`,
`X-Conversation-Request-ID`, `X-Conversation-Message-ID` answers `11128`
`request illegal`. The values are **not validated** (plain non-UUID strings
pass), so they are derived from the conversation rather than stored.

**Gate two — the first message must be `system`.** Otherwise `11128`
`first message is not system prompt`. A placeholder is prepended; your own
messages keep their position and wording.

The domestic gateway needs neither. It applies the *content* fingerprint check
instead, which the global gateway does not — so neither is applied here, because
this proxy sends its own prompts.

An earlier version of this README said the global account could not chat. That
was a conclusion from an incomplete sweep — 15 header combinations, none of them
the three that mattered. See [`docs/findings.md`](docs/findings.md) §4 for the
elimination table.

### Plan limits are a separate failure

Not every model in a catalog is usable by every account. On a
`personal_free_user`, 14 of the 21 global ids answer and 6 are refused:

```
11158  model [gemini-3.5-flash] is not allowed for user type [personal_free_user]
```

The catalog does not distinguish them, so the refusal is reported as-is rather
than retried — it names a plan, and no amount of rotation changes a plan. Check
before building on an id:

---

## Commands

| Command | Purpose |
|---|---|
| `serve` (default) | run the HTTP server |
| `models` | list models with region, cost, context, tool support |
| `doctor` | per-account token, expiry, and a live chat probe |
| `login` | paste a credential into `~/.wbproxy/accounts.json` |
| `--help` | all flags |

Environment:

| Variable | Meaning |
|---|---|
| `WBPROXY_PORT` / `WBPROXY_HOST` | listen address (default `127.0.0.1:8777`) |
| `WBPROXY_REGION` | `cn` or `global` to narrow the pool; unset serves both |
| `WBPROXY_ACCOUNTS` | a credential file to use instead of the discovered ones |
| `WBPROXY_TRACE_BODY` | log the request shape; off by default — it carries the conversation |
| `WBPROXY_FLATTEN_TOOLS` | rewrite tool turns as prose; off by default |

---

## Checking it works

Two scripts, both read-only, both safe to run against a live server:

```bash
python selfcheck.py            # the protocol contract
python selfcheck_autoforge.py  # the autoforge integration
```

`selfcheck.py` proves the wire: health, catalog shape, a plain completion, a
structured tool call, a structured tool **result** (the case that used to fail),
and that an unknown model fails loudly instead of silently.

`selfcheck_autoforge.py` proves the integration: `auto config` accepts the
endpoint, `auto forge` seals a tool through it, all six verification checks run,
and a burst of requests produces no hard `400`.

Last run on a 4-account pool (2026-10-22):

```
selfcheck.py           ALL CHECKS PASSED   (23 checks)
selfcheck_autoforge.py ALL CHECKS PASSED   (14 checks)
```

---

## Known rough edges

- **A forge round occasionally fails its `trigger` check** (`fired 0/1`) and
  passes on the retry. The proxy answers 200 throughout — the model simply
  declined to call a tool it had been given, and autoforge's trigger check is
  `hits > 0` over a single trial, so one "no" is a whole round. Measured
  separately at temperature 0 with a bare tool schema, `hy3` and
  `step-5-preview` both called 5/5; it is the longer system prompt around the
  real forge that makes it wobble. Cost is one extra round, not a failure —
  three consecutive forges after the observed miss all sealed in round 1.

- **Model latency varies by an order of magnitude.** `hy3` answers a short
  completion in ~5-7s; a full forge round takes 75-170s because it makes six-plus
  calls. Budget accordingly.

---

## Limits, stated plainly

- **Not a security boundary.** It holds OAuth tokens in memory and serves them
  to anything that can reach `127.0.0.1:8777`. Bind to loopback only.
- **Your subscription is shared.** If another tool drives the same accounts, both
  compete for the same rate limit. wbproxy retries; it cannot create capacity.
- **Credential refresh writes back.** When a token is renewed it is persisted to
  its source file so the next start is warm. If that file is DSH's cache, the
  plugin may overwrite it — harmless, both sides refresh from the same seat.
- **No image input**, even for models the catalog marks multimodal.
- **`credits` are the gateway's numbers**, not a price. `free` means the catalog
  states a zero multiplier; the gateway may still meter it.

---

## Layout

```
wbproxy.py          the whole thing: pool, protocol, HTTP server, CLI
docs/findings.md    the probes behind every claim above
.gitignore
```

MIT.

# Findings

Every claim in the README comes from a probe against the live gateway, run on
2026-10-22 with a CN account (`www.workbuddy.cn`) and a global one
(`www.workbuddy.ai`). This file records what was asked, what came back, and the
hypotheses that were eliminated — so the next person does not re-run them.

The probes themselves are in `probes/`.

---

## 1. Which paths exist

`POST` with a minimal streaming body, against `copilot.tencent.com` (CN) and
`www.workbuddy.ai` (global):

| Path | CN | Global |
|---|---|---|
| `/v2/chat/completions` | 200 | 400 `11128` |
| `/v3/chat/completions` | — | 404 |
| `/inference/v1/chat/completions` | — | 404 |
| `/v1/chat/completions` | — | 404 |
| `/v2/responses` | 404 | 404 |
| `/v2/messages` | 404 | 404 |

**Conclusion.** Only `/v2/chat/completions` exists. The Anthropic-shaped
(`/v2/messages`) and Responses-shaped (`/v2/responses`) wires are not served on
these hosts, which removes "translate the conversation into another wire" as a
strategy before it is attempted.

---

## 2. Is `role: "tool"` accepted?

A prior version of `wbproxy.py` flattened tool turns into prose, assuming the
chat wire rejected `role: "tool"`. That assumption was never tested. Bisected by
starting from a minimal tool turn and adding one field at a time:

| Variant | Result |
|---|---|
| minimal tool turn (`role: tool` + `tool_call_id` + `tool_calls`) | **200** |
| + system message | 200 |
| + `temperature: 0.0` | 200 |
| + `max_tokens: 32768` | 200 |
| + `max_tokens: 512` | 200 |
| + `stream_options` | 200 |
| + `tool_choice: "auto"` | 200 |
| + assistant text alongside `tool_calls` | 200 |
| + `name` on the tool message | 200 |
| − `tool_call_id` | 200 |
| all of the above together | 200 |

**Conclusion.** Structured tool turns work. The flattening was unnecessary and
cost the model its tool affordance on every turn after the first. It is now
opt-in (`WBPROXY_FLATTEN_TOOLS=1`) rather than the default.

A separate probe also ran the adversarial-gate system prompt verbatim — the one
mentioning `script injection` and `null bytes` — and got 200 as well, which
eliminated content filtering as the explanation for those failures too.

---

## 3. What 11128 actually is

`11128` appears with `msg: "request illegal"` and `displayMsg: "Request blocked.
Please send it again"`.

**It is multiplexed.** At least four distinct causes share it, on two gateways:

| Cause | Gateway | `msg` |
|---|---|---|
| rate limit | both | `request illegal` |
| missing conversation headers | global | `request illegal` |
| first message not `system` | global | `first message is not system prompt` |
| content fingerprint in system/assistant text | domestic | `Illegal API invocation from an unapproved channel` |
| `role: "developer"` | domestic | same as above |

So nothing may branch on the business code alone. The `msg` field is what
separates them, and even that only sometimes.

Three observations that identify the rate-limit case:

1. Every structural hypothesis passes with an identical body.
2. Requests that 11128'd answered 200 minutes later with the **same** body.
3. The failures cluster: a forge run makes six-plus calls in under a minute.

**Conclusion.** On this gateway, `11128` is a rate limit at least as often as it
is a format error, and it always arrives as HTTP 400. Since 400 is not in any
client's retry set, a throttled request is indistinguishable from a caller
mistake — and dies as one.

Handling: `400 + 11128` → `429`, plus cross-account retry with backoff. A genuine
format error still surfaces as 400, after the ladder. The conversation-header and
system-first causes are fixed at the source instead (§4), so what remains under
this translation is the rate limit and the fingerprint.

---

## 4. CN and global are separate pools

Catalogs are nested differently and overlap only partly:

| | CN | Global |
|---|---|---|
| catalog path | `/v2/enterprises/personal/models` | `/v3/config` |
| envelope | `{code, msg, data: {models, agents}}` | same shape, different path |
| model count | 34 | 21 |

| Set | Count | Examples |
|---|---|---|
| CN only | 25 | `auto`, `step-5-preview`, `hunyuan-chat`, `glm-4.7`, `hy3-x` |
| Global only | 12 | `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gemini-3.5-flash` |
| Both | 9 | `hy3`, `hy4-preview`, `deepseek-v4.1-flash`, `glm-5.2/5.3` |

**Conclusion.** Region is a property of the credential, not a setting. A pool may
hold accounts from both, and each request must use its own account's base,
headers and model ids.

### The global account could not chat -- until two gates were found

An earlier revision of this file concluded the global account "cannot stream
chat" and marked it unusable. **That conclusion was wrong.** It was reached from
an incomplete sweep, and the fix is in the code.

What the sweep covered, and why every one of those hypotheses was a dead end:

| Check | Result |
|---|---|
| catalog `/v3/config` | **200** — credential valid, account readable |
| chat `stream: false` | **400 `11101`** "Non-stream chat request is currently not…" |
| chat `stream: true` | **400 `11128`** |
| 12 model ids (incl. global-only ones) | all `11128` |
| 15 header combinations (identity, UA, product, no `X-Domain`, Accept) | all `11128` |
| gzip body, query params, request-id headers, other hostnames | all failed |
| via a different egress (local proxy) | all `11128` |

The `11101` was correctly read as "the request is parsed and routed", which is
why the search stayed on headers. It stopped one dimension short: it varied
`X-IDE-*`, `X-Product`, `X-Domain`, `User-Agent` and `Accept`, but never
**`X-Conversation-*`**.

#### Gate one: three conversation headers

`www.workbuddy.ai` made its channel check hard. A chat request missing **any**
of these is rejected with `11128`:

```
X-Conversation-ID
X-Conversation-Request-ID
X-Conversation-Message-ID
```

Isolated by elimination against a real credential:

| Variant | Result |
|---|---|
| no conversation headers | `11128 request illegal` |
| minus `X-Conversation-ID` | `11128 request illegal` |
| minus `X-Conversation-Request-ID` | `11128 request illegal` |
| minus `X-Conversation-Message-ID` | `11128 request illegal` |
| **all three present** | **error changes** → `first message is not system prompt` |
| all three, plain non-UUID strings | same — the values are **not** validated |

That last row is the useful one: the gateway checks **presence**, not a
signature. So the ids can be derived from the conversation instead of registered
with upstream, and no per-account state is needed.

#### Gate two: the first message must be a system prompt

With gate one passed, the failure changes rather than disappears:

| Variant | Result |
|---|---|
| `user` first | `11128 first message is not system prompt` |
| **placeholder system prepended** | **`200` — stream opens** |

The domestic gateway accepts the same body, so this rule is global-only. The
fix prepends a placeholder and leaves every caller-supplied message at its
original position with its original wording.

#### What the international gateway does NOT check

A system prompt containing `powered by DeepSeek Harness` answers `200`. The
content fingerprint that `dsh-xdpool` rewrites is a **domestic-only** check. The
proxy therefore applies it to neither region rather than guessing.

#### Plan limits, which are a different failure

Not every model in the global catalog is usable by every account. Measured on a
`personal_free_user`:

| Refused (`11158`, `model ... not allowed for user type`) | Usable |
|---|---|
| `primary-model`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4`, `gemini-3.5-flash` | the other 14 |

`deep-model` answers `11102 service info not found` — listed but withdrawn.

This matters for diagnosis because the catalog does not distinguish the two: it
publishes 21 ids and the account can use 14. A refusal names the plan, so it is
reported as-is rather than retried.

---

## 4b. Both regions in one pool

Region is a property of the credential, so a single pool holds both and routes
by model:

| | CN | Global |
|---|---|---|
| accounts here | 3 | 1 |
| catalog | 34 ids | 21 ids |
| distinct ids across both | **46** | |
| published by both | **9** — `deepseek-v4.1-flash`, `glm-5.2`, `glm-5.3`, `glm-5.3-flash`, `hy3`, `hy4-preview`, `kimi-k2.6`, `kimi-k2.8-preview`, `minimax-m3.1-flash` | |

A request for a model only one catalog lists is routed to that region, because
the other region cannot serve it. A request for a shared id is served by
whichever account is idle, across both.

`WBPROXY_REGION=cn|global` narrows the pool to one side for diagnosis. Unset is
the normal case.

**Not verified:** whether a second global account would behave the same. Only one
was available, so the single-account pool has no redundancy: a rate limit on it
is a hard stop until the window reopens.

---

## 5. Why the DSH plugin's cache, not the app's files

`%LOCALAPPDATA%\CodeBuddyExtension\Data\Public\auth\workbuddy-desktop.info`
holds:

```json
"accessToken": { "$wbEncrypted": 1, "envelope": "<base64 of a JSON envelope>" }
```

The envelope decodes to `{suite, keyId, nonce, authTag, ciphertext}` —
AES-256-GCM, with the 32-byte key derived from the OS keystore. `dsh-xdpool`
implements that derivation (`atRestKeyFor`, `encryptedFieldOpener`,
`fieldAad`) and re-emits plaintext into `accounts.json`.

**Conclusion.** Reading the app's files directly means reimplementing per-platform
key derivation (DPAPI / Keychain / Secret Service) for a deliberately
non-portable key. Depending on the plaintext cache is the cheaper, honest
dependency; `wbproxy login` covers users who have neither.

---

## 6. Not established

Recorded so they are not mistaken for solved:

- whether `11128` also covers content refusal (the softened-prompt probe passed,
  so no positive case was produced)
- whether a second global account behaves like the first — only one was available,
  so the global side of the pool has **no redundancy**
- whether `/v2/messages` exists on a different host or version prefix
- **resolved since:** the `-model` aliases (`default-model`, `fast-model`,
  `balanced-model`) answer `200` once §4's two gates are satisfied;
  `primary-model`, `deep-model` and the GPT/Gemini ids are plan-gated (`11158`)
  or withdrawn (`11102`)
- **resolved since:** the global account chats. The earlier "cannot stream chat"
  conclusion came from an incomplete sweep — see §4
- image input: the catalog marks models multimodal, no image was sent
- how the conversation-id triple is used for cache accounting upstream; the three
  are derived locally and never read back

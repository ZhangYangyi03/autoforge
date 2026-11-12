# Reusing a tool that lives on the shelf

`market_search` finds tools on the shared toolmarket shelf. A hit does **not**
mean the tool is registered in the current process: `describe_tool` will report
"no tool named ...". Do not re-forge it and do not hand-roll the same logic.

Call it through the shelf instead:

```python
from autoforge.market_invoke import market_tool_invoke

market_tool_invoke("tool:mail_inbox_query", {
    "account": "gmail",
    "query": 'from:welcome@junehomes.com "Confirmation code"',
    "limit": 3,
})
```

Equivalent raw request:

```
POST http://127.0.0.1:8000/resources/tool:mail_inbox_query/invoke
Content-Type: application/json

{"arguments": {"account": "gmail", "query": "...", "limit": 3}}
```

The response is the shelf's own envelope: `resource_id`, `ok`, `output`,
`error`. Check `ok`, not just the HTTP status.

## Pitfalls measured on 2026-11-12

* `/search` can be slow or time out while `/resources/{id}` and `/invoke`
  answer immediately. A known id does not need the search endpoint.
* `forge_tool` can fail when its code-generation backend is unavailable, but
  the shelf is still callable. Publish a stdlib-only single-file tool if needed;
  `RegisterRequest.effect_signature` must be a string, not a dict, or the
  shelf answers 422.
* A dead shelf returns `{"ok": false, ...}` from this client. It never raises,
  so a cold-tool lookup cannot abort the turn.

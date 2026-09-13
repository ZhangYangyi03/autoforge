"""Time one forge-shaped call and show where the seconds go.

The complaint is "waiting on the model takes minutes". Reading the client rules
out the usual suspects (the CLI passes timeout=600, so a slow answer is not a
timeout being retried), which leaves the endpoint itself. This measures it and,
crucially, prints the token accounting: a `deepseek-chat` that spends its whole
budget on `reasoning_content` is a thinking model wearing a non-thinking name,
and that is the difference between "slow network" and "we are paying for a
chain of thought we never asked for".
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoforge.core.llm import OpenAICompatClient           # noqa: E402
from autoforge.core.message import Message                  # noqa: E402

cfg = json.loads((Path.home() / ".autoforge" / "config.json").read_text())
print(f"model={cfg['model']}  base={cfg['base_url']}  max_tokens={cfg.get('max_tokens')}")

client = OpenAICompatClient(
    model=cfg["model"], base_url=cfg["base_url"], api_key=cfg["api_key"],
    timeout=600, proxies=None,
)

# The forge generator's job: write a Python tool module from a need. Long input,
# long output -- the shape that is actually slow in the log.
need = (
    "Search the host filesystem for any file, directory, or text pattern and "
    "return the matches. Support glob names and literal content search, an "
    "optional root, a result cap, and a readable listing. Fail loudly on a bad "
    "root instead of returning an empty list."
)
messages = [
    Message(role="system", content=(
        "You write a single self-contained Python module implementing exactly "
        "one tool. Define `def run(args: dict) -> dict` returning a dict with "
        "keys ok/result/error. No imports beyond the standard library."
    )),
    Message(role="user", content=f"Tool need: {need}\n\nWrite the module."),
]

for attempt in (1, 2):
    t0 = time.time()
    resp = client.chat(messages)
    dt = time.time() - t0
    content = resp.content or ""
    reasoning = getattr(resp, "reasoning", "") or ""
    usage = (resp.raw or {}).get("usage") or {}
    print(f"\n--- attempt {attempt} ---")
    print(f"wall_clock      : {dt:.1f}s")
    print(f"content_chars   : {len(content)}")
    print(f"reasoning_chars : {len(reasoning)}")
    print(f"tool_calls      : {len(resp.tool_calls)}")
    print(f"usage           : {json.dumps(usage, ensure_ascii=False)}")
    if reasoning:
        print(f"reasoning head  : {reasoning[:160]!r}")
    print(f"content head    : {content[:120]!r}")
    if attempt == 1 and dt < 20:
        print("\n(fast enough that a second sample is not worth the wall clock)")
        break

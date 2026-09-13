"""Run one real forge and attribute the wall clock to each model call.

The log says a single round took 254s and a later one sat "waiting on model" for
80s. A direct probe of the same endpoint answered in 5.5s, so the endpoint is
not slow and the round is not one call. This counts and times them: the round is
generate + adversarial + trigger + negative, where the last two each spin up a
two-turn tool-using sub-agent, and every one of those is a separate round-trip.

Prints a table: sequence, seconds, approx prompt size, and the calling frame.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoforge.core.llm import OpenAICompatClient           # noqa: E402
from autoforge.cli import _build                            # noqa: E402

cfg = json.loads((Path.home() / ".autoforge" / "config.json").read_text())

# `_build` speaks the resolved vocabulary (`base`, `key`, `fast`), not the raw
# config file's (`base_url`, `api_key`). Normalise the same way the CLI does,
# so the probe measures the real assembly rather than a hand-rolled one.
cfg = {
    "model": cfg["model"],
    "base": cfg["base_url"],
    "key": cfg["api_key"],
    "proxy": bool(cfg.get("proxy", False)),
    "max_tokens": int(cfg.get("max_tokens") or 3000),
    "fast": bool(cfg.get("fast", False)) or os.environ.get("AUTOFORGE_FAST") == "1",
    "policy": cfg.get("policy", "full"),
}

_CALLS: list[dict] = []


class TimedClient(OpenAICompatClient):
    """Same client, but every round-trip is measured and attributed."""

    def chat(self, messages, tools=None, **kwargs):
        # The deepest non-autoforge frame is what actually asked, i.e. whether
        # this is the generator, the adversarial gate, or a verification agent.
        caller = "?"
        for fr in reversed(traceback.extract_stack()[:-1]):
            if "autoforge" in fr.filename:
                caller = f"{Path(fr.filename).name}:{fr.lineno} {fr.name}"
                break
        prompt_chars = sum(len(getattr(m, "content", "") or "") for m in messages)
        t0 = time.time()
        resp = super().chat(messages, tools=tools, **kwargs)
        dt = time.time() - t0
        _CALLS.append({
            "s": round(dt, 1),
            "prompt_chars": prompt_chars,
            "tools": len(tools or []),
            "out": len(resp.content or ""),
            "calls": len(resp.tool_calls),
            "who": caller,
        })
        print(f"  [{len(_CALLS):>2}] {dt:6.1f}s  prompt={prompt_chars:>6}c "
              f"tools={len(tools or []):>2}  out={len(resp.content or ''):>5}c  {caller}")
        return resp


agent = _build(cfg)
agent.llm = TimedClient(
    model=cfg["model"], base_url=cfg["base"], api_key=cfg["key"],
    timeout=600, proxies=None,
)
# Re-point every collaborator at the measured client, or the swap only catches
# the top-level agent and silently misses the verifier and the generator.
agent.pipeline.llm = agent.llm
agent.pipeline.generator.llm = agent.llm
agent.pipeline.verifier.llm = agent.llm
if getattr(agent.pipeline.verifier, "adversary", None) is not None:
    agent.pipeline.verifier.adversary.llm = agent.llm

need = ("Search the host filesystem for any file, directory, or text pattern "
        "and return the matches.")

print(f"model={cfg['model']}  max_tokens={cfg.get('max_tokens')}  "
      f"max_rounds={agent.pipeline.config.max_rounds}  fast={cfg.get('fast')}")
print("\n--- forging ---")
t0 = time.time()
result = agent.pipeline.forge(need)
total = time.time() - t0

print(f"\n=== total {total:.1f}s over {len(_CALLS)} model calls ===")
if _CALLS:
    slowest = sorted(_CALLS, key=lambda c: -c["s"])[:3]
    print("slowest: " + ", ".join(f"{c['s']}s @ {c['who']}" for c in slowest))
    print(f"model time: {sum(c['s'] for c in _CALLS):.1f}s "
          f"({100 * sum(c['s'] for c in _CALLS) / total:.0f}% of the round)")
print(f"ok={result.ok}  rounds={len(result.attempts)}")
for a in result.attempts:
    print(f"  round seen: accepted={a.accepted} error={(a.error or '')[:90]}")

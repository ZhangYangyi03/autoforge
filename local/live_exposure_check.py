"""One live turn, to prove the exposure split does not confuse the model.

The injected cold manifest is a `user` message placed immediately after the
system prompt and *before* the task. That is a real risk worth paying for: a
model that reads the first user turn as the task would answer the manifest
instead of the question. This runs one real turn through the real gateway and
prints exactly what went out and what came back.

Needs the sandbox off -- the sandbox SIGTERMs Python processes that make
network calls.

    C:/autoforge/venv/Scripts/python.exe live_exposure_check.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

from autoforge.agent import ForgeAgent
from autoforge.configfile import load
from autoforge.core.llm import OpenAICompatClient
from autoforge.store import ToolStore

TASK = "In one short sentence: what is 17 * 23? Do not call any tool."


def main() -> int:
    cfg = load()
    base = cfg.get("base_url") or cfg.get("base") or ""
    key = cfg.get("api_key") or cfg.get("key") or ""
    model = cfg.get("model") or ""
    print(f"  gateway : {base}")
    print(f"  model   : {model}")
    print(f"  key     : {key[:6]}... (len {len(key)})")
    if not (base and key and model):
        print("  !! config is incomplete")
        return 2

    llm = OpenAICompatClient(
        model=model, base_url=base, api_key=key, timeout=180,
        proxies=None, default_max_tokens=200,
    )

    # A copy of the real ledger, so the split is decided by real history.
    real_db = os.path.expanduser("~/AppData/Local/autoforge/autoforge.db")
    d = tempfile.mkdtemp(prefix="live_exposure_")
    db = os.path.join(d, "agent.db")
    if os.path.exists(real_db):
        import shutil
        shutil.copy2(real_db, db)
    store = ToolStore(db)

    agent = ForgeAgent(llm=llm, store=store)
    reg = agent.registry

    # What the split decided, before the run.
    hot, cold, stats = reg.exposure()
    full = reg.schemas()
    full_chars = sum(len(json.dumps(s, ensure_ascii=False)) for s in full)
    hot_chars = sum(len(json.dumps(s, ensure_ascii=False)) for s in hot)
    cold_chars = sum(len(x) for x in cold)
    print()
    print("  --- what the loop will carry ---")
    print(f"  visible       : {len(full)} schemas, {full_chars:,} chars")
    print(f"  hot (in full) : {len(hot)} schemas, {hot_chars:,} chars")
    print(f"  cold (a line) : {len(cold)} entries, {cold_chars:,} chars")
    print(f"  saving        : {full_chars - hot_chars - cold_chars:,} chars "
          f"({100.0 * (1 - (hot_chars + cold_chars) / full_chars):.1f}%)")

    # Wrap the client so we can see the exact request the loop built.
    seen: dict = {}
    real_chat = llm.chat

    def spy(msgs, *, tools=None, **kw):
        seen["msgs"] = list(msgs)
        seen["tools"] = tools or []
        seen["t0"] = time.time()
        out = real_chat(msgs, tools=tools, **kw)
        seen["dt"] = time.time() - seen["t0"]
        seen["usage"] = getattr(out, "usage", None)
        return out

    llm.chat = spy

    print()
    print(f"  --- running: {TASK!r} ---")
    t0 = time.time()
    result = agent.run(TASK)
    elapsed = time.time() - t0

    print()
    print("  --- the request the loop actually built ---")
    msgs = seen.get("msgs", [])
    for i, m in enumerate(msgs):
        head = m.content.split("\n")[0][:78]
        print(f"    [{i}] {m.role:9s} {head}")
    tools = seen.get("tools", [])
    print(f"    tools sent in full: {len(tools)} "
          f"-> {[(t.get('function') or t).get('name') for t in tools]}")
    if len(msgs) > 1 and "not carrying in full" in msgs[1].content:
        print()
        print("    the injected block, verbatim:")
        for line in msgs[1].content.split("\n")[:4]:
            print(f"      {line[:100]}")
        print(f"      ... ({len(msgs[1].content.splitlines()) - 1} tool lines total)")
    print()
    print("  --- the answer ---")
    print(f"    turns      : {result.turns}")
    print(f"    tool_calls : {result.tool_calls}")
    print(f"    elapsed    : {elapsed:.1f}s")
    print(f"    usage      : {seen.get('usage')}")
    print()
    print("    content:")
    for line in (result.content or "").strip().split("\n"):
        print(f"      {line}")

    print()
    # Did it answer the question, or the manifest?
    text = (result.content or "").lower()
    ok = "391" in text
    confused = "not carrying in full" in text or "describe_tool" in text
    print(f"  answered the arithmetic (391 present): {ok}")
    print(f"  treated the manifest as the task    : {confused}")
    if ok and not confused:
        print("  PASS: the split is engaged and the model is not confused by it.")
    elif confused:
        print("  FAIL: the model answered the manifest instead of the task.")
    else:
        print("  INCONCLUSIVE: no 391 in the answer -- read the content above.")

    try:
        store._conn.close()
    except Exception:
        pass
    return 0 if (ok and not confused) else 1


if __name__ == "__main__":
    sys.exit(main())

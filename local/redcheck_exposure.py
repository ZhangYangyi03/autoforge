"""Red-check for the exposure backport.

A test that passes proves nothing until you have seen it fail for the right
reason. This measures what the loop actually hands the model, then removes the
single line that wires the split in, measures again, and restores.

Two things this script gets right that the first draft did not:

* **Each measurement runs in its own process.** `autoforge.agent` is imported
  once per interpreter, so rewriting the file on disk inside one process
  changes nothing -- the first draft "proved" the split was a no-op because it
  was measuring the same cached module twice.
* **It measures against a copy of the real store**, so the hot set is decided
  by the real ledger (150 tools ever called, 19 of them inside the week) rather
  than by a fresh empty db where everything is cold and the floor is all there
  is to compare.

Expected, on this machine:
    wired    -> ~14 tools in full, plus a cold manifest block
    unwired  ->  59 tools in full, no block
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

AGENT_PY = r"C:\autoforge\venv\Lib\site-packages\autoforge\agent.py"
REAL_DB = os.path.expanduser("~/AppData/Local/autoforge/autoforge.db")
WIRING = "            exposure=self.registry,\n"


# --------------------------------------------------------------------------
# child mode: measure once, in a clean interpreter
# --------------------------------------------------------------------------
def measure_once() -> int:
    from autoforge.agent import ForgeAgent
    from autoforge.core.llm import MockLLMClient
    from autoforge.store import ToolStore

    d = tempfile.mkdtemp(prefix="redcheck_")
    db = os.path.join(d, "agent.db")
    if os.path.exists(REAL_DB):
        shutil.copy2(REAL_DB, db)
    store = ToolStore(db)

    llm = MockLLMClient()
    agent = ForgeAgent(llm=llm, store=store)
    try:
        # A task that names no tool, so `protect` is empty and the hot set is
        # decided purely by the ledger -- which is what we want to compare.
        agent.run("hello there")
    finally:
        try:
            store._conn.close()
        except Exception:
            pass

    msgs, tools = llm.calls[-1]
    tools = tools or []
    tool_chars = sum(len(json.dumps(t, ensure_ascii=False)) for t in tools)
    block = next((m.content for m in msgs
                  if "not carrying in full" in m.content), "")
    json.dump({
        "tools": len(tools),
        "tool_chars": tool_chars,
        "block_chars": len(block),
        "injected": bool(block),
        "visible": len(agent.registry.schemas()),
        "full_chars": sum(len(json.dumps(s, ensure_ascii=False))
                          for s in agent.registry.schemas()),
    }, sys.stdout)
    return 0


# --------------------------------------------------------------------------
# parent mode: orchestrate
# --------------------------------------------------------------------------
def run_child(label: str) -> dict:
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--measure"],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        print(f"  {label}: child failed\n{proc.stderr[-1500:]}")
        raise SystemExit(1)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    total = data["tool_chars"] + data["block_chars"]
    print(f"  {label:9s} tools={data['tools']:3d}  "
          f"tool_chars={data['tool_chars']:6,d}  "
          f"block_chars={data['block_chars']:6,d}  "
          f"total={total:6,d}  injected={data['injected']}")
    return data


def clear_cache() -> None:
    for base in (os.path.dirname(AGENT_PY),
                 os.path.join(os.path.dirname(AGENT_PY), "core")):
        cache = os.path.join(base, "__pycache__")
        if not os.path.isdir(cache):
            continue
        for name in os.listdir(cache):
            if name.startswith("agent.") or name.startswith("registry."):
                try:
                    os.remove(os.path.join(cache, name))
                except OSError:
                    pass


def main() -> int:
    print("=== exposure backport red-check ===")
    print(f"  source : {AGENT_PY}")
    print(f"  ledger : {REAL_DB} "
          f"({'present' if os.path.exists(REAL_DB) else 'MISSING'})")
    print()

    with open(AGENT_PY, encoding="utf-8") as fh:
        original = fh.read()
    if WIRING not in original:
        print("  !! the wiring line is not in agent.py -- nothing to red-check")
        return 2

    try:
        print("  GREEN state (the backport as installed):")
        clear_cache()
        green = run_child("wired")

        with open(AGENT_PY, "w", encoding="utf-8") as fh:
            fh.write(original.replace(WIRING, "", 1))
        print()
        print("  RED state (the one wiring line removed):")
        clear_cache()
        red = run_child("unwired")
    finally:
        with open(AGENT_PY, "w", encoding="utf-8") as fh:
            fh.write(original)
        clear_cache()

    print()
    print("  restored; re-measuring to confirm the restore took:")
    again = run_child("restored")
    print()

    def total(d):
        return d["tool_chars"] + d["block_chars"]

    ok = True
    if red["tools"] != red["visible"]:
        print(f"  FAIL: unwired sent {red['tools']} of {red['visible']} visible "
              "schemas -- it should send all of them")
        ok = False
    if green["tools"] >= red["tools"]:
        print(f"  FAIL: the split did not shrink the tool count "
              f"({green['tools']} vs {red['tools']})")
        ok = False
    if total(green) >= total(red):
        print(f"  FAIL: the split did not shrink the payload "
              f"({total(green):,} vs {total(red):,})")
        ok = False
    if red["injected"]:
        print("  FAIL: the unwired run injected a block it should not have")
        ok = False
    if not green["injected"]:
        print("  FAIL: the wired run did not inject the cold manifest")
        ok = False
    if (again["tools"], total(again)) != (green["tools"], total(green)):
        print(f"  FAIL: the restore did not take "
              f"({again['tools']}/{total(again):,} vs "
              f"{green['tools']}/{total(green):,})")
        ok = False

    if ok:
        saved = total(red) - total(green)
        cut = 100.0 * saved / total(red)
        print(f"  PASS: {total(red):,} -> {total(green):,} chars per request; "
              f"{saved:,} saved ({cut:.1f}% cut).")
        print(f"        {red['tools']} schemas in full -> {green['tools']} "
              f"in full + {red['visible'] - green['tools']} as one line each.")
        return 0
    return 1


if __name__ == "__main__":
    if "--measure" in sys.argv:
        sys.exit(measure_once())
    sys.exit(main())

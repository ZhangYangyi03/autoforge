"""Does the built-in fallback table carry a reasoning ladder at all?

The picker reads the catalog, and the catalog starts as the built-in table
(`source = "fallback"` at construction, lib/index.js:4593) until a live fetch
lands. If a live fetch never lands for a region, the picker renders the built-in
rows -- and `deepseek-v4.1-flash`'s built-in row (lib/index.js:4429-4435) has no
`reasoning` field whatsoever.

This extracts the built-in table from the real build and reports, per model,
whether it declares any reasoning ladder. It is the one piece of the chain that
can be settled from the file on disk.
"""
import re
import os
import sys
from pathlib import Path

#: The built xdpool file this reads. Override with XDPOOL_LIB so the probe is
#: not tied to one machine's profile path.
LIB = Path(os.environ.get(
    "XDPOOL_LIB",
    Path.home() / ".dsh" / "profiles" / "desktop" / "node_modules"
    / "dsh-xdpool" / "lib" / "index.js"))
TARGETS = ["deepseek-v4.1-flash", "step-5-preview", "hy3", "glm-5.3",
           "kimi-k3", "gpt-5.6-terra", "space-bunny"]


def main() -> int:
    src = LIB.read_text(encoding="utf-8")
    lines = src.splitlines()

    # The CN table runs from its declaration to the GLOBAL one, which is derived
    # from it by mapping. Reading the CN table alone answers the question.
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("const FALLBACK_WORKBUDDY_MODELS = ["))
    end = next(i for i, l in enumerate(lines)
               if i > start and l.startswith("const FALLBACK_WORKBUDDY_MODELS_GLOBAL"))
    block = lines[start:end]

    print(f"built-in table: lines {start+1}-{end} ({end-start-2} entries)\n")

    # Split the array literal into per-entry chunks on the object boundaries.
    text = "\n".join(block)
    entries = re.findall(r"\{\s*id:\s*\"([^\"]+)\"(.*?)\n\t\}", text, re.S)
    if not entries:
        # The minified/pretty form may differ; fall back to a line scan.
        entries = []
        cur_id, cur_lines = None, []
        for l in block:
            m = re.match(r'\s*id:\s*"([^"]+)"', l)
            if m:
                if cur_id:
                    entries.append((cur_id, "\n".join(cur_lines)))
                cur_id, cur_lines = m.group(1), [l]
            elif cur_id:
                cur_lines.append(l)
        if cur_id:
            entries.append((cur_id, "\n".join(cur_lines)))

    found = {mid: body for mid, body in entries}
    print(f"parsed {len(found)} entries\n")
    print(f"{'id':24s} {'reasoning ladder':>18s}  detail")
    print("-" * 84)
    for mid in TARGETS:
        body = found.get(mid)
        if body is None:
            print(f"{mid:24s} {'(absent from table)':>18s}")
            continue
        has = "reasoningEfforts" in body or "supportedEfforts" in body \
            or "reasoning" in body
        mark = "present" if has else "MISSING"
        detail = ""
        m = re.search(r"reasoningEfforts:\s*\[([^\]]*)\]", body)
        if m:
            detail = "reasoningEfforts=[" + m.group(1).strip() + "]"
        m2 = re.search(r"supportsReasoning:\s*(\w+)", body)
        if m2:
            detail += f" supportsReasoning={m2.group(1)}"
        print(f"{mid:24s} {mark:>18s}  {detail}")

    print()
    with_ladder = [m for m, b in found.items()
                   if "reasoningEfforts" in b or "supportedEfforts" in b]
    print(f"entries WITH a ladder    : {len(with_ladder):2d}")
    print(f"entries WITHOUT a ladder : {len(found) - len(with_ladder):2d}")
    print()
    print("A row with no ladder makes thinkingLevelMap() return undefined,")
    print("which sets reasoning:false and supportsReasoningEffort:false --")
    print("the picker then shows the level as 'none'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

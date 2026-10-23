"""Does xdpool's effort folding actually produce a ladder for the singular form?

`parseReasoning` (lib/index.js:800) is supposed to fold the singular
`{"effort": "high", "summary": "auto"}` shape into a full ladder via
`singularEffortLadder` -> `SINGULAR_EFFORT_LADDER`. The catalog publishes that
shape for 24 CN models and 10 global ones, including `hy3`,
`deepseek-v4.1-flash` and `glm-5.2` -- exactly the models whose thinking level
reads as "off" in the picker.

So the fold is either not running or not working. Rather than reading the code
again, this calls the real exported module and prints what it returns, per raw
shape, straight from the built file on disk.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

#: The built xdpool file this slices functions out of. Override with
#: XDPOOL_LIB so the probe is not tied to one machine's profile path.
LIB = Path(os.environ.get(
    "XDPOOL_LIB",
    Path.home() / ".dsh" / "profiles" / "desktop" / "node_modules"
    / "dsh-xdpool" / "lib" / "index.js"))

CASES = [
    ("singular, effort=high", {"effort": "high", "summary": "auto"}),
    ("singular, effort=medium", {"effort": "medium", "summary": "auto"}),
    ("singular, bare", {"effort": "low"}),
    ("plural, deepseek-v4.1-flash",
     {"canDisableThinking": True, "defaultEffort": "high",
      "supportedEfforts": ["low", "high", "max"]}),
    ("absent", None),
]


def main() -> int:
    # The library is ESM with a huge import surface; extracting just the two
    # functions under test keeps this runnable without booting DSH.
    src = LIB.read_text(encoding="utf-8")
    lines = src.splitlines()

    # Slice by line number rather than by brace-matching: brace counting is
    # confused by braces inside string literals in the built file, and an
    # over-long slice re-declares an identifier and kills the module.
    spans = {
        "ladder": (765, 771),        # const SINGULAR_EFFORT_LADDER = [...]
        "is_singular": (778, 780),   # function isSingularEffortForm(raw) {...}
        "fold": (789, 793),          # function singularEffortLadder(raw) {...}
        "parse": (800, 814),         # function parseReasoning(value) {...}
        "level_map": (11878, 11893),  # function thinkingLevelMap(info) {...}
    }
    expect = {
        "ladder": "const SINGULAR_EFFORT_LADDER = [",
        "is_singular": "function isSingularEffortForm(",
        "fold": "function singularEffortLadder(",
        "parse": "function parseReasoning(",
        "level_map": "function thinkingLevelMap(",
    }
    parts = []
    for name, (a, b) in spans.items():
        block = "\n".join(lines[a - 1:b])
        if expect[name] not in block:
            raise SystemExit(
                f"{name}: lines {a}-{b} do not contain {expect[name]!r};\n"
                f"got:\n{block[:300]}")
        parts.append(block)

    cases_js = json.dumps([[l, v] for l, v in CASES])
    js = "\n".join(parts + ["""
const cases = CASES_JSON;
const out = cases.map(([label, raw]) => {
  let parsed;
  try { parsed = parseReasoning(raw); } catch (e) { parsed = {ERROR: String(e)}; }
  let map;
  try { map = thinkingLevelMap(parsed || {}); } catch (e) { map = {ERROR: String(e)}; }
  return {label, parsed, map};
});
console.log(JSON.stringify(out, null, 1));
""".replace("CASES_JSON", cases_js)])

    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(js)
        path = fh.name
    try:
        res = subprocess.run(["node", path], capture_output=True, text=True,
                             timeout=60)
        if res.returncode != 0:
            print("node failed:", res.stderr[:600])
            return 1
        for row in json.loads(res.stdout):
            print("=" * 78)
            print(row["label"])
            print("  parseReasoning ->", json.dumps(row["parsed"], ensure_ascii=False))
            map_ = row["map"]
            if isinstance(map_, dict):
                levels = {k: v for k, v in map_.items() if k != "off"}
                usable = [k for k, v in levels.items() if v is not None]
                print(f"  thinkingLevelMap -> {len(usable)} usable level(s): {usable}")
            else:
                print("  thinkingLevelMap ->", map_)
    finally:
        Path(path).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

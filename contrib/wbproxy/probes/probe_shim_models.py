"""Ask the running xdpool shim what it advertises for the affected models.

The reasoning ladder comes from the catalog, and the catalog is either live or
the built-in table. The built-in entry for `deepseek-v4.1-flash` carries no
`reasoning` field at all (lib/index.js:4429-4435), so if the picker is showing
that table, the level reads as "none".

Reading the source cannot settle which one DSH is actually serving. The shim
exposes `/v1/models` on a loopback port, so this asks it directly and prints
what the model rows carry -- which is exactly what the picker renders from.
"""
import json
import re
import sys
import urllib.error
import urllib.request

TARGETS = ["deepseek-v4.1-flash", "step-5-preview", "hy3", "glm-5.3",
           "kimi-k3", "gpt-5.6-terra"]


def find_shim_port() -> int | None:
    """The loopback port xdpool's shim bound, read from the plugin's own state."""
    import os
    from pathlib import Path
    root = Path(os.environ.get("DSH_HOME", Path.home() / ".dsh"))
    for name in ("proxy-origin", "port-zone.json"):
        p = root / name
        if not p.exists():
            continue
        try:
            raw = p.read_text(encoding="utf-8")
        except Exception:
            continue
        m = re.search(r'"lanes"\s*:\s*\{[^}]*"dsh"\s*:\s*(\d+)', raw)
        if m:
            return int(m.group(1))
    return None


def main() -> int:
    port = find_shim_port()
    if port is None:
        print("could not find the xdpool shim port")
        print("(looked for DSH_HOME/proxy-origin and DSH_HOME/port-zone.json)")
        return 1
    print(f"shim port: {port}")

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                    timeout=30) as r:
            doc = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"GET /v1/models -> {e.code}")
        return 1
    except Exception as exc:                          # noqa: BLE001
        print(f"GET /v1/models failed: {type(exc).__name__}: {exc}")
        return 1

    rows = doc.get("data", []) if isinstance(doc, dict) else []
    print(f"models advertised: {len(rows)}\n")

    by_id = {str(r.get("id")): r for r in rows}
    print(f"{'id':24s} {'reasoning':>10s} {'supportsEffort':>15s}  thinkingLevelMap")
    print("-" * 92)
    for mid in TARGETS:
        r = by_id.get(mid)
        if r is None:
            # The shim namespaces its ids; try the namespaced spelling.
            hit = [x for x in rows if str(x.get("id", "")).endswith(mid)]
            r = hit[0] if hit else None
        if r is None:
            print(f"{mid:24s} {'(not offered)':>10s}")
            continue
        keys = [k for k in r if "reason" in k.lower() or "effort" in k.lower()
                or "thinking" in k.lower()]
        reasoning = r.get("reasoning", "-")
        effort = r.get("compat", {}).get("supportsReasoningEffort", "-") \
            if isinstance(r.get("compat"), dict) else "-"
        ladder = r.get("thinkingLevelMap")
        ladder_s = "-"
        if isinstance(ladder, dict):
            usable = [k for k, v in ladder.items() if v is not None and k != "off"]
            ladder_s = ",".join(usable) if usable else "(all null)"
        print(f"{mid:24s} {str(reasoning):>10s} {str(effort):>15s}  {ladder_s}")
        if not keys:
            print(f"{'':24s}   no reasoning-ish key at all on this row")

    # Which region's roster is this? The ids say it.
    has = {str(r.get("id")) for r in rows}
    if any("gpt-5.6" in i or "gemini-3.5" in i for i in has):
        print("\nroster includes global-only ids -> the GLOBAL shim")
    if any(i in ("step-5-preview", "auto") for i in has):
        print("roster includes cn-only ids -> the CN shim")
    return 0


if __name__ == "__main__":
    sys.exit(main())

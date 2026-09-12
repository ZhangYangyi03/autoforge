"""Does omitting `max_tokens` really break the gateway, or was it drift?

Background. Two live forge runs against aiping.cn differed in one respect: the
first sent no `max_tokens`, the second sent 1024. The first failed with HTTP
503; the second did not. One request each, so either the cap changed the
gateway's routing, or the first run simply landed in a bad window.

A first sweep (two consecutive reps per cap, caps ascending) put the failures
at the two ends -- none and 8192 -- which a server degrading over time would
also produce. Ordering cannot be ruled out from that data.

Method. Interleave: cycle through the caps round-robin, three full cycles, so
wall-clock drift lands on every cap equally and cannot impersonate an effect.
Record `finish_reason` too: a cap that routes fine but always truncates cannot
carry a generated tool envelope, so a 200 alone is not success.

Run:
    AIPING_API_KEY=... python probes/probe_gateway_max_tokens.py

Finding (2026-09-12). Interleaved sweep, 3 cycles, caps {2048, 3000, 4096}:
every capped request returned 200 and every uncapped request returned 503 --
9/9 versus 0/3, with the two outcomes adjacent in wall-clock inside a single
cycle. Presence of the field is what the gateway routes on; its *size* is not.
An earlier ascending sweep (not interleaved) put a failure at 4096 and blamed
the size; this sweep returned 200 at 4096 on all three reps, so that reading
was the sweep's own drift. Its 8192 failure was never replicated and is left
open. Both sweeps are recorded in `probes/FINDINGS.md`.

Separate observation, same data: every 200 came back with
`finish_reason=length`, at every cap including 4096. This model always runs
into the ceiling, so a truncated completion is the normal case, not the
exception -- which is why `autoforge.forge.json_repair` and the generator's
"be self-contained and terse" rule are load-bearing rather than belt-and-braces.

`OpenAICompatClient.DEFAULT_MAX_TOKENS` exists because of this probe.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoforge.forge.generator import GENERATOR_SYSTEM  # noqa: E402

URL = os.environ.get("PROBE_URL", "https://aiping.cn/api/v1/chat/completions")
MODEL = os.environ.get("PROBE_MODEL", "DeepSeek-V4.1-Flash")
USER = "Recurring need: normalise ISBN identifiers. Emit the JSON envelope now."
CAPS: list[int | None] = [None, 1024, 2048, 4096]
CYCLES = 3


def once(cap: int | None, key: str) -> str:
    body: dict[str, object] = {
        "model": MODEL,
        "messages": [{"role": "system", "content": GENERATOR_SYSTEM},
                     {"role": "user", "content": USER}],
        "temperature": 0.0,
    }
    if cap is not None:
        body["max_tokens"] = cap
    try:
        resp = requests.post(
            URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=90,
        )
    except Exception as exc:  # noqa: BLE001
        return f"EXC/{type(exc).__name__}"
    if resp.status_code != 200:
        return str(resp.status_code)
    try:
        reason = resp.json()["choices"][0].get("finish_reason") or "?"
    except Exception:  # noqa: BLE001
        reason = "?"
    return f"200/{reason[:4]}"


def main() -> None:
    key = os.environ.get("AIPING_API_KEY") or os.environ.get("PROBE_API_KEY")
    if not key:
        raise SystemExit("set AIPING_API_KEY (or PROBE_API_KEY)")

    def label(cap: int | None) -> str:
        return "none" if cap is None else str(cap)

    print(f"{URL}  model={MODEL}  interleaved, {CYCLES} cycles", flush=True)
    print(f"start {time.strftime('%H:%M:%S')}", flush=True)

    seen: dict[str, list[str]] = defaultdict(list)
    for cycle in range(CYCLES):
        row = []
        for cap in CAPS:
            out = once(cap, key)
            seen[label(cap)].append(out)
            row.append(f"{label(cap)}={out}")
        print(f"  cycle {cycle + 1}: " + "  ".join(row), flush=True)

    print("--- per cap ---", flush=True)
    verdict_ok = True
    for cap in CAPS:
        outs = seen[label(cap)]
        ok = sum(1 for o in outs if o.startswith("200"))
        trunc = sum(1 for o in outs if o.endswith("leng"))
        print(f"  max_tokens={label(cap):<5} 200s {ok}/{len(outs)}  "
              f"truncated {trunc}/{len(outs)}", flush=True)
        if ok != len(outs):
            verdict_ok = False
    print(f"end {time.strftime('%H:%M:%S')}", flush=True)
    print("VERDICT: every cap returned 200 -- max_tokens is not the discriminator"
          if verdict_ok else
          "VERDICT: at least one cap did not return 200 -- cap affects routing",
          flush=True)


if __name__ == "__main__":
    main()

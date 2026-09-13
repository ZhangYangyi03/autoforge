"""Dump the generator's raw reply so the parse failure can be seen, not guessed.

Context: with `Qwen3.5-Flash` the generator finally answers — 2490 characters of
envelope at `finish_reason='stop'` — but `extract_json` rejects it. `stop` means
the model believed it finished, so this is not the truncation problem the
reasoning-wall investigation was about. Something in the text is malformed, and
the only way to know what is to look at the bytes.

Writes the raw reply to `probes/raw_generator_reply.txt` and reports what the
repair ladder makes of it.
"""
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from autoforge.core.llm import OpenAICompatClient
from autoforge.core.message import Message
from autoforge.forge import generator as G

NEED = ("I keep needing to validate and normalise ISBN-10 and ISBN-13 identifiers: "
        "strip hyphens, check the check digit, convert a valid ISBN-10 to ISBN-13")
OUT = pathlib.Path(__file__).with_name("raw_generator_reply.txt")


def main() -> int:
    key = os.environ.get("AIPING_API_KEY") or os.environ.get("AUTOFORGE_API_KEY")
    if not key:
        print("no AIPING_API_KEY in the environment", file=sys.stderr)
        return 2
    # The gateway 503s in bursts, so let the client's own retry ladder do its job
    # rather than hand-rolling a wait here.
    llm = OpenAICompatClient("Qwen3.5-Flash", "https://aiping.cn/api/v1",
                             key, max_attempts=5)
    gen = G.LLMToolGenerator(llm, max_tokens=6000)
    # Same prompt the pipeline builds, so the reply under inspection is the one
    # the failing round actually received.
    prompt = f"Recurring need:\n{NEED}\n\nEmit the JSON envelope now."
    resp = llm.chat([Message.system(gen.system_prompt), Message.user(prompt)],
                    tools=None, max_tokens=6000)

    OUT.write_text(resp.content, encoding="utf-8")
    print(f"finish={resp.finish_reason!r}  content={len(resp.content)}c  "
          f"reasoning={len(resp.reasoning)}c  -> {OUT.name}")

    ok = G.extract_json(resp.content)
    print(f"extract_json -> {'parsed' if ok else 'None'}")

    if ok is None:
        # Where exactly does it stop being JSON? Show the boundary.
        import json
        try:
            json.loads(resp.content)
            print("plain json.loads actually succeeds?!")
        except json.JSONDecodeError as exc:
            print(f"json.loads: {exc.msg} at line {exc.lineno} col {exc.colno} "
                  f"(char {exc.pos} of {len(resp.content)})")
            lo = max(0, exc.pos - 120)
            print("…context around the failure:")
            print(repr(resp.content[lo:exc.pos + 120]))
        print(f"tail of reply: {resp.content[-200:]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

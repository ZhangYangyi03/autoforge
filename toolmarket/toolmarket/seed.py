"""Demo tools for the dashboard, so a fresh `tool-market web` shows something.

These are deliberately tiny and self-contained: the point of the page is the
*substrate* around them — the lifecycle, the event log, the lineage DAG — not the
tools. A tool simple enough to read in one screen keeps the attention where it
belongs.

The three mirror the shapes an agent tool actually takes:
  slugify         pure text transform, no effects
  parse_duration  pure parse, returns a number
  redact_secrets  a transform whose whole job is to remove something

Nothing here is imported by the protocol or the tests; the dashboard is a view
over the registry, and the registry does not know this module exists.
"""
from __future__ import annotations

from typing import Any

SLUGIFY = (
    "import re\n"
    "\n"
    "def slugify(text=''):\n"
    "    text = re.sub(r'[^A-Za-z0-9\\s-]', '', text).strip().lower()\n"
    "    return re.sub(r'[-\\s]+', '-', text)\n"
)

PARSE_DURATION = (
    "import re\n"
    "\n"
    "_UNIT = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}\n"
    "\n"
    "def parse_duration(text=''):\n"
    "    total = 0\n"
    "    for n, u in re.findall(r'(\\d+)\\s*([smhd])', text.lower()):\n"
    "        total += int(n) * _UNIT[u]\n"
    "    return total\n"
)

REDACT_SECRETS = (
    "import re\n"
    "\n"
    "_PATTERNS = [\n"
    "    (re.compile(r'sk-[A-Za-z0-9]{8,}'), '[REDACTED_KEY]'),\n"
    "    (re.compile(r'AKIA[0-9A-Z]{12,}'), '[REDACTED_AWS]'),\n"
    "]\n"
    "\n"
    "def redact_secrets(text=''):\n"
    "    for pat, tag in _PATTERNS:\n"
    "        text = pat.sub(tag, text)\n"
    "    return text\n"
)


def _fn(code: str, name: str) -> Any:
    """Compile the registered source and return the callable defined in it.

    The same string is used for both `code` and `fn`, so the dashboard executes
    exactly the tool it displays — there is no second copy to drift.
    """
    ns: dict[str, Any] = {}
    exec(compile(code, f"<tool:{name}>", "exec"), ns)  # noqa: S102
    return ns[name]


def demo_tools() -> list[Any]:
    """The three seed tools as autoforge ToolSpecs, not yet registered.

    Each carries triggers, and that is not decoration. The regression gate only
    has something to compare against when the base declares what it is supposed
    to respond to: with no triggers on the base, every candidate -- including
    the deliberately sabotaged one `StubProposer` appends -- clears the gate, and
    the console's veto panel shows a row of green forever. Verified by running
    the evolve loop both ways; see the note in `seed()`.
    """
    from autoforge.tools.spec import ToolSpec, TriggerProbe

    def spec(name: str, description: str, code: str, *, invariance: str,
             query: str, negative: str) -> Any:
        return ToolSpec(
            name=name,
            description=description,
            parameters={"type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"]},
            fn=_fn(code, name),
            code=code,
            source="human",
            probes=[TriggerProbe(query=query, expect="call",
                                 negative_query=negative)],
            invariances=[invariance],
            effect_signature="pure",
        )

    return [
        spec("slugify", "Turn arbitrary text into a URL-safe slug.",
             SLUGIFY, invariance="text",
             query="make a slug from 'Hello World!'",
             negative="what is the capital of France?"),
        spec("parse_duration", "Parse '2h30m' into seconds.",
             PARSE_DURATION, invariance="text",
             query="how many seconds is 2h30m?",
             negative="parse this sentiment score"),
        spec("redact_secrets", "Replace API keys and AWS ids with placeholders.",
             REDACT_SECRETS, invariance="text",
             query="strip the key sk-abcdefgh12345 from this log line",
             negative="write me a haiku about keys"),
    ]


def seed(registry: Any) -> list[Any]:
    """Register the demo tools and walk two of them to ACTIVE.

    The third is left at DRAFT on purpose so the dashboard shows a mixed state.

    What DRAFT does *not* mean: the tool is still callable. autoforge's registry
    blocks only QUARANTINED and RETIRED (see `ToolRegistry.call`), so a DRAFT
    resource invokes fine, and an earlier version of this docstring claimed
    otherwise. The state that actually gates a call is quarantine -- which is
    what `/resources/{id}/invoke` without `force` will show you.
    """
    from toolmarket.protocol.lifecycle import ResourceState

    records = []
    for i, spec in enumerate(demo_tools()):
        rec = registry.register(spec)
        if i < 2:
            registry.transition(rec.id, ResourceState.PROBATION,
                                reason="cleared trigger probes")
            registry.promote(rec.id)
        records.append(registry.get(rec.id))
    return records

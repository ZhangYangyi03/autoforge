"""Regression tests for tolerant JSON extraction in the forge generator.

Context: a small model inside the real forge loop can emit an envelope that
ends with finish_reason='stop' yet is not strict JSON. The defects are
*intermittent*, not characteristic of one model -- qwen2.5:7b has produced both
the broken envelope below and a clean one (``probes/raw_generator_reply_clean.txt``),
and Qwen3.5-Flash via aiping produced the stray-brace shape. Capture the sample,
not the model, because the next model will invent a third shape.

One failure mode is the Python source in the ``code`` field containing
unescaped double quotes:

    "code": "... raise ValueError("Invalid ISBN-10 check digit") ..."

The JSON string terminates early and json.loads dies with
"Expecting ',' delimiter". Before the repair layer this aborted the whole
forge; the envelope was thrown away even though it was structurally complete.
"""
from __future__ import annotations

import json
from pathlib import Path

from autoforge.forge.generator import extract_json, extract_json_any


def _strict_loads(text: str):
    """Return the parsed value, or None when the text is not strict JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# The exact shape that broke the real run (transcribed from the captured
# sample), including the unescaped inner quotes in the code field.
BROKEN_ENVELOPE = (
    "{\n"
    '  "name": "isbn_normalizer",\n'
    '  "description": "Normalise and validate ISBN identifiers",\n'
    '  "code": "def isbn_normalizer(isbn: str) -> str:\\n'
    "    if len(isbn) == 10:\\n"
    "        if int(isbn[9]) == 0:\\n"
    '            raise ValueError("Invalid ISBN-10 check digit")\\n'
    "        return isbn\\n"
    '    return isbn",\n'
    '  "entry": "isbn_normalizer",\n'
    '  "parameters": {"type": "object", "properties": {}},\n'
    '  "probes": [{"query": "normalise this isbn", "expect": "call"}]\n'
    "}"
)

BSLASH_QUOTE = "\\'"          # the illegal JSON escape small models emit


def test_broken_envelope_is_not_strict_json():
    """Guard the premise: this fixture must genuinely be unparseable."""
    assert _strict_loads(BROKEN_ENVELOPE) is None


def test_repair_recovers_unescaped_quotes():
    data = extract_json(BROKEN_ENVELOPE)
    assert isinstance(data, dict), "repair layer failed to recover the envelope"
    assert data["name"] == "isbn_normalizer"
    assert data["entry"] == "isbn_normalizer"
    assert len(data["probes"]) == 1


def test_repaired_code_is_valid_python():
    """The recovered ``code`` field must still compile -- repairing escapes
    must not corrupt the source the sandbox is about to execute."""
    data = extract_json(BROKEN_ENVELOPE)
    assert isinstance(data, dict)
    compile(data["code"], "<repaired>", "exec")


def test_clean_json_still_passes_through_unchanged():
    clean = json.dumps({"name": "t", "code": 'x = "ok"', "probes": []})
    assert extract_json(clean) == {"name": "t", "code": 'x = "ok"', "probes": []}


def test_escaped_quotes_are_left_intact():
    """Properly escaped quotes must not be double-escaped by the repair."""
    src = "def f():\\n    raise ValueError(\\\"bad\\\")"
    envelope = json.dumps({"name": "f", "code": src, "probes": []})
    data = extract_json(envelope)
    assert isinstance(data, dict)
    assert data["code"] == src


def test_trailing_comma_tolerated():
    broken = '{"name": "t", "probes": [],}'
    assert _strict_loads(broken) is None
    data = extract_json(broken)
    assert isinstance(data, dict) and data["name"] == "t"


def test_invalid_backslash_escape_dropped():
    """Python-style ``\\'`` is not a legal JSON escape."""
    broken = (
        '{"name": "t", "code": "c in '
        + BSLASH_QUOTE + "0123456789" + BSLASH_QUOTE
        + '"}'
    )
    assert _strict_loads(broken) is None
    data = extract_json(broken)
    assert isinstance(data, dict)
    assert "'0123456789'" in data["code"]


def test_prose_wrapped_envelope_still_extracted():
    text = (
        "Sure! Here is the tool:\n```json\n"
        + json.dumps({"name": "t", "probes": []})
        + "\n```\nHope that helps."
    )
    assert extract_json(text) == {"name": "t", "probes": []}


def test_array_output_supported():
    assert extract_json_any('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]


def test_garbage_returns_none():
    assert extract_json("not json at all, just words") is None
    assert extract_json("") is None


# ---- stray closer: the envelope closed before it was finished ----------------
#
# Transcribed from a real reply (Qwen3.5-Flash via aiping; qwen2.5:7b locally
# produces the same shape). The model writes name/description/code, closes the
# object, then KEEPS GOING with entry/parameters/probes and a second closing
# brace. json.loads says "Extra data: line 5 column 4"; raw_decode stops at the
# stray brace and yields only three keys, silently dropping entry, parameters
# and probes. Recovering just those three would be worse than failing, because
# the pipeline would accept a tool with no probes and never check it.
#
# Note the ``code`` line carries no trailing comma: the model put the stray
# brace exactly where the comma belonged. That is why deleting the brace leaves
# a legal single comma instead of two, and the fixture keeps that detail --
# giving ``code`` a trailing comma makes the text a different defect (``",,``)
# that this repair does not claim to fix.

STRAY_BRACE_ENVELOPE = (
    "{\n"
    '  "name": "normalize_isbn",\n'
    '  "description": "Validate and normalize ISBN identifiers",\n'
    '  "code": "def normalize_isbn(s):\\n    return s"\n'
    "  },\n"
    '  "entry": "normalize_isbn",\n'
    '  "parameters": {\n'
    '    "type": "object",\n'
    '    "properties": {"isbn_input": {"type": "string"}},\n'
    '    "required": ["isbn_input"]\n'
    "  },\n"
    '  "probes": [\n'
    '    {"query": "ISBN-10: 0-306-40615-2", "expect": "call"},\n'
    '    {"query": "abc-def-ghi", "expect": "call"}\n'
    "  ],\n"
    '  "effect_signature": "pure | reads:<input> | writes:<none>",\n'
    '  "tags": ["validation", "isbn"],\n'
    '  "rationale": "ISBNs appear in book systems"\n'
    "}"
)


def test_stray_brace_envelope_is_not_strict_json():
    """Guard the premise: this fixture must genuinely break strict parsing."""
    assert _strict_loads(STRAY_BRACE_ENVELOPE) is None


def test_stray_brace_does_not_cost_the_rest_of_the_envelope():
    """The whole point: the fields after the stray brace must survive.

    A raw_decode-based "take the first object" fix would return three keys and
    look like a success. Assert the later fields are present so that shortcut
    can never be mistaken for the fix.
    """
    data = extract_json(STRAY_BRACE_ENVELOPE)
    assert isinstance(data, dict), "stray-closer repair did not recover the envelope"
    assert data["name"] == "normalize_isbn"
    assert data["entry"] == "normalize_isbn"
    assert data["parameters"]["required"] == ["isbn_input"]
    assert len(data["probes"]) == 2
    assert data["tags"] == ["validation", "isbn"]
    assert data["effect_signature"].startswith("pure")


def test_stray_brace_repaired_code_still_compiles():
    data = extract_json(STRAY_BRACE_ENVELOPE)
    assert isinstance(data, dict)
    compile(data["code"], "<repaired>", "exec")


def test_valid_json_with_two_top_level_objects_is_not_corrupted():
    """Concatenated objects are a different defect; do not invent an envelope.

    Deleting closers must stay conservative: with no envelope to repair the
    repair should decline, not fuse two unrelated objects into one.
    """
    assert extract_json('{"a": 1}{"b": 2}') is None


def test_stray_closer_repair_is_a_noop_on_clean_json():
    clean = '{"name": "t", "code": "x = 1", "probes": []}'
    assert extract_json(clean) == {"name": "t", "code": "x = 1", "probes": []}


def test_a_stray_closing_bracket_is_also_dropped():
    """Same defect, array flavour: the model closes ``probes`` twice."""
    broken = (
        '{"name": "t", "probes": [\n'
        '  {"query": "a", "expect": "call"}\n'
        "  ]],\n"
        '  "entry": "t"\n'
        "}"
    )
    assert _strict_loads(broken) is None
    data = extract_json(broken)
    assert isinstance(data, dict)
    assert data["entry"] == "t"
    assert len(data["probes"]) == 1


def test_unclosed_delimiters_are_left_alone():
    """Truncation is not this repair's job -- it must not guess."""
    truncated = '{"name": "t", "code": "def f():\\n    return {'
    assert _strict_loads(truncated) is None
    assert extract_json(truncated) is None


# ---- real captured replies --------------------------------------------------
# Transcribed fixtures encode what we believed at the time. These run the parser
# over the actual bytes a model returned, so a future capture widens coverage by
# being dropped into probes/ -- no test edit required.

PROBES_DIR = Path(__file__).resolve().parent.parent / "probes"
REQUIRED_KEYS = ("name", "description", "code", "entry", "parameters", "probes")


def _captured_replies():
    """Every ``probes/raw_generator_reply*.txt`` we have, as (name, text)."""
    return sorted((p.name, p.read_text(encoding="utf-8"))
                  for p in PROBES_DIR.glob("raw_generator_reply*.txt"))


def test_at_least_one_captured_reply_exists():
    """A corpus test that silently covers nothing is worse than no corpus test."""
    assert _captured_replies(), f"no captured replies under {PROBES_DIR}"


def test_every_captured_reply_yields_a_usable_envelope():
    """Each real reply must recover into a complete, runnable envelope.

    This is the assertion that matters for the forge loop: not "the parser
    returned something", but "the pipeline got everything it needs to sandbox
    and check the tool". A partial recovery would ship an unchecked tool.
    """
    for name, raw in _captured_replies():
        data = extract_json(raw)
        assert isinstance(data, dict), f"{name}: repair recovered nothing"
        missing = [k for k in REQUIRED_KEYS if k not in data]
        assert not missing, f"{name}: repaired envelope is missing {missing}"
        assert data["probes"], f"{name}: no probes -- the tool would go unchecked"
        compile(data["code"], f"<{name}>", "exec")


def test_clean_captured_reply_is_untouched_by_the_repair():
    """The tolerant path must be a no-op when the model behaved.

    Worth asserting on real bytes rather than a hand-built dict: the clean
    sample is 2.8k of long description and nested schema, exactly the shape
    where an over-eager repair does damage.
    """
    raw = (PROBES_DIR / "raw_generator_reply_clean.txt").read_text(encoding="utf-8")
    assert json.loads(raw) is not None, "premise: this sample is strict JSON"
    assert extract_json(raw) == json.loads(raw)

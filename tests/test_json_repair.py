"""Regression tests for tolerant JSON extraction in the forge generator.

Context: qwen2.5:7b inside the real forge loop reliably emits an envelope that
ends with finish_reason='stop' yet is not strict JSON -- because the Python
source in the ``code`` field contains unescaped double quotes:

    "code": "... raise ValueError("Invalid ISBN-10 check digit") ..."

The JSON string terminates early and json.loads dies with
"Expecting ',' delimiter". Before the repair layer this aborted the whole
forge; the envelope was thrown away even though it was structurally complete.
"""
from __future__ import annotations

import json

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

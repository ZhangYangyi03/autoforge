"""Regression tests: the generator must not crash a round on schema slips.

Observed live with qwen2.5:7b: it emitted `probes` as a list of bare strings,
and the generator raised

    AttributeError: 'str' object has no attribute 'get'

which burned a whole round and surfaced as an opaque failure. A cosmetic schema
slip is not a reason to lose an attempt; coerce what is coercible and raise a
*clear* error only when something essential (name, code) is truly absent.
"""
from __future__ import annotations

import json

import pytest

from autoforge.core.llm import LLMClient, LLMResponse
from autoforge.forge.generator import LLMToolGenerator


class Canned(LLMClient):
    """Returns whatever payload it was handed, as the model's raw content."""

    name = "canned"

    def __init__(self, payload) -> None:
        self.payload = payload

    def chat(self, messages, tools=None, **kwargs):    # noqa: ANN001, ANN003
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return LLMResponse(content=content, raw={"choices": [{"finish_reason": "stop"}]})


def gen(payload) -> LLMToolGenerator:
    return LLMToolGenerator(Canned(payload), max_tokens=100)


BASE = {
    "name": "t",
    "description": "d",
    "code": "def t(x):\n    return x\n",
    "entry": "t",
    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
}


def test_probes_as_bare_strings_do_not_crash():
    """The exact live failure: probes = ["..."] raised AttributeError."""
    tool = gen({**BASE, "probes": ["normalise this ISBN 0306406152"]}).generate("x")
    assert len(tool.probes) == 1
    assert tool.probes[0].query == "normalise this ISBN 0306406152"
    assert tool.probes[0].expect == "call"


def test_probes_as_single_dict_not_list():
    tool = gen({**BASE, "probes": {"query": "q", "negative_query": "n"}}).generate("x")
    assert [p.query for p in tool.probes] == ["q"]
    assert tool.probes[0].negative_query == "n"


def test_probes_alternate_key_names_are_accepted():
    tool = gen({**BASE, "probes": [{"q": "alt", "should_not": "no"}]}).generate("x")
    assert tool.probes[0].query == "alt"
    assert tool.probes[0].negative_query == "no"


def test_non_dict_probe_entries_are_skipped_not_fatal():
    tool = gen({**BASE, "probes": [None, 7, {"query": "keep"}]}).generate("x")
    assert [p.query for p in tool.probes] == ["keep"]


def test_blank_probe_query_is_dropped():
    tool = gen({**BASE, "probes": [{"query": "   "}, {"query": "ok"}]}).generate("x")
    assert [p.query for p in tool.probes] == ["ok"]


def test_tags_as_string_becomes_list():
    tool = gen({**BASE, "tags": "isbn"}).generate("x")
    assert tool.tags == ["isbn"]


def test_non_dict_parameters_falls_back_to_empty_schema():
    tool = gen({**BASE, "parameters": "oops"}).generate("x")
    assert tool.parameters == {"type": "object", "properties": {}}


# -- property schemas ----------------------------------------------------
# The same class of slip, one level deeper. Live with qwen2.5:7b the model
# emitted {"properties": {"isbn": "string"}} -- a bare type *name* where a
# schema belongs. The generator accepted it, and the verifier then died with
# AttributeError: 'str' object has no attribute 'get' inside _infer_args,
# three frames away from the cause.

def test_property_schema_as_bare_type_name_becomes_a_schema():
    tool = gen({**BASE, "parameters": {"properties": {"isbn": "string"}}}).generate("x")
    assert tool.parameters["properties"]["isbn"] == {"type": "string"}


def test_property_schema_keeps_a_real_json_type_name():
    tool = gen({**BASE, "parameters": {"properties": {"n": "integer"}}}).generate("x")
    assert tool.parameters["properties"]["n"] == {"type": "integer"}


def test_property_schema_free_text_defaults_to_string():
    """A description is not a type; do not invent a bogus one."""
    tool = gen({**BASE, "parameters": {"properties": {"isbn": "ISBN-10 or ISBN-13"}}}).generate("x")
    assert tool.parameters["properties"]["isbn"] == {"type": "string"}


def test_non_dict_property_schema_is_coerced():
    tool = gen({**BASE, "parameters": {"properties": {"isbn": None, "n": 7}}}).generate("x")
    assert tool.parameters["properties"] == {"isbn": {"type": "string"}, "n": {"type": "string"}}


def test_properties_as_free_text_becomes_empty_not_fatal():
    tool = gen({**BASE, "parameters": {"properties": "isbn: string"}}).generate("x")
    assert tool.parameters["properties"] == {}


def test_required_as_string_becomes_a_list():
    tool = gen({**BASE, "parameters": {
        "properties": {"isbn": {"type": "string"}}, "required": "isbn"}}).generate("x")
    assert tool.parameters["required"] == ["isbn"]


def test_normalised_parameters_always_carry_a_type():
    tool = gen({**BASE, "parameters": {"properties": {"isbn": {"type": "string"}}}}).generate("x")
    assert tool.parameters["type"] == "object"


def test_property_normalisation_is_not_shared_between_tools():
    """The normaliser must copy, not mutate the caller's schema in place."""
    schema = {"properties": {"isbn": "string"}}
    gen({**BASE, "parameters": schema}).generate("x")
    assert schema == {"properties": {"isbn": "string"}}, "caller's dict was mutated"


def test_non_string_name_is_stringified():
    tool = gen({**BASE, "name": 123}).generate("x")
    assert tool.name == "123"


def test_missing_name_raises_a_clear_error():
    payload = {k: v for k, v in BASE.items() if k != "name"}
    with pytest.raises(ValueError, match="omitted tool name"):
        gen(payload).generate("x")


def test_missing_code_raises_a_clear_error():
    payload = {k: v for k, v in BASE.items() if k != "code"}
    with pytest.raises(ValueError, match="omitted code"):
        gen(payload).generate("x")


def test_top_level_array_is_rejected_with_a_clear_error():
    """extract_json only ever yields dicts, so a bare array is 'no parseable
    JSON' -- the generator's own isinstance guard is unreachable insurance."""
    with pytest.raises(ValueError, match="no parseable JSON|expected an object"):
        gen([1, 2, 3]).generate("x")


def test_entry_defaults_to_name():
    payload = {k: v for k, v in BASE.items() if k != "entry"}
    tool = gen(payload).generate("x")
    assert tool.entry == tool.name

# ---------------------------------------------------------------------------
# A *union* type is legal JSON Schema, and it used to kill the forge pipeline.
# ---------------------------------------------------------------------------

def test_union_type_is_normalised_and_does_not_crash_the_fuzzer():
    """`{"type": ["number", "null"]}` is how a nullable parameter is written.

    Reported 2026-09-21 by a peer session: forging any tool whose parameter was
    declared as a list of types died with

        TypeError: unhashable type: 'list'

    The list survived `normalise_parameters` (which only coerced bare strings and
    non-dicts) and reached `_PROBE_MAP.get(ptype, ...)` in the fuzzer, where a
    list cannot be a dict key. The fix is at the trust boundary where untrusted
    parameters enter, which is what `normalise_parameters` is for -- so this
    test asserts on that function *and* on the fuzzer actually running.
    """
    from autoforge.forge.fuzzer import _reference_args, generate_robustness_probes
    from autoforge.tools.spec import ToolSpec, normalise_parameters

    params = normalise_parameters({
        "type": "object",
        "properties": {"sec": {"type": ["number", "null"]}},
        "required": ["sec"],
    })
    # The first non-null member is the real type. Choosing the *last* member
    # would call a nullable number a null, which is worse than a crash because
    # it is silent, so the order matters and is asserted.
    assert params["properties"]["sec"]["type"] == "number"
    assert params["properties"]["sec"]["type_union"] == ["number", "null"]

    spec = ToolSpec(name="u", description="d", parameters=params, fn=lambda **k: None)
    assert generate_robustness_probes(spec)          # no TypeError
    assert _reference_args(spec) == {"sec": 1}


def test_union_type_does_not_escape_into_a_tool_schema():
    """Nothing downstream should ever see the list again."""
    from autoforge.tools.spec import normalise_parameters

    for declared, expected in [(["null", "string"], "string"),
                               (["integer", "null"], "integer"),
                               (["null"], "null"),       # degenerate but legal: kept, not invented away
                               (["widget", "null"], "string")]:   # not a JSON type at all
        got = normalise_parameters(
            {"type": "object", "properties": {"p": {"type": declared}}}
        )["properties"]["p"]["type"]
        assert got == expected, (declared, got)

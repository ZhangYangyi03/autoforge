"""The exposition format, and the two ways a hand-written one goes wrong.

There is no Prometheus in this test file and there does not need to be: the
contract is bytes on the wire, and the two failure modes that matter — a
duplicated sample and a duplicated `# HELP` — both cost the *entire* scrape
rather than one series. Those are asserted directly, because the symptom shows up
in a browser on a Grafana panel and nowhere near the code that caused it.
"""
from __future__ import annotations

import pytest

from toolmarket.metrics import (
    DEFAULT_BUCKETS,
    Counter,
    Gauge,
    Histogram,
    Registry,
    _dedupe_samples,
    _escape_label_value,
)


def _samples(text: str) -> dict[str, str]:
    """Parse exposition text into {sample_key: value}, ignoring comments."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, value = line.rpartition(" ")
        out[key] = value
    return out


def _a_named_gatherer() -> list[str]:
    """A module-level gatherer, so its qualified name is stable across calls.

    It exists to exercise the default-key path: the same function registered
    twice must replace, not append.
    """
    return ["# HELP named_gauge n", "# TYPE named_gauge gauge", "named_gauge 1"]


# ------------------------------------------------------------ basic types
def test_counter_renders_help_type_and_one_series():
    r = Registry()
    c = r.counter("things_total", "Things, by kind.", ("kind",))
    c.inc(kind="a")
    c.inc(2, kind="a")
    c.inc(kind="b")
    text = r.render()
    assert "# HELP things_total Things, by kind." in text
    assert "# TYPE things_total counter" in text
    samples = _samples(text)
    assert samples['things_total{kind="a"}'] == "3"
    assert samples['things_total{kind="b"}'] == "1"


def test_gauge_set_inc_dec():
    r = Registry()
    g = r.gauge("g", "A gauge.")
    g.set(5)
    g.inc()
    g.dec(2)
    assert _samples(r.render())["g"] == "4"


def test_a_gauge_with_no_observations_emits_no_sample():
    """Before the first request the API has counters at zero and gauges with no
    children. Emitting `h 0` for those would be a claim — "this is zero" — about
    something nothing has measured; the panel would draw a real 0 and hide the
    difference between "no data" and "measured zero"."""
    r = Registry()
    r.gauge("h", "Empty.")
    text = r.render()
    assert "# TYPE h gauge" in text
    assert not [l for l in text.splitlines() if l.startswith("h ")]


def test_histogram_has_inf_bucket_sum_and_count():
    r = Registry()
    h = r.histogram("d", "Duration.", ("route",))
    for v in (0.004, 0.02, 3.0):
        h.observe(v, route="/x")
    samples = _samples(r.render())
    # +Inf must equal _count: `histogram_quantile` reads +Inf as the denominator
    # and returns nonsense without it. This is the assertion that would have
    # caught the bucket list being emitted without the terminal bucket.
    assert samples['d_bucket{route="/x",le="+Inf"}'] == "3"
    assert samples['d_count{route="/x"}'] == "3"
    assert samples['d_sum{route="/x"}'] == "3.024"
    # Cumulative buckets: 0.004 lands in every bucket above its size. The `le`
    # values are rendered by the same float formatter as everything else, so 1.0
    # and 5.0 appear as "1" and "5" — which is what Prometheus expects to see.
    assert samples['d_bucket{route="/x",le="0.005"}'] == "1"
    assert samples['d_bucket{route="/x",le="0.01"}'] == "1"
    assert samples['d_bucket{route="/x",le="5"}'] == "3"
    assert h.count(route="/x") == 3
    assert h.sum(route="/x") == pytest.approx(3.024)


def test_histogram_bucket_boundaries_are_inclusive():
    r = Registry()
    h = r.histogram("d", "Duration.")
    h.observe(0.005)  # exactly the first bucket boundary
    samples = _samples(r.render())
    assert samples['d_bucket{le="0.005"}'] == "1"
    assert samples['d_bucket{le="0.01"}'] == "1"
    assert DEFAULT_BUCKETS[0] == 0.005


def test_duplicate_family_registration_is_refused():
    # Two metrics with one name would emit two `# HELP` lines for the family,
    # which is a scrape-level parse error. Refusing at registration turns it into
    # an import-time exception instead of a dashboard that goes blank later.
    r = Registry()
    r.counter("dup_total", "First.")
    with pytest.raises(ValueError):
        r.counter("dup_total", "Second.")


# ------------------------------------------------------------ label values
def test_label_values_are_escaped_per_spec():
    assert _escape_label_value('a"b') == 'a\\"b'
    assert _escape_label_value("a\\b") == "a\\\\b"
    assert _escape_label_value("a\nb") == "a\\nb"
    r = Registry()
    c = r.counter("c_total", "C.", ("route",))
    c.inc(route='/resources/{id}"quoted"')
    line = [l for l in r.render().splitlines() if l.startswith("c_total{")][0]
    # The quote is escaped and the value is still one line: a raw newline here
    # would split one sample into two unparseable halves.
    assert '\\"' in line
    assert line.count("\n") == 0


def test_float_formatting_is_parseable_and_avoids_a_useless_dot_zero():
    r = Registry()
    g = r.gauge("g", "G.")
    g.set(1.0)
    assert _samples(r.render())["g"] == "1"
    g.set(0.5)
    assert _samples(r.render())["g"] == "0.5"
    # A very small value comes out in exponent form. Prometheus parses that
    # (`strconv.ParseFloat` semantics), so the contract is round-tripping as a
    # float — not a particular spelling, which is what the previous version of
    # this test wrongly pinned.
    g.set(1e-9)
    assert float(_samples(r.render())["g"]) == pytest.approx(1e-9)


# ---------------------------------------------------------------- gatherers
def test_gatherer_output_is_included():
    r = Registry()
    r.add_gatherer(lambda: ["# HELP g_synth x", "# TYPE g_synth gauge", "g_synth 7"])
    assert _samples(r.render())["g_synth"] == "7"


def test_a_broken_gatherer_does_not_cost_the_rest_of_the_scrape():
    r = Registry()
    r.counter("ok_total", "Fine.").inc()

    def explode():
        raise RuntimeError("this gatherer is broken")

    r.add_gatherer(explode)
    text = r.render()  # must not raise
    assert 'ok_total' in _samples(text)


def test_same_gatherer_key_replaces_instead_of_duplicating():
    """The regression this exists for: `create_app()` builds the gatherer, and it
    is called twice per process (module-level `app`, then `served_app`). A blind
    append leaves two gatherers emitting the same family, and a scrape with a
    duplicated sample is rejected *in full* — the endpoint goes dark, including
    the panels that would have shown why.
    """
    r = Registry()

    def make_gatherer(tag: str):
        def gather() -> list[str]:
            return [
                "# HELP g_state current state",
                "# TYPE g_state gauge",
                f'g_state{{tag="{tag}"}} 1',
            ]
        return gather

    r.add_gatherer(make_gatherer("first"), key="same")
    r.add_gatherer(make_gatherer("second"), key="same")
    text = r.render()
    assert text.count("# HELP g_state") == 1
    assert text.count('g_state{tag="first"}') == 0
    assert text.count('g_state{tag="second"}') == 1


def test_two_gatherers_emitting_the_same_sample_are_deduped():
    r = Registry()
    line = 'dup_metric{a="1"} 1'
    r.add_gatherer(lambda: [line], key="one")
    r.add_gatherer(lambda: [line], key="two")
    assert r.render().count('dup_metric{a="1"}') == 1


def test_default_gatherer_key_is_the_qualified_name():
    """So that the common case — the same factory called twice — is idempotent
    without the caller having to invent a key."""
    r = Registry()
    r.add_gatherer(_a_named_gatherer)
    r.add_gatherer(_a_named_gatherer)
    text = r.render()
    # One HELP, one TYPE, one sample: the name appears three times, not six.
    assert text.count("named_gauge") == 3
    assert text.count("# HELP named_gauge") == 1


def test_remove_gatherer():
    r = Registry()
    key = r.add_gatherer(lambda: ["x 1"])
    r.remove_gatherer(key)
    assert "x 1" not in r.render()
    r.remove_gatherer("never-registered")  # must not raise


# ------------------------------------------------------------------ dedupe
def test_dedupe_keeps_first_occurrence_and_preserves_order():
    out = _dedupe_samples(["a 1", "b 2", "a 3", "c 4"])
    assert out.splitlines() == ["a 1", "b 2", "c 4"]


def test_dedupe_keys_on_labels_not_just_name():
    out = _dedupe_samples(['m{a="1"} 1', 'm{a="2"} 2', 'm{a="1"} 3'])
    assert out.splitlines() == ['m{a="1"} 1', 'm{a="2"} 2']


def test_dedupe_drops_a_repeated_help_line():
    # A repeated `# HELP` for one family is itself a parse error, so comments have
    # to be deduped as well — the samples being unique is not sufficient.
    out = _dedupe_samples(["# HELP m h", "# TYPE m gauge", "m 1", "# HELP m h"])
    assert out.count("# HELP m") == 1


def test_dedupe_tolerates_a_label_value_containing_a_space():
    out = _dedupe_samples(['m{a="has space"} 1', 'm{a="has space"} 2'])
    assert out.splitlines() == ['m{a="has space"} 1']


# ------------------------------------------------------------------ reset
def test_reset_drops_per_label_children():
    """`reset` is a test hook and it *clears* rather than zeroing.

    Zeroing would be the wrong contract for the gauge and histogram: a gauge
    whose last observed value is 9 would come back as 0, which is a claim nothing
    measured. Clearing means the next scrape reports only what the next test
    actually observed — which is the property that stops one test's requests from
    appearing in another's assertions.
    """
    r = Registry()
    label_counter = r.counter("c_total", "C.", ("result",))
    label_counter.inc(result="hit")
    bare_counter = r.counter("b_total", "B.")
    bare_counter.inc()
    g = r.gauge("g", "G.")
    g.set(9)
    h = r.histogram("d", "D.")
    h.observe(0.01)

    r.reset()
    text = r.render()
    samples = _samples(text)
    # A labelled counter reports nothing until something is counted again...
    assert not [l for l in text.splitlines() if l.startswith("c_total")]
    # ...while an unlabelled one keeps its family visible at 0.
    assert samples["b_total"] == "0"
    assert "g" not in samples
    assert "d_count" not in samples


def test_names_lists_registered_families():
    r = Registry()
    r.counter("a_total", "A.")
    r.gauge("b", "B.")
    assert sorted(r.names()) == ["a_total", "b"]


def test_hand_built_types_have_the_same_surface():
    # The API and the console both import the module-level singletons; constructing
    # the types directly must behave identically or a test's counter and the
    # server's counter drift apart.
    c, g, h = Counter("c", "C.", ("l",)), Gauge("g", "G."), Histogram("h", "H.")
    c.inc(l="x")
    g.set(2)
    h.observe(1.0)
    assert c.value(l="x") == 1 and g.value() == 2 and h.count() == 1

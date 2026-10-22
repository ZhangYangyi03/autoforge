#!/usr/bin/env python
"""Validate the Grafana dashboard against the provisioning and the code.

Three checks, each of which corresponds to a failure that a Python test suite
cannot see because none of this is Python:

  1. Every panel's datasource uid exists in the provisioned datasources. Grafana
     imports a dashboard whose panels point at a uid that does not exist, and the
     failure surfaces as "Datasource ... was not found" in a browser — silently,
     per panel, with no error anywhere a CI log would show it.
  2. Every `toolmarket_*` metric a panel queries is actually emitted somewhere in
     the package. This is the copy-paste failure: a panel that renders "No data"
     forever, which looks exactly like a service with no traffic.
  3. The alert rules' `expr` reference metrics that exist too, for the same
     reason — an alert on a metric nobody emits is a rule that reads as coverage
     while never being able to fire.

Deliberately no `promtool` here: that needs the whole Prometheus binary, and the
checks above are the ones this repository is actually at risk of. CI runs
`promtool check config` and `promtool check rules` alongside this for the parts
only it can validate (PromQL syntax, rule schema).

    python tools/check_dashboards.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DASH = ROOT / "deploy/grafana/dashboards/toolmarket.json"
DATASOURCES = ROOT / "deploy/grafana/provisioning/datasources/prometheus.yml"
ALERTS = ROOT / "deploy/prometheus/alerts.yml"

#: Where metric names come from. Listed explicitly rather than globbing the
#: package, so that a new file has to be added here on purpose — the failure this
#: guards against is a name that exists nowhere, and a glob over `*.py` would
#: count a comment as an emitter.
SOURCES = (
    "toolmarket/metrics.py",
    "toolmarket/api/main.py",
    "toolmarket/ratelimit.py",
    "toolmarket/worker.py",
    "toolmarket/tasks.py",
    "toolmarket/registry.py",
)

METRIC = re.compile(r"\btoolmarket_[a-z0-9_]+")

#: Suffixes Prometheus generates from a family rather than reading from source.
#: `toolmarket_http_request_duration_seconds_bucket` exists on the wire but the
#: literal appears nowhere in the code — the histogram appends `_bucket`, `_sum`
#: and `_count` when it renders. Without this, the check flags the most important
#: panel in the dashboard (the latency quantiles) as querying a metric nobody
#: emits, and a guard that cries wolf gets switched off.
GENERATED_SUFFIXES = ("_bucket", "_sum", "_count")


def emitted() -> set[str]:
    names: set[str] = set()
    for rel in SOURCES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        names |= set(METRIC.findall(text))
    return names


def unknown_metrics(expr: str, known: set[str]) -> set[str]:
    """Metric names in `expr` that no code emits, allowing generated suffixes."""
    out: set[str] = set()
    for name in METRIC.findall(expr):
        if name in known:
            continue
        base = next((name[: -len(s)] for s in GENERATED_SUFFIXES
                     if name.endswith(s)), None)
        if base and base in known:
            continue
        out.add(name)
    return out


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    sys.exit(1)


def main() -> int:
    import yaml

    dash = json.loads(DASH.read_text(encoding="utf-8"))
    panels = dash.get("panels", [])
    if not panels:
        fail("dashboard has no panels")

    known = emitted()
    if not known:
        fail("found no toolmarket_* metrics in the sources at all")

    # -- 1. datasource uids --------------------------------------------------
    provided = {d["uid"] for d in
                yaml.safe_load(DATASOURCES.read_text(encoding="utf-8"))["datasources"]}
    used: set[str] = set()
    for panel in panels:
        ds = panel.get("datasource")
        if isinstance(ds, dict) and "uid" in ds:
            used.add(ds["uid"])
    missing = used - provided
    if missing:
        fail(f"panels reference datasource uids that are not provisioned: {sorted(missing)}")
    if used != provided:
        print(f"note: provisioned but unused datasources: {sorted(provided - used)}")

    # -- 2. dashboard metrics ------------------------------------------------
    unknown: set[str] = set()
    for panel in panels:
        for target in panel.get("targets", []):
            unknown |= unknown_metrics(target.get("expr", ""), known)
    if unknown:
        fail(f"dashboard queries metrics no code emits: {sorted(unknown)}")

    # -- 3. alert metrics ----------------------------------------------------
    alerts = yaml.safe_load(ALERTS.read_text(encoding="utf-8"))
    rules = [r for group in alerts.get("groups", []) for r in group.get("rules", [])]
    if not rules:
        fail("alerts file defines no rules")
    bad_alerts: set[str] = set()
    for rule in rules:
        bad_alerts |= unknown_metrics(rule.get("expr", ""), known)
    if bad_alerts:
        fail(f"alert rules reference metrics no code emits: {sorted(bad_alerts)}")

    # Every rule must carry a severity, because the routing is by label and a
    # rule without one is an alert that reaches nobody.
    for rule in rules:
        if "severity" not in (rule.get("labels") or {}):
            fail(f"alert {rule.get('alert')} has no severity label")

    print(f"ok: {len(panels)} panels, {len(rules)} alert rules, "
          f"{len(known)} metrics, datasources {sorted(used)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""toolmarket.protocol — the RSPL + SEPL substrate.

This subpackage is the whole point of the project: it takes `autoforge`'s
existing, *enforcing* machinery and re-expresses it in the protocol vocabulary
AGP names, without weakening a single check.

  resources.py  — the RSPL record: contract, state, version, ledger, metadata
  lifecycle.py  — the state machine + AGP version status, and the legal edges
  events.py     — append-only event log (the audit substrate)
  lineage.py    — the DAG of who evolved from whom
  sepl.py       — the closed-loop evolution operator (propose → assess → commit)
"""

from toolmarket.protocol import events, lifecycle, lineage, resources, sepl

__all__ = ["events", "lifecycle", "lineage", "resources", "sepl"]

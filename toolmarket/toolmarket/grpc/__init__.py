"""toolmarket.grpc — the typed, cross-process front door to the substrate.

`server.py` serves the registry over gRPC; `client.py` is a thin caller for it;
`gen/` holds the stubs generated from `proto/toolmarket.proto`.

The rule that governs this package is the same one that governs the REST
surface: a handler calls exactly one registry/operator method, and never
advances a state or writes an event itself. Two front doors, one set of rules —
if they could disagree, the enforcement would be advisory.
"""

from toolmarket.grpc import client, server

__all__ = ["client", "server"]

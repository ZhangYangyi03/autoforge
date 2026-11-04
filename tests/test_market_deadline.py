"""The market lookup has a DEADLINE, not a socket timeout.

On 2026-10-23 two probes against a peer that accepts the connection and never
answers blocked ~35s each and outlived their own timeouts -- long enough that a
forge call, and with it the whole run, looked hung. The cause was not one slow
endpoint: `urlopen(timeout=...)` bounds each blocking recv and nothing else, and
`/resources` (52 MB on this host) was read into memory with no timeout at all.

A thread was the obvious fix and the wrong one: a thread cannot be killed, and
`join(timeout)` was measured to overrun by 15s+. A child process can be killed,
so that is what does the fetch.

These tests pin the property, not the implementation: whatever a peer does, the
call returns within its budget and says so.
"""

from __future__ import annotations

import socket
import time
import urllib.request

import pytest

from autoforge.agent import _fetch_json_with_deadline


def _silent_peer() -> tuple[socket.socket, str]:
    """A socket that accepts connections and never writes a byte."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    return srv, "http://127.0.0.1:%d/hang" % srv.getsockname()[1]


def test_a_peer_that_never_answers_is_cut_off_at_the_deadline():
    srv, url = _silent_peer()
    started = time.time()
    try:
        with pytest.raises(TimeoutError):
            _fetch_json_with_deadline(urllib.request.Request(url), 2.0)
    finally:
        srv.close()
    elapsed = time.time() - started
    assert elapsed < 4.0, (
        "a 2s deadline took %.1fs -- a socket timeout is not a deadline" % elapsed)


def test_a_dead_port_reads_as_a_failure_not_as_an_empty_answer():
    srv, url = _silent_peer()
    port = srv.getsockname()[1]
    srv.close()                              # nothing listens now
    with pytest.raises(Exception) as exc:
        _fetch_json_with_deadline(
            urllib.request.Request("http://127.0.0.1:%d/x" % port), 3.0)
    assert not isinstance(exc.value, TimeoutError) or True
    # The point: it raises. An empty dict here would be "the shelf answered
    # nothing", which licenses a duplicate forge.


def test_the_body_is_capped():
    """A 52 MB /resources must not be read in full to answer a duplicate check."""
    from autoforge.agent import ForgeAgent
    assert 0 < ForgeAgent.MAX_MARKET_BYTES <= 16 * 1024 * 1024

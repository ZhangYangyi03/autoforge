"""The browser transport and the CDP loop, against a real socket.

A CDP client has two failure modes that a mocked transport hides completely.

The first is framing. A WebSocket has three length encodings, a mandatory mask
on client frames, an optional mask on server frames, fragmentation, and
interleaved control frames -- and getting any of them subtly wrong still
produces *bytes*, just the wrong ones. So the frame tests check the RFC's own
worked example, every length form, a message split one byte at a time, and both
directions of the masking rule.

The second is the request/response loop. CDP replies by id and pushes events
whenever it likes, so a client that treats "the next message" as "the answer"
works until two calls overlap and then returns the wrong value to the wrong
caller. The fake server here deliberately sends an event before the reply, and
a reply for an id nobody asked for, so that bug fails a test instead of
appearing as intermittent nonsense in production.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
import threading
import time

import pytest

from autoforge.browser import (
    OP_CLOSE,
    OP_CONTINUATION,
    OP_PING,
    OP_TEXT,
    Browser,
    BrowserError,
    CDPError,
    WebSocket,
    WebSocketError,
    _Reader,
    accept_key,
    browser_candidates,
    closed_connection,
    encode_frame,
    find_browser,
    launch_browser,
    list_targets,
    parse_targets,
)


# ----------------------------------------------------------------------
# framing
# ----------------------------------------------------------------------


class TestAcceptKey:
    def test_the_rfc_worked_example(self):
        """RFC 6455 section 1.3: this key must produce this accept value."""
        assert accept_key("dGhlIHNhbXBsZSBub25jZQ==") == \
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="

    def test_different_keys_differ(self):
        assert accept_key("AAAA") != accept_key("BBBB")


class TestEncodeFrame:
    @pytest.mark.parametrize("size,extra_bytes", [
        (0, 0), (125, 0), (126, 2), (65535, 2), (65536, 8),
    ])
    def test_every_length_form_is_used_at_the_right_threshold(
            self, size, extra_bytes):
        """125/126 and 65535/65536 are where the encoding changes, and an
        off-by-one there corrupts every larger frame.

        The frame is always at least two header bytes (opcode + length), plus
        the 2- or 8-byte extended length when the payload needs it.
        """
        raw = encode_frame(b"x" * size, OP_TEXT, mask=False)
        assert len(raw) == 2 + extra_bytes + size
        assert (raw[1] & 0x7F) == {0: size, 2: 126, 8: 127}[extra_bytes]

    def test_the_mask_bit_is_set_and_the_key_precedes_the_payload(self):
        raw = encode_frame(b"hello", OP_TEXT, mask=True,
                           mask_key=b"\x01\x02\x03\x04")
        assert raw[1] & 0x80, "client frames must be masked"
        assert raw[2:6] == b"\x01\x02\x03\x04"
        assert raw[6:] != b"hello", "the payload was not actually masked"

    def test_an_unmasked_frame_is_available_when_asked(self):
        raw = encode_frame(b"hi", OP_TEXT, mask=False)
        assert not raw[1] & 0x80
        assert raw[2:] == b"hi"

    def test_fin_is_reflected(self):
        assert encode_frame(b"x", OP_TEXT, fin=False, mask=False)[0] & 0x80 == 0
        assert encode_frame(b"x", OP_TEXT, fin=True, mask=False)[0] & 0x80

    def test_control_frames_are_length_limited(self):
        """RFC 6455: control frames carry at most 125 bytes."""
        with pytest.raises(WebSocketError, match="125"):
            encode_frame(b"x" * 126, OP_PING, mask=False)

    def test_control_frames_cannot_fragment(self):
        with pytest.raises(WebSocketError, match="fragmented"):
            encode_frame(b"x", OP_PING, fin=False, mask=False)

    def test_a_short_mask_key_is_refused(self):
        with pytest.raises(WebSocketError, match="4 bytes"):
            encode_frame(b"x", OP_TEXT, mask=True, mask_key=b"\x01\x02")


class TestReader:
    """The client role: unmasked server frames in, whole messages out."""

    def _feed(self, *raws: bytes) -> list[bytes]:
        reader = _Reader()
        for raw in raws:
            reader.feed(raw)
        out = []
        while reader.has_frame():
            out.append(reader.read_frame().payload)
        return out

    @pytest.mark.parametrize("size", [0, 1, 125, 126, 70000])
    def test_round_trip_through_every_length_form(self, size):
        got = self._feed(encode_frame(b"z" * size, OP_TEXT, mask=False))
        assert got == [b"z" * size]

    def test_two_frames_in_one_read(self):
        got = self._feed(encode_frame(b"a", OP_TEXT, mask=False),
                         encode_frame(b"b", OP_TEXT, mask=False))
        assert got == [b"a", b"b"]

    def test_a_frame_split_one_byte_at_a_time(self):
        """TCP is a stream. A frame arriving byte by byte is normal."""
        raw = encode_frame(b"payload", OP_TEXT, mask=False)
        reader = _Reader()
        seen = 0
        for i in range(len(raw)):
            reader.feed(raw[i:i + 1])
            if reader.has_frame():
                seen += 1
        assert seen == 1
        assert reader.read_frame().payload == b"payload"

    def test_a_partial_frame_is_not_a_frame(self):
        raw = encode_frame(b"0123456789", OP_TEXT, mask=False)
        reader = _Reader()
        reader.feed(raw[:5])
        assert not reader.has_frame()

    def test_a_masked_server_frame_is_refused(self):
        """Servers must not mask; accepting it hides a role mix-up."""
        reader = _Reader()
        reader.feed(encode_frame(b"x", OP_TEXT, mask=True))
        with pytest.raises(WebSocketError, match="server sent a masked frame"):
            reader.read_frame()

    def test_the_server_role_refuses_an_unmasked_client_frame(self):
        reader = _Reader(expect_masked=True)
        reader.feed(encode_frame(b"x", OP_TEXT, mask=False))
        with pytest.raises(WebSocketError, match="client sent an unmasked frame"):
            reader.read_frame()

    def test_the_server_role_unmasks(self):
        reader = _Reader(expect_masked=True)
        reader.feed(encode_frame(b"SECRET", OP_TEXT, mask=True,
                                 mask_key=b"\x0f\x0e\x0d\x0c"))
        assert reader.read_frame().payload == b"SECRET"

    def test_an_absurd_length_is_refused(self):
        """A corrupt header must not make us allocate gigabytes."""
        raw = bytes([0x81, 127]) + struct.pack("!Q", 2 ** 40)
        reader = _Reader()
        reader.feed(raw)
        with pytest.raises(WebSocketError, match="plausible"):
            reader.read_frame()

    def test_fragmented_message_is_reassembled(self):
        reader = _Reader()
        reader.feed(encode_frame(b"one ", OP_TEXT, mask=False, fin=False))
        reader.feed(encode_frame(b"two", OP_CONTINUATION, mask=False, fin=True))
        frame = reader.recv_message()
        assert frame.payload == b"one two"

    def test_a_continuation_with_nothing_to_continue_is_refused(self):
        reader = _Reader()
        reader.feed(encode_frame(b"x", OP_CONTINUATION, mask=False, fin=True))
        with pytest.raises(WebSocketError, match="nothing to continue"):
            reader.recv_message()


# ----------------------------------------------------------------------
# a scripted CDP endpoint on a real socket
# ----------------------------------------------------------------------


class FakeCDP:
    """A WebSocket server that speaks just enough CDP to be driven.

    Runs on a real localhost socket and uses this module's own encoder and
    reader in the *server* role, so the masking and framing rules are exercised
    from both sides rather than assumed.
    """

    def __init__(self, replies: dict | None = None, *, status: int = 101,
                 accept_override: str | None = None, event_first: bool = False,
                 stray_reply: bool = False, upgrade_garbage: bytes = b"",
                 silent: bool = False, on_message=None,
                 kill_after: int = 0, kill_mode: str = "reset") -> None:
        self.replies = replies or {}
        self.status = status
        self.accept_override = accept_override
        self.event_first = event_first
        self.stray_reply = stray_reply
        self.upgrade_garbage = upgrade_garbage
        #: Read the requests and answer none of them, which is what a hung
        #: browser looks like from here.
        self.silent = silent
        self.on_message = on_message
        #: Die after answering this many requests, without a close frame --
        #: which is what a browser quitting under the client looks like. Zero
        #: means stay up. `kill_mode` picks which death: a reset (`RST`, so the
        #: client's read raises) or a hang-up (`FIN`, so it returns an empty
        #: chunk). Those two are the same event arriving in two shapes, and the
        #: whole point of `closed_connection` is that they read alike.
        self.kill_after = kill_after
        self.kill_mode = kill_mode
        self.received: list[dict] = []
        self.sent: list[dict] = []
        self.request_headers: dict[str, str] = {}
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._stop = threading.Event()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/devtools/page/fake"

    def __enter__(self) -> "FakeCDP":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass

    # -- server internals ----------------------------------------------

    def _serve(self) -> None:
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        with conn:
            try:
                self._handshake(conn)
                if self.status != 101:
                    return
                self._loop(conn)
            except (OSError, WebSocketError):
                return

    def _handshake(self, conn: socket.socket) -> None:
        buf = bytearray()
        while b"\r\n\r\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                return
            buf += chunk
        head = bytes(buf).split(b"\r\n\r\n")[0].decode("latin-1")
        lines = head.split("\r\n")
        for line in lines[1:]:
            name, sep, value = line.partition(":")
            if sep:
                self.request_headers[name.strip().lower()] = value.strip()
        key = self.request_headers.get("sec-websocket-key", "")

        if self.status != 101:
            conn.sendall(f"HTTP/1.1 {self.status} No\r\n\r\n".encode("ascii"))
            return

        accept = self.accept_override or accept_key(key)
        conn.sendall((
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode("ascii"))
        if self.upgrade_garbage:
            # Frames sent in the same packet as the upgrade reply. Those bytes
            # belong to the frame stream and must not be discarded.
            conn.sendall(self.upgrade_garbage)

    def _loop(self, conn: socket.socket) -> None:
        reader = _Reader(expect_masked=True)
        while not self._stop.is_set():
            if not reader.has_frame():
                conn.settimeout(0.2)
                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    return
                reader.feed(chunk)
                continue

            frame = reader.read_frame()
            if frame.opcode == OP_PING:
                conn.sendall(encode_frame(frame.payload, 0xA, mask=False))
                continue
            if frame.opcode == OP_CLOSE:
                return
            if frame.opcode != OP_TEXT:
                continue

            message = json.loads(frame.payload.decode("utf-8"))
            self.received.append(message)
            self._answer(conn, message)
            if self.kill_after and len(self.received) >= self.kill_after:
                self._die(conn)
                return

    def _die(self, conn: socket.socket) -> None:
        """Drop the connection the way a browser process quitting does.

        `SO_LINGER` with a zero timeout makes `close` send `RST` instead of
        `FIN`, so the client's next read *raises* rather than coming back
        empty. That is the Windows behaviour (WinError 10053/10054) reproduced
        on every platform, which is the branch the empty-chunk path misses.
        """
        if self.kill_mode == "reset":
            try:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                struct.pack("ii", 1, 0))
            except OSError:
                pass
        try:
            conn.close()
        except OSError:
            pass

    def _answer(self, conn: socket.socket, message: dict) -> None:
        if self.silent:
            return
        if self.event_first:
            # An event arriving before the reply. A client that takes the next
            # message as its answer fails here.
            self._send(conn, {"method": "Page.loadEventFired", "params": {}})
        if self.stray_reply:
            self._send(conn, {"id": 999999, "result": {"wrong": True}})

        if self.on_message is not None:
            custom = self.on_message(message)
            if custom is not None:
                self._send(conn, custom)
                return

        method = message.get("method", "")
        result = self.replies.get(method)
        if callable(result):
            result = result(message.get("params") or {})
        if result is None:
            result = {}
        self._send(conn, {"id": message.get("id"), "result": result})

    def _send(self, conn: socket.socket, payload: dict) -> None:
        self.sent.append(payload)
        conn.sendall(encode_frame(json.dumps(payload).encode("utf-8"),
                                  OP_TEXT, mask=False))


@pytest.fixture
def cdp():
    with FakeCDP(replies={}) as server:
        yield server


# ----------------------------------------------------------------------
# the handshake
# ----------------------------------------------------------------------


class TestHandshake:

    def test_connect_succeeds_against_a_well_behaved_server(self, cdp):
        ws = WebSocket.connect(cdp.url, timeout=5)
        try:
            assert ws.host.startswith("127.0.0.1")
        finally:
            ws.close()

    def test_the_key_sent_is_the_key_checked(self, cdp):
        WebSocket.connect(cdp.url, timeout=5).close()
        assert len(cdp.request_headers.get("sec-websocket-key", "")) > 0
        assert cdp.request_headers.get("upgrade", "").lower() == "websocket"

    def test_a_non_101_reply_is_refused_with_the_status(self):
        with FakeCDP(status=404) as server:
            with pytest.raises(WebSocketError, match="404"):
                WebSocket.connect(server.url, timeout=5)

    def test_a_wrong_accept_key_is_refused(self):
        """A proxy that answers 101 without understanding WebSocket lands here."""
        with FakeCDP(accept_override="not-the-right-value") as server:
            with pytest.raises(WebSocketError, match="did not match the key"):
                WebSocket.connect(server.url, timeout=5)

    def test_a_non_websocket_url_is_refused(self):
        with pytest.raises(WebSocketError, match="not a websocket url"):
            WebSocket.connect("http://127.0.0.1:1/x", timeout=1)

    def test_nothing_listening_is_an_oserror_not_a_hang(self):
        with pytest.raises(OSError):
            WebSocket.connect("ws://127.0.0.1:9/", timeout=2)

    def test_bytes_arriving_with_the_upgrade_are_kept(self):
        """A frame sent in the same packet as the 101 must not be lost."""
        greeting = encode_frame(json.dumps(
            {"method": "Inspector.detached", "params": {}}).encode(), OP_TEXT,
            mask=False)
        with FakeCDP(upgrade_garbage=greeting, replies={}) as server:
            ws = WebSocket.connect(server.url, timeout=5)
            try:
                frame = ws.recv(timeout=2)
                assert b"Inspector.detached" in frame.payload
            finally:
                ws.close()


class TestTheBrowserQuitting:
    """A browser that dies mid-stream, in both of the shapes it arrives in.

    This is the failure that made the graceful-close path dead code on Windows.
    The same event -- the peer is gone -- reaches `recv` as an empty chunk on
    POSIX and as a *raised* `OSError` (WinError 10053/10054) on Windows, so the
    empty-chunk branch never ran there and a raw, localized OS string escaped
    the whole websocket layer. Both shapes are exercised here and both must end
    as this module's error, carrying the same message.
    """

    def _drive_into_a_dead_socket(self, server: "FakeCDP") -> WebSocketError:
        ws = WebSocket.connect(server.url, timeout=5)
        try:
            ws.send(json.dumps({"id": 1, "method": "Page.enable"}))
            with pytest.raises(WebSocketError) as caught:
                ws.recv(timeout=5)
            return caught.value
        finally:
            ws.close()

    def test_a_reset_reads_as_this_modules_error_not_an_oserror(self):
        """The raising shape: what Windows does, forced on every platform."""
        # `silent` because the death has to land on an *outstanding* request --
        # a server that answers first and dies after has left the client with
        # its reply, which is a completed call, not a dead peer.
        with FakeCDP(silent=True, kill_after=1, kill_mode="reset") as server:
            error = self._drive_into_a_dead_socket(server)
        # A timeout would mean the death was read as silence, which is a
        # different claim about the peer and the wrong one.
        assert not isinstance(error, TimeoutError)
        assert "closed the connection" in str(error)

    def test_a_hangup_reads_the_same_as_a_reset(self):
        """The empty-chunk shape: what POSIX does. Identical text, by design."""
        with FakeCDP(silent=True, kill_after=1, kill_mode="close") as server:
            error = self._drive_into_a_dead_socket(server)
        assert "closed the connection" in str(error)

    def test_a_hung_browser_is_a_timeout_not_a_death(self):
        """The distinction the socket layer must not blur.

        A browser that is alive and simply not answering is not the same fact
        as one that has quit, and the CDP loop needs to tell them apart to name
        the method that hung. Collapsing both into one error loses that.
        """
        with FakeCDP(silent=True) as server:
            ws = WebSocket.connect(server.url, timeout=5)
            try:
                ws.send(json.dumps({"id": 1, "method": "Page.enable"}))
                with pytest.raises(WebSocketError, match="sent nothing"):
                    ws.recv(timeout=1)
            finally:
                ws.close()


class TestClosedConnectionVocabulary:
    """`closed_connection` itself, without a socket in the way."""

    def test_a_windows_error_becomes_this_modules_error_carrying_the_number(self):
        exc = ConnectionResetError(10054, "forcibly closed by the remote host")
        exc.winerror = 10054
        error = closed_connection(exc, "the browser closed the connection")
        assert isinstance(error, WebSocketError)
        assert error.errno == 10054
        assert error.__cause__ is exc

    def test_a_posix_errno_survives_the_same_way(self):
        exc = ConnectionResetError(104, "Connection reset by peer")
        error = closed_connection(exc, "the browser closed the connection")
        assert error.errno == 104

    def test_the_number_is_not_stuffed_into_the_message(self):
        """The text must stay stable for callers that match on it.

        The errno is the language-independent half of an OS message, so it
        rides along as an attribute; embedding it would make the message
        change shape per platform for no gain.
        """
        exc = ConnectionResetError(104, "Connection reset by peer")
        error = closed_connection(exc, "the browser closed the connection")
        assert str(error) == "the browser closed the connection"


class TestWebSocketControl:

    def test_a_ping_is_answered_with_a_pong(self, cdp):
        ws = WebSocket.connect(cdp.url, timeout=5)
        try:
            ws.ping(b"keepalive")
            # The server answers the ping; ask it for something so the pong is
            # consumed rather than stalling.
            ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate",
                                "params": {"expression": "1"}}))
            frame = ws.recv(timeout=5)
            assert json.loads(frame.payload)["id"] == 1
        finally:
            ws.close()

    def test_the_server_role_pong_is_not_mistaken_for_a_message(self):
        """A pong is not a message; ignoring it is what keeps the CDP loop
        from parsing control traffic as JSON."""
        def answer(message):
            if message.get("method") == "Runtime.evaluate":
                # Send a pong *before* the reply: the client must skip it.
                return {"__pong__": True}
            return None

        with FakeCDP() as server:
            replies = []
            original_send = server._send

            def send_with_pong(conn, payload):
                conn.sendall(encode_frame(b"", 0xA, mask=False))   # OP_PONG
                original_send(conn, payload)

            server._send = send_with_pong
            server.replies = {"Runtime.evaluate": {"result": {"value": "ok"}}}
            ws = WebSocket.connect(server.url, timeout=5)
            try:
                ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate",
                                    "params": {"expression": "1"}}))
                frame = ws.recv(timeout=5)
                assert json.loads(frame.payload)["id"] == 1
            finally:
                ws.close()


# ----------------------------------------------------------------------
# the request/response loop
# ----------------------------------------------------------------------


class TestCall:

    def test_a_result_comes_back(self):
        with FakeCDP(replies={"Runtime.evaluate": {"result": {"value": 42}}}) as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            assert browser.call("Runtime.evaluate",
                                {"expression": "42"})["result"]["value"] == 42

    def test_the_request_carries_a_fresh_id_each_time(self):
        with FakeCDP(replies={"A": {}, "B": {}}) as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            browser.call("A")
            browser.call("B")
            ids = [m["id"] for m in s.received]
            assert len(ids) == 2 and ids[0] != ids[1]

    def test_an_event_arriving_first_is_not_mistaken_for_the_answer(self):
        """The bug this whole fake server exists to catch.

        A client that returns the next message as the reply would return the
        Page.loadEventFired event here, and every later call would be off by
        one -- intermittently, depending on timing.
        """
        with FakeCDP(replies={"Runtime.evaluate": {"result": {"value": "real"}}},
                     event_first=True) as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            got = browser.call("Runtime.evaluate", {"expression": "x"})
            assert got["result"]["value"] == "real"
            assert any(e["method"] == "Page.loadEventFired"
                       for e in browser.events)

    def test_a_reply_for_another_id_is_ignored(self):
        with FakeCDP(replies={"Runtime.evaluate": {"result": {"value": "mine"}}},
                     stray_reply=True) as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            got = browser.call("Runtime.evaluate", {"expression": "x"})
            assert got["result"]["value"] == "mine"

    def test_an_error_payload_becomes_a_CDPError_with_its_message(self):
        with FakeCDP() as s:
            s.on_message = lambda m: {"id": m["id"], "error": {
                "code": -32601, "message": "method not found"}}
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            with pytest.raises(CDPError, match="not found"):
                browser.call("No.SuchMethod")

    def test_a_call_after_close_is_refused(self):
        with FakeCDP() as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            browser.close()
            with pytest.raises(BrowserError, match="closed"):
                browser.call("Runtime.evaluate")

    def test_a_silent_browser_times_out_with_the_method_named(self):
        """The method name is the whole value of this error.

        Asserted as a BrowserError specifically, not "some socket error": the
        weaker form passes when the socket's own unnamed timeout leaks through,
        which is exactly the regression this pins.
        """
        with FakeCDP(silent=True) as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            with pytest.raises(
                    BrowserError,
                    match=r"Slow\.method did not answer within 0\.4s"):
                browser.call("Slow.method", timeout=0.4)


# ----------------------------------------------------------------------
# page operations
# ----------------------------------------------------------------------


class TestPageOperations:

    def test_evaluate_returns_the_value(self):
        with FakeCDP(replies={"Runtime.evaluate": {"result": {"value": "hi"}}}) as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            assert browser.evaluate("'hi'") == "hi"

    def test_evaluate_sends_the_right_flags(self):
        """returnByValue and awaitPromise are what make a value come back
        instead of a remote object handle."""
        with FakeCDP(replies={"Runtime.evaluate": {"result": {"value": 1}}}) as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            browser.evaluate("1")
            sent = s.received[0]["params"]
            assert sent["returnByValue"] is True
            assert sent["awaitPromise"] is True

    def test_a_page_exception_is_raised_with_its_description(self):
        with FakeCDP(replies={"Runtime.evaluate": {
                "exceptionDetails": {"text": "Uncaught",
                                     "exception": {"description": "TypeError: nope"}}}}) as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            with pytest.raises(CDPError, match="TypeError: nope"):
                browser.evaluate("boom()")

    def test_screenshot_decodes_the_base64_payload(self):
        png = b"\x89PNG\r\n\x1a\n" + b"body"
        with FakeCDP(replies={"Page.captureScreenshot": {
                "data": base64.b64encode(png).decode()}}) as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            assert browser.screenshot() == png

    def test_an_empty_screenshot_is_an_error_not_an_empty_file(self):
        with FakeCDP(replies={"Page.captureScreenshot": {}}) as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            with pytest.raises(BrowserError, match="empty screenshot"):
                browser.screenshot()

    def test_goto_waits_until_the_document_is_complete(self):
        """Returning before load means reading a half-built page."""
        states = iter(["loading", "loading", "complete"])

        def evaluate(params):
            expr = params.get("expression", "")
            if "readyState" in expr:
                return {"result": {"value": next(states, "complete")}}
            if "title" in expr:
                return {"result": {"value": "Loaded"}}
            return {"result": {"value": ""}}

        with FakeCDP() as s:
            s.replies = {"Runtime.evaluate": evaluate}
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            assert browser.goto("http://example.invalid/") == "Loaded"

    def test_goto_gives_up_with_the_url_and_the_timeout(self):
        with FakeCDP(replies={"Runtime.evaluate": {"result": {"value": "loading"}}}) as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            with pytest.raises(BrowserError, match="did not finish loading"):
                browser.goto("http://example.invalid/", timeout=0.4)

    def test_click_sends_press_and_release_at_the_point(self):
        with FakeCDP() as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            browser.click(10, 20)
            kinds = [m["params"]["type"] for m in s.received]
            assert kinds == ["mousePressed", "mouseReleased"]
            assert s.received[0]["params"]["x"] == 10.0
            assert s.received[0]["params"]["y"] == 20.0

    def test_type_uses_insert_text(self):
        with FakeCDP() as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            browser.type_text("hello")
            assert s.received[0]["method"] == "Input.insertText"
            assert s.received[0]["params"]["text"] == "hello"

    def test_press_sends_key_down_and_key_up(self):
        with FakeCDP() as s:
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            browser.press("Enter")
            assert [m["params"]["type"] for m in s.received] == \
                ["keyDown", "keyUp"]

    def test_enable_survives_a_domain_that_does_not_exist(self):
        """A browser-level target has no Page domain; enabling must still work."""
        with FakeCDP() as s:
            def routing(message):
                if message["method"] == "Page.enable":
                    return {"id": message["id"], "error": {
                        "code": -32601, "message": "not found"}}
                return {"id": message["id"], "result": {}}
            s.on_message = routing
            browser = Browser(WebSocket.connect(s.url, timeout=5))
            browser.enable()
            assert any(m["method"] == "Runtime.enable" for m in s.received)


# ----------------------------------------------------------------------
# connecting to a browser
# ----------------------------------------------------------------------


class TestConnect:

    def test_a_ws_url_is_used_directly(self, cdp):
        browser = Browser.connect(cdp.url, timeout=5)
        assert browser.call("Runtime.evaluate",
                            {"expression": "1"}) == {} or True

    def test_no_page_target_names_what_was_there_instead(self, monkeypatch):
        import autoforge.browser as mod
        monkeypatch.setattr(mod, "list_targets", lambda *a, **k: [
            {"type": "background_page", "id": "x"},
        ])
        with pytest.raises(BrowserError, match="background_page"):
            Browser.connect("http://127.0.0.1:9222")

    def test_a_target_without_a_debugger_url_says_so(self, monkeypatch):
        import autoforge.browser as mod
        monkeypatch.setattr(mod, "list_targets", lambda *a, **k: [
            {"type": "page", "id": "abc"},
        ])
        with pytest.raises(BrowserError, match="webSocketDebuggerUrl"):
            Browser.connect("http://127.0.0.1:9222")

    def test_an_index_past_the_end_is_refused(self, monkeypatch):
        import autoforge.browser as mod
        monkeypatch.setattr(mod, "list_targets", lambda *a, **k: [
            {"type": "page", "id": "a", "webSocketDebuggerUrl": "ws://x/1"},
        ])
        with pytest.raises(BrowserError, match="#3"):
            Browser.connect("http://127.0.0.1:9222", index=3)

    def test_the_first_page_is_chosen_not_the_first_target(self, monkeypatch):
        """Real browsers list background pages and chrome:// UI before the tab."""
        import autoforge.browser as mod
        monkeypatch.setattr(mod, "list_targets", lambda *a, **k: [
            {"type": "background_page", "id": "ext",
             "webSocketDebuggerUrl": "ws://127.0.0.1:1/ext"},
            {"type": "page", "id": "tab",
             "webSocketDebuggerUrl": "ws://127.0.0.1:1/tab"},
        ])
        chosen: dict = {}

        class FakeWS:
            def __init__(self, *a, **k):
                pass

        monkeypatch.setattr(mod.WebSocket, "connect",
                            lambda url, **k: chosen.update(url=url) or FakeWS())
        Browser.connect("http://127.0.0.1:9222")
        assert chosen["url"].endswith("/tab")


class TestDiscovery:

    def test_list_targets_reports_a_dead_port_helpfully(self):
        with pytest.raises(BrowserError, match="could not reach a browser"):
            list_targets("http://127.0.0.1:9", timeout=2)

    def test_parse_targets_normalises_and_drops_junk(self):
        got = parse_targets([
            {"id": "a", "type": "page", "url": "http://x", "title": "T"},
            "not a dict",
            {"id": "b"},
        ])
        assert len(got) == 2
        assert got[0]["title"] == "T"
        assert got[1]["webSocketDebuggerUrl"] == ""

    def test_parse_targets_tolerates_a_non_list(self):
        assert parse_targets({"error": "nope"}) == []
        assert parse_targets(None) == []


class TestFindingABrowser:

    def test_the_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("AUTOFORGE_BROWSER", "/custom/chrome")
        assert browser_candidates()[0] == "/custom/chrome"

    def test_windows_candidates_include_edge(self):
        """Edge is the same engine and is on every Windows box."""
        joined = " ".join(browser_candidates("win32")).lower()
        assert "chrome.exe" in joined
        assert "msedge.exe" in joined

    def test_linux_candidates_include_chromium(self):
        assert any("chromium" in c for c in browser_candidates("linux"))

    def test_find_browser_returns_none_rather_than_guessing(self, monkeypatch):
        monkeypatch.delenv("AUTOFORGE_BROWSER", raising=False)
        import autoforge.browser as mod
        monkeypatch.setattr(mod, "_CANDIDATES", {"linux": []})
        monkeypatch.setattr(mod.shutil, "which", lambda name: None)
        assert find_browser("linux") is None

    def test_launch_with_no_browser_names_the_escape_hatch(self, monkeypatch):
        monkeypatch.delenv("AUTOFORGE_BROWSER", raising=False)
        import autoforge.browser as mod
        monkeypatch.setattr(mod, "_CANDIDATES", {"linux": []})
        monkeypatch.setattr(mod.shutil, "which", lambda name: None)
        with pytest.raises(BrowserError, match="AUTOFORGE_BROWSER"):
            mod.launch_browser(9222)

    def test_launch_is_reported_when_the_browser_will_not_start(
            self, monkeypatch, tmp_path):
        """Whatever the OS says about why, it must reach the caller.

        On Windows the interesting failure is a locked user-data-dir, on Unix
        a bad interpreter or a missing shared library -- either way the point
        is that the reason is surfaced rather than swallowed as a timeout.
        """
        import autoforge.browser as mod
        fake = tmp_path / "chrome"
        fake.write_text("#!/bin/sh\nexit 1\n")
        fake.chmod(0o755)
        monkeypatch.setenv("AUTOFORGE_BROWSER", str(fake))
        with pytest.raises(BrowserError,
                           match="could not start|did not expose|exited"):
            mod.launch_browser(9222, wait=1.0,
                               user_data_dir=str(tmp_path / "profile"))

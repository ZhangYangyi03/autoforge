"""Driving a real browser over the DevTools protocol.

An agent that can only read files and run shell commands cannot check its own
work when the work is a web page, and cannot read a page whose content only
exists after its scripts run. That is most of the web. This module is the part
that closes the gap: it speaks CDP to a real Chrome.

**The WebSocket client is here, not in a dependency.** CDP's transport is a
WebSocket, and the alternative was another entry in someone's install list for
a protocol that is a handshake plus a length-prefixed frame. So the handshake
and the frames are implemented against RFC 6455 directly, on a plain socket:
masking, the 7/16/64-bit length forms, fragmentation, and the ping/pong that
keeps a long-running session alive. Everything above it -- the id-keyed
request/response, the event buffer, the page operations -- is CDP.

Two rules the implementation keeps, because violating either produces a
failure that looks like something else:

  * Client frames are always masked and server frames must never be. A server
    that masks, or a client that does not, is a protocol error, not a decode
    oddity to paper over.
  * Every request carries an id and its response is matched by that id. A
    reply to a different call, or an event arriving mid-call, must not be
    mistaken for the answer. Events go to a buffer; the loop keeps reading
    until the id it is waiting for arrives.

Nothing here launches a browser unless asked to. `connect` attaches to one
already running with `--remote-debugging-port`, which is the safer default.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlparse

__all__ = [
    "Browser", "BrowserError", "CDPError", "Frame", "WebSocket", "WebSocketError",
    "WebSocketTimeout",
    "accept_key", "browser_candidates", "encode_frame", "find_browser",
    "launch_browser", "parse_targets",
]

#: RFC 6455 section 1.3. Fixed, and required to be exactly this.
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONTINUATION = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

#: A single frame this large is a protocol problem, not a payload. CDP screenshots
#: are the biggest thing that legitimately arrives, and they are megabytes.
MAX_FRAME = 64 * 1024 * 1024

#: What a client may send unmasked before it is clearly not a browser talking.
MAX_HANDSHAKE = 16 * 1024


class WebSocketError(RuntimeError):
    """A transport-level failure: handshake, framing, or the peer hanging up."""


def _secs(value: float) -> str:
    """A duration a human can act on: 0.4s, not "0s".

    `{:.0f}` on a sub-second budget reports a zero-second timeout, which reads
    as a bug in the caller rather than as the deadline it actually was.
    """
    return f"{value:.0f}s" if value >= 10 else f"{value:.1f}s"


class WebSocketTimeout(WebSocketError):
    """Nothing arrived before the deadline.

    Separate from its parent because the two mean opposite things to a caller:
    a closed socket is a fact about the peer, while silence is only a fact about
    *this* request. The CDP loop needs to catch the latter to name the method
    that hung, and must not relabel the former -- "goto did not answer in 30s"
    is a lie when what actually happened is the browser quit.
    """


def closed_connection(exc: OSError, message: str) -> WebSocketError:
    """Name a dropped socket in this module's vocabulary, on every platform.

    The same event -- the browser died mid-stream -- reaches `recv` in two
    shapes. On POSIX the read returns `b""` and the caller raises on the empty
    chunk. On Windows it *raises* instead: WinError 10053
    (`ConnectionAbortedError`) when the local end tears down, 10054
    (`ConnectionResetError`) when the peer vanishes. Both are `OSError`
    subclasses, so the empty-chunk branch never runs and the raw exception --
    carrying a localized OS string -- escaped the whole websocket layer. The
    graceful-close path was dead code on Windows.

    Normalizing it here means a dead browser reads the same everywhere. The
    errno rides along as an attribute rather than in the text: it is the
    language-independent half of the OS message, and the text stays stable for
    callers that match on it.
    """
    error = WebSocketError(message)
    error.errno = getattr(exc, "winerror", None) or exc.errno  # type: ignore[attr-defined]
    error.__cause__ = exc
    return error


class CDPError(RuntimeError):
    """The browser answered, and the answer was an error."""


class BrowserError(RuntimeError):
    """No browser, no page, or a target that cannot be driven."""


# ----------------------------------------------------------------------
# framing
# ----------------------------------------------------------------------


def accept_key(key: str) -> str:
    """The `Sec-WebSocket-Accept` a given `Sec-WebSocket-Key` must produce.

    Sent to the server and, more usefully, checked on the answer: a proxy that
    answers 101 without understanding the protocol fails here rather than
    producing a stream of nonsense frames later.
    """
    digest = hashlib.sha1((key + GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


@dataclass
class Frame:
    """One WebSocket frame, as read off the wire."""

    opcode: int
    payload: bytes = b""
    fin: bool = True

    @property
    def is_control(self) -> bool:
        # Control frames may be interleaved between the fragments of a message
        # and must never be fragmented themselves.
        return self.opcode >= OP_CLOSE


def encode_frame(payload: bytes, opcode: int = OP_TEXT, *,
                 mask: bool = True, fin: bool = True,
                 mask_key: bytes | None = None) -> bytes:
    """Build one frame. Client frames are masked; that is not optional.

    A server that receives an unmasked client frame is required to close the
    connection, and Chrome does.
    """
    if opcode >= OP_CLOSE and len(payload) > 125:
        raise WebSocketError("control frames carry at most 125 bytes")
    if opcode >= OP_CLOSE and not fin:
        raise WebSocketError("control frames cannot be fragmented")

    header = bytearray()
    header.append((0x80 if fin else 0x00) | opcode)

    length = len(payload)
    mask_bit = 0x80 if mask else 0x00
    if length < 126:
        header.append(mask_bit | length)
    elif length < 65536:
        header.append(mask_bit | 126)
        header += struct.pack("!H", length)
    else:
        header.append(mask_bit | 127)
        header += struct.pack("!Q", length)

    if not mask:
        return bytes(header) + payload

    key = mask_key if mask_key is not None else os.urandom(4)
    if len(key) != 4:
        raise WebSocketError("a masking key is exactly 4 bytes")
    masked = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
    return bytes(header) + key + masked


class _Reader:
    """Accumulates bytes and hands back whole frames as they complete.

    Incremental on purpose: a frame can arrive in as many TCP segments as the
    network likes, and one read can carry several frames plus the start of
    another. Both are normal.

    `expect_masked` states which side of the protocol this reader is on. RFC
    6455 requires clients to mask and servers not to, and a reader that
    accepted either would let a mismatched role produce a stream of plausible
    nonsense instead of a protocol error. The default is the client role, which
    is what this module implements; a test server passes True.
    """

    def __init__(self, expect_masked: bool = False) -> None:
        self._buf = bytearray()
        self.expect_masked = expect_masked
        # Fragmentation is state of the frame *stream*, not of the connection,
        # so it lives with the reader that consumes the stream.
        self._fragments: list[bytes] = []
        self._fragment_opcode = 0

    def feed(self, data: bytes) -> None:
        self._buf += data

    def has_frame(self) -> bool:
        return self._peek() is not None

    def _peek(self) -> tuple[int, bool, int, int, bytes] | None:
        """(opcode, fin, length, payload_offset, mask_key) or None if partial."""
        buf = self._buf
        if len(buf) < 2:
            return None
        b0, b1 = buf[0], buf[1]
        opcode = b0 & 0x0F
        fin = bool(b0 & 0x80)
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F
        offset = 2

        if length == 126:
            if len(buf) < offset + 2:
                return None
            length = struct.unpack_from("!H", buf, offset)[0]
            offset += 2
        elif length == 127:
            if len(buf) < offset + 8:
                return None
            length = struct.unpack_from("!Q", buf, offset)[0]
            offset += 8

        if length > MAX_FRAME:
            raise WebSocketError(f"frame of {length} bytes is not plausible")

        if masked and not self.expect_masked:
            # A server must not mask. Refusing is clearer than guessing whether
            # the bytes are masked or the stream is corrupt.
            raise WebSocketError("server sent a masked frame")
        if self.expect_masked and not masked:
            # A client must mask; accepting an unmasked one would silently
            # accept a peer that is not following the protocol at all.
            raise WebSocketError("client sent an unmasked frame")

        key = b""
        if masked:
            if len(buf) < offset + 4:
                return None
            key = bytes(buf[offset:offset + 4])
            offset += 4                       # the masking key precedes the payload

        if len(buf) < offset + length:
            return None
        return opcode, fin, length, offset, key

    def read_frame(self) -> Frame:
        info = self._peek()
        if info is None:
            raise WebSocketError("no complete frame to read")
        opcode, fin, length, offset, key = info
        payload = bytes(self._buf[offset:offset + length])
        del self._buf[:offset + length]
        if key:
            # Undo the mask. A masked payload left as-is decodes to noise that
            # looks like a different bug entirely.
            payload = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
        if opcode >= OP_CLOSE:
            # Control frames carry at most 125 bytes and are never fragmented.
            if length > 125:
                raise WebSocketError("control frame too long")
            if not fin:
                raise WebSocketError("fragmented control frame")
        return Frame(opcode=opcode, payload=payload, fin=fin)

    def recv_message(self) -> Frame:
        """The next whole message, reassembling fragments.

        A control frame is returned the moment it appears rather than being
        buffered behind a half-finished message: RFC 6455 allows one to be
        interleaved precisely so that a ping can be answered while a large
        message is still arriving, and holding it would defeat that.

        Fragmentation state lives here rather than in the socket layer because
        it is a property of the frame stream, not of the connection.
        """
        while True:
            frame = self.read_frame()
            if frame.opcode >= OP_CLOSE:
                return frame

            if frame.opcode == OP_CONTINUATION:
                if not self._fragments:
                    raise WebSocketError(
                        "continuation frame with nothing to continue")
                self._fragments.append(frame.payload)
                if not frame.fin:
                    continue
                whole = b"".join(self._fragments)
                opcode = self._fragment_opcode
                self._fragments, self._fragment_opcode = [], 0
                return Frame(opcode=opcode, payload=whole, fin=True)

            if frame.fin:
                return frame

            # First fragment of a message.
            self._fragment_opcode = frame.opcode
            self._fragments = [frame.payload]


# ----------------------------------------------------------------------
# transport
# ----------------------------------------------------------------------


@dataclass
class WebSocket:
    """A minimal RFC 6455 client, sufficient for CDP.

    Not a general-purpose library: it does not do permessage-deflate, and it
    assumes a single reader. Both are true of how CDP is used here.
    """

    sock: Any = None
    host: str = ""
    path: str = "/"
    timeout: float = 30.0
    _reader: _Reader = field(default_factory=_Reader)
    #: Messages read while looking for something else. CDP ends a *response* by
    #: id, so events that arrive first must be kept rather than discarded.
    inbox: list[Frame] = field(default_factory=list)

    # -- connecting -----------------------------------------------------

    @classmethod
    def connect(cls, url: str, timeout: float = 30.0,
                headers: dict[str, str] | None = None) -> "WebSocket":
        parsed = urlparse(url)
        if parsed.scheme not in ("ws", "wss"):
            raise WebSocketError(f"not a websocket url: {url!r}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        raw = socket.create_connection((host, port), timeout=timeout)

        if parsed.scheme == "wss":
            import ssl
            context = ssl.create_default_context()
            raw = context.wrap_socket(raw, server_hostname=host)

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = [
            f"GET {path} HTTP/1.1",
            f"Host: {host}:{port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
        ]
        for name, value in (headers or {}).items():
            request.append(f"{name}: {value}")
        raw.sendall(("\r\n".join(request) + "\r\n\r\n").encode("ascii"))

        status, reply_headers, leftover = _read_handshake(raw)
        if status != 101:
            raw.close()
            raise WebSocketError(
                f"the server refused the upgrade (HTTP {status}); "
                f"a CDP endpoint answers 101, and a plain web server does not")
        expected = accept_key(key)
        if reply_headers.get("sec-websocket-accept", "") != expected:
            raw.close()
            raise WebSocketError(
                "the upgrade answer did not match the key sent, so whatever "
                "answered is not speaking WebSocket")

        sock = cls(sock=raw, host=f"{host}:{port}", path=path, timeout=timeout)
        if leftover:
            sock._reader.feed(leftover)
        return sock

    # -- sending --------------------------------------------------------

    def send(self, data: str | bytes, opcode: int | None = None) -> None:
        if opcode is None:
            opcode = OP_BINARY if isinstance(data, (bytes, bytearray)) else OP_TEXT
        payload = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        self._sendall(encode_frame(payload, opcode))

    def ping(self, payload: bytes = b"") -> None:
        self._sendall(encode_frame(payload, OP_PING))

    def _sendall(self, raw: bytes) -> None:
        if self.sock is None:
            raise WebSocketError("this socket is not connected")
        self.sock.sendall(raw)

    # -- receiving ------------------------------------------------------

    def recv(self, timeout: float | None = None) -> Frame:
        """The next *message* frame, with control frames handled here.

        Pings are answered and closes raise. Fragments are reassembled by the
        reader, so a caller never sees half a message and never has to think
        about opcode 0.
        """
        while True:
            frame = self._next_message(timeout=timeout)
            if frame.opcode == OP_PING:
                self._sendall(encode_frame(frame.payload, OP_PONG))
                continue
            if frame.opcode == OP_PONG:
                continue
            if frame.opcode == OP_CLOSE:
                code = struct.unpack("!H", frame.payload[:2])[0] if len(frame.payload) >= 2 else 1005
                self._sendall(encode_frame(b"", OP_CLOSE))
                self.close()
                raise WebSocketError(f"the peer closed the connection (code {code})")
            return frame

    def _next_message(self, timeout: float | None = None) -> Frame:
        """A whole message: buffered first, then read off the socket."""
        if self.inbox:
            return self.inbox.pop(0)
        if self.sock is None:
            raise WebSocketError("this socket is not connected")
        limit = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + limit
        while True:
            if self._reader.has_frame():
                return self._reader.recv_message()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WebSocketTimeout(
                    f"the browser sent nothing for {_secs(limit)}")
            self.sock.settimeout(remaining)
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                raise WebSocketTimeout(
                    f"the browser sent nothing for {_secs(limit)}") from None
            except OSError as exc:
                # Windows reports a dead peer by raising (10053/10054) where
                # POSIX returns b"" -- same event, so the same error. Must come
                # after the timeout clause: on 3.10+ socket.timeout is
                # TimeoutError, an OSError subclass.
                raise closed_connection(
                    exc, "the browser closed the connection")
            if not chunk:
                self.close()
                raise WebSocketError("the browser closed the connection")
            self._reader.feed(chunk)

    def recv_text(self, timeout: float | None = None) -> str:
        frame = self.recv(timeout=timeout)
        if frame.opcode != OP_TEXT:
            raise WebSocketError(
                f"expected a text message, got opcode {frame.opcode}")
        return frame.payload.decode("utf-8", "replace")

    def close(self) -> None:
        if self.sock is None:
            return
        sock, self.sock = self.sock, None
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


def _read_handshake(raw: Any) -> tuple[int, dict[str, str], bytes]:
    """Read the HTTP upgrade reply, returning (status, headers, leftover bytes).

    Leftover matters: a server may answer 101 and begin sending frames in the
    same packet, and bytes left in that buffer belong to the frame stream.
    """
    buf = bytearray()
    while b"\r\n\r\n" not in buf:
        if len(buf) > MAX_HANDSHAKE:
            raise WebSocketError("the upgrade reply never ended")
        try:
            chunk = raw.recv(4096)
        except socket.timeout:
            raise WebSocketError("timed out waiting for the upgrade reply") from None
        except OSError as exc:
            # Same Windows/POSIX split as `_next_message`: a browser that dies
            # during the upgrade raises here too, and that must read as this
            # module's error rather than surfacing a raw WinError.
            raise closed_connection(
                exc, "the connection closed during the upgrade")
        if not chunk:
            raise WebSocketError("the connection closed during the upgrade")
        buf += chunk

    head, _, rest = bytes(buf).partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    try:
        status = int(lines[0].split(" ")[1])
    except (IndexError, ValueError):
        raise WebSocketError(f"unreadable status line: {lines[0]!r}") from None
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if sep:
            headers[name.strip().lower()] = value.strip()
    return status, headers, rest


# ----------------------------------------------------------------------
# CDP
# ----------------------------------------------------------------------


class Browser:
    """A CDP session against one page (or browser) target."""

    def __init__(self, ws: WebSocket, target: dict | None = None) -> None:
        self.ws = ws
        self.target = target or {}
        self._next_id = 0
        self.events: list[dict] = []
        self._closed = False

    # -- connecting -----------------------------------------------------

    @classmethod
    def connect(cls, endpoint: str = "http://127.0.0.1:9222",
                *, target_type: str = "page", index: int = 0,
                timeout: float = 30.0) -> "Browser":
        """Attach to a running browser's debugging endpoint.

        `endpoint` may be the HTTP discovery root (the usual
        `--remote-debugging-port` case) or a `ws://` URL pointing straight at a
        target, which is what an already-attached client hands around.
        """
        if endpoint.startswith(("ws://", "wss://")):
            return cls(WebSocket.connect(endpoint, timeout=timeout))

        targets = list_targets(endpoint, timeout=timeout)
        wanted = [t for t in targets if t.get("type") == target_type]
        if not wanted:
            kinds = sorted({t.get("type", "?") for t in targets}) or ["nothing"]
            raise BrowserError(
                f"no {target_type!r} target at {endpoint}. Found: "
                f"{', '.join(kinds)}. A browser started for CDP needs at least "
                f"one open tab.")
        if index >= len(wanted):
            raise BrowserError(
                f"asked for {target_type} #{index}, but only {len(wanted)} exist")
        chosen = wanted[index]
        url = chosen.get("webSocketDebuggerUrl")
        if not url:
            raise BrowserError(
                f"target {chosen.get('id')} has no webSocketDebuggerUrl, so it "
                f"cannot be driven")
        return cls(WebSocket.connect(url, timeout=timeout), target=chosen)

    # -- the request/response loop --------------------------------------

    def call(self, method: str, params: dict | None = None, *,
             timeout: float | None = None) -> dict:
        """Send one CDP command and return its result.

        Matched by id: replies to other calls, and events that arrive while
        waiting, are handled separately. Treating the next message as the
        answer is the bug that makes a client work until two calls overlap.
        """
        if self._closed:
            raise BrowserError("this browser session is closed")
        self._next_id += 1
        wanted = self._next_id
        self.ws.send(json.dumps({"id": wanted, "method": method,
                                 "params": params or {}}))
        budget = self.ws.timeout if timeout is None else timeout
        deadline = time.monotonic() + budget

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BrowserError(
                    f"{method} did not answer within {_secs(budget)}")
            try:
                message = json.loads(self.ws.recv_text(timeout=remaining))
            except WebSocketTimeout as exc:
                # The socket's timeout carries no method name, leaving the
                # caller unable to say *which* call hung. Named by type, not by
                # comparing clocks: the inner deadline closes microseconds
                # before this one, so a comparison here loses the race under
                # load and the silent-browser report reverts to "nothing was
                # sent". A closed socket is deliberately not caught -- that is
                # a fact about the browser, not about this request.
                raise BrowserError(
                    f"{method} did not answer within {_secs(budget)}") from exc

            if "method" in message and "id" not in message:
                self.events.append(message)
                if len(self.events) > 500:
                    del self.events[:250]
                continue
            if message.get("id") != wanted:
                # Someone else's answer, or an id we never sent. Keep it rather
                # than failing: CDP is allowed to interleave.
                continue

            error = message.get("error")
            if error:
                raise CDPError(
                    f"{method} failed: {error.get('message', error)}"
                    + (f" (code {error['code']})" if "code" in error else ""))
            return message.get("result", {})

    # -- page operations ------------------------------------------------

    def enable(self) -> None:
        """Turn on the domains the page operations rely on."""
        for domain in ("Page", "Runtime"):
            try:
                self.call(f"{domain}.enable")
            except CDPError:
                # A browser-level target has no Page domain. Not fatal.
                pass

    def evaluate(self, expression: str, *, await_promise: bool = True,
                 timeout: float | None = None) -> Any:
        result = self.call("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": await_promise,
        }, timeout=timeout)
        if result.get("exceptionDetails"):
            details = result["exceptionDetails"]
            described = (details.get("exception") or {}).get("description") \
                or details.get("text") or "unknown error"
            raise CDPError(f"the page raised: {described}")
        return (result.get("result") or {}).get("value")

    def goto(self, url: str, *, wait: bool = True,
             timeout: float = 30.0) -> str:
        """Navigate, and by default wait for the document to finish loading."""
        self.call("Page.navigate", {"url": url}, timeout=timeout)
        if not wait:
            return ""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                state = self.evaluate("document.readyState", await_promise=False,
                                      timeout=max(1.0, deadline - time.monotonic()))
            except (CDPError, BrowserError, WebSocketError):
                # Mid-navigation the context is torn down and re-created, so a
                # failed evaluate here is expected rather than an error.
                time.sleep(0.05)
                continue
            if state == "complete":
                return self.title()
            time.sleep(0.05)
        raise BrowserError(f"{url} did not finish loading within {timeout:.0f}s")

    def title(self) -> str:
        try:
            return self.evaluate("document.title", await_promise=False) or ""
        except (CDPError, BrowserError):
            return ""

    def current_url(self) -> str:
        try:
            return self.evaluate("location.href", await_promise=False) or ""
        except (CDPError, BrowserError):
            return ""

    def screenshot(self, *, full_page: bool = False,
                   timeout: float = 30.0) -> bytes:
        """A PNG of the viewport, or of the whole scrollable page."""
        params: dict[str, Any] = {"format": "png"}
        if full_page:
            metrics = self.call("Page.getLayoutMetrics")
            size = metrics.get("cssContentSize") or metrics.get("contentSize") or {}
            width, height = size.get("width"), size.get("height")
            if width and height:
                params["captureBeyondViewport"] = True
                params["clip"] = {"x": 0, "y": 0, "width": width,
                                  "height": height, "scale": 1}
        result = self.call("Page.captureScreenshot", params, timeout=timeout)
        data = result.get("data")
        if not data:
            raise BrowserError("the browser returned an empty screenshot")
        return base64.b64decode(data)

    # -- input ----------------------------------------------------------

    def click(self, x: float, y: float, *, button: str = "left",
              clicks: int = 1) -> None:
        """Click at a viewport point. Coordinates are CSS pixels from the
        top-left of the viewport, which is what `getBoundingClientRect` gives."""
        base = {"x": float(x), "y": float(y), "button": button,
                "clickCount": clicks}
        self.call("Input.dispatchMouseEvent", {**base, "type": "mousePressed"})
        self.call("Input.dispatchMouseEvent", {**base, "type": "mouseReleased"})

    def move(self, x: float, y: float) -> None:
        self.call("Input.dispatchMouseEvent", {"type": "mouseMoved",
                                               "x": float(x), "y": float(y)})

    def type_text(self, text: str) -> None:
        """Insert text as if typed into the focused element.

        `Input.insertText` is used rather than per-key events: it is one round
        trip, it does not depend on which physical layout the browser thinks
        the user has, and it does not fire a keydown for every character.
        """
        self.call("Input.insertText", {"text": text})

    def press(self, key: str, *, modifiers: int = 0) -> None:
        """Press and release one key by its CDP name (Enter, ArrowDown, a...)."""
        for kind in ("keyDown", "keyUp"):
            self.call("Input.dispatchKeyEvent", {"type": kind, "key": key,
                                                 "modifiers": modifiers,
                                                 "text": key if len(key) == 1 else ""})

    def scroll(self, dy: float, *, x: float = 0, y: float = 0) -> None:
        self.call("Input.dispatchMouseEvent", {
            "type": "mouseWheel", "x": float(x), "y": float(y),
            "deltaX": 0, "deltaY": float(dy)})

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        self._closed = True
        self.ws.close()


# ----------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------


def list_targets(endpoint: str = "http://127.0.0.1:9222",
                 *, timeout: float = 10.0) -> list[dict]:
    """Ask a debugging endpoint what it has open."""
    import requests

    url = endpoint.rstrip("/") + "/json/list"
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise BrowserError(
            f"could not reach a browser at {endpoint}: "
            f"{type(exc).__name__}: {exc}. Start one with "
            f"`--remote-debugging-port=9222`.") from exc
    if resp.status_code != 200:
        raise BrowserError(
            f"{endpoint} answered HTTP {resp.status_code} for {url}; a CDP "
            f"endpoint serves /json/list")
    try:
        body = resp.json()
    except ValueError as exc:
        raise BrowserError(
            f"{endpoint} did not answer JSON for {url}; whatever is on that "
            f"port is not a browser debugging endpoint") from exc
    if not isinstance(body, list):
        raise BrowserError(f"expected a list of targets, got {type(body).__name__}")
    return body


def parse_targets(payload: Any) -> list[dict]:
    """Normalise a target listing into dicts with the fields we use."""
    if not isinstance(payload, list):
        return []
    out = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        out.append({
            "id": entry.get("id", ""),
            "type": entry.get("type", ""),
            "title": entry.get("title", ""),
            "url": entry.get("url", ""),
            "webSocketDebuggerUrl": entry.get("webSocketDebuggerUrl", ""),
        })
    return out


#: Where Chrome and Edge live, per platform. Edge is listed because it is the
#: same engine and is already installed on every Windows machine.
_CANDIDATES = {
    "win32": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ],
    "darwin": [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ],
    "linux": [
        "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium", "/usr/bin/chromium-browser",
        "/usr/bin/microsoft-edge",
    ],
}


def browser_candidates(platform: str | None = None) -> list[str]:
    """Executables to try, in order, for this platform."""
    plat = (platform or sys.platform).lower()
    if plat.startswith("win"):
        key = "win32"
    elif plat.startswith("darwin"):
        key = "darwin"
    else:
        key = "linux"
    found = list(_CANDIDATES.get(key, []))
    env = os.environ.get("AUTOFORGE_BROWSER")
    if env:
        found.insert(0, env)
    return found


def find_browser(platform: str | None = None) -> str | None:
    """The first browser that actually exists, or None.

    Also consults PATH, so a machine that installed Chrome somewhere unusual
    works without configuration.
    """
    for path in browser_candidates(platform):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    for name in ("chrome", "google-chrome", "chromium", "msedge", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    return None


def launch_browser(port: int = 9222, *, headless: bool = True,
                   executable: str | None = None,
                   user_data_dir: str | None = None,
                   wait: float = 20.0,
                   extra_args: Iterable[str] = ()) -> tuple[subprocess.Popen, str]:
    """Start a browser with debugging on and wait for it to answer.

    Returns (process, endpoint). A separate profile directory is the default,
    because Chrome refuses to expose the debug port on the profile a person is
    already using -- and quietly attaching to someone's logged-in session is
    not something to do by accident.
    """
    exe = executable or find_browser()
    if not exe:
        raise BrowserError(
            "no Chrome, Chromium or Edge found. Set AUTOFORGE_BROWSER to the "
            "executable, or start a browser yourself with "
            "--remote-debugging-port=9222 and connect to it.")

    profile = user_data_dir or os.path.join(
        os.environ.get("AUTOFORGE_HOME") or os.path.expanduser("~"),
        ".autoforge-browser-profile")

    argv = [
        exe,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-background-networking",
    ]
    if headless:
        argv.append("--headless=new")
    argv.extend(extra_args)
    argv.append("about:blank")

    kwargs: dict[str, Any] = {"stdout": subprocess.DEVNULL,
                              "stderr": subprocess.DEVNULL}
    if sys.platform.startswith("win"):
        # Detached so closing the agent does not kill the browser it started,
        # and no console window flashes up.
        kwargs["creationflags"] = 0x00000008 | 0x08000000
    else:
        kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(argv, **kwargs)
    except OSError as exc:
        raise BrowserError(
            f"could not start {exe}: {type(exc).__name__}: {exc}") from exc

    endpoint = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise BrowserError(
                f"{exe} exited immediately (code {proc.returncode}); the "
                f"profile at {profile} may be locked by another run")
        try:
            list_targets(endpoint, timeout=1.0)
            return proc, endpoint
        except BrowserError:
            time.sleep(0.2)

    proc.terminate()
    raise BrowserError(
        f"{exe} did not expose a debugging endpoint on port {port} within "
        f"{wait:.0f}s")

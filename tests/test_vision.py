"""Looking at a picture, and being honest about what came back.

The interesting property of this module is not that it can POST an image. It is
that every way of *failing* gets named precisely, because the failure mode of a
vision call is a model cheerfully describing something that is not there.

Three specific traps are pinned below:

* **The media type.** A wrong `image/...` header comes back as a complaint about
  the model or the prompt, which sends the reader somewhere useless. So the type
  is read off the bytes, and a type that cannot be read is refused here rather
  than guessed.

* **The model that cannot see.** On a single-provider setup the endpoint is
  shared with the text model, so `base_url` falls back to the main one. The
  *model* must not: inheriting a text model id sends the image to something that
  will answer anyway, with a description invented from the filename. Naming the
  vision model is what turns this on, and an unconfigured vision is silent
  rather than an error, because most runs never look at an image.

* **The empty answer.** A gateway that returns HTTP 200 with no content, or a
  reasoning model that spends its whole budget thinking, both read as "the image
  is blank" unless the empty answer is raised.

The HTTP tests use a real server on a real socket, so the status code, the body
and the connection failure are all genuinely produced rather than mocked.
"""

from __future__ import annotations

import base64
import json
import socket
import threading

import pytest

from autoforge.vision import (
    DEFAULT_QUESTION,
    IMAGE_LIMIT,
    VisionClient,
    VisionError,
    _read_answer,
    load_image,
    sniff_media_type,
    vision_from_config,
)

from autoforge.configfile import PROXIES


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
GIF = b"GIF89a" + b"\x00" * 64
BMP = b"BM" + b"\x00" * 64
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 64


# ----------------------------------------------------------------------
# a real HTTP endpoint
# ----------------------------------------------------------------------


class FakeEndpoint:
    """A one-request-per-call HTTP server that records what it was sent."""

    def __init__(self, *, status: int = 200, body: bytes | str = b"{}",
                 content_type: str = "application/json",
                 accept_connections: bool = True) -> None:
        self.status = status
        self.body = body.encode("utf-8") if isinstance(body, str) else body
        self.content_type = content_type
        self.accept_connections = accept_connections
        self.requests: list[dict] = []
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if accept_connections:
            self._sock.bind(("127.0.0.1", 0))
            self._sock.listen(8)
            self.port = self._sock.getsockname()[1]
        else:
            self.port = 1
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> "FakeEndpoint":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                try:
                    self._handle(conn)
                except OSError:
                    pass

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(5)
        buf = bytearray()
        while b"\r\n\r\n" not in buf:
            chunk = conn.recv(65536)
            if not chunk:
                return
            buf += chunk
        head, _, rest = bytes(buf).partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        headers = {}
        for line in lines[1:]:
            name, sep, value = line.partition(":")
            if sep:
                headers[name.strip().lower()] = value.strip()
        length = int(headers.get("content-length", "0") or 0)
        body = rest
        while len(body) < length:
            chunk = conn.recv(65536)
            if not chunk:
                break
            body += chunk

        self.requests.append({
            "request_line": lines[0],
            "headers": headers,
            "body": body[:length].decode("utf-8", "replace"),
        })

        conn.sendall((
            f"HTTP/1.1 {self.status} X\r\n"
            f"Content-Type: {self.content_type}\r\n"
            f"Content-Length: {len(self.body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii") + self.body)

    def last_json(self) -> dict:
        return json.loads(self.requests[-1]["body"])


def completion(text: str = "a red square") -> bytes:
    return json.dumps({
        "choices": [{"message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
    }).encode()


# ----------------------------------------------------------------------
# what the bytes are
# ----------------------------------------------------------------------


class TestSniffMediaType:
    @pytest.mark.parametrize("data,expected", [
        (PNG, "image/png"),
        (JPEG, "image/jpeg"),
        (GIF, "image/gif"),
        (BMP, "image/bmp"),
        (WEBP, "image/webp"),
    ])
    def test_the_format_is_read_from_the_bytes(self, data, expected):
        assert sniff_media_type(data) == expected

    def test_an_unknown_format_is_none_rather_than_a_guess(self):
        """Guessing means the far end blames the model, not the file."""
        assert sniff_media_type(b"not an image at all") is None
        assert sniff_media_type(b"") is None

    def test_a_riff_that_is_not_webp_is_not_webp(self):
        assert sniff_media_type(b"RIFF\x00\x00\x00\x00AVI ") is None

    def test_a_pdf_is_not_an_image(self):
        assert sniff_media_type(b"%PDF-1.7\n") is None


class TestLoadImage:

    def test_a_png_becomes_a_data_url(self, tmp_path):
        path = tmp_path / "shot.png"
        path.write_bytes(PNG)
        url, size, media_type = load_image(str(path))
        assert url.startswith("data:image/png;base64,")
        assert size == len(PNG)
        assert media_type == "image/png"
        assert base64.b64decode(url.split(",", 1)[1]) == PNG

    def test_an_http_url_is_passed_through_untouched(self):
        """Re-uploading means this process downloads bytes for no reason."""
        url, size, media_type = load_image("https://example.com/a.png")
        assert url == "https://example.com/a.png"
        assert (size, media_type) == (0, "")

    def test_a_data_url_is_already_a_data_url(self):
        assert load_image("data:image/png;base64,AAAA")[0] == \
            "data:image/png;base64,AAAA"

    def test_no_source_is_refused(self):
        with pytest.raises(VisionError, match="no image"):
            load_image("")

    def test_a_missing_file_names_the_path(self):
        with pytest.raises(VisionError, match="no such image"):
            load_image("/nope/missing.png")

    def test_a_non_image_says_what_it_expected(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("just text")
        with pytest.raises(VisionError, match="expected PNG, JPEG"):
            load_image(str(path))

    def test_an_empty_file_is_refused(self, tmp_path):
        path = tmp_path / "empty.png"
        path.write_bytes(b"")
        with pytest.raises(VisionError, match="empty"):
            load_image(str(path))

    def test_an_oversized_file_says_its_size_and_the_limit(self, tmp_path):
        """An image that is too big must say so, not fail inside the request."""
        path = tmp_path / "huge.png"
        path.write_bytes(PNG + b"\x00" * 2000)
        with pytest.raises(VisionError, match="MiB limit"):
            load_image(str(path), limit=1000)

    def test_a_directory_is_not_a_file(self, tmp_path):
        with pytest.raises(VisionError, match="no such image"):
            load_image(str(tmp_path))


# ----------------------------------------------------------------------
# the call
# ----------------------------------------------------------------------


class TestDescribe:

    def test_a_good_answer_comes_back(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(PNG)
        with FakeEndpoint(body=completion("a red square")) as server:
            client = VisionClient(model="vlm", base_url=server.base_url)
            assert client.describe(str(path)) == "a red square"

    def test_the_request_is_openai_shaped_and_carries_the_image(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(PNG)
        with FakeEndpoint(body=completion()) as server:
            client = VisionClient(model="vlm", base_url=server.base_url,
                                  api_key="sk-test")
            client.describe(str(path), "what is this?")
            sent = server.last_json()
            assert sent["model"] == "vlm"
            parts = sent["messages"][0]["content"]
            assert parts[0] == {"type": "text", "text": "what is this?"}
            image = parts[1]["image_url"]["url"]
            assert image.startswith("data:image/png;base64,")
            assert server.requests[-1]["headers"]["authorization"] == \
                "Bearer sk-test"

    def test_it_posts_to_chat_completions(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(PNG)
        with FakeEndpoint(body=completion()) as server:
            VisionClient(model="vlm",
                         base_url=server.base_url).describe(str(path))
            assert server.requests[-1]["request_line"].startswith(
                "POST /v1/chat/completions")

    def test_the_default_question_is_asked_when_none_is_given(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(PNG)
        with FakeEndpoint(body=completion()) as server:
            VisionClient(model="vlm",
                         base_url=server.base_url).describe(str(path))
            assert server.last_json()["messages"][0]["content"][0]["text"] == \
                DEFAULT_QUESTION

    def test_a_trailing_slash_on_the_base_url_does_not_double(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(PNG)
        with FakeEndpoint(body=completion()) as server:
            VisionClient(model="vlm",
                         base_url=server.base_url + "/").describe(str(path))
            assert "//chat/completions" not in server.requests[-1]["request_line"]

    def test_no_api_key_sends_no_authorization_header(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(PNG)
        with FakeEndpoint(body=completion()) as server:
            VisionClient(model="vlm",
                         base_url=server.base_url).describe(str(path))
            assert "authorization" not in server.requests[-1]["headers"]

    def test_an_http_error_names_the_status_the_model_and_the_body(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(PNG)
        with FakeEndpoint(status=404, body="model not found") as server:
            client = VisionClient(model="ghost-vlm", base_url=server.base_url)
            with pytest.raises(VisionError) as info:
                client.describe(str(path))
        message = str(info.value)
        assert "404" in message
        assert "ghost-vlm" in message
        assert "model not found" in message

    def test_a_non_json_body_says_it_was_not_json(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(PNG)
        with FakeEndpoint(body="<html>gateway</html>",
                          content_type="text/html") as server:
            client = VisionClient(model="vlm", base_url=server.base_url)
            with pytest.raises(VisionError, match="non-JSON"):
                client.describe(str(path))

    def test_nothing_listening_is_reported_as_such(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(PNG)
        client = VisionClient(model="vlm", base_url="http://127.0.0.1:9/v1",
                              timeout=2)
        with pytest.raises(VisionError, match="could not reach"):
            client.describe(str(path))

    def test_an_unconfigured_client_refuses_before_reaching_the_network(self):
        client = VisionClient(model="", base_url="")
        with pytest.raises(VisionError, match="no vision endpoint"):
            client.describe("whatever.png")

    def test_a_bad_image_is_refused_before_any_request_is_made(self, tmp_path):
        """The upload is the expensive part; a bad file must not reach it."""
        with FakeEndpoint(body=completion()) as server:
            client = VisionClient(model="vlm", base_url=server.base_url)
            with pytest.raises(VisionError, match="no such image"):
                client.describe(str(tmp_path / "gone.png"))
            assert server.requests == []


# ----------------------------------------------------------------------
# reading the answer
# ----------------------------------------------------------------------


class TestReadAnswer:

    def test_a_plain_string_answer(self):
        body = {"choices": [{"message": {"content": "hello"}}]}
        assert _read_answer(body, "vlm", "m") == "hello"

    def test_a_list_of_content_parts_is_joined(self):
        body = {"choices": [{"message": {"content": [
            {"type": "text", "text": "one "},
            {"type": "image_url", "image_url": {}},
            {"type": "text", "text": "two"},
        ]}}]}
        assert _read_answer(body, "vlm", "m") == "one two"

    def test_surrounding_whitespace_is_stripped(self):
        body = {"choices": [{"message": {"content": "  hi  "}}]}
        assert _read_answer(body, "vlm", "m") == "hi"

    def test_an_empty_answer_explains_the_reasoning_model_case(self):
        """The trap: HTTP 200, no content, and it reads as a blank image."""
        body = {"choices": [{"message": {"content": ""},
                             "finish_reason": "length"}]}
        with pytest.raises(VisionError, match="empty answer"):
            _read_answer(body, "vlm", "m")
        with pytest.raises(VisionError, match="length"):
            _read_answer(body, "vlm", "m")
        with pytest.raises(VisionError, match="raise max_tokens"):
            _read_answer(body, "vlm", "m")

    def test_a_null_content_is_empty_too(self):
        body = {"choices": [{"message": {"content": None}}]}
        with pytest.raises(VisionError, match="empty answer"):
            _read_answer(body, "vlm", "m")

    def test_no_choices_names_the_model(self):
        with pytest.raises(VisionError, match="ghost"):
            _read_answer({"choices": []}, "vlm", "ghost")

    def test_a_missing_choices_key_is_no_choices(self):
        with pytest.raises(VisionError, match="no choices"):
            _read_answer({"id": "x"}, "vlm", "m")

    def test_an_error_object_is_raised_with_its_message(self):
        body = {"error": {"message": "quota exceeded", "code": 429}}
        with pytest.raises(VisionError, match="quota exceeded"):
            _read_answer(body, "vlm", "m")

    def test_a_string_error_is_raised_too(self):
        with pytest.raises(VisionError, match="bad request"):
            _read_answer({"error": "bad request"}, "vlm", "m")

    def test_a_non_object_body_is_named(self):
        with pytest.raises(VisionError, match="not an object"):
            _read_answer(["unexpected"], "vlm", "m")


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------


class TestVisionFromConfig:

    def test_an_explicit_model_turns_vision_on(self):
        client, problem = vision_from_config({
            "base_url": "https://gw/v1", "key": "k",
            "vision": {"model": "qwen-vl"},
        })
        assert problem == ""
        assert client is not None and client.model == "qwen-vl"
        assert client.base_url == "https://gw/v1"

    def test_no_vision_section_is_silently_off(self):
        """Most runs never look at an image; that is not a problem to report."""
        client, problem = vision_from_config({"base_url": "https://gw/v1"})
        assert client is None
        assert problem == ""

    def test_a_text_model_id_is_never_inherited(self):
        """The trap this exists to prevent.

        A shared gateway makes the *endpoint* reusable, but the model id is not:
        sending an image to a text model returns a confident description of
        nothing. So vision stays off until a vision model is named.
        """
        client, problem = vision_from_config({
            "base_url": "https://gw/v1", "key": "k",
            "model": "deepseek-chat",
        })
        assert client is None
        assert problem == ""

    def test_disabled_is_off_even_with_a_model(self):
        client, problem = vision_from_config({
            "base_url": "https://gw/v1",
            "vision": {"model": "qwen-vl", "enabled": False},
        })
        assert client is None
        assert "disabled" in problem

    def test_a_model_with_no_reachable_endpoint_is_a_reported_problem(self):
        client, problem = vision_from_config({"vision": {"model": "qwen-vl"}})
        assert client is None
        assert "no vision.base_url" in problem

    def test_a_non_table_vision_section_is_reported(self):
        client, problem = vision_from_config({"vision": "qwen-vl"})
        assert client is None
        assert "table" in problem

    def test_the_vision_endpoint_wins_over_the_main_one(self):
        client, _ = vision_from_config({
            "base_url": "https://main/v1",
            "vision": {"model": "vlm", "base_url": "https://vision/v1"},
        })
        assert client.base_url == "https://vision/v1"

    def test_the_vision_key_wins_over_the_main_key(self):
        client, _ = vision_from_config({
            "key": "main-key", "base_url": "https://gw/v1",
            "vision": {"model": "vlm", "key": "vision-key"},
        })
        assert client.api_key == "vision-key"

    def test_the_main_key_is_used_when_vision_has_none(self):
        client, _ = vision_from_config({
            "key": "main-key", "base_url": "https://gw/v1",
            "vision": {"model": "vlm"},
        })
        assert client.api_key == "main-key"

    def test_the_env_var_turns_it_on(self, monkeypatch):
        monkeypatch.setenv("AUTOFORGE_VISION_MODEL", "env-vlm")
        client, _ = vision_from_config({"base_url": "https://gw/v1"})
        assert client is not None and client.model == "env-vlm"

    def test_timeout_and_max_tokens_are_honoured(self):
        client, _ = vision_from_config({
            "base_url": "https://gw/v1",
            "vision": {"model": "vlm", "timeout": 7, "max_tokens": 55},
        })
        assert client.timeout == 7.0
        assert client.max_tokens == 55

    def test_proxies_are_passed_through(self):
        """A gateway behind a proxy is the normal case on some networks."""
        client, _ = vision_from_config({
            "base_url": "https://gw/v1", "proxies": {"https": "socks5://h:1"},
            "vision": {"model": "vlm"},
        })
        assert client.proxies == {"https": "socks5://h:1"}

    # The `proxy` switch is the one the text client reads. Vision sharing the
    # relay is not a nicety: on a network where the text side needs it, vision
    # without it fails in the least diagnosable way -- the page loads, the
    # screenshot is fine, and only the description call dies.
    def test_the_proxy_switch_reaches_vision(self):
        client, _ = vision_from_config({
            "base_url": "https://gw/v1", "proxy": True,
            "vision": {"model": "vlm"},
        })
        assert client.proxies == PROXIES

    def test_the_switch_being_off_leaves_vision_direct(self):
        client, _ = vision_from_config({
            "base_url": "https://gw/v1", "proxy": False,
            "vision": {"model": "vlm"},
        })
        assert client.proxies is None

    def test_an_explicit_vision_proxy_beats_the_switch(self):
        client, _ = vision_from_config({
            "base_url": "https://gw/v1", "proxy": True,
            "vision": {"model": "vlm", "proxies": {"https": "http://alt:9"}},
        })
        assert client.proxies == {"https": "http://alt:9"}

    def test_vision_inherits_the_main_proxy_table_when_present(self):
        client, _ = vision_from_config({
            "base_url": "https://gw/v1",
            "proxies": {"https": "socks5://shared:1"},
            "vision": {"model": "vlm"},
        })
        assert client.proxies == {"https": "socks5://shared:1"}


class TestReport:

    def test_an_unconfigured_report_names_what_to_set(self):
        client = VisionClient(model="", base_url="")
        report = client.report()
        assert "not configured" in report
        assert "vision.model" in report

    def test_a_configured_report_names_the_model_and_endpoint(self):
        client = VisionClient(model="qwen-vl",
                              base_url="https://gw/v1")
        report = client.report()
        assert "qwen-vl" in report
        assert "https://gw/v1" in report

    def test_a_client_with_a_model_but_no_endpoint_is_falsy(self):
        assert not VisionClient(model="vlm", base_url="")
        assert VisionClient(model="vlm", base_url="https://x")


class TestImageLimit:
    def test_the_limit_is_a_sane_number_of_megabytes(self):
        assert 1_000_000 <= IMAGE_LIMIT <= 50_000_000

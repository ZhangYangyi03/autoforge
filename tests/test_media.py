"""Making a picture or a clip, and being honest about when it did not.

The property worth pinning here is not that the module can POST a prompt. It is
that the *asynchronous* half reports the truth, because the two halves of this
API fail in opposite ways:

* An image request that is refused comes back as an HTTP error, so the status
  code is the whole story and the gateway's own message is the useful part --
  it lists the model ids that would have been accepted.

* A video request that is refused comes back as **HTTP 200 with a task id**.
  The refusal only appears later, on the poll, as `status: failed` with an
  `error` string. A client that trusts the create call's status code reports
  success for a job that never ran, and the operator sees a green tick next to
  nothing. `test_a_video_task_that_fails_raises_with_the_servers_reason` is the
  test for that, and it is the reason the poll loop exists at all.

The size separator is also pinned, because it is genuinely inconsistent between
the two endpoints and sending the wrong one is refused: images spell a size
`1024x1024`, video spells it `1280*720`.

The HTTP tests use a real server on a real socket, so the status code, the body
and the connection failure are all genuinely produced rather than mocked.
"""

from __future__ import annotations

import json
import socket
import threading

import pytest

from autoforge.media import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_VIDEO_MODEL,
    IMAGE_SIZES,
    VIDEO_SIZES,
    Generation,
    MediaClient,
    MediaError,
    _slug,
    _unique,
    media_from_config,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 400


# ----------------------------------------------------------------------
# a real HTTP endpoint, scripted per route
# ----------------------------------------------------------------------


class ScriptedEndpoint:
    """A socket HTTP server whose answer per (method, path) is written down.

    Unlike a single-response fake, this can answer the same path differently on
    successive calls, which is what a poll loop needs: `queued`, then
    `in_progress`, then `completed`. That sequence is the whole subject of the
    video tests, so faking it with one fixed response would test nothing.
    """

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], list] = {}
        self.requests: list[dict] = []
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def script(self, method: str, path: str, *answers) -> None:
        """Queue answers for a route. The last one repeats forever."""
        self.routes.setdefault((method.upper(), path), []).extend(answers)

    def __enter__(self) -> "ScriptedEndpoint":
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
        method, path, _ = (lines[0].split(" ") + ["", "", ""])[:3]
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
            "method": method, "path": path, "headers": headers,
            "body": body[:length].decode("utf-8", "replace"),
        })

        answers = self.routes.get((method.upper(), path))
        if not answers:
            self._reply(conn, 404, b'{"error":"no such route"}')
            return
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        status, payload, content_type = answer
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self._reply(conn, status, payload, content_type)

    @staticmethod
    def _reply(conn: socket.socket, status: int, payload: bytes,
               content_type: str = "application/json") -> None:
        reason = {200: "OK", 401: "Unauthorized", 422: "Unprocessable Entity",
                  503: "Service Unavailable"}.get(status, "Error")
        head = (f"HTTP/1.1 {status} {reason}\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Length: {len(payload)}\r\n"
                "Connection: close\r\n\r\n")
        conn.sendall(head.encode("latin-1") + payload)


def client_for(endpoint: ScriptedEndpoint, **kw) -> MediaClient:
    """A client pointed at the fake, with proxying off.

    `proxies={}` rather than the default: the test environment sets an
    `http_proxy`, and a request to 127.0.0.1 that goes through it never arrives.
    """
    kw.setdefault("out_dir", None)
    return MediaClient(base_url=endpoint.base_url, api_key="k",
                       poll_seconds=0.01, proxies={}, **kw)


# ----------------------------------------------------------------------
# filenames: a prompt is not a path
# ----------------------------------------------------------------------


def test_a_slug_cannot_escape_the_output_directory():
    """The prompt reaches the filesystem, so a slash in it must not."""
    assert "/" not in _slug("../../etc/passwd")
    assert "\\" not in _slug("..\\..\\windows")
    assert ".." not in _slug("..")


def test_a_slug_keeps_cjk_readable():
    """A Chinese prompt should produce a filename a person can recognise."""
    assert "红色圆" in _slug("一个红色圆 在白色背景上")


def test_an_existing_file_is_not_overwritten(tmp_path):
    """Two runs are not the same bytes; the first must survive the second."""
    first = tmp_path / "image.png"
    first.write_bytes(b"first")
    assert _unique(first).name == "image-2.png"
    (tmp_path / "image-2.png").write_bytes(b"second")
    assert _unique(first).name == "image-3.png"


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------


def test_the_top_level_api_key_is_used():
    """The regression: the config field is `api_key`, not `key`.

    A media section that names no key of its own must inherit the main one.
    Reading only `key` produces a client that is configured, reports itself
    ready, and answers 401 on every call -- which is exactly the failure this
    pins.
    """
    client, problem = media_from_config(
        {"base_url": "https://x/v1", "api_key": "top-level"})
    assert problem == ""
    assert client is not None
    assert client.api_key == "top-level"


def test_the_media_key_wins_over_the_top_level_one():
    client, _ = media_from_config({
        "base_url": "https://x/v1", "api_key": "top-level",
        "media": {"api_key": "media-level"}})
    assert client is not None
    assert client.api_key == "media-level"


def test_an_unset_media_section_is_silent():
    """A setup that never generates anything should not be nagged."""
    client, problem = media_from_config({"base_url": "https://x/v1"})
    assert problem == ""
    assert client is not None


def test_a_disabled_media_section_says_so():
    client, problem = media_from_config(
        {"base_url": "https://x/v1", "media": {"enabled": False}})
    assert client is None
    assert "disabled" in problem


def test_the_endpoint_falls_back_to_the_main_base_url():
    client, _ = media_from_config(
        {"base_url": "https://x/v1", "api_key": "k",
         "media": {"image_model": "Some-Image"}})
    assert client is not None
    assert client.base_url == "https://x/v1"


def test_the_model_ids_do_not_fall_back_to_the_text_model():
    """The trap: a text model id sent to the image endpoint is refused by name.

    The endpoint is shared on a single-provider setup, so it falls back. The
    model must not, and the default has to be a real image id rather than the
    configured text model.
    """
    client, _ = media_from_config(
        {"base_url": "https://x/v1", "api_key": "k", "model": "DeepSeek-V4.1-Flash"})
    assert client is not None
    assert client.image_model == DEFAULT_IMAGE_MODEL
    assert client.video_model == DEFAULT_VIDEO_MODEL
    assert client.image_model != "DeepSeek-V4.1-Flash"


def test_the_size_separator_is_not_uniform():
    """Images use 'x', video uses '*'. Both spellings are offered so a caller
    that copies one into the other has a documented place to notice."""
    assert all("x" in s and "*" not in s for s in IMAGE_SIZES)
    assert all("*" in s and "x" not in s for s in VIDEO_SIZES)


def test_an_unconfigured_client_refuses_before_it_calls_anything():
    client = MediaClient(base_url="", api_key="k")
    assert not client
    with pytest.raises(MediaError) as caught:
        client.generate_image("a circle")
    assert "media.base_url" in str(caught.value)


# ----------------------------------------------------------------------
# images: synchronous
# ----------------------------------------------------------------------


def test_generate_image_writes_the_bytes_and_reports_the_path(tmp_path):
    with ScriptedEndpoint() as ep:
        ep.script("POST", "/v1/images/generations",
                  (200, json.dumps({"model": "Qwen-Image",
                                    "data": [{"url": f"http://127.0.0.1:"
                                                     f"{ep.port}/a.png"}]}), "application/json"))
        ep.script("GET", "/a.png", (200, PNG, "image/png"))
        made = client_for(ep, out_dir=tmp_path).generate_image("a red circle")

    assert isinstance(made, Generation)
    assert made.kind == "image"
    assert made.bytes == len(PNG)
    assert open(made.path, "rb").read() == PNG
    assert made.model == "Qwen-Image"


def test_an_empty_prompt_costs_no_request(tmp_path):
    """Refused here, because the far side charges for the round trip."""
    with ScriptedEndpoint() as ep:
        ep.script("POST", "/v1/images/generations",
                  (200, json.dumps({"data": [{"url": "http://x/a.png"}]}), "application/json"))
        with pytest.raises(MediaError):
            client_for(ep, out_dir=tmp_path).generate_image("   ")
        assert ep.requests == []


def test_a_refused_image_carries_the_gateways_own_message(tmp_path):
    """The allowlist in the message is the useful part; it must survive."""
    with ScriptedEndpoint() as ep:
        ep.script("POST", "/v1/images/generations",
                  (422, json.dumps({"detail": {
                      "message": "__bogus__ not in ['Qwen-Image', 'Kolors'], "
                                 "please input valid image model name."}}),
                   "application/json"))
        with pytest.raises(MediaError) as caught:
            client_for(ep, out_dir=tmp_path).generate_image("x", model="__bogus__")
    message = str(caught.value)
    assert "422" in message
    assert "Qwen-Image" in message


def test_a_success_that_carries_no_url_is_an_error(tmp_path):
    """HTTP 200 with an empty data list reads as success and is not one."""
    with ScriptedEndpoint() as ep:
        ep.script("POST", "/v1/images/generations",
                  (200, json.dumps({"data": []}), "application/json"))
        with pytest.raises(MediaError) as caught:
            client_for(ep, out_dir=tmp_path).generate_image("x")
    assert "no URL" in str(caught.value)


def test_a_failed_download_leaves_no_file(tmp_path):
    """A truncated artifact must not be left behind looking finished."""
    with ScriptedEndpoint() as ep:
        ep.script("POST", "/v1/images/generations",
                  (200, json.dumps({"data": [{"url": f"http://127.0.0.1:"
                                                     f"{ep.port}/missing.png"}]}), "application/json"))
        ep.script("GET", "/missing.png", (404, b"gone", "text/plain"))
        with pytest.raises(MediaError):
            client_for(ep, out_dir=tmp_path).generate_image("a red circle")
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------
# video: a task, and the failure that only shows up on the poll
# ----------------------------------------------------------------------


def test_generate_video_polls_until_completed(tmp_path):
    with ScriptedEndpoint() as ep:
        ep.script("POST", "/v1/videos",
                  (200, json.dumps({"id": "t-1", "status": "queued",
                                    "model": "Wan2.6-T2V"}), "application/json"))
        ep.script("GET", "/v1/videos/t-1",
                  (200, json.dumps({"id": "t-1", "status": "in_progress"}), "application/json"),
                  (200, json.dumps({"id": "t-1", "status": "in_progress"}), "application/json"),
                  (200, json.dumps({"id": "t-1", "status": "completed",
                                    "video_url": f"http://127.0.0.1:{ep.port}/v.mp4",
                                    "usage": {"seconds": 5.0}}), "application/json"))
        ep.script("GET", "/v.mp4", (200, MP4, "video/mp4"))
        made = client_for(ep, out_dir=tmp_path).generate_video("a red circle, static")

    assert made.kind == "video"
    assert made.task_id == "t-1"
    assert made.seconds == 5.0
    assert open(made.path, "rb").read() == MP4
    polls = [r for r in ep.requests if r["path"] == "/v1/videos/t-1"]
    assert len(polls) == 3, "the loop must poll past the non-terminal statuses"


def test_a_video_task_that_fails_raises_with_the_servers_reason(tmp_path):
    """The defect this module exists to not have.

    The create call answers 200. The refusal arrives on the poll. Reporting the
    create's status code would call this a success.

    `video_timeout` is deliberately short. The point of this test is that the
    *server's* reason comes out; if the failure branch were ever dropped, the
    loop would spin until the timeout and the test would have to wait 15
    minutes to notice. With a 5-second ceiling it fails in seconds, and the
    assertion on the message is what distinguishes "raised the reason" from
    "gave up".
    """
    with ScriptedEndpoint() as ep:
        ep.script("POST", "/v1/videos",
                  (200, json.dumps({"id": "t-2", "status": "queued"}), "application/json"))
        ep.script("GET", "/v1/videos/t-2",
                  (200, json.dumps({"id": "t-2", "status": "failed",
                                    "error": "prompt must contain words"}),
                   "application/json"))
        with pytest.raises(MediaError) as caught:
            client_for(ep, out_dir=tmp_path,
                       video_timeout=5.0).generate_video("x")
    assert "prompt must contain words" in str(caught.value)
    assert list(tmp_path.iterdir()) == [], "a failed task writes no file"


def test_a_video_task_that_never_finishes_times_out(tmp_path):
    """A job stuck in `queued` must end the wait, not hang the turn."""
    with ScriptedEndpoint() as ep:
        ep.script("POST", "/v1/videos",
                  (200, json.dumps({"id": "t-3", "status": "queued"}), "application/json"))
        ep.script("GET", "/v1/videos/t-3",
                  (200, json.dumps({"id": "t-3", "status": "queued"}), "application/json"))
        with pytest.raises(MediaError) as caught:
            client_for(ep, out_dir=tmp_path,
                       video_timeout=0.05).generate_video("x")
    assert "t-3" in str(caught.value)


def test_a_create_that_returns_no_task_id_is_an_error(tmp_path):
    with ScriptedEndpoint() as ep:
        ep.script("POST", "/v1/videos",
                  (200, json.dumps({"status": "queued"}), "application/json"))
        with pytest.raises(MediaError) as caught:
            client_for(ep, out_dir=tmp_path).generate_video("x")
    assert "no task id" in str(caught.value)


def test_video_task_asks_without_waiting(tmp_path):
    """A task id from an earlier turn should be re-readable, not re-run."""
    with ScriptedEndpoint() as ep:
        ep.script("GET", "/v1/videos/t-9",
                  (200, json.dumps({"id": "t-9", "status": "in_progress"}),
                   "application/json"))
        seen = client_for(ep, out_dir=tmp_path).video_task("t-9")
    assert seen["status"] == "in_progress"


def test_a_progress_callback_that_raises_does_not_fail_the_job(tmp_path):
    """A printing callback is a convenience; it must not be able to kill a job
    that is already paid for."""
    with ScriptedEndpoint() as ep:
        ep.script("POST", "/v1/videos",
                  (200, json.dumps({"id": "t-4", "status": "queued"}), "application/json"))
        ep.script("GET", "/v1/videos/t-4",
                  (200, json.dumps({"id": "t-4", "status": "in_progress"}), "application/json"),
                  (200, json.dumps({"id": "t-4", "status": "completed",
                                    "video_url": f"http://127.0.0.1:{ep.port}/v.mp4"}),
                   "application/json"))
        ep.script("GET", "/v.mp4", (200, MP4, "video/mp4"))

        def angry(_elapsed: float) -> None:
            raise RuntimeError("the progress printer is broken")

        made = client_for(ep, out_dir=tmp_path).generate_video(
            "x", on_wait=angry)
    assert made.kind == "video"


def test_a_generation_reports_enough_to_act_on():
    made = Generation(kind="image", path="/tmp/a.png", bytes=10, model="M",
                      provider="-")
    text = made.report()
    assert "/tmp/a.png" in text and "10 bytes" in text
    assert "via -" not in text, "an unreported provider is not a provider"


# ----------------------------------------------------------------------
# wired, not merely available
# ----------------------------------------------------------------------


def test_the_media_tools_reach_a_real_agent():
    """The framework's own complaint: a capability that nothing consults.

    That `media.py` is correct and tested is not the claim being made here. The
    claim is that a `ForgeAgent` actually offers `draw`, so the loop can reach
    it. If the `_tool_media()` line is dropped from `_register_meta_tools`,
    every other test in this file still passes and this one fails -- which is
    the same shape as `tests/test_compaction_wiring.py`, and for the same
    reason.
    """
    from autoforge.agent import ForgeAgent
    from autoforge.core.llm import LLMResponse, MockLLMClient

    agent = ForgeAgent(llm=MockLLMClient(script=[LLMResponse(content="ok")]))
    for name in ("draw", "animate", "media_status"):
        assert agent.registry.get(name) is not None, f"{name} is not reachable"


def test_a_media_tool_explains_itself_when_nothing_is_configured():
    """An unconfigured endpoint must produce a sentence, not a traceback.

    The tool stays registered whether or not media is configured. Removing it
    would make the failure read "no such tool", which names the wrong problem:
    the agent can draw, it just has nowhere to send the request.
    """
    from autoforge.agent import ForgeAgent
    from autoforge.core.llm import LLMResponse, MockLLMClient

    agent = ForgeAgent(llm=MockLLMClient(script=[LLMResponse(content="ok")]))
    agent.media = None
    agent._media_problem = "media is disabled in the config"

    said = agent.registry.get("draw").fn(prompt="a red circle")
    assert "media is disabled in the config" in said
    assert "media.base_url" in said

    status = agent.registry.get("media_status").fn()
    assert "media is disabled in the config" in status

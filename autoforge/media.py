"""Making a picture or a clip, instead of only reading one.

`vision.py` is the half of multimodality that turns pixels into words. This is
the other half: words into pixels. The two are deliberately separate modules
because they fail for different reasons and are configured by different model
ids -- a gateway that can see is not the same gateway that can draw, and
inheriting one id for the other produces a failure that blames the prompt.

Two endpoints, because the providers do not agree on one:

    an image is synchronous. POST, wait, get a URL back.
    a video is a task. POST, get an id, poll the id until it finishes.

The video side is worth being explicit about, because it breaks an assumption
the rest of the framework makes: **a rejected video request is not an HTTP
error.** The create call answers 200 and hands back a task id even for a prompt
the model will refuse; the refusal only appears later, on the poll, as
`{"status": "failed", "error": "prompt must contain words"}`. So a client that
trusts the status code of the create call reports success for a job that never
ran. This module treats a task as unfinished until a poll says otherwise, and
carries the server's own `error` string out rather than inventing one.

**What this cannot do, and the loop that partly fixes it.** The forge battery
verifies a tool by running it and checking the effect. That works for a parser
and does not work here: there is no assertion that says an image is the image
the caller meant. Generated output is only *checkable by looking*. That is not
a gap this module can close on its own, and pretending otherwise would be the
decorative-oracle mistake -- a check that always passes is worse than no check.
What it can do is hand the file back with its path, so the natural partner is
`see`: draw, then look at what was drawn, then say whether it is blank, or
clipped, or the wrong subject. `draw` and `see` compose into the only closed
loop available for generative output, and neither is worth much alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .configfile import PROXIES

__all__ = [
    "DEFAULT_IMAGE_MODEL", "DEFAULT_VIDEO_MODEL", "IMAGE_SIZES", "VIDEO_SIZES",
    "Generation", "MediaClient", "MediaError", "media_from_config",
]

#: Proven against the gateway this was written for. A model id that does not
#: exist is rejected by name, so the default only has to be *a* real one -- the
#: caller who cares passes their own.
DEFAULT_IMAGE_MODEL = "Qwen-Image"
DEFAULT_VIDEO_MODEL = "Wan2.6-T2V"

#: The separator is not a typo and is not uniform: the image endpoint spells a
#: size `1024x1024` and the video endpoint spells the same idea `1280*720`.
#: Sending the wrong one is refused, so the two spellings live here together
#: where the difference is visible.
IMAGE_SIZES = ("1024x1024", "768x768", "1280x720", "720x1280")
VIDEO_SIZES = ("1280*720", "720*1280", "1920*1080", "1080*1920")

#: Statuses a task passes through. `queued` and `in_progress` are both "not
#: yet"; only the two terminal ones stop the poll.
RUNNING_STATUSES = frozenset({"queued", "in_progress", "pending", "running",
                              "processing", "submitted"})
DONE_STATUSES = frozenset({"completed", "succeeded", "success", "done"})
FAILED_STATUSES = frozenset({"failed", "error", "cancelled", "canceled"})

#: How long to wait between polls. Video jobs took ~35s end to end against the
#: gateway this was written for; a tight loop would spend more on requests than
#: the job spends on compute, and a loose one makes a 40-second wait feel like
#: a hang.
POLL_SECONDS = 5.0


class MediaError(RuntimeError):
    """No endpoint, a refused request, or a task that failed on the far side."""


@dataclass
class Generation:
    """One finished artifact, and enough about it to be worth reporting.

    `path` is the point of the object: a URL expires and a remote URL is not
    something a later tool can open. The file is written before this is
    returned, so a caller that gets a Generation has bytes on disk.
    """

    kind: str                 # "image" | "video"
    path: str
    url: str = ""
    model: str = ""
    bytes: int = 0
    seconds: float = 0.0      # video only: what the provider says it rendered
    provider: str = ""
    task_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def report(self) -> str:
        bits = [f"{self.kind} written to {self.path}"]
        if self.bytes:
            bits.append(f"{self.bytes} bytes")
        if self.model:
            bits.append(f"model {self.model}")
        if self.seconds:
            bits.append(f"{self.seconds:g}s")
        if self.provider and self.provider != "-":
            bits.append(f"via {self.provider}")
        return ", ".join(bits)


def _slug(text: str, limit: int = 40) -> str:
    """A filename that says what it is without saying too much.

    The prompt goes in the name because a directory of `image_1.png` is
    useless to a person and to the agent on the next turn. It is truncated and
    stripped of separators so a prompt containing a slash cannot escape the
    output directory.
    """
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", text.strip())
    cleaned = cleaned.strip("-")[:limit].strip("-")
    return cleaned or "prompt"


def _default_out_dir() -> Path:
    """Where artifacts go when the caller does not say.

    Next to the rest of the agent's state rather than the working directory: a
    generated file that lands in whatever directory the agent happened to be
    started from is a file nobody finds again.
    """
    home = os.environ.get("AUTOFORGE_HOME") or os.path.join(
        os.path.expanduser("~"), ".autoforge")
    return Path(home) / "media"


def _unique(path: Path) -> Path:
    """A path that does not already exist.

    Overwriting is the wrong default for generated output: the two runs are not
    the same bytes, so silently replacing one with the other loses the thing the
    caller may have been comparing against.
    """
    if not path.exists():
        return path
    stem, suffix, n = path.stem, path.suffix, 2
    while True:
        candidate = path.with_name(f"{stem}-{n}{suffix}")
        if not candidate.exists():
            return candidate
        n += 1


class MediaClient:
    """An OpenAI-compatible gateway, asked to make something.

    `out_dir` is where finished files land. `on_wait` is called on each poll of
    a video task with the seconds elapsed so far -- the caller that wants to
    print a progress line gets one, and the caller that does not passes nothing
    and pays nothing.
    """

    def __init__(self, *, image_model: str = DEFAULT_IMAGE_MODEL,
                 video_model: str = DEFAULT_VIDEO_MODEL,
                 base_url: str, api_key: str = "",
                 timeout: float = 300.0, video_timeout: float = 900.0,
                 poll_seconds: float = POLL_SECONDS,
                 out_dir: str | os.PathLike[str] | None = None,
                 proxies: dict[str, str] | None = None) -> None:
        self.image_model = image_model
        self.video_model = video_model
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.timeout = timeout
        self.video_timeout = video_timeout
        self.poll_seconds = max(0.5, float(poll_seconds))
        self.out_dir = Path(out_dir) if out_dir else _default_out_dir()
        self.proxies = proxies

    def __bool__(self) -> bool:
        return bool(self.base_url and (self.image_model or self.video_model))

    # -- transport -----------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: dict[str, Any],
              timeout: float | None = None) -> dict[str, Any]:
        import requests

        url = f"{self.base_url}{path}"
        try:
            resp = requests.post(url, headers=self._headers(), json=payload,
                                 timeout=timeout or self.timeout,
                                 proxies=self.proxies)
        except requests.RequestException as exc:
            raise MediaError(
                f"could not reach {self.base_url}: {type(exc).__name__}: {exc}"
            ) from exc
        return self._read(resp, url, payload.get("model", ""))

    def _get(self, path: str, timeout: float | None = None) -> dict[str, Any]:
        import requests

        url = f"{self.base_url}{path}"
        try:
            resp = requests.get(url, headers=self._headers(),
                                timeout=timeout or self.timeout,
                                proxies=self.proxies)
        except requests.RequestException as exc:
            raise MediaError(
                f"could not reach {self.base_url}: {type(exc).__name__}: {exc}"
            ) from exc
        return self._read(resp, url, "")

    @staticmethod
    def _read(resp: Any, url: str, model: str) -> dict[str, Any]:
        """A JSON body, or an error that names the endpoint that failed.

        The gateway's own validation message is far more useful than a status
        code -- it lists the model ids it would have accepted -- so it is
        carried through verbatim rather than flattened into "HTTP 422".
        """
        try:
            body = resp.json()
        except ValueError:
            raise MediaError(
                f"{url} returned a non-JSON body (HTTP {resp.status_code}): "
                f"{(resp.text or '')[:200]!r}") from None
        if resp.status_code != 200:
            detail = body
            if isinstance(body, dict):
                detail = body.get("detail") or body.get("error") or body
                if isinstance(detail, dict):
                    detail = detail.get("message", detail)
            raise MediaError(
                f"{url} answered HTTP {resp.status_code}"
                + (f" for model {model!r}" if model else "")
                + f": {str(detail)[:400]}")
        if not isinstance(body, dict):
            raise MediaError(f"{url} returned {type(body).__name__}, not an object")
        return body

    def _download(self, url: str, dest: Path, timeout: float) -> int:
        """Fetch the artifact and write it. Returns the byte count.

        Written through a temporary name and renamed, so a download that dies
        halfway does not leave a truncated file that later looks like a
        finished one.
        """
        import requests

        if not url:
            raise MediaError("the gateway reported success but returned no URL")
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name + ".part")
        try:
            resp = requests.get(url, timeout=timeout, proxies=self.proxies,
                                stream=True)
            if resp.status_code != 200:
                raise MediaError(
                    f"could not download the artifact: HTTP {resp.status_code} "
                    f"from {url[:120]}")
            written = 0
            with open(temp, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)
        except requests.RequestException as exc:
            raise MediaError(
                f"could not download the artifact: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if temp.exists() and temp.stat().st_size == 0:
                temp.unlink(missing_ok=True)
        if not temp.exists():
            raise MediaError("the download produced no bytes")
        os.replace(temp, dest)
        return written

    def _destination(self, kind: str, model: str, prompt: str,
                     save: str | os.PathLike[str] | None) -> Path:
        if save:
            return Path(save)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        suffix = ".png" if kind == "image" else ".mp4"
        return _unique(self.out_dir / f"{kind}-{_slug(prompt)}-{digest}{suffix}")

    # -- the two verbs -------------------------------------------------
    def generate_image(self, prompt: str, *, model: str = "", size: str = "",
                       save: str | os.PathLike[str] | None = None) -> Generation:
        """A picture, synchronously. Returns once the bytes are on disk."""
        if not self:
            raise MediaError(self._unconfigured())
        if not prompt.strip():
            raise MediaError("a prompt is required; an empty one is refused by "
                             "the model after the request has been paid for")
        model = model or self.image_model
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "n": 1}
        if size:
            payload["size"] = size
        body = self._post("/images/generations", payload)

        data = body.get("data")
        url = ""
        if isinstance(data, list) and data and isinstance(data[0], dict):
            url = str(data[0].get("url") or data[0].get("b64_json") or "")
        if not url:
            raise MediaError(
                f"the image endpoint returned no URL for {model!r}: "
                f"{json.dumps(body, ensure_ascii=False)[:300]}")
        dest = self._destination("image", model, prompt, save)
        written = self._download(url, dest, self.timeout)
        return Generation(
            kind="image", path=str(dest), url=url,
            model=str(body.get("model") or model),
            bytes=written,
            provider=str(body.get("provider") or ""),
            detail={"usage": body.get("usage"), "size": payload.get("size")},
        )

    def generate_video(self, prompt: str, *, model: str = "", size: str = "",
                       save: str | os.PathLike[str] | None = None,
                       on_wait: Callable[[float], None] | None = None) -> Generation:
        """A clip, asynchronously: create a task, poll it, fetch the result.

        The create call answering 200 means *accepted*, not *will succeed* --
        see the module docstring. Nothing here returns a file until a poll has
        said `completed`, and a `failed` poll raises with the server's own
        reason rather than a generic timeout.
        """
        if not self:
            raise MediaError(self._unconfigured())
        if not prompt.strip():
            raise MediaError("a prompt is required; the task is created anyway "
                             "and fails later, which wastes the round trip")
        model = model or self.video_model
        payload: dict[str, Any] = {"model": model, "prompt": prompt}
        if size:
            payload["size"] = size
        body = self._post("/videos", payload)
        task_id = str(body.get("id") or "")
        if not task_id:
            raise MediaError(
                f"the video endpoint returned no task id for {model!r}: "
                f"{json.dumps(body, ensure_ascii=False)[:300]}")

        started = time.monotonic()
        last = body
        while True:
            elapsed = time.monotonic() - started
            status = str(last.get("status") or "")
            if status in DONE_STATUSES:
                break
            if status in FAILED_STATUSES:
                raise MediaError(
                    f"the video task failed after {elapsed:.0f}s: "
                    f"{last.get('error') or status}")
            if elapsed > self.video_timeout:
                raise MediaError(
                    f"the video task was still {status or 'unreported'} after "
                    f"{elapsed:.0f}s (task {task_id}); it may still finish, but "
                    f"nothing here is waiting for it")
            if on_wait is not None:
                try:
                    on_wait(elapsed)
                except Exception:
                    pass          # a progress printer must not fail the job
            time.sleep(self.poll_seconds)
            last = self._get(f"/videos/{task_id}", timeout=self.timeout)

        url = str(last.get("video_url") or last.get("url") or "")
        dest = self._destination("video", model, prompt, save)
        written = self._download(url, dest, self.video_timeout)
        usage = last.get("usage") or {}
        return Generation(
            kind="video", path=str(dest), url=url,
            model=str(last.get("model") or model),
            bytes=written,
            seconds=float(usage.get("seconds") or 0.0),
            provider=str(last.get("provider") or ""),
            task_id=task_id,
            detail={"usage": usage, "size": last.get("size"),
                    "elapsed": round(time.monotonic() - started, 1)},
        )

    def video_task(self, task_id: str) -> dict[str, Any]:
        """What the far side says about a task, without waiting for it.

        Exposed because a caller that has a task id from a previous turn should
        be able to ask about it again -- the alternative is re-generating work
        that is already done.
        """
        if not task_id:
            raise MediaError("a task id is required")
        return self._get(f"/videos/{task_id}")

    def _unconfigured(self) -> str:
        return ("no media endpoint is configured. Set media.base_url in the "
                "config file (or AUTOFORGE_MEDIA_BASE_URL).")

    def report(self) -> str:
        if not self:
            return ("media: not configured -- set media.base_url to let the "
                    "agent draw or render")
        return (f"media: images via {self.image_model}, video via "
                f"{self.video_model}, at {self.base_url}"
                + ("" if self.api_key else " (no key set)")
                + f", output to {self.out_dir}")


def media_from_config(cfg: dict[str, Any]) -> tuple[MediaClient | None, str]:
    """Build the media client the config describes. Returns (client, problem).

    The endpoint falls back to the main provider for the same reason vision
    does -- one gateway usually serves everything -- but the *model ids* do not
    fall back to the text model. A text model id sent to the image endpoint is
    refused by name, and the refusal lists the ids that would have worked.

    An unset section is silent, not a problem: a setup that never generates
    anything should not be told about a feature it does not use.
    """
    media = cfg.get("media") or {}
    if not isinstance(media, dict):
        return None, "the `media` section must be a table/object"
    if not media.get("enabled", True):
        return None, "media is disabled in the config"

    base_url = (media.get("base_url") or os.environ.get("AUTOFORGE_MEDIA_BASE_URL")
                or cfg.get("base_url") or "")
    image_model = (media.get("image_model")
                   or os.environ.get("AUTOFORGE_MEDIA_IMAGE_MODEL")
                   or DEFAULT_IMAGE_MODEL)
    video_model = (media.get("video_model")
                   or os.environ.get("AUTOFORGE_MEDIA_VIDEO_MODEL")
                   or DEFAULT_VIDEO_MODEL)
    if not base_url and not (image_model or video_model):
        return None, ""

    if not base_url:
        return None, ("media is configured with model ids but there is no "
                      "media.base_url and no main base_url to fall back to")

    api_key = (media.get("key") or media.get("api_key")
               or os.environ.get("AUTOFORGE_MEDIA_API_KEY")
               or cfg.get("api_key") or cfg.get("key") or "")

    # Same relay logic as vision: `media.proxies` for an explicit relay,
    # otherwise the same on/off switch the text client uses.
    proxies = media.get("proxies") or cfg.get("proxies")
    if proxies is None and cfg.get("proxy"):
        proxies = dict(PROXIES)

    out_dir = (media.get("out_dir")
               or os.environ.get("AUTOFORGE_MEDIA_OUT_DIR"))

    return MediaClient(
        image_model=str(image_model),
        video_model=str(video_model),
        base_url=base_url,
        api_key=api_key,
        timeout=float(media.get("timeout", 300.0)),
        video_timeout=float(media.get("video_timeout", 900.0)),
        poll_seconds=float(media.get("poll_seconds", POLL_SECONDS)),
        out_dir=out_dir,
        proxies=proxies,
    ), ""

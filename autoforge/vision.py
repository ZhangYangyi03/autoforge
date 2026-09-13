"""Looking at a picture instead of being told what is in it.

Every other tool here reduces the world to text, which means anything that is
only visible -- a chart's shape, a screenshot's layout, whether the render is
blank -- is invisible to the agent unless a person describes it. This module
sends the bytes to a vision-capable model and comes back with words.

The media type is sniffed from the bytes, never taken from the file name. An
extension is a claim about the contents, and a `.png` that is really a JPEG is
common enough that believing it produces a confusing rejection from the far
end. What the bytes say is what gets sent.

The endpoint is configured rather than assumed, because vision models are
separate from text models on most gateways and the ids differ. When it is not
configured the error says exactly which key to set, since the alternative is a
404 from a URL the caller never chose.
"""

from __future__ import annotations

import base64
import os
from typing import Any

from .configfile import PROXIES

__all__ = [
    "DEFAULT_QUESTION", "IMAGE_LIMIT", "VisionClient", "VisionError",
    "load_image", "sniff_media_type", "vision_from_config",
]

#: A model will not accept an unbounded upload, and base64 inflates by a third.
#: 8 MiB of source image is already a 4000x4000 photograph.
IMAGE_LIMIT = 8 * 1024 * 1024

DEFAULT_QUESTION = (
    "Describe what is in this image. If it is a screenshot of an interface, "
    "say what the interface is, what state it is in, and anything that looks "
    "like an error, an empty result, or a control that did not render."
)

#: Magic bytes, longest first: a prefix that is a prefix of a longer one must
#: not win. WEBP is checked by its RIFF container below rather than here.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


class VisionError(RuntimeError):
    """No endpoint, no image, or an answer that is not an answer."""


def sniff_media_type(data: bytes) -> str | None:
    """The media type the bytes actually are, or None.

    Returns None rather than guessing: a wrong type is rejected by the far end
    with a message about the model, which sends the reader looking in the wrong
    place.
    """
    for signature, media_type in _SIGNATURES:
        if data.startswith(signature):
            return media_type
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def load_image(source: str, *, limit: int = IMAGE_LIMIT) -> tuple[str, int, str]:
    """A data URL for an image the caller named.

    Returns (data_url, byte_length, media_type). An http(s) URL is passed
    through as-is: the far end can fetch it, and re-uploading it would mean
    this process downloading bytes it has no reason to hold.
    """
    if not source:
        raise VisionError("no image was given")

    if source.startswith(("http://", "https://", "data:")):
        return source, 0, ""

    if not os.path.isfile(source):
        raise VisionError(f"no such image: {source}")
    try:
        size = os.path.getsize(source)
    except OSError as exc:
        raise VisionError(f"could not stat {source}: {exc}") from exc
    if size > limit:
        raise VisionError(
            f"{source} is {size / 1048576:.1f} MiB, over the {limit / 1048576:.0f} "
            f"MiB limit; crop or downscale it first")

    try:
        with open(source, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise VisionError(f"could not read {source}: {exc}") from exc
    if not data:
        raise VisionError(f"{source} is empty")

    media_type = sniff_media_type(data)
    if media_type is None:
        head = data[:12]
        raise VisionError(
            f"{source} is not an image this can identify "
            f"(first bytes {head!r}); expected PNG, JPEG, GIF, BMP or WebP")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{encoded}", len(data), media_type


class VisionClient:
    """An OpenAI-compatible chat endpoint, asked about an image."""

    def __init__(self, *, model: str, base_url: str, api_key: str = "",
                 timeout: float = 120.0, name: str = "vision",
                 max_tokens: int | None = 1024,
                 proxies: dict[str, str] | None = None) -> None:
        self.model = model
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.timeout = timeout
        self.name = name
        self.max_tokens = max_tokens
        self.proxies = proxies

    def __bool__(self) -> bool:
        return bool(self.model and self.base_url)

    def describe(self, source: str, question: str = "",
                 *, limit: int = IMAGE_LIMIT) -> str:
        import requests

        if not self:
            raise VisionError(
                "no vision endpoint is configured. Set vision.model and "
                "vision.base_url in the config file (or AUTOFORGE_VISION_MODEL "
                "and AUTOFORGE_VISION_BASE_URL).")
        data_url, size, media_type = load_image(source, limit=limit)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question or DEFAULT_QUESTION},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.post(f"{self.base_url}/chat/completions",
                                 headers=headers, json=payload,
                                 timeout=self.timeout, proxies=self.proxies)
        except requests.RequestException as exc:
            raise VisionError(
                f"could not reach {self.base_url}: {type(exc).__name__}: {exc}"
            ) from exc

        if resp.status_code != 200:
            detail = (resp.text or "")[:300].strip()
            raise VisionError(
                f"{self.name} answered HTTP {resp.status_code} for model "
                f"{self.model!r}: {detail}")

        try:
            body = resp.json()
        except ValueError as exc:
            raise VisionError(
                f"{self.name} returned a non-JSON body (HTTP {resp.status_code}): "
                f"{(resp.text or '')[:200]!r}") from exc

        return _read_answer(body, self.name, self.model)

    def report(self) -> str:
        if not self:
            return ("vision: not configured -- set vision.model and "
                    "vision.base_url to let the agent look at images")
        return (f"vision: {self.model} at {self.base_url}"
                + ("" if self.api_key else " (no key set)"))


def _read_answer(body: Any, name: str, model: str) -> str:
    """Pull the text out of a completion, saying which shape was wrong."""
    if not isinstance(body, dict):
        raise VisionError(f"{name} returned {type(body).__name__}, not an object")

    if body.get("error"):
        error = body["error"]
        message = error.get("message", error) if isinstance(error, dict) else error
        raise VisionError(f"{name} reported an error for {model!r}: {message}")

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VisionError(
            f"{name} returned no choices for {model!r}: "
            f"{str(body)[:200]}")

    message = choices[0].get("message") or {}
    content = message.get("content")

    # Some gateways answer with a list of content parts instead of a string.
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content
                       if isinstance(part, dict) and part.get("type") == "text")
    elif isinstance(content, str):
        text = content
    else:
        text = ""

    if not text.strip():
        finish = choices[0].get("finish_reason")
        raise VisionError(
            f"{name} returned an empty answer for {model!r}"
            + (f" (finish_reason: {finish})" if finish else "")
            + ". A reasoning model that spends its whole budget thinking "
              "does this; raise max_tokens or use a model that answers.")
    return text.strip()


def vision_from_config(cfg: dict[str, Any]) -> tuple[VisionClient | None, str]:
    """Build the vision client the config describes.

    Returns (client, problem).

    The *endpoint* falls back to the main provider, because on a single-provider
    setup the same gateway usually serves both. The *model* deliberately does
    not: a vision model is a different id from the text model on essentially
    every provider, so inheriting `gpt-4o` for text would send an image to a
    model that cannot see, and the failure that comes back blames the image.
    Naming the model is what turns vision on.
    """
    vision = cfg.get("vision") or {}
    if not isinstance(vision, dict):
        return None, "the `vision` section must be a table/object"

    model = (vision.get("model") or os.environ.get("AUTOFORGE_VISION_MODEL") or "")
    base_url = (vision.get("base_url") or os.environ.get("AUTOFORGE_VISION_BASE_URL")
                or cfg.get("base_url") or "")
    api_key = (vision.get("key") or vision.get("api_key")
               or os.environ.get("AUTOFORGE_VISION_API_KEY")
               or cfg.get("key") or "")

    if not vision.get("enabled", True):
        return None, "vision is disabled in the config"
    if not model:
        # Silence here is the right answer: on a setup that never looks at
        # images, an unset vision model is not a problem to report.
        return None, ""

    if not base_url:
        return None, ("vision.model is set but there is no vision.base_url and "
                      "no main base_url to fall back to")

    # `vision.proxies` for an explicit relay; otherwise the same on/off switch
    # the text client uses. Without this, vision silently goes direct while the
    # text side is proxied -- the calls that fail are the ones with no visible
    # cause, because the page loads fine everywhere else.
    proxies = vision.get("proxies") or cfg.get("proxies")
    if proxies is None and cfg.get("proxy"):
        proxies = dict(PROXIES)

    return VisionClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=float(vision.get("timeout", 120.0)),
        max_tokens=vision.get("max_tokens", 1024),
        proxies=proxies,
    ), ""

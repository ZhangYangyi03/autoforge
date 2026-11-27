"""Searching the Earth's surface by words, by example, or by coordinate.

The other tools here read text and pictures. This one reads a *place*. It
talks to EarthEmbeddingExplorer, the cross-modal satellite retrieval service
built on the ESA MajorTOM Core-S2L2A collection: the whole land surface of the
planet as Sentinel-2 imagery, ~249k sampled tiles, each encoded into an
embedding by one of several open encoders. A query -- a phrase, a picture, or a
lat/lon pair -- becomes a vector, and the nearest tiles come back as
coordinates, product ids and scores.

Why it is worth carrying rather than doing by hand: the numbers a remote-sensing
task needs (what land cover is at this coordinate, where in the world does this
look like, which tiles resemble this photo) are one HTTP round trip, where the
alternative is downloading terabytes of imagery or training a model.

Two facts about the far end, measured on 2026-11-21 and encoded here:

  * The service is a Gradio app, so the transport is Gradio's own call protocol
    -- POST /gradio_api/call/<api>, then stream
    GET /gradio_api/call/<api>/<event_id> as server-sent events. There is no
    plain JSON endpoint to fall back to.
  * Not every model serves every query. Text-to-image works with the
    text-image encoders (SigLIP, FarSLIP); image-to-image works with all four;
    location-to-image is model-free. Asking SigLIP to do a location search is
    not an error on the client side -- it is a request the far end has no
    handler for, so the model is validated here rather than sent hopefully.

The returned image URLs are rewritten, not used as given. Gradio publishes them
as `<base>/gradio/gradio_api/file=<path>`, which now serves an HTML page rather
than the bytes; the same path under `<base>/gradio_api/file=<path>` serves the
image. Guessing which one a deployment publishes is how a saved thumbnail turns
out to be a landing page, so the path is rebuilt from the file path.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import uuid
from typing import Any

__all__ = [
    "DEFAULT_BASE_URL", "TEXT_MODELS", "IMAGE_MODELS", "EarthExplorerClient",
    "EarthExplorerError", "explorer_from_config", "parse_sse",
]

#: The Hugging Face Space, which is free and needs no account. The ModelScope
#: studio at Major-TOM/EarthEmbeddingExplorer carries all eight models instead
#: of four, but gates its GPU queue behind a login, so it is only useful with a
#: session cookie -- configure it explicitly if you have one.
DEFAULT_BASE_URL = "https://ml4sustain-earthexplorer.hf.space"

#: The two encoders that share a text space with images. The other two encode
#: images only, so a text query against them has no meaning.
TEXT_MODELS = ("SigLIP", "FarSLIP")

#: Every model the service can search images with.
IMAGE_MODELS = ("SigLIP", "FarSLIP", "SatCLIP", "DINOv2")

#: The Modelscope deployment carries four more (TIPSv2, Qwen3-VL, Clay,
#: OlmoEarth). Named so an operator who has a session there knows the ids are
#: accepted rather than guessed.
EXTENDED_MODELS = ("TIPSv2", "Qwen3VL", "Clay", "OlmoEarth-v1_2")

def _looks_like_image(head: bytes) -> bool:
    """Does this look like an image the explorer can decode?

    Sniffed from the magic bytes rather than the extension, because the
    far end accepts any upload and answers a non-image with an empty
    result -- the same silence a genuine no-match gives. Measured
    2026-11-27 against the Hugging Face Space.
    """
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if head.startswith(b"\xff\xd8\xff"):          # jpeg
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return True
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return True
    if head.startswith(b"BM"):                    # bmp
        return True
    if head[:4] in (b"II*\x00", b"MM\x00*"):        # tiff
        return True
    return False


DEFAULT_MODEL = "FarSLIP"

#: Per-mille of the index returned. The app's own slider runs 1..30.
DEFAULT_TOP_PERMILLE = 7

#: A text query spends ~60-80 s on the free queue -- encode, similarity over
#: 249k vectors, then fetch thumbnails over HTTP Range. This is the budget for
#: the whole stream, not per byte.
DEFAULT_TIMEOUT = 300.0

#: Two shapes for the same fact, because the service sends both. The gallery
#: captions carry it as three lines ("Score / Lat, Lon / ID"); the status text
#: carries it as one comma-separated sentence ("Product ID: X, Location: (lat,
#: lon), Score: s"). Which one arrives depends on whether the visualization step
#: ran, so neither can be assumed -- the first real call after this module was
#: written returned only the second.
_CAPTION = re.compile(
    r"Score:\s*(?P<score>[-0-9.]+)\s*\\n?\s*"
    r"Lat:\s*(?P<lat>[-0-9.]+),\s*Lon:\s*(?P<lon>[-0-9.]+)\s*\\n?\s*"
    r"ID:\s*(?P<id>\S+)"
)
_SENTENCE = re.compile(
    r"Product ID:\s*(?P<id>[^,\s]+),\s*Location:\s*\(\s*"
    r"(?P<lat>[-0-9.]+)\s*,\s*(?P<lon>[-0-9.]+)\s*\),\s*Score:\s*"
    r"(?P<score>[-0-9.]+)"
)


class EarthExplorerError(RuntimeError):
    """The service was unreachable, refused, or answered with no search."""


def _rewrite_file_url(base_url: str, url: str) -> str:
    """A URL that serves the image bytes rather than Gradio's HTML wrapper."""
    if not url:
        return ""
    marker = "/file="
    if marker not in url:
        return url
    path = url.split(marker, 1)[1]
    return f"{base_url.rstrip('/')}/gradio_api/file={path}"


def parse_sse(text: str) -> list[Any]:
    """Every `data:` payload of a Gradio event stream, as decoded JSON.

    Gradio interleaves `event:` and `data:` lines and may send heartbeats; only
    the data lines carry the answer. A malformed payload is kept as its raw
    string rather than dropped -- a truncated final event is exactly what a
    timeout looks like, and hiding it would turn "cut off" into "no results".
    """
    out: list[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "null":
            continue
        try:
            out.append(json.loads(payload))
        except json.JSONDecodeError:
            out.append(payload)
    return out


def _captions(node: Any, found: list[str]) -> None:
    """Collect every gallery caption in a nested Gradio payload."""
    if isinstance(node, dict):
        caption = node.get("caption")
        if isinstance(caption, str) and "Score:" in caption:
            found.append(caption)
        for value in node.values():
            _captions(value, found)
    elif isinstance(node, list):
        for value in node:
            _captions(value, found)


class EarthExplorerClient:
    """One search service, spoken to over Gradio's call protocol."""

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL,
                 top_permille: int = DEFAULT_TOP_PERMILLE,
                 timeout: float = DEFAULT_TIMEOUT,
                 out_dir: str | None = None,
                 proxies: dict[str, str] | None = None,
                 headers: dict[str, str] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.top_permille = top_permille
        self.timeout = timeout
        self.out_dir = out_dir
        self.proxies = proxies or {}
        #: A deployment may gate its queue behind a login (ModelScope's xGPU
        #: does). The cookie that satisfies it can only come from a browser
        #: session, so it is accepted here rather than earned here.
        self.headers = headers or {}
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler(self.proxies))
        self._opener.addheaders = [
            ("User-Agent",
             self.headers.get("User-Agent")
             or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
            ("Accept", "*/*"),
            ("Referer", self.base_url + "/"),
            ("Origin", self.base_url),
        ] + [(k, v) for k, v in self.headers.items()
             if k.lower() not in ("user-agent", "accept", "referer", "origin")]

    # -- transport ---------------------------------------------------------

    def _request(self, path: str, *, data: bytes | None = None,
                 headers: dict[str, str] | None = None,
                 timeout: float | None = None) -> bytes:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, data=data, headers=headers or {})
        try:
            with self._opener.open(req, timeout=timeout or self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:400]
            except OSError:
                pass
            if exc.code == 403:
                raise EarthExplorerError(
                    f"{self.base_url} refused the request ({exc.code}): {body}. "
                    f"A deployment that gates its GPU queue behind a login "
                    f"answers exactly this way; use the public Hugging Face "
                    f"mirror, or set a session cookie in "
                    f"earthexplorer.headers.") from exc
            raise EarthExplorerError(
                f"{self.base_url}{path} -> HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EarthExplorerError(
                f"could not reach {url}: {exc}") from exc

    def _call(self, api: str, payload: list[Any],
              timeout: float | None = None) -> list[Any]:
        """One Gradio call: submit, then stream events to the end."""
        body = self._request(
            f"/gradio_api/call/{api}",
            data=json.dumps({"data": payload}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=min(60.0, timeout or self.timeout))
        try:
            event_id = json.loads(body)["event_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EarthExplorerError(
                f"{api}: the service did not return an event id "
                f"({body[:200]!r})") from exc
        stream = self._request(
            f"/gradio_api/call/{api}/{event_id}", timeout=timeout or self.timeout)
        return parse_sse(stream.decode("utf-8", "replace"))

    # -- results -----------------------------------------------------------

    @staticmethod
    def _collect(events: list[Any]) -> dict[str, Any]:
        """The useful parts of a finished stream: hits, status, images."""
        captions: list[str] = []
        status = ""
        images: list[str] = []
        error = ""
        for event in events:
            if isinstance(event, dict):
                for key in ("error", "message"):
                    if isinstance(event.get(key), str) and event[key]:
                        error = event[key]
                continue
            if not isinstance(event, list):
                continue
            for value in event:
                if isinstance(value, str) and value.strip() and not value.startswith("http"):
                    status = value
                elif isinstance(value, dict):
                    if isinstance(value.get("error"), str):
                        error = value["error"]
                    _captions(value, captions)
                    url = value.get("url")
                    if isinstance(url, str) and "file=" in url:
                        images.append(url)

        hits = []
        seen = set()
        matches = []
        for caption in captions:
            flat = caption.replace("\\n", "\n")
            match = _CAPTION.search(flat)
            if match:
                matches.append(match)
        # The status text repeats the same rows in prose. Parsed only when the
        # galleries carried nothing, so a duplicate can never double a hit.
        if not matches and status:
            matches = list(_SENTENCE.finditer(status.replace("\\n", "\n")))
        for match in matches:
            key = (match.group("id"), match.group("lat"), match.group("lon"))
            if key in seen:
                continue
            seen.add(key)
            hits.append({
                "score": float(match.group("score")),
                "lat": float(match.group("lat")),
                "lon": float(match.group("lon")),
                "product_id": match.group("id"),
            })
        return {"hits": hits, "status": status, "images": images,
                "error": error}

    def _download(self, url: str, name: str) -> str:
        target_dir = self.out_dir or os.environ.get("TEMP") or "."
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, name)
        data = self._request(
            _rewrite_file_url(self.base_url, url).replace(self.base_url, ""),
            timeout=min(120.0, self.timeout))
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def _report(self, found: dict[str, Any], query: str, model: str,
                save: str) -> str:
        hits = found["hits"]
        if not hits and found["error"]:
            raise EarthExplorerError(f"{query}: {found['error']}")
        if not hits:
            status = found["status"] or "no results and no status"
            # The hint has to match the query actually sent. Measured
            # 2026-11-27: a text file handed to image search comes back as
            # an empty stream, and the old text-encoder hint pointed the
            # reader at the wrong end of the call.
            if model in TEXT_MODELS:
                hint = f"Text queries need one of {', '.join(TEXT_MODELS)}."
            else:
                hint = (f"{model} reads images, not text: if this was an "
                        f"image search, the upload may not have decoded "
                        f"-- pass a png/jpeg/webp/tif file")
            raise EarthExplorerError(
                f"{query}: the service returned nothing ({status}). {hint}")
        lines = [f"{len(hits)} tile(s) for {query} (model {model})"]
        for hit in hits:
            lines.append(
                f"  {hit['score']:.4f}  lat {hit['lat']:8.4f}  "
                f"lon {hit['lon']:9.4f}  {hit['product_id']}")
        lines.append("  nearest first: the first row is the best match; "
                     "product_id is the Sentinel-2 granule to pull the pixels "
                     "from")
        if save and found["images"]:
            written = []
            for index, url in enumerate(found["images"][:2]):
                try:
                    written.append(self._download(url, f"{save}-{index}.webp"))
                except EarthExplorerError as exc:
                    written.append(f"(could not save image {index}: {exc})")
            lines.append("  thumbnails: " + ", ".join(written)
                         + " -- open with `see`")
        return "\n".join(lines)

    # -- the three searches ------------------------------------------------

    def search_text(self, query: str, *, model: str = "",
                    top_permille: int = 0, save: str = "") -> str:
        """Where on Earth looks like this phrase."""
        if not query.strip():
            raise EarthExplorerError("an empty query matches nothing")
        model = model or self.model
        if model not in TEXT_MODELS:
            raise EarthExplorerError(
                f"{model} does not encode text; text search works with "
                f"{', '.join(TEXT_MODELS)}. Use search_image for "
                f"{', '.join(IMAGE_MODELS)}.")
        events = self._call("search_text", [
            query, int(top_permille or self.top_permille), model,
        ])
        return self._report(self._collect(events), f"text {query!r}", model, save)

    def search_image(self, image: str, *, model: str = "",
                     top_permille: int = 0, save: str = "") -> str:
        """Where on Earth looks like this picture."""
        if not image:
            raise EarthExplorerError("no image was given")
        model = model or self.model
        if model not in IMAGE_MODELS:
            raise EarthExplorerError(
                f"{model} is not one of {', '.join(IMAGE_MODELS)}")
        far_path = self._upload(image)
        events = self._call("search_image", [
            {"path": far_path, "meta": {"_type": "gradio.FileData"}},
            int(top_permille or self.top_permille), model,
        ])
        return self._report(self._collect(events), f"image {os.path.basename(image)}",
                            model, save)

    def search_location(self, lat: float, lon: float, *,
                        top_permille: int = 0, save: str = "") -> str:
        """What the tiles near this coordinate look like -- model-free."""
        if not -90.0 <= float(lat) <= 90.0:
            raise EarthExplorerError(f"lat {lat} is outside -90..90")
        if not -180.0 <= float(lon) <= 180.0:
            raise EarthExplorerError(f"lon {lon} is outside -180..180")
        events = self._call("search_location", [
            float(lat), float(lon), int(top_permille or self.top_permille),
        ])
        return self._report(self._collect(events),
                            f"location ({lat}, {lon})", "coordinate", save)

    def _upload(self, image: str) -> str:
        """Put a local image on the service and return its remote path."""
        if image.startswith(("http://", "https://")):
            raise EarthExplorerError(
                "image search uploads a local file; download the URL first and "
                "pass the path")
        if not os.path.isfile(image):
            raise EarthExplorerError(f"no such image: {image}")
        if os.path.getsize(image) > 32 * 1024 * 1024:
            raise EarthExplorerError(f"{image} is over 32 MiB; downscale it")
        with open(image, "rb") as peek:
            head = peek.read(12)
        # Sniffed, not guessed from the extension: the far end accepts the
        # upload and then returns an empty result for anything it cannot
        # decode, which is indistinguishable from "nothing looks like this".
        if not _looks_like_image(head):
            raise EarthExplorerError(
                f"{image} is not an image the service can decode "
                f"(first bytes {head[:8]!r}); pass a png, jpeg, webp, gif, "
                f"bmp or tiff file")
        with open(image, "rb") as handle:
            blob = handle.read()
        boundary = "----autoforge" + uuid.uuid4().hex
        name = os.path.basename(image) or "upload.png"
        head = (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="files"; '
                f'filename="{name}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n").encode()
        tail = f"\r\n--{boundary}--\r\n".encode()
        body = head + blob + tail
        raw = self._request(
            "/gradio_api/upload", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=min(120.0, self.timeout))
        try:
            paths = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EarthExplorerError(
                f"upload did not return a path: {raw[:200]!r}") from exc
        if isinstance(paths, list) and paths:
            return str(paths[0])
        raise EarthExplorerError(f"upload did not return a path: {raw[:200]!r}")

    def models(self) -> str:
        """Which encoders this deployment will actually search with.

        Asked, not assumed: the ModelScope build carries four more than the
        public mirror, and a list printed from a constant would be a claim
        about the far end rather than a reading of it.
        """
        raw = self._request("/config", timeout=min(60.0, self.timeout))
        try:
            config = json.loads(raw)
        except json.JSONDecodeError:
            return f"{self.base_url}: /config did not return JSON"
        choices: list[str] = []
        for component in config.get("components", []):
            props = component.get("props") or {}
            if props.get("label") == "Model":
                for choice in props.get("choices") or []:
                    label = choice[0] if isinstance(choice, list) else choice
                    if isinstance(label, str) and label not in choices:
                        choices.append(label)
        text_space = [c for c in choices if c in TEXT_MODELS
                      or c in EXTENDED_MODELS]
        return (f"{self.base_url}: {len(choices)} encoder(s): "
                f"{', '.join(choices) or '(none advertised)'}\n"
                f"  text->image: {', '.join(c for c in choices if c in TEXT_MODELS) or '(none)'}\n"
                f"  image->image: {', '.join(choices) or '(none)'}\n"
                f"  location->image: model-free\n"
                f"  note: {len(text_space)} of these advertise a text space; a "
                f"text query against the rest has nothing to match")

    def report(self) -> str:
        return (f"earthexplorer: {self.base_url}, default model {self.model}, "
                f"top {self.top_permille} per-mille, timeout "
                f"{self.timeout:.0f}s, output to {self.out_dir or '(temp)'}")


def explorer_from_config(cfg: dict[str, Any]) -> tuple[EarthExplorerClient | None, str]:
    """Build the client the config describes. Returns (client, problem).

    An absent section still yields a client, because the public mirror is free
    and account-free -- the whole capability is one HTTP endpoint, so requiring
    configuration before the first query would gate a working feature behind a
    step that changes nothing. Set `earthexplorer.enabled` to false to turn it
    off deliberately.
    """
    section = cfg.get("earthexplorer") or cfg.get("earth_explorer") or {}
    if not isinstance(section, dict):
        return None, "the `earthexplorer` section must be a table/object"
    if not section.get("enabled", True):
        return None, "earthexplorer is disabled in the config"

    base_url = (section.get("base_url")
                or os.environ.get("AUTOFORGE_EXPLORER_BASE_URL")
                or DEFAULT_BASE_URL)
    model = (section.get("model")
             or os.environ.get("AUTOFORGE_EXPLORER_MODEL")
             or DEFAULT_MODEL)
    if model not in IMAGE_MODELS + EXTENDED_MODELS:
        return None, (f"earthexplorer.model {model!r} is not one of "
                      f"{', '.join(IMAGE_MODELS + EXTENDED_MODELS)}")

    try:
        top = int(section.get("top_permille", DEFAULT_TOP_PERMILLE))
    except (TypeError, ValueError):
        return None, "earthexplorer.top_permille must be a whole number"
    try:
        timeout = float(section.get("timeout", DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        return None, "earthexplorer.timeout must be a number of seconds"

    proxies = section.get("proxies") or cfg.get("proxies")
    if proxies is None and cfg.get("proxy"):
        from .configfile import PROXIES
        proxies = dict(PROXIES)

    return EarthExplorerClient(
        base_url=str(base_url).rstrip("/"),
        model=str(model),
        top_permille=max(1, min(30, top)),
        timeout=timeout,
        out_dir=section.get("out_dir") or os.environ.get("AUTOFORGE_EXPLORER_OUT_DIR"),
        proxies=proxies,
        headers={str(k): str(v) for k, v in (section.get("headers") or {}).items()},
    ), ""

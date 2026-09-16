"""The web: asking it a question, and reading one page of it.

An agent that can only touch what is already on the disk is answering from
whatever it happened to be trained on. Everything else in this framework is
about the agent's own artifacts — its tools, its skills, its memory. This
module is the one place it can look *outside*, and it is the difference
between an agent that knows and one that guesses confidently.

Ported from the parent framework's `tools/web_tools.py` and its `anysearch`
plugin, with the parts that were built for a different runtime left behind.
What came across is the reasoning, which is the expensive part:

1. **Head+tail, never head-only.** A truncated page that keeps only its first
   N characters loses the conclusion — the thing at the bottom. Pages over the
   budget are cut ~75% head / ~25% tail on a line boundary, and the model is
   told in a footer exactly how much it is seeing. A model that does not know
   it is looking at an excerpt will reason about the excerpt as if it were the
   whole document, and that is how a confident wrong answer happens.

2. **The full text is still reachable.** Truncation that discards the middle
   makes it unanswerable whether or not it matters. The complete page is
   written to the cache and the footer carries the concrete `read_file` call
   that pages through the omitted middle. Context is spent, not data lost.

3. **The footer is deterministic.** No model decides what to keep. Two runs on
   one page produce one answer, so a wrong reading of a page is reproducible
   and therefore fixable.

4. **Base64 images never reach the model.** A single inline PNG is tens of
   thousands of tokens of noise. The image is replaced with a placeholder that
   keeps its alt text, so the *fact* of the image survives and the bytes do
   not. Real image URLs are left intact, because those can be fetched.

5. **A URL is not a resource.** Before anything is fetched it is checked for an
   embedded credential (which must not be handed to a third-party reader) and
   for a private-network target (which must not be reachable from here). This
   is a tripwire, not a sandbox — it is checked on every redirect hop too,
   since that is the obvious way around a single check.

Two capabilities, and they are not equally served:

  * **search** — answered by a provider that needs a key. `anysearch` ships
    here because it is the one the operator has, and its endpoint list is
    kept: this machine's DNS is hijacked often enough that a single hardcoded
    hostname is a single point of failure.
  * **extract** — done locally, with no key at all. The parent framework
    delegates this to a paid reader (Firecrawl/Tavily/Exa). Requiring a
    subscription to read a public page is a bad trade, so the default here is
    a plain fetch plus a stdlib HTML-to-text pass. It cannot run JavaScript;
    when a page comes back empty because of that, the error says so and names
    `browser_goto` instead, which can.

Declared scope note: both tools declare `network`, matching `notify_send`'s
precedent. `extract` also writes the fetched page into this program's own
cache, which is a storage detail of keeping the full text retrievable rather
than a workspace side effect — declaring `system` to cover it would make the
gate ask about arbitrary code execution in order to download a page, which is
true but useless.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import urllib.parse
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from .configfile import PROXIES
from .store import _default_home as autoforge_home

__all__ = [
    "DEFAULT_CHAR_LIMIT", "MAX_STORED_CHARS", "AnySearchProvider",
    "LocalFetchProvider", "Page", "SearchHit", "WebError", "WebProvider",
    "cache_dir", "extract", "get_provider", "html_to_text", "providers",
    "register_provider", "resolve_provider", "safe_url", "search",
    "search_from_config", "strip_base64_images", "truncate_with_footer",
]

#: What this program calls itself when a server asks. Read from the installed
#: distribution so it cannot drift from `pyproject.toml`, with a fallback for the
#: source-checkout case where nothing is installed. The distribution is
#: `autoforge-agent` (the bare name `autoforge` on the index is somebody else's
#: package), so looking up `autoforge` here would not raise -- it would find
#: nothing installed and quietly serve a stale version forever.
#:
#: The fallback is deliberately not a plausible version. It used to be the
#: release number, which is worse than useless: on release day the fallback and
#: the real answer are the same string, so a broken lookup looks exactly like a
#: working one -- and a test that compares the two cannot tell them apart.
try:
    from importlib.metadata import PackageNotFoundError as _NotFound
    from importlib.metadata import version as _pkg_version
    try:
        _VERSION = _pkg_version("autoforge-agent")
    except _NotFound:
        _VERSION = "0.0.0+unknown"
except ImportError:                                    # pragma: no cover
    _VERSION = "0.0.0+unknown"

#: Per-page budget for what the model sees. Generous because this spends
#: context rather than API dollars, and a page cut to 5k is a page whose
#: conclusion was cut off.
DEFAULT_CHAR_LIMIT = 15000

#: Ceiling on the copy written to disk. The model only ever sees
#: `char_limit`, so an unbounded write is pure liability: one pathological
#: page should not be able to fill the disk.
MAX_STORED_CHARS = 2_000_000

#: How long to wait. A page that takes longer than this is a page this tool
#: is not going to read usefully, and the agent needs to get on with it.
DEFAULT_TIMEOUT = 20

#: Redirect hops followed before giving up. Each hop is re-checked, so this is
#: also the bound on how much a redirect chain can be used to walk somewhere.
MAX_REDIRECTS = 5


class WebError(RuntimeError):
    """Something about the web request could not be done, said in one line."""


# ---------------------------------------------------------------------------
# URL safety
# ---------------------------------------------------------------------------
#: Query-parameter names that carry credentials. A URL containing one of these
#: is a URL the operator pasted out of a browser, not one they meant to hand to
#: a third party.
_SENSITIVE_PARAMS = frozenset({
    "access_token", "api_key", "apikey", "auth", "authorization", "code",
    "id_token", "key", "password", "passwd", "refresh_token", "secret",
    "session", "sig", "signature", "token",
})

#: Substrings that look like a secret in a URL path or query. Deliberately
#: broad: the cost of a false positive is one confusing refusal, and the cost
#: of a false negative is the operator's key arriving in someone's log.
_SECRET_SHAPES = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|as_sk_[A-Za-z0-9_\-]{6,}|ghp_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9\-]{10,}"
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.)"
)


def _blocked_reason(url: str) -> str | None:
    """Why this URL must not be fetched, or None if it may be.

    Says which of the two rules it hit, because "blocked" without a reason
    teaches the operator nothing and they will simply try the other form.
    """
    if _SECRET_SHAPES.search(url) or _SECRET_SHAPES.search(urllib.parse.unquote(url)):
        return ("it contains what appears to be an API key or token; secrets "
                "must not be sent in a URL")
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError as exc:
        return f"it is not a URL this program can read ({exc})"
    if parts.scheme not in ("http", "https"):
        # file:, data:, gopher: and friends are not "the web", and one of them
        # reaching a fetch routine is how a local file ends up in a transcript.
        return f"only http and https can be fetched, not {parts.scheme or 'a bare path'!r}"
    for name, _ in urllib.parse.parse_qsl(parts.query, keep_blank_values=True):
        if name.lower() in _SENSITIVE_PARAMS:
            return (f"it carries a credential-like parameter ({name!r}); web "
                    f"readers are third parties, so remove it or read this "
                    f"page with the browser instead")
    return None


def _is_private_host(host: str) -> bool:
    """Whether a hostname resolves only to addresses on this machine's side
    of the network.

    Resolution, not string matching: `localtest.me` and `127.0.0.1.nip.io` are
    public names that point at loopback, and a check that only looks for
    "localhost" is a check that misses them. Every resolved address must be
    public for the host to pass — a name with one public and one private
    address is a coin flip, and there is no reason to take it.
    """
    if not host:
        return True
    # A literal address needs no lookup, and `getaddrinfo` on a literal is
    # just a slower way to read the same string.
    try:
        return not ipaddress.ip_address(host.strip("[]")).is_global
    except ValueError:
        pass
    if host.lower().rstrip(".").endswith((".local", ".internal", ".localhost")):
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        # A name that does not resolve is not a name this program can fetch.
        # Reporting it as private would be a lie about why; the fetch itself
        # will fail with the real reason a moment later.
        return False
    if not infos:
        return False
    return all(not ipaddress.ip_address(info[4][0]).is_global for info in infos)


def safe_url(url: str, *, allow_private: bool = False) -> str:
    """The URL if it may be fetched, else a `WebError` explaining why not."""
    url = str(url or "").strip()
    if not url:
        raise WebError("no URL was given")
    reason = _blocked_reason(url)
    if reason:
        raise WebError(f"refusing to fetch {url!r}: {reason}")
    if not allow_private and _is_private_host(urllib.parse.urlsplit(url).hostname or ""):
        raise WebError(
            f"refusing to fetch {url!r}: it is a private or loopback address. "
            f"This agent has a shell and a browser for this machine; a web "
            f"reader must not be turned into one for the network it sits on."
        )
    return url


def safe_redirects(url: str, *, allow_private: bool = False) -> list[str]:
    """Follow redirects manually, checking each hop, and return the chain.

    Manual rather than `allow_redirects=True`, because a library following
    redirects for us does the check exactly once, on the URL the caller gave —
    which is the one URL a hostile page does not need to control.
    """
    import requests

    seen: list[str] = []
    current = url
    from . import egress

    for _ in range(MAX_REDIRECTS):
        seen.append(current)
        # Per hop, not once up front. `safe_redirects` exists because a library
        # following redirects checks exactly one URL -- the one a hostile page
        # need not control -- and an egress check done only on the caller's URL
        # has the identical hole. A refusal raises rather than returning a short
        # chain: a caller that got back a partial chain would report "no
        # redirects" for what was actually "the second hop was refused".
        egress.admit(current, tool="webtools")
        response = requests.get(
            current, timeout=DEFAULT_TIMEOUT, allow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code not in (301, 302, 303, 307, 308):
            return seen
        location = response.headers.get("Location")
        if not location:
            return seen
        current = safe_url(urllib.parse.urljoin(current, location),
                           allow_private=allow_private)
    raise WebError(f"{url!r} redirected more than {MAX_REDIRECTS} times")


USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


# ---------------------------------------------------------------------------
# text shaping
# ---------------------------------------------------------------------------
_MD_BASE64_IMAGE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)"
)


def strip_base64_images(text: str) -> str:
    """Replace inline base64 images with a placeholder that keeps the alt text.

    Called on everything before it is returned, including snippets, because a
    search result's description is a string this program did not write.
    """
    def _keep_alt(m: re.Match[str]) -> str:
        alt = (m.group("alt") or "").strip()
        return f"[IMAGE: {alt}]" if alt else "[IMAGE]"

    out = _MD_BASE64_IMAGE.sub(_keep_alt, text)
    out = re.sub(r"\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)", "[IMAGE]", out)
    out = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE]", out)
    return out


def cache_dir() -> Path:
    """Where a fetched page is kept so its full text stays reachable."""
    path = Path(autoforge_home()) / "cache" / "web"
    path.mkdir(parents=True, exist_ok=True)
    return path


def store_full_text(url: str, content: str) -> str | None:
    """Write the complete page under the cache and return its path.

    Best-effort: a cache that cannot be written must not fail the fetch. The
    caller gets the truncated text either way, and the footer says the full
    copy is unavailable rather than pointing at a file that is not there.
    """
    try:
        host = (urllib.parse.urlsplit(url).hostname or "page").replace(":", "_")
        slug = re.sub(r"[^A-Za-z0-9._-]", "-", host)[:60].strip("-") or "page"
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
        path = cache_dir() / f"{slug}-{digest}.md"
        if len(content) > MAX_STORED_CHARS:
            content = (
                content[:MAX_STORED_CHARS]
                + f"\n\n[... stored copy truncated at {MAX_STORED_CHARS:,} of "
                f"{len(content):,} chars ...]"
            )
        path.write_text(content, encoding="utf-8")
        return str(path)
    except OSError:
        return None


def truncate_with_footer(content: str, url: str, char_limit: int,
                         *, store: bool = True) -> tuple[str, bool]:
    """`(text for the model, was_truncated)` for one page.

    The two cut points are snapped to newlines so neither end of the excerpt
    begins or ends mid-word — an excerpt that reads as broken text invites the
    model to guess at what the breakage meant.
    """
    if len(content) <= char_limit:
        return content, False

    head_budget = int(char_limit * 0.75)
    tail_budget = char_limit - head_budget
    head, tail = content[:head_budget], content[-tail_budget:]

    cut = head.rfind("\n")
    if cut > head_budget * 0.5:
        head = head[:cut]
    cut = tail.find("\n")
    if 0 <= cut < tail_budget * 0.5:
        tail = tail[cut + 1:]

    stored = store_full_text(url, content) if store else None
    footer = [
        "",
        "─" * 8 + " [TRUNCATED] " + "─" * 8,
        f"Showing {len(head):,} chars (head) + {len(tail):,} chars (tail) of "
        f"{len(content):,} total clean characters.",
    ]
    if stored:
        # The omitted middle starts right after the head already shown, so the
        # model's first read lands in the gap instead of re-reading the top.
        # read_file is 1-indexed: head newlines + 2 is the first unseen line.
        footer.append(f"Full text saved to: {stored}")
        footer.append(
            f'To read the omitted middle: read_file path="{stored}" '
            f"offset={head.count(chr(10)) + 2} limit=200  (the file is the "
            f"complete page; raise or lower offset to page through it)."
        )
        footer.append(
            "Or call web_extract again on a more specific URL. Do not answer "
            "from this excerpt alone if the omitted middle could hold the point."
        )
    else:
        footer.append(
            "The full text could not be stored, so the omitted middle is not "
            "recoverable from here. Re-extract a more specific URL, or use "
            "browser_goto for the complete page."
        )
    footer.append("─" * 29)

    text = head + "\n\n[... middle omitted — see footer ...]\n\n" + tail
    return text + "\n" + "\n".join(footer), True


# ---------------------------------------------------------------------------
# HTML -> text, with no dependency and no service
# ---------------------------------------------------------------------------
#: Elements whose entire subtree is dropped. These nest, so `_skip_depth` is a
#: counter rather than a flag — `<div><style>..</style></div>` must not clear
#: the skip when the inner tag closes.
_DROP_CONTAINERS = frozenset({
    "script", "style", "noscript", "template", "svg", "iframe",
    "form", "nav", "footer", "aside", "button", "select", "option",
})

#: Elements with no end tag and no content to skip. Dropped without touching
#: the depth counter: `<meta>` never closes, so counting it would leave the
#: counter stuck above zero and silently discard the whole rest of the page.
#: This was a real bug here, and it is why the two sets are separate.
#: `head` is deliberately absent from both: dropping it takes `<title>` with
#: it, and the title is the one piece of metadata worth keeping.
_DROP_VOID = frozenset({"meta", "link", "base", "source", "track"})

_BLOCK_ELEMENTS = frozenset({
    "p", "div", "section", "article", "main", "header", "blockquote", "tr",
    "table", "ul", "ol", "dl", "dt", "dd", "figure", "figcaption", "hr",
})
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


class _TextExtractor(HTMLParser):
    """Turns a page into markdown-ish text: structure kept, chrome dropped.

    Not a readability implementation and not trying to be. What it does is
    keep the things a reader would notice missing — headings, list items,
    links, code — and throw away the things a reader would skip. `nav`,
    `footer` and `aside` go because on most sites they are identical on every
    page, which makes them pure context cost.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._pre_depth = 0
        self._in_title = False
        self._link_href: str | None = None
        self._link_text: list[str] = []
        self._list_depth = 0

    # -- helpers ----------------------------------------------------------
    def _newline(self, count: int = 1) -> None:
        self._chunks.append("\n" * count)

    def _text(self, data: str) -> None:
        text = unescape(data)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        if text.strip():
            self._chunks.append(text)

    def _current(self) -> str:
        return "".join(self._chunks)

    # -- parser hooks -----------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k.lower(): (v or "") for k, v in attrs}
        if tag in _DROP_VOID:
            return
        if tag in _DROP_CONTAINERS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _HEADINGS:
            self._newline(2)
            self._chunks.append("#" * _HEADINGS[tag] + " ")
            return
        if tag == "pre":
            self._pre_depth += 1
            self._newline(2)
            self._chunks.append("```\n")
            return
        if tag == "code" and not self._pre_depth:
            self._chunks.append("`")
            return
        if tag in ("strong", "b"):
            self._chunks.append("**")
            return
        if tag in ("em", "i"):
            self._chunks.append("*")
            return
        if tag == "a":
            self._link_href = attr.get("href") or ""
            self._link_text = []
            return
        if tag == "img":
            alt = attr.get("alt", "").strip()
            src = attr.get("src", "")
            if src.startswith("data:"):
                self._chunks.append(f"[IMAGE: {alt}]" if alt else "[IMAGE]")
            elif src:
                self._chunks.append(f"![{alt}]({src})" if alt else f"![]({src})")
            return
        if tag == "br":
            self._newline()
            return
        if tag == "li":
            self._newline()
            self._chunks.append("  " * max(0, self._list_depth - 1) + "- ")
            return
        if tag in ("ul", "ol"):
            self._list_depth += 1
            self._newline()
            return
        if tag in _BLOCK_ELEMENTS:
            self._newline(2)

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP_VOID:
            return
        if tag in _DROP_CONTAINERS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            # A page that never closes its `<script>` would otherwise swallow
            # everything after it. The document ending is the one signal that
            # means no further content is coming, so the skip is cleared there.
            if tag in ("body", "html"):
                self._skip_depth = 0
            return
        if tag == "title":
            self._in_title = False
            self._newline(2)
            return
        if tag == "pre":
            self._pre_depth = max(0, self._pre_depth - 1)
            self._chunks.append("\n```")
            self._newline(2)
            return
        if tag == "code" and not self._pre_depth:
            self._chunks.append("`")
            return
        if tag in ("strong", "b"):
            self._chunks.append("**")
            return
        if tag in ("em", "i"):
            self._chunks.append("*")
            return
        if tag == "a":
            label = "".join(self._link_text).strip()
            href = (self._link_href or "").strip()
            if href and label:
                self._chunks.append(f"[{label}]({href})")
            elif label:
                self._chunks.append(label)
            self._link_href = None
            self._link_text = []
            return
        if tag in ("ul", "ol"):
            self._list_depth = max(0, self._list_depth - 1)
            self._newline(2)
            return
        if tag in _HEADINGS or tag in _BLOCK_ELEMENTS:
            self._newline(2)

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        if self._pre_depth:
            # Inside <pre> the whitespace is the content, so it is the one
            # place the collapsing above must not run.
            self._chunks.append(data)
            return
        if self._link_href is not None:
            self._link_text.append(unescape(data))
            return
        self._text(data)

    # -- result -----------------------------------------------------------
    def result(self) -> str:
        text = self._current()
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return strip_base64_images(text).strip()


def _preferred_fragment(html: str) -> str:
    """The part of the page a reader would have read, when it is marked up.

    `main` and `article` are how a site says "this is the page". When one is
    present the rest is chrome, and keeping it costs context on every fetch.
    A regex rather than a parse because this is a hint, not a decision: if it
    is wrong the fallback is the whole document, which is only ever more text.
    """
    for tag in ("main", "article"):
        match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", html,
                          re.IGNORECASE | re.DOTALL)
        if match and len(match.group(1)) > 500:
            return match.group(1)
    return html


def html_to_text(html: str, url: str = "") -> tuple[str, str]:
    """`(title, text)` for one page's HTML. Never raises on bad markup.

    `HTMLParser` is forgiving by design — a stray tag is a stray tag, not an
    exception — which matters because the pages this runs on are the ones
    nobody validated.
    """
    if not html:
        return "", ""
    body = _preferred_fragment(html)
    parser = _TextExtractor()
    try:
        parser.feed(body)
        parser.close()
    except Exception:                              # noqa: BLE001
        # A malformed page must degrade to "less text", never to "no fetch".
        # What was parsed before the failure is still better than nothing.
        pass
    return (parser.title or "").strip(), parser.result()


def _resolve_relative_links(text: str, url: str) -> str:
    """Make markdown links absolute so they can be fetched in turn."""
    if not url:
        return text
    return re.sub(
        r"\]\((?!/|https?://|mailto:|#)([^)\s]+)\)",
        lambda m: "](" + urllib.parse.urljoin(url, m.group(1)) + ")",
        text,
    )


# ---------------------------------------------------------------------------
# what a provider is
# ---------------------------------------------------------------------------
@dataclass
class SearchHit:
    """One result. `position` is kept so output ordering is not a guess."""

    title: str = ""
    url: str = ""
    description: str = ""
    position: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "url": self.url,
                "description": self.description, "position": self.position}


@dataclass
class Page:
    """One extracted page. `error` is set instead of content on failure."""

    url: str = ""
    title: str = ""
    content: str = ""
    error: str = ""
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "title": self.title, "content": self.content,
                "error": self.error, "truncated": self.truncated}


class WebProvider:
    """A way of reaching the web. Two capabilities, asked separately.

    Search and extract are separate questions because they are separately
    answered in practice: a key that searches rarely also reads, and pretending
    one implies the other makes an operator configure two things that look like
    one. `supports_*` is what the resolver consults, so a provider that only
    searches is never picked to read a page.
    """

    name = "provider"

    def is_available(self) -> bool:
        return True

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int) -> list[SearchHit]:
        raise WebError(f"{self.name} cannot search")

    def extract(self, url: str, *, timeout: int = DEFAULT_TIMEOUT,
                allow_private: bool = False) -> Page:
        raise WebError(f"{self.name} cannot read pages")


#: Registration order is also resolution order for the availability walk.
_PROVIDERS: dict[str, WebProvider] = {}


def register_provider(provider: WebProvider) -> WebProvider:
    """Add a provider. Same name twice replaces, so a test can stand one in."""
    _PROVIDERS[provider.name] = provider
    return provider


def providers() -> list[WebProvider]:
    return [_PROVIDERS[k] for k in sorted(_PROVIDERS)]


def get_provider(name: str) -> WebProvider | None:
    return _PROVIDERS.get(str(name or "").strip())


#: Searched in this order when the config names no backend. The reason `ddgs`
#: is first is not quality — a keyed provider often returns better results —
#: but that it cannot fail on the first run of a fresh install. A default that
#: needs an account is a default that is broken on the machine where nobody has
#: read the README yet. An operator who wants the other one names it in
#: `web.search_backend`, and then step 1 of `resolve_provider` puts it first.
PREFERENCE = ("ddgs", "anysearch", "local-fetch", "jina-reader")


def resolve_provider(capability: str, configured: str = "") -> WebProvider:
    """The provider that will answer `capability`, or a `WebError` naming what
    to set.

    Three rules, in order:

    1. A named provider wins even when it is unavailable, so the failure is
       "ANYSEARCH_API_KEY is not set" instead of silently going somewhere the
       operator did not choose.
    2. Otherwise the first provider in `PREFERENCE` that supports the
       capability and reports itself available.
    3. Otherwise an error that says the one thing to do about it.
    """
    if not capability in ("search", "extract"):
        raise WebError(f"unknown capability {capability!r}")

    capable = (lambda p: p.supports_search()) if capability == "search" \
        else (lambda p: p.supports_extract())

    if configured:
        named = get_provider(configured)
        if named is None:
            known = ", ".join(p.name for p in providers()) or "none"
            raise WebError(f"no web provider named {configured!r}. Known: {known}.")
        if not capable(named):
            raise WebError(f"provider {configured!r} cannot {capability} pages")
        if not named.is_available():
            raise WebError(_unavailable_hint(named, capability))
        return named

    for name in PREFERENCE:
        candidate = get_provider(name)
        if candidate is not None and capable(candidate) and candidate.is_available():
            return candidate

    for candidate in providers():
        if capable(candidate) and candidate.is_available():
            return candidate

    hints = [p.name for p in providers() if capable(p)]
    named = ", ".join(hints) or "none"
    raise WebError(
        f"no web provider can {capability}. Providers that could: {named}. "
        f"Set web.{capability}_backend (or web.backend) to one of them and "
        f"supply its key."
    )


def _unavailable_hint(provider: WebProvider, capability: str) -> str:
    """The one sentence that says what to set. A generic 'not available' sends
    the operator to read this file; the variable name does not."""
    envs = getattr(provider, "env_vars", ())
    if envs:
        return (f"{provider.name} is selected for {capability} but is not "
                f"available — set {', '.join(envs)}.")
    return f"{provider.name} is selected for {capability} but is not available."


# ---------------------------------------------------------------------------
# provider: ddgs (needs nothing at all)
# ---------------------------------------------------------------------------
#: The child program that performs one search. Kept as source here rather than
#: as a second file so the worker and its caller cannot drift apart, and run
#: through `sys.executable -c` so it needs no import path of its own.
_DDGS_WORKER = r'''
import json, sys
out = []
try:
    from ddgs import DDGS
    query = sys.stdin.read()
    limit = int(sys.argv[1])
    with DDGS(timeout=10) as client:
        for index, hit in enumerate(client.text(query, max_results=limit)):
            if index >= limit:
                break
            out.append({
                "title": str(hit.get("title") or ""),
                "url": str(hit.get("href") or hit.get("url") or ""),
                "description": str(hit.get("body") or hit.get("description") or ""),
                "position": index + 1,
            })
    print(json.dumps({"ok": True, "hits": out}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
'''


class DdgsProvider(WebProvider):
    """Search DuckDuckGo. No key, no account, no configuration.

    This is the default searcher, and the reason is not that it is the best
    source — it is that it is the only one that cannot fail for a reason the
    operator has to go and fix. A searcher with no key works on the first run
    of a fresh install, which is the only run most tools get.

    The whole of the interesting code here is the subprocess. `ddgs` scrapes a
    site that throttles scrapers, and when it is throttled it does not raise —
    it *hangs*, retrying across DuckDuckGo's backends. An agent tool call that
    never returns is worse than one that fails, because the agent has no move
    to make: it cannot retry, cannot fall back, and cannot tell the operator
    anything is wrong. A child process is the only bound that works — a thread
    cannot be killed in Python, and `DDGS(timeout=…)` covers each individual
    HTTP request rather than the retry loop across all of them.

    The escalation is terminate, then kill, then wait. Skipping the final wait
    leaves a zombie holding the pipe open on POSIX and a stray process on
    Windows, and this runs once per query.
    """

    name = "ddgs"
    env_vars = ()
    #: Generous relative to the query itself, because DuckDuckGo is slow when
    #: it is busy and a false timeout costs the agent its only searcher.
    timeout = 45

    def __init__(self, *, timeout: int | None = None,
                 runner: Callable[[str, int, int], str] | None = None) -> None:
        if timeout is not None:
            self.timeout = int(timeout)
        # Injected for tests: a unit test must not reach DuckDuckGo.
        self._runner = runner

    def is_available(self) -> bool:
        if self._runner is not None:
            return True
        try:
            import ddgs                                    # noqa: F401
        except ImportError:
            return False
        return True

    def supports_search(self) -> bool:
        return True

    def search(self, query: str, limit: int) -> list[SearchHit]:
        query = str(query or "").strip()
        if not query:
            raise WebError("no query was given")
        # DuckDuckGo answers in pages of about ten; asking for more than that
        # costs extra round trips for results nobody reads.
        wanted = max(1, min(int(limit or 5), 20))

        if self._runner is not None:
            raw = self._runner(query, wanted, self.timeout)
        else:
            raw = self._run_bounded(query, wanted)

        try:
            body = json.loads(raw)
        except ValueError:
            raise WebError(f"the search worker returned nothing readable: "
                           f"{raw[:200]!r}") from None
        if not body.get("ok"):
            raise WebError(f"search failed — {body.get('error') or 'no reason given'}")
        return [SearchHit(**{k: hit.get(k, "") for k in
                             ("title", "url", "description", "position")})
                for hit in body.get("hits") or []]

    def _run_bounded(self, query: str, limit: int) -> str:
        """Run the worker with a wall-clock cap, and make sure it dies."""
        import subprocess
        import sys

        proc = subprocess.Popen(
            [sys.executable, "-c", _DDGS_WORKER, str(limit)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8",
        )
        try:
            out, err = proc.communicate(query, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            _terminate_and_reap(proc)
            raise WebError(
                f"the search did not answer within {self.timeout}s. DuckDuckGo "
                f"throttles scrapers and ddgs retries instead of failing, so a "
                f"hung search is expected occasionally rather than a bug. Try "
                f"again, narrow the query, or configure a keyed provider."
            ) from None
        if proc.returncode != 0 and not out.strip():
            raise WebError(f"the search worker exited {proc.returncode}"
                           + (f" — {(err or '').strip()[:200]}" if err else ""))
        return out.strip()


def _terminate_and_reap(proc: "Any", grace: float = 3.0) -> None:
    """Terminate, escalate to kill, and wait. Never leaves a live child.

    The wait is the part that matters and the part that is easy to omit: a
    terminated process that is not waited on stays a zombie holding the pipe,
    and on Windows it keeps running long enough to answer the *next* query
    with this one's results.
    """
    import time as _time

    def _dead(seconds: float) -> bool:
        deadline = _time.monotonic() + seconds
        while _time.monotonic() < deadline:
            if proc.poll() is not None:
                return True
            _time.sleep(0.05)
        return proc.poll() is not None

    try:
        if proc.poll() is None:
            proc.terminate()
            _dead(grace)
        if proc.poll() is None:
            proc.kill()
            _dead(grace)
    except OSError:
        pass
    finally:
        try:
            proc.wait(timeout=grace)
        except Exception:                              # noqa: BLE001
            pass
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# provider: anysearch (needs a key)
# ---------------------------------------------------------------------------
class AnySearchProvider(WebProvider):
    """Search over `search.intelligence.pub`, ported from the parent
    framework's plugin.

    The endpoint list is the point of this provider rather than an afterthought.
    The machine this runs on has its DNS hijacked often enough that any single
    hostname is a coin flip, so a query is tried against each endpoint in turn
    and the first real answer wins. What was *not* ported from the original is
    the three nested loops of endpoint patterns — they retried the same URLs
    with different spellings and logged a warning per attempt. One ordered
    list, tried once each, and a report of which endpoint answered.
    """

    name = "anysearch"
    env_vars = ("ANYSEARCH_API_KEY",)
    timeout = 20

    #: Tried in order. The first three are the operator's own relay; the last
    #: two are the vendor's direct hostnames. Order matters: when the relay is
    #: up it is the one that has been observed to work from here.
    ENDPOINTS = (
        "https://search-proxy-api.workbuddy.ai/api/search",
        "https://anysearch.workbuddy.io/api/v1/search/query",
        "https://hermes-proxy.net/api/anysearch/search",
        "https://search.intelligence.pub/search/query",
        "https://api-v2.search.intelligence.pub/api/v1/search/query",
    )

    def __init__(self, api_key: str = "", *, timeout: int | None = None,
                 proxy: bool = False) -> None:
        self._api_key = api_key or os.environ.get("ANYSEARCH_API_KEY", "")
        if timeout is not None:
            self.timeout = int(timeout)
        # Off by default and set by config: sending every web query through a
        # local relay is the operator's decision, not this module's.
        self.proxies = dict(PROXIES) if proxy else None

    def is_available(self) -> bool:
        return bool(self._api_key)

    def supports_search(self) -> bool:
        return True

    def search(self, query: str, limit: int) -> list[SearchHit]:
        if not self._api_key:
            raise WebError("ANYSEARCH_API_KEY is not set")
        query = str(query or "").strip()
        if not query:
            raise WebError("no query was given")

        import requests

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "x-api-version": "v1",
            "User-Agent": USER_AGENT,
        }
        # The API caps at 10 per call; asking for more and trimming is how a
        # caller ends up believing it got 20 results.
        payload = {"query": query, "num": max(1, min(int(limit or 5), 10))}

        failures: list[str] = []
        for endpoint in self.ENDPOINTS:
            try:
                response = requests.post(endpoint, json=payload, headers=headers,
                                         timeout=self.timeout,
                                         proxies=self.proxies)
            except requests.RequestException as exc:
                failures.append(f"{urllib.parse.urlsplit(endpoint).netloc}: {exc}")
                continue
            if response.status_code != 200:
                failures.append(f"{urllib.parse.urlsplit(endpoint).netloc}: "
                                f"HTTP {response.status_code}")
                continue
            try:
                body = response.json()
            except ValueError:
                failures.append(f"{urllib.parse.urlsplit(endpoint).netloc}: "
                                f"not JSON")
                continue
            hits = _anysearch_hits(body)
            if hits is not None:
                return hits[: payload["num"]]
            failures.append(f"{urllib.parse.urlsplit(endpoint).netloc}: "
                            f"no 'results' in the response")
        raise WebError(
            "every anysearch endpoint failed. Tried:\n  " + "\n  ".join(failures)
        )


def _anysearch_hits(body: Any) -> list[SearchHit] | None:
    """Parse a search response, or None when it is not one.

    Returns None rather than an empty list on a response with no `results` key,
    because the two mean different things: an empty list is a real answer that
    says nothing matched, and a missing key means this endpoint is not the API.
    Collapsing them makes the fallback loop stop on a wrong-shaped reply.
    """
    if not isinstance(body, dict) or "results" not in body:
        return None
    raw = body.get("results")
    if not isinstance(raw, list):
        return None
    hits: list[SearchHit] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        hits.append(SearchHit(
            title=strip_base64_images(str(item.get("title") or "")),
            url=str(item.get("url") or ""),
            description=strip_base64_images(
                str(item.get("snippet") or item.get("description") or "")),
            position=int(item.get("position") or index),
        ))
    return hits


# ---------------------------------------------------------------------------
# provider: plain fetch (needs nothing)
# ---------------------------------------------------------------------------
class LocalFetchProvider(WebProvider):
    """Read a page with one HTTP GET and no service in the middle.

    This is the default extractor because reading a public page should not
    require a subscription. The honest limit is JavaScript: a page that renders
    its content client-side comes back as an empty shell, and the error says
    exactly that rather than returning "" for the agent to interpret as an
    empty page. `browser_goto` is the answer in that case, and it is named.
    """

    name = "local-fetch"
    env_vars = ()

    def __init__(self, *, timeout: int = DEFAULT_TIMEOUT, proxy: bool = False) -> None:
        self.timeout = int(timeout)
        self.proxies = dict(PROXIES) if proxy else None

    def supports_extract(self) -> bool:
        return True

    def extract(self, url: str, *, timeout: int | None = None,
                allow_private: bool = False) -> Page:
        import requests

        try:
            url = safe_url(url, allow_private=allow_private)
        except WebError as exc:
            return Page(url=str(url), error=str(exc))

        try:
            chain = safe_redirects(url, allow_private=allow_private)
        except (WebError, Exception) as exc:        # noqa: BLE001
            return Page(url=url, error=f"could not reach the page — {exc}")

        final = chain[-1]
        # The hop check above validated each URL as it was followed; this
        # fetch repeats the last one so the body is what was validated.
        try:
            response = requests.get(
                final, timeout=timeout or self.timeout, proxies=self.proxies,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
            )
        except Exception as exc:                    # noqa: BLE001
            return Page(url=final, error=f"could not reach the page — {exc}")

        if response.status_code >= 400:
            return Page(url=final,
                        error=f"the page answered HTTP {response.status_code}")

        ctype = (response.headers.get("Content-Type") or "").lower()
        text = response.text or ""
        if "json" in ctype:
            return Page(url=final, title="", content=strip_base64_images(text))
        # PDFs and images are handled by their own tools; saying so beats
        # handing back a page of binary as if it were text.
        if not ("html" in ctype or "xml" in ctype or "text" in ctype or not ctype):
            return Page(url=final, error=(
                f"this is {ctype or 'an unknown type'}, not a page. "
                f"Use the tool that reads that format."
            ))
        if "xml" in ctype and "html" not in ctype:
            # A feed is worth having as-is; there is no markup to strip.
            return Page(url=final, content=strip_base64_images(text))

        title, body = html_to_text(text, final)
        body = _resolve_relative_links(body, final)
        # A short page and a JavaScript shell look the same from the text
        # alone — both come back nearly empty — and they are opposite problems.
        # The discriminator is the markup that surrounded the text: a stub page
        # ships almost no HTML too, while a client-rendered app ships tens of
        # kilobytes of script that produced nothing. Misreading the first as
        # the second turns "this page is short" into "fetch failed", which is
        # how an agent ends up refusing to read a page that was right there.
        if len(body.strip()) < 200 and len(text) > 5000:
            return Page(url=final, title=title, error=(
                f"only {len(body.strip())} characters of text came back from "
                f"{len(text):,} characters of markup. The page most likely "
                f"renders its content with JavaScript, which a plain fetch "
                f"cannot run. Two ways out, and they are not the same trade: "
                f"`browser_goto` reads it here with a real browser (nothing "
                f"leaves this machine), or set web.extract_backend to "
                f"`jina-reader`, which renders it on someone else's server."
            ))
        return Page(url=final, title=title, content=body)


# ---------------------------------------------------------------------------
# provider: jina reader (needs nothing, but a third party sees the URL)
# ---------------------------------------------------------------------------
class JinaReaderProvider(WebProvider):
    """Read a page by asking a public reader service to render it.

    Named for what it is: `r.jina.ai` fetches the URL, runs a headless browser
    over it, and hands back clean markdown. It needs no key, which is why it is
    here — it is the free answer to the one thing a local fetch cannot do,
    which is JavaScript.

    It is **not** the default, and that is a deliberate trade rather than a
    ranking. The URL being read leaves this machine and reaches a third party,
    and an agent reading an internal wiki page, a staging host, or a signed URL
    would be publishing it without the operator having agreed to that. So it is
    opt-in: `web.extract_backend: jina-reader`, or the operator reads the error
    local-fetch returns on a JavaScript page and chooses.

    Bypasses `safe_url`'s private-address check for the *remote* reader's sake
    but not its own — a private URL is still refused here, because a reader
    service is exactly the wrong thing to point at an internal host.
    """

    name = "jina-reader"
    env_vars = ()
    #: The reader drives a browser before answering, so it is slower than a
    #: plain GET by an order of magnitude.
    timeout = 45
    ENDPOINT = "https://r.jina.ai/"

    def __init__(self, *, base: str = "", timeout: int | None = None,
                 api_key: str = "") -> None:
        self.base = (base or self.ENDPOINT).rstrip("/") + "/"
        if timeout is not None:
            self.timeout = int(timeout)
        # Optional: an unauthenticated request works and is rate-limited. A key
        # raises the limit, and is read from the environment like everything
        # else so it never has to be written into a config file.
        self.api_key = api_key or os.environ.get("JINA_API_KEY", "")

    def supports_extract(self) -> bool:
        return True

    def extract(self, url: str, *, timeout: int | None = None,
                allow_private: bool = False) -> Page:
        import requests

        try:
            url = safe_url(url, allow_private=allow_private)
        except WebError as exc:
            return Page(url=str(url), error=str(exc))

        # The reader blocks browser-looking user agents outright — verified:
        # a Chrome UA gets HTTP 403 and a Cloudflare page, the default
        # `python-requests` UA gets 200 on the same URL. It is guarding against
        # scrapers pretending to be browsers, which is exactly what a browser
        # UA is. So this identifies itself honestly instead of imitating one,
        # and that is the version that works rather than the version that
        # ought to.
        headers = {"User-Agent": f"autoforge/{_VERSION}",
                   "Accept": "text/plain"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.get(self.base + url,
                                    timeout=timeout or self.timeout,
                                    headers=headers)
        except Exception as exc:                       # noqa: BLE001
            return Page(url=url, error=f"the reader could not be reached — {exc}")

        if response.status_code >= 400:
            return Page(url=url, error=(
                f"the reader answered HTTP {response.status_code}. It rate-limits "
                f"anonymous callers, so try again shortly or supply JINA_API_KEY."
            ))
        text = response.text or ""
        # The reader prefixes its markdown with a metadata header — `Title:`,
        # `URL Source:`, `Published Time:`, a cache warning — separated by
        # blank lines. All of it is stripped, blank lines included: the header
        # is metadata about the request rather than content of the page, and
        # leaving it in teaches the model to open its answer with a heading it
        # did not find anywhere on the page. Stopping at the first blank line
        # (the obvious way to write this loop) leaves the rest behind, because
        # the header is *made* of blank lines.
        title = ""
        lines = text.strip().splitlines()
        prefixes = ("Title:", "URL Source:", "Published Time:", "Warning:",
                    "Markdown Content:", "X-Return-Format:")
        while lines and (not lines[0].strip() or lines[0].startswith(prefixes)):
            line = lines.pop(0)
            if line.startswith("Title:"):
                title = line[len("Title:"):].strip()
        body = "\n".join(lines).strip()
        if not body:
            return Page(url=url, title=title,
                        error="the reader returned no content for this URL")
        return Page(url=url, title=title, content=body)


#: Registered at import so `resolve_provider` works with no setup step. The
#: registry is in memory only: a provider is code, not configuration.
register_provider(DdgsProvider())
register_provider(AnySearchProvider())
register_provider(LocalFetchProvider())
register_provider(JinaReaderProvider())


# ---------------------------------------------------------------------------
# config -> provider
# ---------------------------------------------------------------------------
def _web_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    block = (cfg or {}).get("web")
    return block if isinstance(block, dict) else {}


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "0", "no", "off", "none"}
    return bool(value)


def _backend_for(cfg: dict[str, Any] | None, capability: str) -> str:
    """`web.<capability>_backend` first, then `web.backend`, then ''. """
    block = _web_config(cfg)
    for key in (f"{capability}_backend", "backend"):
        value = str(block.get(key) or "").strip()
        if value:
            return value
    return ""


def search_from_config(cfg: dict[str, Any] | None) -> WebProvider:
    return resolve_provider("search", _backend_for(cfg, "search"))


def extract_from_config(cfg: dict[str, Any] | None) -> WebProvider:
    return resolve_provider("extract", _backend_for(cfg, "extract"))


def char_limit_from_config(cfg: dict[str, Any] | None,
                           override: int | None = None) -> int:
    """Per-page budget. Floored at 2k, below which the footer dominates the
    thing it is describing."""
    value = override
    if value is None:
        raw = _web_config(cfg).get("char_limit")
        try:
            value = int(raw) if raw is not None else DEFAULT_CHAR_LIMIT
        except (TypeError, ValueError):
            value = DEFAULT_CHAR_LIMIT
    return max(2000, min(int(value), 500_000))


# ---------------------------------------------------------------------------
# the two things an agent calls
# ---------------------------------------------------------------------------
def search(query: str, limit: int = 5, *,
           cfg: dict[str, Any] | None = None,
           provider: WebProvider | None = None) -> list[SearchHit]:
    """Search the web. Returns hits; raises `WebError` when nothing can."""
    chosen = provider or search_from_config(cfg)
    hits = chosen.search(str(query or ""), int(limit or 5))
    return [h for h in hits if h.url or h.title]


def extract(urls: Any, char_limit: int | None = None, *,
            cfg: dict[str, Any] | None = None,
            provider: WebProvider | None = None) -> list[Page]:
    """Read pages. One page failing never costs the others.

    Takes a single URL, a list of URLs, or a list of search hits — the last
    because it is the one call the agent actually wants to make, and making it
    reshape its own results first is where a round trip disappears.
    """
    if isinstance(urls, (str, bytes)):
        items: list[Any] = [urls]
    elif isinstance(urls, (list, tuple, set)):
        items = list(urls)
    else:
        items = [urls]

    chosen = provider or extract_from_config(cfg)
    limit = char_limit_from_config(cfg, char_limit)
    allow_private = _truthy(_web_config(cfg).get("allow_private_urls"))
    store = _truthy(_web_config(cfg).get("store_full_text", True))

    pages: list[Page] = []
    for item in items:
        url = _url_of(item)
        if not url:
            pages.append(Page(error=(
                "expected a URL string, or an object with a 'url' field, "
                f"got {type(item).__name__}"
            )))
            continue
        page = chosen.extract(url, allow_private=allow_private)
        if page.content and not page.error:
            page.content, page.truncated = truncate_with_footer(
                page.content, page.url or url, limit, store=store)
        pages.append(page)
    return pages


def _url_of(item: Any) -> str:
    """A URL out of whatever the caller passed: a string, a dict, a hit."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, SearchHit):
        return item.url.strip()
    if isinstance(item, Page):
        return item.url.strip()
    if isinstance(item, dict):
        for key in ("url", "href", "link"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("url", "href"):
        value = getattr(item, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def search_json(query: str, limit: int = 5,
                cfg: dict[str, Any] | None = None) -> str:
    """Search and render it the way a tool must: a JSON string.

    Kept here rather than in the agent so the shape the model sees is testable
    without building an agent.
    """
    try:
        hits = search(query, limit, cfg=cfg)
    except WebError as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)
    return json.dumps({
        "success": True,
        "data": {"web": [h.to_dict() for h in hits]},
        "provider": search_from_config(cfg).name,
    }, ensure_ascii=False, indent=2)


def extract_json(urls: Any, char_limit: int | None = None,
                 cfg: dict[str, Any] | None = None) -> str:
    """Read pages and render them as the tool's JSON string."""
    try:
        chosen = extract_from_config(cfg)
    except WebError as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)
    pages = extract(urls, char_limit, cfg=cfg, provider=chosen)
    if not pages:
        return json.dumps({"success": False, "error": "no URL was given"},
                          ensure_ascii=False)
    payload = {
        "success": True,
        "provider": chosen.name,
        "results": [p.to_dict() for p in pages],
    }
    payload["pages_extracted"] = sum(1 for p in pages if p.content and not p.error)
    payload["pages_failed"] = sum(1 for p in pages if p.error)
    return json.dumps(payload, ensure_ascii=False, indent=2)

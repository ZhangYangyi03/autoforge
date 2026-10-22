"""wbproxy — an OpenAI-compatible front door for WorkBuddy subscription accounts.

Reads the account pool dsh-xdpool already harvested, refreshes OAuth tokens
when they expire, and serves ``POST /v1/chat/completions`` in plain OpenAI JSON
so any OpenAI-compatible client can talk to it. Forged-tool frameworks such as
autoforge need no code change: point their base_url at this server.

Why a proxy instead of a native client: autoforge's OpenAICompatClient carries
retry ladders, max_tokens clamping, failover and usage accounting worth keeping.
Reimplementing those against WorkBuddy's SSE-only endpoint would mean either
duplicating them or losing them. This speaks WorkBuddy on one side and OpenAI on
the other, and the client stays untouched.

WorkBuddy protocol facts, all measured against the live gateway (see docs/):
  * the chat path is /v2/chat/completions -- not /v1, and not /responses or
    /messages, both of which answer 404 on this host
  * `stream: true` is required; a non-stream body answers 11101
  * `role: "tool"` and structured `tool_calls` ARE accepted -- the flattening
    some earlier versions did was unnecessary and cost the model its tools
  * business code 11128 covers two unrelated failures: a rate limit and a
    content refusal. 400 is in nobody's retry set while 429 is in everybody's,
    so it is translated, not forwarded
  * the catalog lives at /v2/enterprises/personal/models and nests its data
    under `data.models`; the global host uses /v3/config instead

Usage:
    python wbproxy.py                 # serve on 127.0.0.1:8777
    python wbproxy.py models          # list models with cost/context/tools
    python wbproxy.py models --free   # only the x0.00 ones
    python wbproxy.py models --tools  # only models that support tool calls
    python wbproxy.py login           # paste a credential by hand
    python wbproxy.py doctor          # check accounts, tokens and endpoints

Client setup (autoforge, or anything OpenAI-compatible):
    base_url  http://127.0.0.1:8777/v1
    model     hy3                     (or any id from `models`)
    api_key   wb                      (any non-empty string; not verified)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# WorkBuddy constants -- transcribed from dsh-xdpool/lib/index.js so the two
# cannot drift silently. Every path here was re-checked against the live host;
# the ones that answer 404 are listed so nobody adds them back on a hunch.
# --------------------------------------------------------------------------
CN_CHAT_BASE = "https://copilot.tencent.com"
CN_BILLING_BASE = "https://www.codebuddy.cn"
GLOBAL_BASE = "https://www.workbuddy.ai"
CLIENT_UA = "CLI/2.63.2 CodeBuddy/2.63.2"

MODELS_PATH = "/v2/enterprises/personal/models"    # cn
GLOBAL_CONFIG_PATH = "/v3/config"                  # global
REFRESH_PATH = "/v2/plugin/auth/token/refresh"
CHAT_PATH = "/v2/chat/completions"

#: Paths that do NOT exist on these hosts. Kept as a record: an earlier version
#: of this file assumed the messages/responses wires were available and built a
#: whole translation layer for them. Measured: all four are 404.
ABSENT_PATHS = (
    "/v2/responses", "/v2/messages", "/v1/chat/completions",
    "/inference/v1/chat/completions",
)

GLOBAL_HOSTS = ("workbuddy.ai", "workbuddy.cc", "codebuddy.ai")

CHAT_IDENTITY_HEADERS = {
    "X-IDE-Name": "WorkBuddy",
    "X-IDE-Type": "WorkBuddy",
    "X-IDE-Version": "5.5.6",
}

#: Used only when the live catalog cannot be read. `auto` is first because the
#: gateway routes it; the rest are the ids observed to answer on a CN account.
FALLBACK_MODELS = [
    {"id": "auto", "name": "Auto", "context_window": 256000,
     "supports_tools": True, "credits": None},
    {"id": "hy3", "name": "Hy3", "context_window": 192000,
     "supports_tools": True, "credits": 0.0},
    {"id": "default", "name": "Default", "context_window": 200000,
     "supports_tools": True, "credits": 2.2},
]

#: Credential sources, most preferred first.
#:
#: The WorkBuddy desktop app's own files are NOT readable here: their tokens are
#: `$wbEncrypted` AES-256-GCM envelopes whose key lives in the OS keystore, and
#: opening them means reimplementing that derivation per platform. dsh-xdpool
#: does that work and re-emits plaintext, so its cache is the source.
def account_sources() -> list[Path]:
    out: list[Path] = []
    override = os.environ.get("WBPROXY_ACCOUNTS")
    if override:
        out.append(Path(override).expanduser())
    dsh_home = os.environ.get("DSH_HOME") or (Path.home() / ".dsh")
    out.append(Path(dsh_home) / ".xdpool" / "accounts.json")
    # A pool this tool wrote itself, for users with no DSH install.
    out.append(Path.home() / ".wbproxy" / "accounts.json")
    return out


LISTEN_HOST = os.environ.get("WBPROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("WBPROXY_PORT", "8777"))
REFRESH_MARGIN_S = 120.0

#: Restrict the pool to one region. `cn`, `global`, or unset for both.
#:
#: A mixed pool spreads requests over accounts with different plans, catalogs and
#: rate limits, which is fine for throughput and confusing for diagnosis: a
#: failure names a model that the account which served it may not even offer.
#: Pinning the region makes "which plan am I on" a property of the deployment.
WBPROXY_REGION = os.environ.get("WBPROXY_REGION", "").strip().lower()

#: The last request's shape, kept so a rejection can be read against what was
#: actually sent. Not the body itself -- it may carry a conversation.
_LAST_BODY: dict[str, Any] = {}


def log(msg: str) -> None:
    print(f"[wbproxy] {msg}", flush=True)


# --------------------------------------------------------------------------
# conversation identity
# --------------------------------------------------------------------------
class Conversation:
    """The three ids the global gateway demands, stable within a conversation.

    The upstream treats the triple as one session, so regenerating them per
    request would read as a new conversation on every turn. Derived from a key
    rather than stored, so identity survives a restart with no registry.

    Only presence and consistency are checked -- the gateway does not validate
    the values -- but they are still derived deterministically, because a
    conversation that changes its id mid-thread is asking for a cache miss.
    """

    __slots__ = ("key", "_request", "_message")

    def __init__(self, key: str):
        self.key = key
        self._request = ""
        self._message = ""

    @classmethod
    def new(cls) -> "Conversation":
        return cls(str(uuid.uuid4()))

    @classmethod
    def for_conversation(cls, key: str | None) -> "Conversation":
        """One conversation per key, so a thread keeps its identity."""
        return cls(key) if key else cls.new()

    def _derive(self, suffix: str) -> str:
        digest = hashlib.sha256(
            f"conversation:{suffix}:{self.key}".encode("utf-8")).hexdigest()
        return (f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-"
                f"{digest[16:20]}-{digest[20:32]}")

    def rotate_request(self) -> None:
        """A new per-request id. The conversation id stays put."""
        self._request = str(uuid.uuid4())
        self._message = str(uuid.uuid4())

    def headers(self) -> dict[str, str]:
        if not self._request:
            self.rotate_request()
        return {
            "X-Conversation-ID": self._derive("id"),
            "X-Conversation-Request-ID": self._request,
            "X-Conversation-Message-ID": self._message,
        }


# --------------------------------------------------------------------------
# accounts
# --------------------------------------------------------------------------
class Account:
    """One WorkBuddy credential, with lazy token refresh."""

    def __init__(self, raw: dict[str, Any]):
        self.id: str = str(raw.get("id") or "")
        self.label: str = str(raw.get("label") or self.id)
        cred = raw.get("credential") or {}
        self.access_token: str = str(cred.get("accessToken") or "")
        self.refresh_token: str = str(cred.get("refreshToken") or "")
        self.expires_at_ms: int = int(cred.get("expiresAtMs") or 0)
        self.domain: str = str(cred.get("domain") or "")
        self.uid: str = str(cred.get("uid") or "")
        self.enterprise_id: str = str(cred.get("enterpriseId") or "")
        self._lock = threading.Lock()
        self._last_error = ""
        #: Set when the account has been proven unable to chat (as opposed to
        #: merely unable to authenticate). Excluded from rotation, kept visible.
        self.chat_unsupported: str = ""
        self.cooldown_until: float = 0.0
        self.last_used: float = 0.0

    # -- region / endpoints ------------------------------------------------
    @property
    def region(self) -> str:
        lowered = self.domain.strip().lower()
        for host in GLOBAL_HOSTS:
            if lowered == host or lowered.endswith(f".{host}"):
                return "global"
        return "cn"

    @property
    def chat_base(self) -> str:
        # A token is not interchangeable across brand domains, so the base
        # follows the credential's own domain rather than one fixed host.
        return CN_CHAT_BASE if self.region == "cn" else GLOBAL_BASE

    @property
    def origin(self) -> str:
        return CN_BILLING_BASE if self.region == "cn" else GLOBAL_BASE

    # -- headers -----------------------------------------------------------
    def chat_headers(self, conversation: "Conversation | None" = None) -> dict[str, str]:
        h = {
            "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": self.origin,
            "Referer": f"{self.origin}/",
            "User-Agent": CLIENT_UA,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
            "X-Product": "SaaS",
        }
        if self.uid:
            h["X-User-Id"] = self.uid
        else:
            h["X-No-User-Id"] = "1"
        if self.enterprise_id:
            h["X-Enterprise-Id"] = self.enterprise_id
        else:
            h["X-No-Enterprise-Id"] = "1"
        if self.domain:
            h["X-Domain"] = self.domain
        else:
            h["X-No-Department-Info"] = "1"
        h.update(CHAT_IDENTITY_HEADERS)
        # The global gateway made channel verification a hard check: a chat
        # request missing any of these three is rejected with 11128. Narrowed by
        # elimination against a real credential -- deleting one turns a 200 into
        # a 400, and the values are not validated (plain non-UUID strings pass),
        # so this is a presence check, not a signature.
        #
        # Sent to BOTH gateways: the domestic one ignores them, and sending them
        # keeps the code path single.
        conv = conversation or Conversation.new()
        h.update(conv.headers())
        return h

    # -- token lifecycle ---------------------------------------------------
    def expires_in_s(self) -> float:
        return (self.expires_at_ms / 1000.0) - time.time() if self.expires_at_ms else 0.0

    def needs_refresh(self) -> bool:
        return self.expires_in_s() < REFRESH_MARGIN_S

    def refresh(self) -> bool:
        if not self.refresh_token:
            self._last_error = "no refresh token"
            return False
        headers = {
            "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": self.origin,
            "Referer": f"{self.origin}/",
            "User-Agent": CLIENT_UA,
            "X-Refresh-Token": self.refresh_token,
            "X-Auth-Refresh-Source": "workbuddy",
        }
        req = urllib.request.Request(
            f"{self.chat_base}{REFRESH_PATH}", data=b"", headers=headers,
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            self._last_error = f"refresh http {exc.code}: {detail}"
            return False
        except Exception as exc:                       # noqa: BLE001
            self._last_error = f"refresh failed: {type(exc).__name__}: {exc}"
            return False

        if not isinstance(body, dict) or body.get("code") != 0:
            self._last_error = "refresh rejected"
            return False
        data = body.get("data") or {}
        token = str(data.get("accessToken") or "")
        if not token:
            self._last_error = "refresh returned no accessToken"
            return False
        self.access_token = token
        if data.get("refreshToken"):
            self.refresh_token = str(data["refreshToken"])
        expires_in = data.get("expiresIn")
        if isinstance(expires_in, (int, float)) and expires_in > 0:
            self.expires_at_ms = int((time.time() + float(expires_in)) * 1000)
        if data.get("domain"):
            self.domain = str(data["domain"])
        self._last_error = ""
        log(f"refreshed {self.label} (region={self.region}, "
            f"valid {self.expires_in_s() / 3600:.1f}h)")
        return True

    def usable_token(self) -> str:
        """A live token, refreshing once if needed. Empty string on failure."""
        with self._lock:
            if self.needs_refresh() and not self.refresh():
                return ""
            return self.access_token


def load_accounts(verbose: bool = True) -> list[Account]:
    """Every account the sources describe, optionally narrowed to one region.

    `WBPROXY_REGION` is applied here rather than at selection time so that
    `doctor`, `models` and `serve` all agree about which pool they are talking
    about: a diagnostic that inspects accounts the server will never use is
    worse than no diagnostic.
    """
    for path in account_sources():
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:                       # noqa: BLE001
            if verbose:
                log(f"{path}: unreadable — {exc}")
            continue
        entries = doc.get("accounts") if isinstance(doc, dict) else None
        if not isinstance(entries, list):
            # A hand-written pool file may be a bare list.
            entries = doc if isinstance(doc, list) else []
        out: list[Account] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cred = entry.get("credential") or {}
            if not cred.get("accessToken") and not cred.get("refreshToken"):
                continue
            out.append(Account(entry))
        if out:
            if WBPROXY_REGION in ("cn", "global"):
                kept = [a for a in out if a.region == WBPROXY_REGION]
                if verbose and len(kept) != len(out):
                    log(f"WBPROXY_REGION={WBPROXY_REGION}: kept {len(kept)} "
                        f"of {len(out)} account(s)")
                out = kept
            if verbose:
                log(f"{len(out)} account(s) from {path}")
            return out
    return []


# --------------------------------------------------------------------------
# the bridge
# --------------------------------------------------------------------------
class Bridge:
    """Account pool + protocol translation.

    Both regions live in one pool. A request picks an account by *model* first:
    a model id that only one region's catalog offers can only be served by that
    region's accounts, so the choice is forced. When both regions offer the same
    id -- `hy3`, `deepseek-v4.1-flash`, `glm-5.3` and six others are on both --
    the eligible set is the union and rotation spreads across it.

    That is the whole point of one process: the caller names a model, not a
    region, and the pool does the routing. `WBPROXY_REGION` still pins the pool
    to one side for diagnosis, but leaving it unset is the normal case.
    """

    def __init__(self, accounts: list[Account]):
        self.accounts = accounts
        self._cursor = 0
        #: region -> (rows, fetched_at). One catalog per region, because the two
        #: gateways publish different ids and different plans.
        self._catalogs: dict[str, list[dict[str, Any]]] = {}
        self._catalog_at: dict[str, float] = {}
        #: model id -> the set of regions whose catalog lists it. Derived from
        #: the catalogs on demand; a miss means "no catalog has told us yet",
        #: which must not narrow the pool.
        self._index: dict[str, set[str]] = {}

    # -- rotation ----------------------------------------------------------
    def eligible(self, region: str | None = None) -> list[Account]:
        now = time.time()
        return [a for a in self.accounts
                if not a.chat_unsupported and a.cooldown_until <= now
                and (region is None or a.region == region)]

    def regions_for(self, model: str) -> set[str] | None:
        """Which regions publish `model`. None means "not known", not "none".

        The distinction matters: an empty set would exclude every account, while
        an unknown model should still be attempted -- a catalog can lag an id
        the gateway accepts, and refusing to try would turn that lag into an
        outage.
        """
        if not self._index:
            return None
        return self._index.get(model) or None

    def pick(self, model: str | None = None) -> Account | None:
        """Least-recently-used account among those that can serve `model`."""
        regions = self.regions_for(model) if model else None
        pool = self.eligible()
        if regions:
            narrowed = [a for a in pool if a.region in regions]
            # A model the catalogs place in one region while that region's
            # accounts are all cooling is still worth trying elsewhere: the id
            # may be served on both and only published on one.
            pool = narrowed or pool
        if not pool:
            return None
        pool.sort(key=lambda a: a.last_used)
        acct = pool[0]
        acct.last_used = time.time()
        return acct

    # -- model catalog -----------------------------------------------------
    def models(self, refresh: bool = False, region: str | None = None) -> list[dict[str, Any]]:
        """Every model either region offers, each row carrying its region.

        One row per (id, region). The catalog is fetched once per region and
        cached, so three domestic accounts do not triple the listing -- and an
        id published by both regions appears twice rather than merged, because
        the prices differ (a free row here, a paid twin there) and a merged row
        would have to invent which price to show.
        """
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for acct in self.accounts:
            if region is not None and acct.region != region:
                continue
            for row in self._catalog_for(acct.region, refresh):
                key = (row["id"], row["region"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
        # Rebuild the id -> regions index from whatever was actually read.
        self._index = {}
        for row in rows:
            self._index.setdefault(row["id"], set()).add(row["region"])
        if not rows:
            return FALLBACK_MODELS
        return rows

    def _catalog_for(self, region: str, refresh: bool = False) -> list[dict[str, Any]]:
        """One region's catalog, cached for an hour."""
        if not refresh and region in self._catalogs \
                and time.time() - self._catalog_at.get(region, 0) < 3600:
            return self._catalogs[region]
        for acct in self.accounts:
            if acct.region != region:
                continue
            if not acct.usable_token():
                continue
            rows = self._fetch_catalog(acct)
            if rows:
                self._catalogs[region] = rows
                self._catalog_at[region] = time.time()
                return rows
        return self._catalogs.get(region, [])

    def _fetch_catalog(self, acct: Account) -> list[dict[str, Any]]:
        url = (f"{GLOBAL_BASE}{GLOBAL_CONFIG_PATH}" if acct.region == "global"
               else f"{acct.chat_base}{MODELS_PATH}")
        req = urllib.request.Request(url, headers=acct.chat_headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                doc = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:                       # noqa: BLE001
            log(f"catalog {acct.label}: {type(exc).__name__}: {exc}")
            return []
        return self._parse_catalog(doc, acct.region)

    @staticmethod
    def _parse_catalog(doc: Any, region: str) -> list[dict[str, Any]]:
        """`data.models`, normalised.

        Two fields arrive in two shapes: the context window is sometimes a
        number and sometimes `{defaultLength, supportedLengths}`, and `credits`
        is sometimes `"x0.11 credits"` and sometimes a bare number. Both are
        normalised here rather than at every call site.
        """
        if not isinstance(doc, dict):
            return []
        data = doc.get("data")
        raw = data.get("models") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            raw = doc.get("models")

        def _int(value: Any) -> int | None:
            try:
                return int(value) or None
            except (TypeError, ValueError):
                return None

        def _ctx(value: Any) -> int | None:
            if isinstance(value, dict):
                return _int(value.get("defaultLength")) or _int(
                    (value.get("supportedLengths") or [None])[0])
            return _int(value)

        def _credits(value: Any) -> float | None:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                digits = "".join(ch for ch in value.split("credits")[0]
                                 if ch.isdigit() or ch == ".")
                try:
                    return float(digits) if digits else None
                except ValueError:
                    return None
            return None

        rows: list[dict[str, Any]] = []
        for m in raw if isinstance(raw, list) else []:
            if not isinstance(m, dict):
                continue
            mid = str(m.get("id") or m.get("model") or "")
            if not mid:
                continue
            rows.append({
                "id": mid,
                "name": str(m.get("name") or mid),
                "context_window": _ctx(m.get("contextWindow")
                                       or m.get("maxInputTokens")),
                "max_output_tokens": _int(m.get("maxOutputTokens")),
                "credits": _credits(m.get("credits")),
                "supports_tools": bool(m.get("supportsToolCall", True)),
                "supports_images": bool(m.get("supportsImages")),
                "supports_reasoning": bool(m.get("supportsReasoning")),
                "vendor": str(m.get("vendor") or ""),
                "description": str(m.get("descriptionEn") or "")[:120],
                "region": region,
            })
        return rows

    # -- chat --------------------------------------------------------------
    def chat(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """One OpenAI request, retried across accounts and across time.

        11128 covers both a rate limit and a content refusal, so a rejection is
        not proof the request is wrong. The ladder walks every account that can
        serve the requested model once -- a different key arriving at the same
        speed is not a retry, it is a different request -- then backs off.

        The account set is narrowed by model first: a request for an id only the
        global catalog lists should not spend a turn failing against a domestic
        account, and the reverse.
        """
        model = str(payload.get("model") or "")
        regions = self.regions_for(model)
        candidate = self.eligible()
        if regions:
            narrowed = [a for a in candidate if a.region in regions]
            candidate = narrowed or candidate
        attempts = max(1, len(candidate)) + 1
        last: tuple[int, dict[str, Any]] = (503, {"error": {
            "message": "no usable WorkBuddy account", "type": "wbproxy_no_account"}})
        for attempt in range(attempts):
            code, body = self._chat_once(payload)
            if code == 200:
                return code, body
            last = (code, body)
            if attempt < attempts - 1:
                delay = 4.0 * (attempt + 1)
                log(f"attempt {attempt + 1}/{attempts} -> {code}; "
                    f"retrying in {delay:.0f}s")
                time.sleep(delay)
        return last

    def _chat_once(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        model = str(payload.get("model") or "auto")
        acct = self.pick(model)
        if acct is None:
            return 503, {"error": {"message": "no usable WorkBuddy account",
                                   "type": "wbproxy_no_account"}}

        messages = []
        for m in payload.get("messages") or []:
            m = dict(m)
            if m.get("role") == "developer":
                # The gateway rejects the `developer` role (11128).
                m["role"] = "system"
            messages.append(m)

        # The global gateway requires the first message to be a system prompt
        # and answers 11128 "first message is not system prompt" otherwise --
        # while the domestic gateway accepts the same body. A placeholder is
        # prepended rather than the caller's text being rewritten: their words
        # and their position are untouched.
        if acct.region == "global" and not (
                messages and str(messages[0].get("role")) == "system"):
            messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,                 # a non-stream body answers 11101
            "stream_options": {"include_usage": True},
        }
        if payload.get("temperature") is not None:
            body["temperature"] = payload["temperature"]
        if payload.get("max_tokens") is not None:
            body["max_tokens"] = int(payload["max_tokens"])

        tools = payload.get("tools")
        if tools:
            body["tools"] = tools
            choice = payload.get("tool_choice", "auto")
            if isinstance(choice, dict):
                fn = choice.get("function") or {}
                choice = fn.get("name") or choice.get("name") or "auto"
            choice = str(choice or "auto").strip().lower()
            if choice in ("none", ""):
                body.pop("tools", None)
                body.pop("tool_choice", None)
            elif choice in ("auto", "required"):
                # Measured: an explicit "auto" round-trips fine, but omitting it
                # asks for the same behaviour with one less field to be wrong.
                pass
            else:
                body["tool_choice"] = choice

        url = f"{acct.chat_base}{CHAT_PATH}"
        conv = Conversation.for_conversation(_conversation_key(messages))
        req = urllib.request.Request(
            url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=acct.chat_headers(conv), method="POST")
        _LAST_BODY.clear()
        _LAST_BODY.update({
            "model": model,
            "roles": [str(m.get("role")) for m in messages],
            "keys": sorted(body.keys()),
            "max_tokens": body.get("max_tokens"),
            "n_tools": len(body.get("tools") or []),
        })
        if os.environ.get("WBPROXY_TRACE_BODY"):
            log("BODY " + json.dumps(body, ensure_ascii=False)[:4000])

        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:600]
            except Exception:
                pass
            log(f"upstream {exc.code} for {acct.label} "
                f"(model={model}, roles={_LAST_BODY.get('roles')}): {detail[:180]}")
            out_code = _translated_status(exc.code, detail)
            return out_code, {"error": {
                "message": f"workbuddy http {exc.code}: {detail}",
                "type": "wbproxy_upstream"}, "_status": out_code}
        except Exception as exc:                       # noqa: BLE001
            return 502, {"error": {
                "message": f"workbuddy unreachable: {exc}",
                "type": "wbproxy_transport"}, "_status": 502}

        try:
            content, tool_calls, usage, finish = _consume_sse(resp)
        finally:
            try:
                resp.close()
            except Exception:
                pass

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return 200, {
            "id": f"chatcmpl-wb-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": finish or ("tool_calls" if tool_calls else "stop")}],
            "usage": usage or {},
            "_status": 200,
        }


def _conversation_key(messages: list[dict[str, Any]]) -> str | None:
    """A stable key for this conversation, derived from its opening turn.

    The upstream treats the conversation-id triple as one session, so the ids
    must not change between turns or every request reads as a new conversation.
    The first user message is what identifies a thread here: it is stable across
    a retry, stable across a provider failover, and does not require the proxy
    to keep state or the client to send an id it does not have.

    Returning None makes the caller mint a random key, which is the right answer
    for a one-shot call with no history.
    """
    for m in messages:
        if str(m.get("role")) not in ("user", "system"):
            continue
        content = m.get("content")
        if isinstance(content, list):
            text = "".join(str(b.get("text") or "") for b in content
                           if isinstance(b, dict))
        else:
            text = str(content or "")
        text = text.strip()
        if text:
            return hashlib.sha256(text[:600].encode("utf-8")).hexdigest()[:32]
    return None


def _translated_status(upstream: int, detail: str) -> int:
    """The status to hand the client for an upstream failure.

    The gateway answers a rate limit with HTTP 400 and business code 11128 --
    the same code it uses for a malformed request. Forwarding that as 400 is
    the one thing that guarantees no client ever recovers: 400 is in nobody's
    retry set, 429 is in everybody's. Measured on this host: requests that
    11128'd all succeeded minutes later with an identical body, which is the
    signature of throttling, not of a defect in the request.

    A genuine format error still surfaces as a 400 -- the code alone cannot
    tell the two apart -- but after the retry ladder has had its chance. That
    is the right order: retrying a correct request costs seconds, dropping a
    throttled one costs a whole tool forge.
    """
    if upstream == 400 and ("11128" in detail or "request illegal" in detail):
        return 429
    return upstream


def _consume_sse(resp: Any) -> tuple[str, list[dict[str, Any]], dict[str, int], str]:
    """Fold a WorkBuddy SSE stream into one OpenAI-shaped completion.

    Two streams are assembled: `delta.content` for the text, and
    `delta.tool_calls`, which arrive as fragments -- the first chunk carries
    id/name, later ones only argument slices. Concatenating by index is what
    reassembles a whole tool call.

    The read is threaded because `select` does not work on Windows pipes and a
    server that accepts a request then stalls must not stall the proxy with it.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tc_acc: dict[int, dict[str, Any]] = {}
    usage: dict[str, int] = {}
    finish = ""

    q: "queue.Queue[Any]" = queue.Queue()

    def _reader() -> None:
        try:
            for chunk in resp:
                q.put(chunk)
        except Exception as exc:                       # noqa: BLE001
            q.put(exc)
        finally:
            q.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    buf = b""
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            log(f"sse reader: {item}")
            break
        buf += item if isinstance(item, (bytes, bytearray)) else str(item).encode()
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line or line.startswith(b":") or not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                frame = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if not isinstance(frame, dict):
                continue
            if isinstance(frame.get("usage"), dict) and frame["usage"]:
                usage = {
                    "prompt_tokens": int(frame["usage"].get("prompt_tokens") or 0),
                    "completion_tokens": int(frame["usage"].get("completion_tokens") or 0),
                }
            for choice in frame.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                if choice.get("finish_reason"):
                    finish = str(choice["finish_reason"])
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(str(delta["content"]))
                if delta.get("reasoning_content"):
                    reasoning_parts.append(str(delta["reasoning_content"]))
                for tc in delta.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    idx = int(tc.get("index") or 0)
                    slot = tc_acc.setdefault(idx, {"id": "", "name": "", "args": []})
                    if tc.get("id"):
                        slot["id"] = str(tc["id"])
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = str(fn["name"])
                    if fn.get("arguments"):
                        slot["args"].append(str(fn["arguments"]))

    tool_calls: list[dict[str, Any]] = []
    for idx in sorted(tc_acc):
        slot = tc_acc[idx]
        if not slot["name"] and not slot["args"]:
            continue
        tool_calls.append({
            "index": idx,
            "id": slot["id"] or f"call_{slot['name'] or idx}",
            "type": "function",
            "function": {"name": slot["name"], "arguments": "".join(slot["args"])},
        })
    if reasoning_parts and not content_parts:
        # A reasoning-only answer is a truncated one; keep it visible rather
        # than returning an empty string the caller cannot explain.
        content_parts.append("".join(reasoning_parts))
    return "".join(content_parts), tool_calls, usage, finish


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    bridge: Bridge = None                # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        log(f"{self.address_string()} {fmt % args}")

    def _json(self, code: int, body: dict[str, Any]) -> None:
        # `_status` is the proxy's own bookkeeping and must not reach the client.
        body = {k: v for k, v in body.items() if not k.startswith("_")}
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = self.path.rstrip("/")
        if path in ("/v1/models", "/models"):
            rows = self.bridge.models()
            data = []
            for r in rows:
                entry = {"id": r["id"], "object": "model", "created": 0,
                         "owned_by": r.get("vendor") or "workbuddy"}
                # Extra fields are legal here and OpenAI-shaped clients ignore
                # what they do not know. A chooser needs them.
                for k in ("name", "context_window", "max_output_tokens",
                          "credits", "supports_tools", "supports_images",
                          "supports_reasoning", "region", "description"):
                    if r.get(k) not in (None, ""):
                        entry[k] = r[k]
                data.append(entry)
            self._json(200, {"object": "list", "data": data})
            return
        if path in ("/healthz", "/health", "/", ""):
            by_region: dict[str, int] = {}
            for a in self.bridge.accounts:
                by_region[a.region] = by_region.get(a.region, 0) + 1
            self._json(200, {
                "ok": True,
                "accounts": [{"label": a.label, "region": a.region,
                              "expires_in_s": round(a.expires_in_s(), 1),
                              "chat_unsupported": a.chat_unsupported or None,
                              "last_error": a._last_error or None}
                             for a in self.bridge.accounts],
                "accounts_by_region": by_region,
                "models": len({r["id"] for r in self.bridge.models()}),
            })
            return
        self._json(404, {"error": {"message": f"unknown path {self.path}"}})

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, {"error": {"message": f"unknown path {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:                       # noqa: BLE001
            self._json(400, {"error": {"message": f"bad body: {exc}"}})
            return
        code, body = self.bridge.chat(payload)
        self._json(code, body)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_models(args: argparse.Namespace) -> int:
    accounts = load_accounts()
    if not accounts:
        print("no accounts found — run `wbproxy login`, or install the DSH "
              "xdpool plugin once so its cache exists", file=sys.stderr)
        return 1
    rows = Bridge(accounts).models(refresh=args.refresh)
    if args.tools:
        rows = [r for r in rows if r.get("supports_tools")]
    if args.free:
        rows = [r for r in rows if (r.get("credits") or 0) == 0]
    if args.region:
        rows = [r for r in rows if r.get("region") == args.region]
    if args.sort == "cost":
        rows.sort(key=lambda r: (r.get("credits") is None, r.get("credits") or 0))
    elif args.sort == "context":
        rows.sort(key=lambda r: r.get("context_window") or 0, reverse=True)

    if not rows:
        print("no model matched the filter", file=sys.stderr)
        return 1

    def fmt_ctx(n: int | None) -> str:
        if not n:
            return "-"
        return f"{n // 1000}k" if n >= 1000 else str(n)

    def fmt_credits(c: float | None) -> str:
        if c is None:
            return "-"
        return "free" if c == 0 else f"x{c:g}"

    head = (f"{'id':24s} {'region':>7s} {'ctx':>7s} {'out':>7s} {'cost':>7s} "
            f"{'tools':>5s} {'img':>4s}  description")
    print(head)
    print("-" * (len(head) + 8))
    for r in rows:
        print(f"{r['id']:24s} {r.get('region', '?'):>7s} "
              f"{fmt_ctx(r.get('context_window')):>7s} "
              f"{fmt_ctx(r.get('max_output_tokens')):>7s} "
              f"{fmt_credits(r.get('credits')):>7s} "
              f"{'y' if r.get('supports_tools') else '-':>5s} "
              f"{'y' if r.get('supports_images') else '-':>4s}  "
              f"{r.get('description') or ''}")
    # An id both regions publish gets two rows, because the prices differ; say
    # so, otherwise the duplicate looks like a bug.
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["id"]] = counts.get(r["id"], 0) + 1
    shared = sorted(k for k, v in counts.items() if v > 1)
    print(f"\n{len(rows)} row(s), {len(counts)} distinct id(s), "
          f"{len({r.get('region') for r in rows})} region(s).")
    if shared:
        print(f"on both regions: {', '.join(shared)}")
    print(f"Pick one with `--model <id>` or the client's model setting.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    accounts = load_accounts()
    if not accounts:
        print("no accounts found in any source:", file=sys.stderr)
        for p in account_sources():
            print(f"  {'ok  ' if p.exists() else 'miss'} {p}", file=sys.stderr)
        return 1
    bridge = Bridge(accounts)
    rows = bridge.models(refresh=True)
    by_region: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_region.setdefault(r["region"], []).append(r)
    print("catalog: " + ", ".join(
        f"{len(v)} model(s) on {k}" for k, v in sorted(by_region.items()))
        or "catalog: unavailable")
    print()
    for acct in accounts:
        token = acct.usable_token()
        state = "ok" if token else "NO TOKEN"
        print(f"{acct.label:26s} region={acct.region:6s} {state:10s} "
              f"expires {acct.expires_in_s() / 3600:+.1f}h"
              + (f"  error={acct._last_error}" if acct._last_error else ""))
        if not token:
            continue
        # Probe with a model this account's OWN catalog lists, so a failure
        # means the account cannot chat rather than that the id was wrong.
        mine = by_region.get(acct.region) or rows
        probe = next((r["id"] for r in mine if r.get("credits") == 0),
                     mine[0]["id"] if mine else "hy3")
        code, body = bridge._chat_once(
            {"model": probe, "messages": [{"role": "user", "content": "ok"}],
             "max_tokens": 16})
        if code == 200:
            print(f"{'':26s} chat ok (model={probe})")
        else:
            msg = (body.get("error") or {}).get("message", "")[:110]
            print(f"{'':26s} chat FAILED {code}: {msg}")
            if "11128" in msg and "conversation" in msg.lower():
                acct.chat_unsupported = "channel verification refused"
    return 0


def cmd_login(args: argparse.Namespace) -> int:
    """Add a credential by hand, for users with no DSH install.

    The desktop app's own files cannot be read: their tokens are AES-256-GCM
    envelopes keyed from the OS keystore, and opening them means reimplementing
    that derivation per platform. Pasting is the honest alternative.
    """
    print("Paste the WorkBuddy credential fields. Leave a field empty to stop.")
    print("They are stored under ~/.wbproxy/accounts.json (not world-readable).")
    print()
    access = input("accessToken: ").strip()
    if not access:
        print("nothing entered")
        return 1
    refresh = input("refreshToken (optional): ").strip()
    domain = input("domain [www.workbuddy.cn]: ").strip() or "www.workbuddy.cn"
    uid = input("uid (optional): ").strip()
    label = input("label [manual]: ").strip() or "manual"

    target = Path.home() / ".wbproxy" / "accounts.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"accounts": []}
    if target.exists():
        try:
            doc = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            pass
    doc.setdefault("accounts", []).append({
        "id": f"manual-{int(time.time())}",
        "label": label,
        "credential": {"accessToken": access, "refreshToken": refresh,
                       "domain": domain, "uid": uid},
    })
    target.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(target, 0o600)
    except Exception:
        pass
    print(f"\nwritten to {target}")
    print("check it with: python wbproxy.py doctor")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    accounts = load_accounts()
    if not accounts:
        log("no accounts found — run `wbproxy login`, or let the DSH xdpool "
            "plugin harvest them once")
        log("sources checked:")
        for p in account_sources():
            log(f"  {'exists' if p.exists() else 'absent'}  {p}")
        return 1
    log(f"{len(accounts)} account(s): "
        + ", ".join(f"{a.label}({a.region})" for a in accounts))
    if WBPROXY_REGION in ("cn", "global"):
        log(f"region pinned to {WBPROXY_REGION} (WBPROXY_REGION)")

    bridge = Bridge(accounts)
    Handler.bridge = bridge

    rows = bridge.models()
    unique = sorted({r["id"] for r in rows})
    per_region: dict[str, int] = {}
    for r in rows:
        per_region[r["region"]] = per_region.get(r["region"], 0) + 1
    log("catalog: " + ", ".join(f"{n} on {k}" for k, n in sorted(per_region.items()))
        + f" — {len(unique)} distinct id(s)")
    shared = sorted({r["id"] for r in rows
                     if sum(1 for x in rows if x["id"] == r["id"]) > 1})
    if shared:
        log(f"both regions serve: {', '.join(shared)}")
    live = bridge.pick()
    if live is None:
        log("WARNING: no account holds a usable token — requests will 503")
    else:
        log(f"first request goes to {live.label}")

    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    log(f"listening on http://{LISTEN_HOST}:{LISTEN_PORT}/v1")
    log(f"client base_url = http://{LISTEN_HOST}:{LISTEN_PORT}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("stopped")
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wbproxy",
        description="OpenAI-compatible proxy for WorkBuddy subscription accounts.")
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="serve the API (default)")
    p_serve.set_defaults(func=cmd_serve)

    p_models = sub.add_parser("models", help="list models with cost and context")
    p_models.add_argument("--tools", action="store_true",
                          help="only models that support tool calls")
    p_models.add_argument("--free", action="store_true",
                          help="only models with a zero credit multiplier")
    p_models.add_argument("--region", choices=("cn", "global"),
                          help="only models from this region")
    p_models.add_argument("--sort", choices=("cost", "context", "id"), default="id")
    p_models.add_argument("--refresh", action="store_true",
                          help="re-read the catalog instead of using the cache")
    p_models.set_defaults(func=cmd_models)

    p_doctor = sub.add_parser("doctor", help="check accounts, tokens and endpoints")
    p_doctor.set_defaults(func=cmd_doctor)

    p_login = sub.add_parser("login", help="add a credential by hand")
    p_login.set_defaults(func=cmd_login)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        args = parser.parse_args(["serve"])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

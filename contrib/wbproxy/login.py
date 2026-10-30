"""QR login for WorkBuddy, both regions.

Ported from dsh-xdpool's `vendor/channel-pack/src/buddy-auth.ts`
(`fetchAuthState` / `loopGetToken` / `getAccount` / `runBuddyLoginFlow`), with
the product tables copied out of `pack.js` rather than guessed.

WHY THIS EXISTS, and what it is not
----------------------------------
The desktop app's own credential files are AES-256-GCM envelopes keyed from the
OS keystore. Reading them means reimplementing that derivation per platform --
which is exactly the work `dsh-xdpool` already did, and which is why this
program reads *its* cache instead. That path needs DSH installed.

A scan login needs nobody's keystore. It is three HTTP calls, and the token it
returns is minted for us. So this module is the one way to add an account on a
machine with no DSH at all.

THE THREE CALLS (measured 2026-10-30 against both live hosts)
-------------------------------------------------------------
    POST {endpoint}/v2/plugin/auth/state?platform={platform}
        -> {"code":0,"msg":"OK","data":{"state":"<uuid>","authUrl":"..."}}
    GET  {endpoint}/v2/plugin/auth/token?state={state}        (poll, 1s)
        -> access_token once the person has scanned
    GET  {endpoint}/v2/plugin/login/account?state={state}
        -> nickname, credits, expiry

The `authUrl` that comes back is already complete: it carries `platform` and
`state` itself. xdpool's `decorateLoginUrl` appends `version` and a fresh
`loginSessionId` on top, but only for the international product
(`appendSessionParams: true`). That is reproduced here for the same reason
xdpool does it -- an extra parameter the server does not require is cheaper to
send than to be wrong about -- and it is skipped for CN because
`CODEBUDDY.appendSessionParams` is `false`.

A 400 measured on the way in, kept as a note so nobody re-learns it: omitting
`platform` for CN returns
    {"code":10001,"msg":"10001:platform is empty"}
`CODEBUDDY.platform` is `"ide"`, not absent -- an earlier draft of this table
had it as None and the first request failed.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

# -- product tables -----------------------------------------------------------
#
# Names are xdpool's: `buddy` is the CN product and `workbuddy` is the
# international one. They are NOT the same service behind two domains -- a token
# issued at one is rejected by the other -- so they are kept as two products,
# never as two settings on one.

AUTH_STATE_PATH = "/v2/plugin/auth/state"
AUTH_TOKEN_PATH = "/v2/plugin/auth/token"
LOGIN_ACCOUNT_PATH = "/v2/plugin/login/account"

HDR_DOMAIN = "X-Domain"
HDR_NO_AUTH = "X-No-Authorization"
HDR_NO_UID = "X-No-User-Id"
HDR_NO_EID = "X-No-Enterprise-Id"
HDR_NO_DEPT = "X-No-Department-Info"

#: Poll cadence and ceiling, from `buddy-auth.ts` (`POLL_INTERVAL_MS`,
#: `LOGIN_TIMEOUT_MS`). Five minutes is what a person takes to find a phone.
POLL_INTERVAL_S = 1.0
LOGIN_TIMEOUT_S = 5 * 60
REQUEST_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class Product:
    """One WorkBuddy product line. Two of them exist; neither serves the other."""

    id: str                 # `buddy` | `workbuddy`  (xdpool's provider ids)
    label: str
    endpoint: str
    api_domain: str
    platform: str
    user_agent: str
    append_session_params: bool
    credential_ref: str

    #: The region name this program's own settings already use.
    @property
    def aliases(self) -> tuple[str, ...]:
        return ("cn", "intl") if self.id == "buddy" else ("global", "intl")

    def chat_base(self) -> str:
        return self.endpoint


#: `CODEBUDDY` in pack.js. platform is "ide" -- see the module docstring for the
#: 400 that a missing platform earns.
CN_PRODUCT = Product(
    id="buddy",
    label="WorkBuddy 国内版",
    endpoint="https://copilot.tencent.com",
    api_domain="copilot.tencent.com",
    platform="ide",
    user_agent="CodeBuddyIDE/1.106.1",
    append_session_params=False,
    credential_ref="BUDDY_ACCESS_TOKEN",
)

#: `WORKBUDDY` in pack.js -- the international line.
INTL_PRODUCT = Product(
    id="workbuddy",
    label="WorkBuddy 国际版",
    endpoint="https://www.workbuddy.ai",
    api_domain="www.workbuddy.ai",
    platform="workbuddy-ai",
    user_agent="WorkBuddy/5.5.2 WorkBuddy AI/5.5.2 CLI/5.5.2",
    append_session_params=True,
    credential_ref="WORKBUDDY_ACCESS_TOKEN",
)

PRODUCTS: dict[str, Product] = {p.id: p for p in (CN_PRODUCT, INTL_PRODUCT)}

#: What a caller may type for `region`. xdpool calls the CN lane `buddy` and the
#: other `workbuddy`; this program's own CLI already said `cn` / `global`, so
#: every spelling anyone has used is accepted rather than one of them being
#: declared correct and the rest a bug report.
REGION_ALIASES: dict[str, str] = {
    "cn": CN_PRODUCT.id,
    "china": CN_PRODUCT.id,
    "buddy": CN_PRODUCT.id,
    "codebuddy": CN_PRODUCT.id,
    "intl": INTL_PRODUCT.id,
    "global": INTL_PRODUCT.id,
    "workbuddy": INTL_PRODUCT.id,
    "international": INTL_PRODUCT.id,
    "ai": INTL_PRODUCT.id,
}


def product_for(region: str) -> Product:
    """Resolve a loose region name to a product, or raise with the list."""
    key = (region or "").strip().lower()
    pid = REGION_ALIASES.get(key)
    if pid is None:
        raise KVError(
            "unknown region %r; use one of: %s"
            % (region, ", ".join(sorted(REGION_ALIASES)))
        )
    return PRODUCTS[pid]


def region_label(product: Product) -> str:
    """`cn` or `intl` -- the short name this program stores in settings."""
    return "cn" if product.id == CN_PRODUCT.id else "intl"


class KVError(RuntimeError):
    """A login step said no. Carries the reason, never a bare failure."""


# -- the three calls ----------------------------------------------------------


def _request(
    method: str,
    url: str,
    headers: dict[str, str],
    *,
    timeout: float = REQUEST_TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """One HTTP hop, returning (status, decoded envelope). Raises on no-network."""
    data = b"" if method == "POST" else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        status = exc.code
    except urllib.error.URLError as exc:
        raise KVError("could not reach %s: %s" % (url, exc.reason)) from exc
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        body = {"raw": raw[:400]}
    return status, body


def _data_of(body: dict[str, Any]) -> dict[str, Any]:
    """The envelope's `data`, or raise naming what came back instead."""
    data = body.get("data")
    if not isinstance(data, dict):
        code = body.get("code")
        msg = body.get("msg") or body.get("message") or ""
        raise KVError("upstream returned no data (code=%r msg=%r)" % (code, msg))
    return data


def fetch_auth_state(product: Product) -> tuple[str, str]:
    """Start a login. Returns (state, authUrl).

    `authUrl` is a plain web link, already carrying platform and state. The
    caller hands it to the person; nothing needs to render a QR image unless the
    caller chooses to.
    """
    url = "%s%s?platform=%s" % (product.endpoint, AUTH_STATE_PATH,
                                urllib.parse.quote(product.platform, safe=""))
    headers = {
        HDR_DOMAIN: product.api_domain,
        HDR_NO_AUTH: "true",
        HDR_NO_UID: "true",
        HDR_NO_EID: "true",
        HDR_NO_DEPT: "true",
        "User-Agent": product.user_agent,
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
    }
    status, body = _request("POST", url, headers)
    if status != 200:
        raise KVError("auth/state HTTP %d: %s"
                      % (status, body.get("msg") or body.get("message") or body))
    data = _data_of(body)
    state = str(data.get("state") or "")
    auth_url = str(data.get("authUrl") or "")
    if not state:
        raise KVError("auth/state returned no state field")
    if not auth_url:
        raise KVError("auth/state returned no authUrl field")
    return state, auth_url


def decorate_login_url(auth_url: str, product: Product) -> str:
    """Add the session parameters xdpool adds, for products that ask for them.

    `loginSessionId` is a fresh random id, not a constant: it is a marker for
    this attempt, and reusing a previous one would be indistinguishable from
    "the same login retried" if the server ever looks at it. Measured: the URL
    from `auth/state` works without these, so this only keeps parity with
    xdpool's international path.
    """
    if not product.append_session_params:
        return auth_url
    try:
        parts = urllib.parse.urlsplit(auth_url)
        query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        query.append(("loginSessionId", str(uuid.uuid4())))
        query.append(("version", str(uuid.uuid4())[:8]))
        return urllib.parse.urlunsplit((
            parts.scheme, parts.netloc, parts.path,
            urllib.parse.urlencode(query), parts.fragment,
        ))
    except Exception:                                        # noqa: BLE001
        return auth_url


def _parse_token(data: dict[str, Any]) -> dict[str, Any]:
    """The token envelope, in the shape this program stores."""
    access = data.get("accessToken") or data.get("access_token") or ""
    refresh = data.get("refreshToken") or data.get("refresh_token") or ""
    if not access:
        raise KVError("auth/token answered without an access token")
    return {
        "accessToken": str(access),
        "refreshToken": str(refresh),
        "domain": str(data.get("domain") or ""),
        "expiresAt": data.get("expiresAt") or data.get("expires_at") or 0,
    }


def poll_for_token(
    product: Product,
    state: str,
    *,
    timeout_s: float = LOGIN_TIMEOUT_S,
    interval_s: float = POLL_INTERVAL_S,
    stop: threading.Event | None = None,
) -> dict[str, Any]:
    """Poll `auth/token` until a token lands, the deadline passes, or `stop` set.

    A poll that fails is not a poll that ended: a dropped socket between two
    attempts is normal while somebody is finding their phone, so transport
    errors are swallowed and retried. Only a 200 without a token is treated as
    "still waiting", and only the deadline as failure.
    """
    url = "%s%s?state=%s" % (product.endpoint, AUTH_TOKEN_PATH,
                             urllib.parse.quote(state, safe=""))
    headers = {
        HDR_NO_AUTH: "true",
        HDR_DOMAIN: product.api_domain,
        "User-Agent": product.user_agent,
        "Accept": "application/json, text/plain, */*",
    }
    deadline = time.monotonic() + timeout_s
    last_note = "waiting for the scan"
    while time.monotonic() < deadline:
        if stop is not None and stop.is_set():
            raise KVError("login cancelled")
        time.sleep(interval_s)
        if stop is not None and stop.is_set():
            raise KVError("login cancelled")
        try:
            status, body = _request("GET", url, headers)
        except KVError as exc:
            last_note = str(exc)
            continue
        if status != 200:
            last_note = "auth/token HTTP %d" % status
            continue
        data = body.get("data")
        if not isinstance(data, dict):
            last_note = "auth/token 200 without data (code=%r)" % body.get("code")
            continue
        if not (data.get("accessToken") or data.get("access_token")):
            continue
        return _parse_token(data)
    raise KVError("login timed out after %.0fs (%s)" % (timeout_s, last_note))


def fetch_account(product: Product, state: str, token: dict[str, Any]) -> dict[str, Any]:
    """Read the account's own facts: nickname, credits, expiry.

    `X-Domain` prefers the token's own domain over the product's, because the
    server just told us which brand issued it: a `codebuddy.ai` token sent with
    a `workbuddy.ai` X-Domain is rejected, and the token is the authority on
    which one it is. This is the same `token.domain || product.apiDomain`
    fallback `getAccount` uses, including the empty-string case -- the upstream
    parser returns "" rather than None for a missing field, so a falsy check is
    the one that works.
    """
    url = "%s%s?state=%s" % (product.endpoint, LOGIN_ACCOUNT_PATH,
                             urllib.parse.quote(state, safe=""))
    domain = str(token.get("domain") or "") or product.api_domain
    headers = {
        HDR_DOMAIN: domain,
        "Authorization": "Bearer %s" % token.get("accessToken", ""),
        HDR_NO_UID: "true",
        HDR_NO_EID: "true",
        "User-Agent": product.user_agent,
        "Accept": "application/json, text/plain, */*",
    }
    deadline = time.monotonic() + LOGIN_TIMEOUT_S
    last_note = "waiting for the account"
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_S)
        try:
            status, body = _request("GET", url, headers)
        except KVError as exc:
            last_note = str(exc)
            continue
        if status != 200:
            last_note = "login/account HTTP %d" % status
            continue
        data = body.get("data")
        if not isinstance(data, dict):
            continue
        return data
    raise KVError("account read timed out (%s)" % last_note)


# -- the whole flow -----------------------------------------------------------


@dataclass
class LoginSession:
    """One attempt, in the three states a caller can act on.

    The service cannot block a request for five minutes while a person scans, so
    the flow is split: `start()` returns the URL immediately, and `wait()` is
    what a background thread calls. That is the only difference from
    `runBuddyLoginFlow`, which is one synchronous function because a browser
    window is already in front of the user there.
    """

    product: Product
    state: str
    auth_url: str
    started_at: float = field(default_factory=time.monotonic)
    #: Filled by `wait()`; until then the session is pending.
    credential: dict[str, Any] | None = None
    account: dict[str, Any] | None = None
    error: str = ""
    _stop: threading.Event = field(default_factory=threading.Event,
                                   repr=False, compare=False)

    @property
    def region(self) -> str:
        return region_label(self.product)

    @property
    def done(self) -> bool:
        return self.credential is not None or bool(self.error)

    def snapshot(self) -> dict[str, Any]:
        """What a status poll is told. Never includes the token itself."""
        out: dict[str, Any] = {
            "region": self.region,
            "product": self.product.id,
            "state": self.state,
            "authUrl": self.auth_url,
            "pending": not self.done,
            "elapsed_s": round(time.monotonic() - self.started_at, 1),
        }
        if self.error:
            out["error"] = self.error
        if self.account:
            out["account"] = {
                "nickname": self.account.get("nickname") or self.account.get("nickName"),
                "userId": self.account.get("userId") or self.account.get("uid"),
                "credits": self.account.get("credits"),
                "expiresAt": self.account.get("expiresAt"),
            }
        return out

    def cancel(self) -> None:
        self._stop.set()

    def wait(self) -> dict[str, Any]:
        """Block until the scan lands. Called from a worker, not a request."""
        try:
            token = poll_for_token(self.product, self.state, stop=self._stop)
            self.account = fetch_account(self.product, self.state, token)
            self.credential = token
            return token
        except KVError as exc:
            self.error = str(exc)
            raise
        finally:
            pass


def start(product: Product) -> LoginSession:
    """Begin a login and hand back the URL to show."""
    state, auth_url = fetch_auth_state(product)
    return LoginSession(product=product, state=state,
                        auth_url=decorate_login_url(auth_url, product))


def login_blocking(
    product: Product,
    *,
    open_url: Callable[[str], Any] | None = None,
    timeout_s: float = LOGIN_TIMEOUT_S,
) -> LoginSession:
    """The CLI's form: start, show the URL, wait. One call, one account.

    `open_url` is a convenience for a terminal that can launch a browser; the
    URL is printed either way, because a headless box has no browser and must
    still be able to complete this by pasting it somewhere else.
    """
    session = start(product)
    print("")
    print("  %s  (%s)" % (product.label, region_label(product)))
    print("  Open this and sign in:")
    print("")
    print("    %s" % session.auth_url)
    print("")
    if open_url is not None:
        try:
            open_url(session.auth_url)
        except Exception:                                    # noqa: BLE001
            pass
    print("  waiting for the scan (up to %.0fs) ..." % timeout_s)
    token = poll_for_token(product, session.state, timeout_s=timeout_s)
    session.credential = token
    try:
        session.account = fetch_account(product, session.state, token)
    except KVError as exc:
        # The token is the point; account facts are a nicety. A failed read here
        # must not throw away a credential that already works.
        session.error = "account read failed (%s); the token is still usable" % exc
    return session

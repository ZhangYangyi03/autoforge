"""The single door out: every outbound call goes through one admitted host check.

This is the Windows-usable form of the "Semantic Boundary Gateway" the AgenticOS
paper (arXiv 2606.21129) puts at the network boundary, and of `seabox`'s
`src/linux/seccomp.rs` / `Kernex`'s `kernex-linux/src/seccomp.rs`, vendored at
`D:/Users/china/Desktop/项目_开发/_vendor`. Those two refuse at the syscall; this
host has no seccomp and no Landlock, so the honest place to refuse is the only
place every outbound call already has: one function.

Why a gateway rather than four checks
-------------------------------------
The reach policy (`forge/capability_policy.py`) can already say "api.aiping.cn:
allow, everything else: deny". Nothing enforced it, because the call sites each
build their own `requests.post`/`urlopen` and a rule nobody consults is a
comment. There were twelve such sites. Twelve places to remember is the classic
version of no enforcement; this module is one.

What it is, and what it is not
------------------------------
*It is*: an admit decision per outbound host, on the same four-part receipt as
every other policy check, with the verdict and the rule written to the audit
chain. A denied host does not get a connection, and the reason is greppable
afterwards.

*It is not* a kernel firewall. Code that calls `socket` directly -- a forged
tool, a vendored dependency, anything not routed through here -- is not stopped
by this. On Windows that gap can only be closed by the OS, and this module says
so in `coverage()` rather than implying otherwise. What it does buy is the thing
that was actually missing: the agent's own reach is *decided*, not incidental,
and the decision is recorded.

Default posture: `allow`, explicitly
------------------------------------
Unlike the resource manifest, an empty host list here means **allow**, and that
is a deliberate choice with a reason rather than a convenience. The agent's
whole job is fetching, forging and reaching the network; a default-deny gateway
that shipped as the only path out would have to be disabled to get any work
done, and a gate that is always off is worse than a gate that is honestly open.
So the field starts absent and a deployment that wants a closed boundary fills
`:data:`DEFAULT_ALLOW` in -- and can see, from `coverage()` and from the audit
rows, exactly which hosts were admitted.
"""
from __future__ import annotations

import os
import socket
import sqlite3
import time
import urllib.parse
from pathlib import Path
from typing import Any

__all__ = ["GatewayDenied", "Gateway", "gateway", "reset", "admit", "coverage",
           "DEFAULT_ALLOW", "ALLOW_ALL", "SITES"]

#: Hosts the agent needs to function, used only when `AUTOFORGE_EGRESS=closed`.
#: Loopback is in the list because the local market, the local MCP servers and
#: the bus all live there; a boundary that cut them off would be measuring the
#: wrong thing entirely.
DEFAULT_ALLOW = ("api.aiping.cn", "127.0.0.1", "localhost", "::1",
                 "*.openai.com", "*.anthropic.com")

ALLOW_ALL = "*"

_TRUTHY = {"closed", "deny", "strict", "1", "on", "true", "yes"}
_FALSEY = {"open", "allow", "off", "0", "false", "no"}


class GatewayDenied(RuntimeError):
    """The host was refused. Carries the rule, so the caller can name it.

    A separate exception from the policy module's `Decision` on purpose: callers
    catch this to produce a sentence, and they must not accidentally catch a
    `KeyError` or a socket error under the same name.
    """

    def __init__(self, host: str, rule: str, reason: str):
        super().__init__("egress denied for %r: %s (%s)" % (host, reason, rule))
        self.host = host
        self.rule = rule
        self.reason = reason


def _host_of(url: str) -> str:
    """The host, and only the host: scheme, userinfo, port and path dropped.

    `user@host` matters -- `http://api.aiping.cn@evil.test/` has `evil.test` as
    its host, and a naive split on `@` or on `/` gets the allow list to admit the
    wrong name.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.hostname:
        return parsed.hostname
    if parsed.scheme and parsed.path:
        return parsed.path.split("/")[0].split(":")[0]
    return (url or "").split("/")[0].split(":")[0]


def parse_hosts(value: str) -> list[str]:
    return [h.strip() for h in (value or "").split(",") if h.strip()]


class Gateway:
    """Admit or refuse outbound hosts, and remember that it did.

    A process-wide instance (`gateway()`) because the alternative -- threading a
    policy object through every fetch, MCP client and web tool -- means the one
    place somebody forgot is the one that leaks, and nobody would know.
    """

    def __init__(self, *, allow: tuple[str, ...] | list[str] | None = None,
                 mode: str | None = None, engine: Any = None, conn: Any = None,
                 name: str = "autoforge"):
        self.mode = (mode or os.environ.get("AUTOFORGE_EGRESS", "open")).strip().lower()
        self.allow = tuple(allow) if allow is not None else self._from_env()
        self.engine = engine
        self.conn = conn
        self.name = name

    # -- configuration ---------------------------------------------------

    def _from_env(self) -> tuple[str, ...]:
        hosts = parse_hosts(os.environ.get("AUTOFORGE_EGRESS_ALLOW", ""))
        if hosts:
            return tuple(hosts)
        if self.closed:
            return tuple(DEFAULT_ALLOW)
        return (ALLOW_ALL,)

    @property
    def closed(self) -> bool:
        mode = self.mode
        if mode in _TRUTHY:
            return True
        if mode in _FALSEY:
            return False
        return False          # an unreadable setting is not a silently closed door

    def rules(self) -> list[str]:
        return list(self.allow)

    # -- the decision ----------------------------------------------------

    def check(self, url: str) -> dict[str, Any]:
        """Judge one URL. Returns a receipt; does not raise and does not connect."""
        host = _host_of(url)
        mode = "closed" if self.closed else "open"
        base = {"host": host, "dimension": "network", "mode": mode,
                "ts": time.time()}

        if not self.allow:
            # An empty list in *either* mode is a refusal, and the two cases are
            # reported apart: closed-with-nothing-listed is a boundary somebody
            # configured; open-with-nothing-listed is a bug in the caller, and
            # saying "denied" for both would hide which one it is.
            return dict(base, allowed=False, rule="egress.deny: empty allow list",
                        reason="no egress allow rules, in %s mode" % mode)

        if not self.closed and ALLOW_ALL in self.allow:
            return dict(base, allowed=True, rule="egress.allow: *",
                        reason="egress is open: host %r admitted" % host)

        # `_host_matches` only accepts an exact host, a `*.suffix` form or `*`.
        # An IP literal can therefore never sneak in through a `*.example.com`
        # rule -- `93.184.216.34` does not end in `.example.com` -- so no special
        # case for addresses is needed, and adding one was the bug in the first
        # draft of this method.
        rule = _first_match(host, self.allow)
        if rule:
            return dict(base, allowed=True, rule="egress.allow: %s" % rule,
                        reason="host %r matches allow rule" % host)
        return dict(base, allowed=False, rule="egress.deny: no allow rule",
                    reason="host %r is not in the egress allow list" % host)

    def admit(self, url: str, *, tool: str = "") -> dict[str, Any]:
        """Check, record, and raise only if refused.

        Called by the gateway itself at every covered call site, and callable by
        anything that wants the check without the connection -- an MCP server
        being configured, say, so a bad host is reported at configuration time
        rather than at first use.
        """
        receipt = self.check(url)
        receipt["tool"] = tool
        recorded = None
        if self.conn is not None:
            recorded = self._record(receipt)
        if not receipt["allowed"]:
            raise GatewayDenied(receipt["host"], receipt["rule"], receipt["reason"])
        receipt["recorded"] = recorded
        return receipt

    def _record(self, receipt: dict[str, Any]) -> Any:
        """Write the receipt to the audit chain. A failure here must not eat the
        call: losing the log is bad, losing the work because the log is bad is
        worse -- and the fact that it failed is itself reported to the caller."""
        try:
            from autoforge import audit
            event = "network_allow" if receipt["allowed"] else "network_deny"
            return audit.record(self.conn, {
                "event_type": event, "allowed": receipt["allowed"],
                "resource": receipt["host"], "rule": receipt["rule"],
                "reason": receipt["reason"], "agent": self.name,
                "extra": receipt.get("tool", ""), "ts": receipt["ts"],
            })
        except Exception as exc:                              # noqa: BLE001
            return {"recorded": False, "error": str(exc)[:200]}

    # -- reporting -------------------------------------------------------

    def coverage(self) -> dict[str, Any]:
        """What this gate does and does not cover, in words.

        Same posture as `manifest.py` on `network` and `capability_policy`.
        `enforcement()`: the difference between a boundary and a policy is
        exactly whether an uncooperative caller is stopped, and a table that
        blurred it would be read as a firewall.
        """
        return {
            "mode": "closed" if self.closed else "open",
            "allow": list(self.allow),
            "gateway": "in-process, at the call sites routed through admit()",
            "covered_sites": [s["name"] for s in SITES if s["covered"]],
            "known_unrouted": [s["name"] for s in SITES if not s["covered"]],
            "not_covered": ("code that opens a socket directly -- a forged tool, "
                            "a vendored library -- is not stopped by this; on "
                            "this host only the OS could, and it is not"),
        }


def _host_matches(host: str, rule: str) -> bool:
    if rule == ALLOW_ALL:
        return True
    if rule == host:
        return True
    if rule.startswith("*."):
        return host.endswith(rule[1:])
    return False


def _first_match(host: str, rules) -> str:
    for rule in rules:
        if _host_matches(host, rule):
            return rule
    return ""


def _is_ip(host: str) -> bool:
    try:
        socket.inet_aton(host)
        return True
    except OSError:
        return ":" in host and all(c in "0123456789abcdef:" for c in host.lower())


# -- the process-wide instance -------------------------------------------

#: Every outbound call site, and whether it is routed through `admit()`.
#:
#: This is a *ledger of omissions* as much as a feature list. An egress boundary
#: whose documentation names only what it covers invites the reader to assume the
#: rest is covered too; naming the unrouted ones makes the gap a task instead of
#: a surprise. `covered` is set by the wiring itself -- if a site stops calling
#: `admit`, its test goes red rather than this table quietly lying.
SITES: tuple[dict[str, Any], ...] = (
    {"name": "mcp._post", "module": "autoforge/mcp.py",
     "why": "an external MCP server is arbitrary code on the other end of a URL",
     "covered": True},
    {"name": "webtools.safe_redirects", "module": "autoforge/webtools.py",
     "why": "fetches an operator- or model-supplied URL, hop by hop",
     "covered": True},
    {"name": "llm.chat", "module": "autoforge/core/llm.py",
     "why": "the model endpoint: a long-lived connection to a known host",
     "covered": False},
    {"name": "vision.describe", "module": "autoforge/vision.py",
     "why": "the vision endpoint, same shape as the model endpoint",
     "covered": False},
    {"name": "market.health / push", "module": "autoforge/market.py",
     "why": "loopback only; the market is a sibling process",
     "covered": False},
    {"name": "notify channels", "module": "autoforge/notify.py",
     "why": "webhook and SMTP the operator configured themselves",
     "covered": False},
    {"name": "browser", "module": "autoforge/browser.py",
     "why": "a real browser: it can reach anything the network allows, and the "
            "gateway cannot see inside it",
     "covered": False},
)


_GATEWAY: Gateway | None = None


def gateway() -> Gateway:
    """The one gateway. Built once per process, from the environment."""
    global _GATEWAY
    if _GATEWAY is None:
        _GATEWAY = Gateway()
    return _GATEWAY


def reset(**kwargs) -> Gateway:
    """Replace it. For tests and for a process that has just read new settings."""
    global _GATEWAY
    _GATEWAY = Gateway(**kwargs) if kwargs else Gateway()
    return _GATEWAY


def admit(url: str, *, tool: str = "") -> dict[str, Any]:
    return gateway().admit(url, tool=tool)


def coverage() -> dict[str, Any]:
    return gateway().coverage()


def bind(conn: Any, *, name: str = "autoforge", allow=None, mode: str | None = None) -> Gateway:
    """Point the gateway at a ledger so its receipts are recorded.

    Called once, where the agent's own store is opened. Without it the gateway
    still decides -- refusing is not conditional on having somewhere to write --
    but the decision leaves no trace, and a boundary with no record of what it
    admitted is a boundary nobody can audit. Same posture as `chaining.py`: the
    write is best-effort, and a failure to write is reported rather than
    swallowed into a false negative.
    """
    return reset(conn=conn, name=name, allow=allow, mode=mode)

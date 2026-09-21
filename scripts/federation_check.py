"""Is a peer's shelf actually usable from here -- and is the forge looking at it?

The one command to run on the other machine after pointing at this one. It
answers, in order, the four questions that are easy to confuse with each other:

  1. is the peer reachable at all          (TCP, because HTTP errors and a dead
                                            host look identical in a log)
  2. does its shelf answer /resources      (and how many entries it carries)
  3. does /search find a tool for a need   (the endpoint the forge pre-lookup
                                            actually asks; a shelf with no
                                            /search still works, via /resources)
  4. does the AGENT's pre-forge lookup see it -- with the ledger lines the
     lookup wrote, so "nothing there" and "I never asked" cannot be confused

That last one is the point. Checks 1-3 say the peer is up; only check 4 says the
two machines are one market. A peer can pass 1-3 and still be invisible to the
forge because AUTOFORGE_PEER_MARKETS was set in a shell that has since closed.

Read-only against the peer: it searches and reads, it never posts. Invoking is
attempted only when asked for with --invoke.

Usage
    python scripts/federation_check.py --peer http://192.168.1.108:8000
    python scripts/federation_check.py --peer http://192.168.1.107:8077/market \
        --token-file C:\\Users\\china\\autoforge_node\\node.token
    python scripts/federation_check.py --peer ... --need "convert seconds to HH:MM:SS" --invoke
    python scripts/federation_check.py            # use AUTOFORGE_PEER_MARKETS
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

OK, BAD, WARN = "  ok ", " FAIL", " warn"


def _parse_host(url: str):
    u = urllib.parse.urlsplit(url)
    host = u.hostname or ""
    port = u.port or (443 if u.scheme == "https" else 80)
    return host, port


def reachable(url: str, timeout: float = 6.0) -> tuple[bool, str]:
    host, port = _parse_host(url)
    if not host:
        return False, "no host in url"
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True, f"TCP {host}:{port} open"
    except Exception as exc:                                   # noqa: BLE001
        return False, f"TCP {host}:{port} {type(exc).__name__}: {exc}"


def get(url: str, token: str | None, path: str, timeout: float = 15.0):
    req = urllib.request.Request(
        url.rstrip("/") + path,
        headers={"User-Agent": "autoforge-federation-check/1.0",
                 **({"Authorization": "Bearer " + token} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:200]
    except Exception as exc:                                   # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def post(url: str, token: str | None, path: str, payload: dict, timeout: float = 30.0):
    req = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "autoforge-federation-check/1.0",
                 **({"Authorization": "Bearer " + token} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:200]
    except Exception as exc:                                   # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer", help="peer shelf base URL (or label=url)")
    ap.add_argument("--token-file", help="file holding the bearer token, if the peer needs one")
    ap.add_argument("--need", default="convert a number of seconds into HH:MM:SS",
                    help="the need to test the pre-forge lookup with")
    ap.add_argument("--invoke", action="store_true", help="also call the top hit")
    a = ap.parse_args()

    peer = a.peer
    if not peer:
        configured = [e for e in os.environ.get("AUTOFORGE_PEER_MARKETS", "").split(",") if e.strip()]
        if not configured:
            print("no --peer and AUTOFORGE_PEER_MARKETS is empty:")
            print("  set AUTOFORGE_PEER_MARKETS=\"kos=http://192.168.1.108:8000\"")
            return 2
        peer = configured[0]
    label, _, url = peer.partition("=") if "=" in peer else (peer, "", peer)
    url = url.split("|FILE:", 1)[0].rstrip("/")

    token = None
    if a.token_file:
        try:
            token = open(a.token_file, encoding="utf-8").read().strip() or None
        except OSError as exc:
            print(f"{WARN} token file unreadable: {exc} (trying without it)")
        if token:
            print(f"  ok  bearer token loaded from {a.token_file}")

    print(f"\npeer: {label}  ->  {url}\n")
    failures = 0

    good, why = reachable(url)
    print(f"{OK if good else BAD} reachability: {why}")
    failures += 0 if good else 1
    if not good:
        print("\nthe peer is not answering: it is switched off, firewalled, or the")
        print("address is wrong. Nothing below can be concluded from this.")
        return 1

    st, body = get(url, token, "/health")
    print(f"{OK if st == 200 else BAD} /health: {st} {body[:140]}")
    failures += 0 if st == 200 else 1

    st, body = get(url, token, "/resources")
    count = None
    if st == 200:
        try:
            d = json.loads(body)
            items = d.get("resources") if isinstance(d, dict) else d
            count = len(items or [])
        except ValueError:
            count = None
    print(f"{OK if count is not None else BAD} /resources: {st} entries={count}")
    failures += 0 if count is not None else 1

    qs = urllib.parse.urlencode({"q": a.need, "k": 5})
    # Generous on purpose: a cold hybrid search with the rerank measures ~22s
    # on this shelf. A short timeout here reports a working endpoint as absent,
    # which is the confusion this script exists to remove.
    t0 = time.time()
    st, body = get(url, token, "/search?" + qs, timeout=90)
    elapsed = time.time() - t0
    hits, confident = [], None
    if st == 200:
        try:
            d = json.loads(body)
            hits = [h.get("name") for h in (d.get("results") or [])]
            conf = d.get("confidence") or {}
            confident = conf.get("confident") if isinstance(conf, dict) else None
        except ValueError:
            pass
    note = "" if st == 200 else " (an older shelf may have no /search; /resources still counts)"
    if st == 200 and elapsed > 10:
        note += f"  [slow: {elapsed:.1f}s -- cold rerank, not a failure]"
    print(f"{OK if st == 200 else WARN} /search: {st} {elapsed:.1f}s"
          f" confident={confident} hits={hits[:4]}{note}")

    if a.invoke and hits:
        st, body = post(url, token, f"/resources/tool:{hits[0]}/invoke", {"arguments": {}})
        print(f"{OK if st and st < 400 else WARN} invoke {hits[0]}: {st} {body[:180]}")

    print("\n--- the question that matters: does the AGENT see this peer? ---")
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from autoforge.agent import ForgeAgent
    except Exception as exc:                                   # noqa: BLE001
        print(f"{WARN} cannot import ForgeAgent from here ({exc}); skipping check 4")
        return 0 if not failures else 1

    class _Probe(ForgeAgent):
        def __init__(self):
            self.trace = []

        def _record(self, kind, payload=None):
            self.trace.append({"kind": kind, **(payload or {})})

    env_markets = os.environ.get("AUTOFORGE_PEER_MARKETS", "")
    os.environ["AUTOFORGE_PEER_MARKETS"] = (
        f"{label}={url}" + ("|FILE:" + a.token_file if a.token_file else ""))
    os.environ.setdefault("TOOLMARKET_URL", "http://127.0.0.1:8000")
    probe = _Probe()
    verdict = probe._prelookup_market(a.need)
    for ev in probe.trace:
        if ev["kind"] in ("market_peer_lookup", "market_prelookup", "market_semantic_lookup"):
            print("    ledger:", json.dumps(ev, ensure_ascii=False)[:190])
    print(f"{OK if verdict else BAD} verdict: {verdict or '(no answer -- the forge would not be licensed)'}")
    if not env_markets:
        print(f"{WARN} AUTOFORGE_PEER_MARKETS was NOT set before this ran; the check set it")
        print("       in-process only. Set it permanently or the forge will not see the peer:")
        print(f'       setx AUTOFORGE_PEER_MARKETS "{label}={url}"')
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

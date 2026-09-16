"""Push to GitHub from this machine, which needs the direct-IP route.

`git push` here fails more often than it succeeds: github.com resolves to
20.205.243.166, which times out, while 140.82.112.3/113.3/114.3/116.4/112.4
answer and reset a plain connection partway through the pack. Twice today a
push was reported as a failed command when what had actually happened was a
dropped connection after the objects were already sent.

So this wraps the retry that worked:

  * pick a GitHub edge IP that actually accepts a connection right now, rather
    than trusting DNS;
  * forward it through a loopback CONNECT proxy we run ourselves, because the
    sandbox has no ambient proxy and git needs a URL;
  * fetch first, then push -- a push rejected as non-fast-forward is a
    different failure from a network one, and the difference matters;
  * retry the push, and afterwards compare local HEAD with origin's, because
    "the command failed" and "the commit is not on the remote" are not the same
    claim.

    python scripts/github_push.py --repo <path> [--branch main] [--remote origin]
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import time

CANDIDATES = ("140.82.112.3", "140.82.113.3", "140.82.114.3",
              "140.82.116.4", "140.82.112.4")


def reachable(ip: str, port: int = 443, timeout: float = 6.0) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def pick_ip(port: int = 443) -> str | None:
    for ip in CANDIDATES:
        if reachable(ip, port):
            return ip
    return None


def start_proxy(upstream: tuple[str, int], port: int = 8899):
    """A minimal CONNECT forwarder on loopback. Returns (stop_event, actual_port)."""
    import select

    stop = threading.Event()

    def pump(a, b):
        try:
            while not stop.is_set():
                r, _, _ = select.select([a], [], [], 1.0)
                if not r:
                    continue
                data = a.recv(65536)
                if not data:
                    break
                b.sendall(data)
        except OSError:
            pass
        finally:
            for s in (a, b):
                try:
                    s.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

    def handle(client):
        """One connection, start to finish, on its own thread.

        The first version did the CONNECT handshake inline in the accept loop,
        and a single idle client -- git leaves one behind after a fetch -- parked
        the loop inside recv() so no later connection was ever accepted. git
        reported that as "Proxy CONNECT aborted", which reads like the proxy
        refusing when in fact it had stopped listening. A handshake is a wait,
        and a wait must not be on the accept path.
        """
        try:
            client.settimeout(15)
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = client.recv(4096)
                if not chunk:
                    return
                head += chunk
            if not head.startswith(b"CONNECT"):
                return
            up = socket.create_connection(upstream, timeout=15)
            client.settimeout(None)
            client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            threading.Thread(target=pump, args=(client, up), daemon=True).start()
            threading.Thread(target=pump, args=(up, client), daemon=True).start()
        except OSError:
            pass
        finally:
            if not head or not head.startswith(b"CONNECT"):
                try:
                    client.close()
                except OSError:
                    pass

    def serve(srv):
        while not stop.is_set():
            try:
                client, _ = srv.accept()
            except OSError:
                break
            threading.Thread(target=handle, args=(client,), daemon=True).start()

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(16)
    threading.Thread(target=serve, args=(srv,), daemon=True).start()
    return stop, srv.getsockname()[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--port", type=int, default=8899)
    a = ap.parse_args()

    ip = pick_ip()
    if ip is None:
        print("NO ROUTE: none of the GitHub edge IPs accepted a connection.")
        return 2
    print("using %s:443" % ip)
    stop, port = start_proxy((ip, 443), a.port)
    G = ["git", "-c", "http.version=HTTP/1.1",
         "-c", "http.proxy=http://127.0.0.1:%d" % port]
    local = subprocess.run(["git", "rev-parse", "HEAD"], cwd=a.repo,
                           capture_output=True, text=True).stdout.strip()
    remote_sha, checked = "", False
    try:
        f = subprocess.run(G + ["fetch", a.remote], cwd=a.repo,
                           capture_output=True, text=True, timeout=90)
        print("fetch rc", f.returncode, f.stderr.strip()[-160:].replace("\n", " "))
        rc = 1
        for attempt in range(3):
            p = subprocess.run(G + ["push", a.remote, a.branch], cwd=a.repo,
                               capture_output=True, text=True, timeout=120)
            print("push %d rc=%d %s" % (attempt + 1, p.returncode,
                                        p.stderr.strip()[-160:].replace("\n", " ")))
            rc = p.returncode
            if rc == 0:
                break
            time.sleep(4)

        # Ask the remote, THROUGH THE PROXY, and ask before the proxy is shut
        # down. The first version checked after stop.set(), so the comparison
        # used a direct connection that this machine cannot make -- it timed
        # out, returned an empty list, and the tool announced "NOT on the
        # remote" for a commit that had in fact landed. A verification step
        # that can fail for its own reasons is worse than none: it manufactures
        # a false alarm and teaches the operator to ignore the real one.
        ls = subprocess.run(G + ["ls-remote", a.remote, "refs/heads/" + a.branch],
                            cwd=a.repo, capture_output=True, text=True, timeout=60)
        remote_sha = ls.stdout.split()[0] if ls.stdout.split() else ""
        checked = bool(remote_sha)
    finally:
        stop.set()

    print("local  HEAD : %s" % local[:12])
    print("remote %-5s: %s" % (a.branch, remote_sha[:12] or "(could not be read)"))
    if not checked:
        print("RESULT: UNKNOWN -- the remote could not be reached to confirm. "
              "The push exit code above is not evidence either way.")
        return 3
    same = local == remote_sha
    print("RESULT: %s" % ("the commit is on the remote" if same else
                          "NOT on the remote -- the push did not land"))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())

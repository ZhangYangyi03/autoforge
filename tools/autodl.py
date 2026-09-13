"""Local driver: push a file to the AutoDL box and run a command.

Credentials come from the env, never from this file:
    AUTODL_HOST / AUTODL_IP / AUTODL_PORT / AUTODL_USER / AUTODL_PASS

AutoDL's DNS is flaky, so we dial the resolved IP but keep the hostname for
the SNI/known-hosts name (see autodl-dns-bypass).
"""
from __future__ import annotations

import os
import posixpath
import sys

import paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.bjb1.seetacloud.com")
IP = os.environ.get("AUTODL_IP", "123.127.15.155")
PORT = int(os.environ.get("AUTODL_PORT", "20552"))
USER = os.environ.get("AUTODL_USER", "root")
PASS = os.environ.get("AUTODL_PASS", "")

if not PASS:
    sys.exit("AUTODL_PASS is not set")


def connect(timeout: int = 60) -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(IP, port=PORT, username=USER, password=PASS,
              timeout=timeout, banner_timeout=timeout, auth_timeout=timeout,
              look_for_keys=False, allow_agent=False)
    return c


def run(cmd: str, timeout: int = 900) -> tuple[int, str, str]:
    c = connect()
    try:
        _, out, err = c.exec_command(cmd, timeout=timeout)
        o = out.read().decode("utf-8", "replace")
        e = err.read().decode("utf-8", "replace")
        return out.channel.recv_exit_status(), o, e
    finally:
        c.close()


def push(local: str, remote_dir: str = "/root/labs/autoforge_gpu") -> str:
    """SFTP a local file to the box, return the remote path."""
    name = posixpath.basename(local)
    remote = posixpath.join(remote_dir, name)
    c = connect()
    try:
        _, out, _ = c.exec_command(f"mkdir -p {remote_dir}", timeout=60)
        out.channel.recv_exit_status()
        sftp = c.open_sftp()
        sftp.put(local, remote)
        sftp.close()
        return remote
    finally:
        c.close()


def main() -> None:
    args = sys.argv[1:]
    timeout = 900
    if args and args[0] == "--timeout":
        timeout = int(args[1])
        args = args[2:]
    if args and args[0] == "--push":
        for f in args[1:]:
            print("pushed:", push(f))
        return
    rc, o, e = run(" ".join(args), timeout)
    sys.stdout.write(o)
    if e:
        sys.stderr.write(e)
    sys.exit(rc)


if __name__ == "__main__":
    main()

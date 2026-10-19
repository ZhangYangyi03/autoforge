"""Install REA as an autoforge-bundled MCP server, and make its Java side work.

REA (rea-agents) is a reverse-engineering MCP server that drives Ghidra, Hopper,
IDA and jadx. It is not vendored here -- it is npm-installed, because it is a
large third-party tool whose own updater should own it. What this script does is
the part that is specific to this machine:

  1. Build a private JDK home (`java-home/`) that is a full copy of a real JDK 21
     with a shim at `bin/java.exe`.

     Why a shim: HotSpot's default heap is a quarter of physical RAM, and on a
     machine whose commit charge is already high that reservation fails with
     Win32 errno 1455 *before main() runs*. Every JVM launch that names no -Xmx
     dies at startup -- which is exactly how Ghidra's LaunchSupport helper and
     REA's own Java probe launch. REA blanks JAVA_TOOL_OPTIONS / _JAVA_OPTIONS /
     JDK_JAVA_OPTIONS before probing, so no environment variable can reach them;
     with the probe dead, REA reports `runtime_unavailable` and refuses to use
     the ghidra provider at all. The shim is the only place left to cap the heap,
     and it lives inside the JDK home so `java.home` stays correct and Ghidra's
     own version check still passes.

  2. Register the server in `~/.autoforge/config.json` under `mcp.servers.rea`,
     with GHIDRA_HEADLESS_MAXMEM bounded.

     REA launches every provider through a Windows Job Object with a 1 GiB
     committed-memory limit per owned process tree, and Ghidra's launcher
     defaults to -Xmx2g, which cannot be committed under that cap. 1 GiB is what
     REA imposes and it is a fixed number in its source
     (`nativeBound: { memoryLimitBytes: 1024 ** 3 }`), not a setting, so the heap
     is the only lever: GHIDRA_HEADLESS_MAXMEM=1024m, and the analysis JVM's real
     ceiling is the host's own commit headroom underneath it.

  3. Refuse to run the analysis inside autoforge's own sandbox.

     This is the one that cost the most time, so it is written down plainly.
     autoforge's containment (`forge/containment.py`) puts every spawned child in
     a Windows Job Object with a 512 MiB commit limit. A JVM of any usefulness
     cannot live in 512 MiB -- measured, a bare allocation loop inside that job
     tops out at 480 MiB -- so Ghidra dies with OutOfMemoryError or errno 1455 no
     matter how the heap is tuned, and the failure surfaces as
     "Ghidra exited before bridge startup". The Job Object is created without
     BREAKAWAY_OK, so a child cannot leave it (CreateProcess with
     CREATE_BREAKAWAY_FROM_JOB returns ERROR_ACCESS_DENIED, measured).

     The consequence: the MCP server must be started by the agent process itself,
     not from inside run_python or a forged tool. Started from outside the job,
     a full analysis of C:/Windows/System32/where.exe completes in ~70 s.

  4. Leave an honest report of what was verified.

Run:  python scripts/rea/install_rea.py
      python scripts/rea/install_rea.py --check      (verify only, change nothing)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIM_SRC = HERE / "java_shim.c"
JAVA_HOME = Path.home() / "AppData" / "Local" / "autoforge" / "jdk-hl"
CONFIG = Path.home() / ".autoforge" / "config.json"
NODE = Path("C:/Program Files/nodejs/node.EXE")
REA_MJS = Path.home() / "AppData" / "Roaming" / "npm" / "node_modules" / "rea-agents" / "scripts" / "rea.mjs"
GHIDRA = Path("D:/ProgramFiles/ghidra_12.1.4_PUBLIC")
MAXMEM = "1024m"
TOOLS = ("java", "javac", "jar", "javaw")


def find_real_jdk() -> Path | None:
    """A real 64-bit full JDK 21+ to copy the shim into."""
    for root in (Path("C:/Program Files/Java"), Path("C:/Program Files/Eclipse Adoptium"),
                 Path("C:/Program Files/Microsoft")):
        if not root.is_dir():
            continue
        for cand in sorted(root.iterdir(), reverse=True):
            if (cand / "bin" / "javac.exe").exists() and (cand / "lib" / "modules").exists():
                return cand
    jh = os.environ.get("JAVA_HOME")
    if jh and (Path(jh) / "bin" / "javac.exe").exists():
        return Path(jh)
    return None


def build_java_home(real: Path, force: bool = False) -> str:
    """Copy the JDK beside the shim, compile the shim, put it at bin/java.exe."""
    if force and JAVA_HOME.exists():
        shutil.rmtree(JAVA_HOME)
    shutil.copytree(real, JAVA_HOME, dirs_exist_ok=True)
    bin_dir = JAVA_HOME / "bin"
    for name in TOOLS:
        src = real / "bin" / f"{name}.exe"
        if src.exists():
            shutil.copy2(src, bin_dir / f"real_{name}.exe")
    # The launcher finds the JVM at <java.home>/bin/server/jvm.dll.
    for sub in ("server", "client"):
        if (real / "bin" / sub).is_dir():
            shutil.copytree(real / "bin" / sub, bin_dir / sub, dirs_exist_ok=True)
    gcc = shutil.which("gcc")
    if not gcc:
        raise SystemExit("gcc is not on PATH; it is needed to build the java shim")
    for name in TOOLS:
        out = bin_dir / f"{name}.exe"
        subprocess.run([gcc, "-O2", "-municode", "-o", str(out), str(SHIM_SRC)],
                       check=True, capture_output=True)
    return str(JAVA_HOME)


def probe_java() -> str:
    """The exact command REA's probe runs; must print java.version and os.arch."""
    exe = JAVA_HOME / "bin" / "java.exe"
    r = subprocess.run(
        [str(exe), "-Dstdout.encoding=UTF-8", "-Dstderr.encoding=UTF-8",
         "-XX:-UsePerfData", "-XshowSettings:properties", "-version"],
        capture_output=True, text=True, errors="ignore", timeout=120)
    text = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "java.version = " not in text:
        raise SystemExit(f"java probe failed (rc={r.returncode}): {text[:400]}")
    version = [l.split("=", 1)[1].strip() for l in text.splitlines() if l.strip().startswith("java.version ")][0]
    return version


def register(enable: bool = True) -> dict:
    cfg: dict = {}
    if CONFIG.exists():
        try:
            cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        except ValueError:
            cfg = {}
    servers = cfg.setdefault("mcp", {}).setdefault("servers", {})
    servers["rea"] = {
        "command": str(NODE),
        "args": [str(REA_MJS), "--mcp"],
        "env": {
            "GHIDRA_INSTALL_DIR": str(GHIDRA),
            "JAVA_HOME": str(JAVA_HOME),
            # Bounded: REA's per-provider Job Object is 1 GiB of committed
            # memory, and Ghidra's launcher default of 2G cannot fit inside it.
            "GHIDRA_HEADLESS_MAXMEM": MAXMEM,
        },
        "enabled": enable,
        "timeout": 120,
        "call_timeout": 900,
    }
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return servers["rea"]


def in_sandbox_job() -> bool:
    """Whether this process is inside a commit-limited Windows Job Object.

    True means: do not start REA from here. See the module docstring, point 3.
    """
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes
    flag = wintypes.DWORD(0)
    ctypes.windll.kernel32.IsProcessInJob(
        ctypes.windll.kernel32.GetCurrentProcess(), None, ctypes.byref(flag))
    return bool(flag.value)


def doctor() -> str:
    env = dict(os.environ)
    env.update({"GHIDRA_INSTALL_DIR": str(GHIDRA), "JAVA_HOME": str(JAVA_HOME),
                "GHIDRA_HEADLESS_MAXMEM": MAXMEM})
    for k in ("_JAVA_OPTIONS", "JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS"):
        env.pop(k, None)
    r = subprocess.run([str(NODE), str(REA_MJS), "doctor"], capture_output=True,
                       text=True, errors="ignore", env=env, timeout=600)
    return r.stdout or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, change nothing")
    ap.add_argument("--force", action="store_true", help="rebuild the private JDK home")
    args = ap.parse_args()

    if args.check:
        if not (JAVA_HOME / "bin" / "java.exe").exists():
            print("java home: MISSING", JAVA_HOME)
            return 1
        print("java home:", JAVA_HOME, "version", probe_java())
        out = doctor()
        for line in out.splitlines():
            if "java" in line or "healthy" in line or "ghidra" in line.lower():
                print(" ", line.strip())
        return 0

    real = find_real_jdk()
    if real is None:
        print("no real JDK 21+ found; set JAVA_HOME to one and retry", file=sys.stderr)
        return 1
    print("real JDK:", real)
    print("java home:", build_java_home(real, force=args.force))
    print("java probe: version", probe_java())
    entry = register()
    print("registered mcp.servers.rea with GHIDRA_HEADLESS_MAXMEM=%s" % entry["env"]["GHIDRA_HEADLESS_MAXMEM"])
    if in_sandbox_job():
        print("\nNOTE: this shell is inside autoforge's sandbox Job Object (512 MiB commit).")
        print("      Do not analyse from here -- start the agent itself instead; see the docstring.")
    print("\nrea doctor (java + ghidra lines):")
    for line in doctor().splitlines():
        if "java" in line or "ghidra" in line.lower():
            print(" ", line.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

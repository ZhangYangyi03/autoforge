# REA as a bundled autoforge component

REA (`rea-agents`) is a reverse-engineering MCP server: 207 tools over Ghidra,
jadx, Frida, rizin, radare2, objdump, gdb, pwn tools, Android/ADB, Electron and
browser analysis. It is installed with npm rather than vendored, because it is a
large third-party tool whose own updater should own it -- what lives here is the
machine-specific glue, and there is a lot of it.

## Install

```
npm install -g rea-agents
python scripts/rea/install_rea.py
```

`install_rea.py` builds a private JDK home and registers
`mcp.servers.rea` in `~/.autoforge/config.json`. Re-run it whenever the JDK or
Ghidra moves.

## Why there is a java.exe shim

HotSpot's ergonomic heap is a quarter of physical RAM, and its initial heap is
RAM/64. On this host (14 GiB, commit charge usually 25 of 28 GiB) the first
reservation fails with Win32 errno 1455 before `main()` runs, so every JVM that
names no `-Xmx` dies at startup. Two launches matter:

```
java.exe -XshowSettings:properties -version          # REA's Java probe
java.exe -cp LaunchSupport.jar LaunchSupport ...     # Ghidra's launcher helper
```

REA blanks `JAVA_TOOL_OPTIONS`, `_JAVA_OPTIONS` and `JDK_JAVA_OPTIONS` before it
probes (`ghidraJavaEnvironment`), so no environment variable can reach them, and
the probe then reports `runtime_unavailable` and REA refuses the ghidra provider
outright.

So `java-home/` is a full copy of a real JDK 21 with `bin/java.exe` replaced by a
small C shim (`java_shim.c`) that prepends `-Xms16m -Xss512k -XX:+UseSerialGC
-XX:ReservedCodeCacheSize=48m` and execs `real_java.exe`. Because the copy is a
complete JDK home, `java.home` resolves to itself and Ghidra's version check still
passes.

No `-Xmx` is set in the shim on purpose. REA's native launcher creates its Job
Object with breakaway forbidden -- `Job Object policy permits unexpected
breakaway` is a hard error in `rea-windows-x64.node` -- so a shim cannot lift the
JVM out of the 1 GiB cap, and a heap ceiling here would only fight the `-Xmx` that
Ghidra passes. What the shim supplies is the part Ghidra never sets.

Metaspace is deliberately uncapped: Ghidra reserves ~112 MiB of class space
before analysis starts, so any `-XX:MaxMetaspaceSize` below 128m makes the next
commit fail with `mmap failed to map 65536 bytes for Failed to commit metaspace`,
which reads as "out of memory" and is really "we capped the wrong pool".

## The limit that took the longest: autoforge's own sandbox job

autoforge's containment (`forge/containment.py`) assigns every spawned child to a
Windows Job Object with a 512 MiB commit limit. Measured, an allocation loop
inside that job tops out at **480 MiB**. A JVM that can run Ghidra does not fit,
so the analysis dies at import with

```
# Native memory allocation (malloc) failed to allocate 193936 bytes for Chunk::new
#  Out of Memory Error (virtualSpace.cpp)
```

and REA reports `Ghidra exited before bridge startup`, which points at Ghidra
when the fault is the outer job. The job is created without `BREAKAWAY_OK`
(`CreateProcess` with `CREATE_BREAKAWAY_FROM_JOB` returns `ERROR_ACCESS_DENIED`,
measured), so a child cannot escape it.

The consequence is operational, not a bug to patch: **start the agent, and let it
start REA.** Do not launch an analysis from `run_python` or a forged tool; those
inherit the 512 MiB job. `install_rea.py --check` prints a warning when it detects
you are inside one. Started from the agent process, a full analysis of
`C:\Windows\System32\where.exe` finishes in ~70 s and returns real Ghidra
evidence (loader, language x86:LE:64:default:windows, sections, imports, functions).

## Verified

What was measured on 2026-10-19, and what it means:

```
java probe (shim, inside the 512 MiB job)      rc 0, java.version 21.0.1, arch 64
java probe (REA's own flags, inside the job)   rc 0
Ghidra LaunchSupport -envvars / -vmargs        both answered, unsplit paths
analyzeHeadless -import where.exe -noanalysis  rc 0, 21 s, 350 MB peak in a 1 GiB job
REA binary_overview, started outside any job   70.8 s, isError false, evidence bundle
REA binary_overview, started inside the job    OutOfMemoryError at import, every time
```

The last two rows are the whole story: same code, same target, same JDK; the only
difference is the commit ceiling above it.

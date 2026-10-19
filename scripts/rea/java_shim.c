/* A java.exe stand-in that caps the JVM heap before HotSpot chooses one.
 *
 * Why this exists, measured on this host:
 *
 *   java.exe -XshowSettings:properties -version
 *   # There is insufficient memory for the Java Runtime Environment to continue.
 *   # Native memory allocation (mmap) failed to map 230686720 bytes for G1
 *   # virtual space ... (DOS error/errno=1455)
 *
 * HotSpot's ergonomic default heap is a quarter of physical RAM. On a 14 GiB
 * box that is a 3.6 GiB reservation, and with commit charge already high the
 * reservation fails before main() runs. So every JVM launch that names no -Xmx
 * dies at startup -- which is exactly how Ghidra's LaunchSupport helper and
 * REA's own Java probe launch:
 *
 *     java.exe -Dstdout.encoding=UTF-8 ... -XshowSettings:properties -version
 *     java.exe -cp LaunchSupport.jar LaunchSupport <install> -envvars
 *
 * Neither can be steered from outside. REA blanks JAVA_TOOL_OPTIONS,
 * _JAVA_OPTIONS and JDK_JAVA_OPTIONS before it probes (ghidraJavaEnvironment),
 * so an environment variable cannot reach them, and the probe then reports
 * "runtime_unavailable: Java exited 1" and REA refuses the ghidra provider.
 *
 * This shim sits at bin/java.exe of a private JDK home that is a full copy of a
 * real JDK 21, so the java.home the JDK computes for itself is still correct
 * and Ghidra's version check still passes. It prepends a bounded heap when the
 * caller named none, and stays out of the way when the caller did name one:
 * Ghidra Headless passes -Xmx<GHIDRA_HEADLESS_MAXMEM> (2G by default) and that
 * later option wins, so real analysis still gets a real heap.
 *
 * The options must come FIRST. The java launcher stops reading VM options at
 * the first terminal option (such as -version), so anything appended after it
 * is handed to the application and silently ignored.
 *
 * Note: no -Xmx here on purpose. REA's native launcher creates the Job Object
 * with breakaway explicitly forbidden ("Job Object policy permits unexpected
 * breakaway" is a hard error in rea-windows-x64.node), so a shim cannot lift the
 * JVM out of the 1 GiB cap and a heap ceiling in this file would only fight
 * Ghidra's own -Xmx. What the shim supplies is the part Ghidra never sets: a
 * small initial heap (HotSpot's ergonomic -Xms is RAM/64, and reserving that is
 * what fails first), small thread stacks and a bounded code cache.
 *
 * The budget is deliberately small. REA launches every provider through a
 * Windows Job Object capped at 1 GiB of committed memory per process tree, so
 * heap + metaspace + code cache + thread stacks all have to fit inside that;
 * a larger default heap reserves memory this host cannot commit and the probe
 * dies again with errno 1455 -- which reads as "unsupported java version".
 *
 * Metaspace is deliberately NOT capped. Ghidra reserves ~112 MiB of class space
 * before analysis starts, so any -XX:MaxMetaspaceSize below 128m makes the very
 * next commit fail; hs_err then reads
 *   mmap failed to map 65536 bytes for Failed to commit metaspace
 * which looks exactly like "out of memory" and is really "we capped the wrong
 * pool".
 *
 * Ghidra is metaspace-hungry (thousands of classes, ~120 MiB of class space
 * reserved before analysis starts), so the cap has to be generous even when the
 * heap is small: a 128m MaxMetaspaceSize turns into
 *   mmap failed to map 65536 bytes for Failed to commit metaspace
 * a few seconds into the import.
 *
 * real_java.exe and the rest of the JDK's bin sit beside this file.
 */
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include <psapi.h>

#ifndef DEFAULT_HEAP
#define DEFAULT_HEAP L"-Xms16m -Xss512k -XX:+UseSerialGC -XX:ReservedCodeCacheSize=48m "
#endif

static void own_dir(wchar_t *out, DWORD cap)
{
    DWORD n = GetModuleFileNameW(NULL, out, cap);
    if (n == 0 || n >= cap) { out[0] = 0; return; }
    wchar_t *slash = wcsrchr(out, L'\\');
    if (slash) *slash = 0;
}

/* Measure the committed-memory ceiling by committing until it fails.
 *
 * A Job Object cap is invisible from inside the process -- there is no API to
 * read "my own job" without a handle to it -- so the only honest way to report
 * it is to walk into it. Each step commits and touches 32 MiB; the last
 * successful total is the ceiling, and the difference from system commit is the
 * share the launcher imposed. Only compiled in for diagnosis (JAVA_SHIM_PROBE). */
static unsigned long long measure_commit_ceiling(void)
{
    const SIZE_T step = 32ull * 1024 * 1024;
    unsigned long long total = 0;
    for (int i = 0; i < 512; i++) {
        void *p = VirtualAlloc(NULL, step, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (p == NULL) break;
        /* Touch it: reserved-but-uncommitted pages do not count against the cap. */
        for (SIZE_T off = 0; off < step; off += 4096) ((volatile char *)p)[off] = 1;
        total += step;
    }
    return total;
}

int wmain(int argc, wchar_t **argv)
{
    wchar_t dir[MAX_PATH];
    own_dir(dir, MAX_PATH);

    const wchar_t *tool = L"java";
    if (argc > 0) {
        const wchar_t *base = wcsrchr(argv[0], L'\\');
        base = base ? base + 1 : argv[0];
        if (_wcsicmp(base, L"javac.exe") == 0) tool = L"javac";
        else if (_wcsicmp(base, L"jar.exe") == 0) tool = L"jar";
        else if (_wcsicmp(base, L"javaw.exe") == 0) tool = L"javaw";
    }

    /* The caps go on unconditionally, ahead of the caller's own options. The
     * initial-heap reservation is the thing that fails (HotSpot's ergonomic -Xms
     * is 1/64 of RAM ~ 220 MiB, and that is the reservation errno 1455 rejects),
     * and Ghidra Headless only passes -Xmx. Since a later option wins, a caller
     * that does name a heap still gets its own value; what it cannot undo is the
     * small -Xms that made the launch possible. */

    wchar_t cmd[65536];
    cmd[0] = 0;
    wcscat(cmd, L"\"");
    wcscat(cmd, dir);
    wcscat(cmd, L"\\real_");
    wcscat(cmd, tool);
    wcscat(cmd, L".exe\" ");
    if (wcscmp(tool, L"javac") != 0)
        wcscat(cmd, DEFAULT_HEAP);

    for (int i = 1; i < argc; i++) {
        int needs_quote = (argv[i][0] == 0 ||
                           wcschr(argv[i], L' ') || wcschr(argv[i], L'\t') ||
                           wcschr(argv[i], L'"'));
        if (needs_quote) {
            wcscat(cmd, L"\"");
            for (const wchar_t *p = argv[i]; *p; p++) {
                if (*p == L'"') wcscat(cmd, L"\\");
                wchar_t one[2] = { *p, 0 };
                wcscat(cmd, one);
            }
            wcscat(cmd, L"\"");
        } else {
            wcscat(cmd, argv[i]);
        }
        wcscat(cmd, L" ");
    }

    /* The caller may hand us std handles that are not inheritable (REA starts
       providers through its own native Windows runner). Duplicating them as
       inheritable and naming them keeps the child's stdout on the caller's
       pipe -- without this, Ghidra's launch.bat reads an empty answer from
       `LaunchSupport -save` and falls back to its interactive JDK prompt. */
    STARTUPINFOW si; PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si)); si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    HANDLE cOut = NULL, cErr = NULL, cIn = NULL;
    DuplicateHandle(GetCurrentProcess(), si.hStdInput, GetCurrentProcess(), &cIn, 0, TRUE, DUPLICATE_SAME_ACCESS);
    DuplicateHandle(GetCurrentProcess(), si.hStdOutput, GetCurrentProcess(), &cOut, 0, TRUE, DUPLICATE_SAME_ACCESS);
    DuplicateHandle(GetCurrentProcess(), si.hStdError, GetCurrentProcess(), &cErr, 0, TRUE, DUPLICATE_SAME_ACCESS);
    if (cIn) si.hStdInput = cIn;
    if (cOut) si.hStdOutput = cOut;
    if (cErr) si.hStdError = cErr;

    /* REA caps every owned process tree at 1 GiB of committed memory through a
       Windows Job Object, and a Ghidra analysis JVM does not fit in that: even
       with the heap at 256m it dies in native code ("malloc failed ...
       Chunk::new") seconds into an import. That cap is the launcher's judgement
       about a tool it did not write, not a property of the JVM, so the JVM asks
       to start outside the job. If the job forbids breakaway the plain launch is
       retried and the cap stands -- degraded, but honest. */
    DWORD flags = CREATE_BREAKAWAY_FROM_JOB;
    DWORD firstError = 0;
    if (!CreateProcessW(NULL, cmd, NULL, NULL, TRUE, flags, NULL, NULL, &si, &pi)) {
        firstError = GetLastError();
        if (firstError != ERROR_ACCESS_DENIED ||
            !CreateProcessW(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
            fwprintf(stderr, L"java-shim: CreateProcess failed %lu\n", GetLastError());
            return 127;
        }
    }
    if (GetEnvironmentVariableW(L"JAVA_SHIM_LOG", NULL, 0) > 0) {
        FILE *lf = _wfopen(L"C:\\Users\\china\\AppData\\Local\\autoforge\\tmp\\javashim.log", L"a, ccs=UTF-8");
        if (lf) {
            BOOL inj = FALSE; IsProcessInJob(GetCurrentProcess(), NULL, &inj);
            /* How much room the OS says is left, so a failed launch can be told
               apart from a job cap: these are committed-memory facts. */
            MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms); GlobalMemoryStatusEx(&ms);
            unsigned long long ceiling = 0, used = 0;
            PROCESS_MEMORY_COUNTERS pmc; ZeroMemory(&pmc, sizeof(pmc));
            if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
                used = (unsigned long long)pmc.PagefileUsage;
            if (GetEnvironmentVariableW(L"JAVA_SHIM_PROBE", NULL, 0) > 0)
                ceiling = measure_commit_ceiling();
            fwprintf(lf, L"selfInJob=%d breakawayErr=%lu physAvailMB=%llu commitAvailMB=%llu commitCeilingMB=%llu usedMB=%llu\nCMD=<%ls>\n",
                     inj, firstError,
                     (unsigned long long)(ms.ullAvailPhys / 1048576),
                     (unsigned long long)(ms.ullAvailPageFile / 1048576),
                     ceiling / 1048576, used / 1048576, cmd);
            fclose(lf);
        }
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return (int)code;
}

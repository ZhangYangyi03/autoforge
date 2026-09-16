/* Does this kernel actually refuse the syscalls, or does the seccomp library
 * only claim it will?
 *
 * The distinction the whole sandbox argument rests on. Reading `Seccomp: 2`
 * out of /proc/self/status says the kernel supports a filter; it says nothing
 * about whether YOUR filter is installed, whether the syscall is on the right
 * side of it, or whether the action you asked for is the action you get. This
 * binary is small enough to check all three by running the syscall and looking
 * at what comes back.
 *
 * Build:  gcc -O2 -o seccomp_probe seccomp_probe.c -lseccomp
 * Run:    ./seccomp_probe
 *
 * Exit:   0 when every expectation held, 1 when one did not, 2 when the
 *         environment cannot run the test at all.
 */
#include <errno.h>
#include <seccomp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/wait.h>

static int failures = 0;

static void expect(int ok, const char *what, const char *detail) {
    printf("  [%s] %s%s%s\n", ok ? "ok" : "FAIL", what,
           detail && *detail ? " -- " : "", detail ? detail : "");
    if (!ok) failures++;
}

/* A raw syscall through the filter, reporting errno rather than dying: the
 * interesting answer is EPERM, and a process that dies on SIGSYS tells you
 * which action was configured only by its tombstone. */
static long raw_socket(void) {
    errno = 0;
    long r = syscall(SYS_socket, AF_INET, SOCK_STREAM, 0);
    return r < 0 ? -errno : r;
}

int main(void) {
    printf("seccomp probe\n");

    int sup = seccomp_api_get();
    expect(sup >= 0, "libseccomp is usable", NULL);
    if (sup < 0) {
        printf("seccomp_api_get failed: %s\n", strerror(errno));
        return 2;
    }
    char msg[64];
    snprintf(msg, sizeof msg, "API level %d", sup);
    expect(sup >= 3, "kernel offers the seccomp API levels this build expects", msg);

    /* 1. Before any filter: the syscall must succeed. Without this control the
     *    later EPERM proves nothing -- a socket() that always fails would look
     *    exactly like a working filter. */
    long before = raw_socket();
    snprintf(msg, sizeof msg, "socket() -> %ld%s%s", before,
             before < 0 ? " errno=" : "", before < 0 ? strerror((int)-before) : "");
    expect(before >= 0, "control: socket() works with no filter installed", msg);
    if (before >= 0) close((int)before);

    /* 2. Install a filter that denies exactly one syscall, with a log action
     *    that does not kill the process, so the refusal is observable. */
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_ALLOW);
    if (!ctx) { printf("seccomp_init failed\n"); return 2; }
    int rc = seccomp_rule_add(ctx, SCMP_ACT_ERRNO(EPERM), SCMP_SYS(socket), 0);
    expect(rc == 0, "rule added: socket -> EPERM", rc == 0 ? "" : strerror(-rc));
    rc = seccomp_load(ctx);
    expect(rc == 0, "filter installed in this process", rc == 0 ? "" : strerror(-rc));
    if (rc != 0) { seccomp_release(ctx); return 2; }

    /* 3. The same call again, now through the filter. */
    long after = raw_socket();
    snprintf(msg, sizeof msg, "socket() -> %ld%s", after,
             after < 0 ? strerror((int)-after) : " (unexpectedly allowed)");
    expect(after == -EPERM, "the filter refused socket() with EPERM", msg);

    /* 4. And a syscall the filter says nothing about still works: a filter that
     *    breaks the process generally is not isolation, it is a crash. */
    errno = 0;
    long pid = syscall(SYS_getpid);
    expect(pid > 0, "an unrelated syscall still works (getpid)", NULL);

    /* 5. A second process must be unaffected: the filter is per-process, and a
     *    sandbox that leaked across fork would be a different thing entirely. */
    pid_t child = fork();
    if (child == 0) {
        errno = 0;
        long r = syscall(SYS_socket, AF_INET, SOCK_STREAM, 0);
        _exit(r >= 0 ? 0 : 1);
    } else if (child > 0) {
        int st = 0;
        waitpid(child, &st, 0);
        int child_ok = WIFEXITED(st) && WEXITSTATUS(st) == 0;
        expect(!child_ok, "the filter is inherited by fork (child was refused too)", NULL);
    } else {
        expect(0, "fork for the inheritance check", strerror(errno));
    }

    seccomp_release(ctx);
    printf("failures: %d\n", failures);
    return failures == 0 ? 0 : 1;
}

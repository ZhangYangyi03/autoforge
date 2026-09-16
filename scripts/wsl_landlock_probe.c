/* Landlock, asked directly: is it in the kernel, and does a rule actually bite?
 *
 * The WSL kernel exposes no /sys/kernel/security/lsm file, so the usual "is
 * Landlock enabled" check answers nothing here. The only honest test is to
 * build a ruleset, restrict this process to one directory, and then try to
 * read a file outside it. If Landlock is absent the syscall returns ENOMSG or
 * EOPNOTSUPP and the code says so instead of pretending.
 *
 * Filesystem confinement that survives a compromised child is the thing a
 * seccomp filter cannot give you: seccomp reasons about syscall *numbers*,
 * Landlock reasons about *paths*, and a policy written as "you may read this
 * tree and nothing else" cannot be expressed in seccomp at all.
 *
 * Build: gcc -O2 -o landlock_probe landlock_probe.c
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <linux/landlock.h>

#ifndef SYS_landlock_create_ruleset
#define SYS_landlock_create_ruleset 444
#define SYS_landlock_add_rule 445
#define SYS_landlock_restrict_self 446
#endif

static int failures = 0;

static void expect(int ok, const char *what, const char *detail) {
    printf("  [%s] %s%s%s\n", ok ? "ok" : "FAIL", what,
           detail && *detail ? " -- " : "", detail ? detail : "");
    if (!ok) failures++;
}

int main(void) {
    printf("landlock probe\n");

    struct landlock_ruleset_attr attr;
    memset(&attr, 0, sizeof attr);
    attr.handled_access_fs = LANDLOCK_ACCESS_FS_READ_FILE |
                             LANDLOCK_ACCESS_FS_WRITE_FILE |
                             LANDLOCK_ACCESS_FS_READ_DIR;

    /* First question: is Landlock present at all? ABI 0 or a negative return
     * with ENOMSG/EOPNOTSUPP means this kernel cannot do path confinement. */
    long abi = syscall(SYS_landlock_create_ruleset, NULL, 0, LANDLOCK_CREATE_RULESET_VERSION);
    char msg[128];
    if (abi < 0) {
        snprintf(msg, sizeof msg, "landlock_create_ruleset(VERSION) -> %s", strerror(errno));
        expect(0, "Landlock is present in this kernel", msg);
        printf("failures: %d\n", failures);
        return 1;
    }
    snprintf(msg, sizeof msg, "ABI version %ld", abi);
    expect(abi >= 1, "Landlock is present in this kernel", msg);

    int rs = syscall(SYS_landlock_create_ruleset, &attr, sizeof attr, 0);
    if (rs < 0) { snprintf(msg, sizeof msg, "%s", strerror(errno));
                  expect(0, "ruleset created", msg);
                  printf("failures: %d\n", failures); return 1; }
    expect(1, "ruleset created", NULL);

    /* Allow exactly one directory, read-only. */
    struct landlock_path_beneath_attr path;
    memset(&path, 0, sizeof path);
    int allowed = open("/root/ll_allowed", O_PATH | O_CLOEXEC);
    if (allowed < 0) { mkdir("/root/ll_allowed", 0700); allowed = open("/root/ll_allowed", O_PATH | O_CLOEXEC); }
    path.allowed_access = LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR;
    path.parent_fd = allowed;
    int rc = syscall(SYS_landlock_add_rule, rs, LANDLOCK_RULE_PATH_BENEATH, &path, 0);
    expect(rc == 0, "rule added: read-only under /root/ll_allowed", rc == 0 ? "" : strerror(errno));

    rc = syscall(SYS_landlock_restrict_self, rs, 0);
    expect(rc == 0, "restriction applied to this process", rc == 0 ? "" : strerror(errno));

    /* Inside the allowance: still readable. A policy that denies everything is
     * not a sandbox, it is a crash -- same control as the seccomp probe. */
    int inside = open("/root/ll_allowed/../ll_allowed", O_RDONLY | O_DIRECTORY);
    expect(inside >= 0, "control: the allowed directory is still readable", NULL);
    if (inside >= 0) close(inside);

    /* Outside it: refused, and refused with EACCES rather than by killing us. */
    errno = 0;
    int outside = open("/etc/hostname", O_RDONLY);
    snprintf(msg, sizeof msg, "open(/etc/hostname) -> %s", outside < 0 ? strerror(errno) : "ALLOWED");
    expect(outside < 0 && errno == EACCES, "a file outside the allowance is refused", msg);
    if (outside >= 0) close(outside);

    printf("failures: %d\n", failures);
    return failures == 0 ? 0 : 1;
}

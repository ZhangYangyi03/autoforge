"""Assert the engine that actually got installed is the release the pin names.

`pyproject.toml` pins autoforge to an exact release (`autoforge-agent==X.Y.Z`)
so that a fresh clone reproduces the README's measurements rather than whatever
`main` looks like today. A pin that silently resolves to something else is worse
than no pin: the reproduction instructions would appear to work while measuring
a different engine.

Three outcomes, and the distinction between them is the point:

  0  the installed engine is the pinned release
  1  it is a *different* version -- a red build
  2  the version matches but the origin is a local checkout, so the question
     "which commit is this" cannot be answered

Outcome 2 is normal for a developer, who installs the engine editable from the
sibling checkout (see the README quickstart) and may well be sitting on
uncommitted edits under the same version number. In CI the engine always
arrives from the index through the pin, so 2 there means something is genuinely
wrong -- which is why the workflow treats any non-zero exit as a failure, and
the message, not the exit code, is what tells a human which of the two they are
looking at.

Two independent signals count as "a checkout": the metadata's own
`direct_url.json`, and the path the import actually resolves to. The second is
not redundant -- pip leaves `direct_url.json` out when an install is interrupted
(observed: a locked console script aborts the write), and a check that vouches
for unidentified code because a file is merely absent is worse than no check.

Run from the repository root: python tools/check_pinned_engine.py
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

# The distribution is `autoforge-agent`; the import is `autoforge`. The pin is a
# version, so the checker compares versions rather than VCS commit ids. A local
# checkout at the same version is still not evidence of the same *code* -- which
# is why outcome 2 exists rather than being folded into 0.
PIN = re.compile(r"autoforge-agent\s*==\s*([0-9][0-9A-Za-z.\-]*)")


def _from_index(origin: str | None) -> bool:
    """True when the import resolved into an installed location.

    `site-packages` and `dist-packages` are what a pip install writes; anything
    else is a working tree. An unknown origin is taken as installed: the
    distribution lookup already succeeded, so the package is on the path.
    """
    if origin is None:
        return True
    return any(p.endswith("-packages") for p in pathlib.Path(origin).parts)


def classify(
    declared: str, installed: str | None, direct_url: dict | None,
    origin: str | None = None,
) -> tuple[int, str]:
    """Decide the verdict from the pinned version and what is installed.

    `direct_url` is the parsed `direct_url.json` of the installed distribution,
    or None when it records none at all; `origin` is the file the import of
    `autoforge` resolves to, or None when it cannot be resolved. Split out from
    the environment probe so the decision can be tested without a pip install.
    """
    if installed is None:
        return 2, (
            "autoforge-agent is not installed -- this check cannot tell you "
            "which engine the suite would measure"
        )

    if installed != declared:
        return 1, (
            f"FATAL: the installed engine is {installed}, but pyproject.toml "
            f"pins {declared}. The reproduction would be measuring a different "
            "engine."
        )

    if (direct_url or {}).get("dir_info", {}).get("editable"):
        # The common developer case. Say so plainly rather than reporting a
        # mismatch that has not been established: a check that cries wolf on the
        # normal path is a check that gets ignored. Same version, unknown code.
        return 2, (
            f"installed editable from {(direct_url or {}).get('url')} at version "
            f"{declared} -- the version matches but this check cannot tell you "
            "which commit you have"
        )

    if not _from_index(origin):
        # Same version, metadata silent, code demonstrably not from the index.
        return 2, (
            f"the import resolves to {origin}, outside site-packages, at version "
            f"{declared} -- the version matches but this check cannot tell you "
            "which commit you have"
        )

    return 0, f"ok: the installed engine is the pinned release ({declared})"


def main() -> int:
    pyproject = pathlib.Path("pyproject.toml")
    if not pyproject.exists():
        print("run me from the repository root", file=sys.stderr)
        return 2

    m = PIN.search(pyproject.read_text(encoding="utf-8"))
    if not m:
        print("pyproject.toml has no pinned autoforge release", file=sys.stderr)
        return 2
    declared = m.group(1)

    # Ask the installed metadata what it has, rather than guessing at a path:
    # site-packages is outside the repo, so a glob from the working directory
    # would silently find nothing.
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        dist = distribution("autoforge-agent")
    except PackageNotFoundError:
        print("autoforge-agent is not installed", file=sys.stderr)
        return 2

    installed = dist.version
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:  # noqa: BLE001 - absence is the normal index-install case
        raw = None
    direct_url = json.loads(raw) if raw else None

    # Where the import actually lands, which is the one signal metadata cannot
    # fake. find_spec runs the editable finders without importing the engine.
    try:
        spec = importlib.util.find_spec("autoforge")
        origin = spec.origin if spec else None
    except Exception:  # noqa: BLE001 - the verdict reports an unusable engine
        origin = None

    code, message = classify(declared, installed, direct_url, origin)

    print(f"pinned    {declared}")
    print(f"installed {installed}")
    print(f"source    {(direct_url or {}).get('url')}")
    print(f"import    {origin}")
    print()
    print(message, file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

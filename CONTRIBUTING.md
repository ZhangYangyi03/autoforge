# Contributing to autoforge

Thanks for looking. This document is what you need to get a change merged, and
nothing else: the conventions below exist because they caught a real bug at some
point, not because they are house style.

## Ground rules

- MIT licensed. By opening a pull request you agree your contribution is MIT too.
- Bug reports and design arguments are contributions. You do not have to write
  code to be useful here.
- One change per pull request. A pull request that fixes a bug and reformats a
  module gets reviewed as two things at once and lands as neither.

## Get it running

Python 3.10 or newer. No API key is needed to run the test suite.

```bash
git clone https://github.com/ZhangYangyi03/autoforge
cd autoforge
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

The suite is large (about 1900 tests) and it is the definition of "working" for
this project. If it is red on `main`, that is a bug worth an issue on its own.

Run a subset while you work; run the whole thing before you open a pull request:

```bash
python -m pytest tests/test_forge.py -q
python -m pytest tests -q -x -n auto
```

Some tests exercise platform-specific paths (Landlock/seccomp on Linux, Windows
job objects). They skip themselves where they cannot apply; a skip is fine, a
failure on your platform is not.

## Where things live

    autoforge/          the package: forge pipeline, verification, ledger, CLI
    autoforge/web/      the single-file web UI (the one non-Python asset)
    tests/              the suite; fixtures and fakes live in tests/conftest.py
    .github/workflows/  CI (tests + artifact checks) and tag-driven publishing
    docs/               design notes and reports
    examples/ probes/   runnable samples and fixture data
    vendor/             third-party code, upstream and unedited

Do not edit anything under `vendor/`. It is a verbatim copy of upstream code, and
a change there cannot be told apart from an upstream change on the next sync.

## Making a change

1. Fork, then branch off `main`. Name the branch for the change, not for you:
   `fix/verify-skip-symlink`, `feat/windows-job-objects`,
   `docs/mission-timing`. Existing branches use both `feat/` and `local/`
   prefixes; either is fine, `fix/`, `feat/`, `docs/` read best.
2. Write the test first where you can. A behaviour with no test is a behaviour
   the next refactor is free to delete.
3. Keep the diff readable: matching the surrounding comment style matters more
   than any external style guide. Comments in this codebase explain why a line
   is the way it is, especially where the obvious version was tried and failed.
   A comment that restates the code will be asked to leave.
4. Commit messages follow the conventional-commit shape already in the log:

       fix(rea): a docstring path that python warns about is a path with a backslash
       feat(media): image and video tools, behind a status probe

   The scope is the module you touched. The subject says what changed for the
   user, in the past tense, with no trailing period.
5. Push to your fork and open the pull request against `main`. Describe the
   problem before the patch, and say how you checked it (`pytest tests -q` output
   is enough).

## What CI will check

`.github/workflows/ci.yml` runs on every pull request:

- the full suite on Python 3.10, 3.11 and 3.12;
- both release artifacts, because the checkout hides what a wheel is missing:
  the wheel is installed into a fresh venv and the web UI is served out of it,
  and the sdist is unpacked and its tests are collected.

A pull request that is green locally but red in the `artifacts` job is usually a
missing `package-data` entry or a file that exists only in the source tree.
Please do not disable a check to make a pull request pass; say what broke.

## Review and merge

- `main` is the only long-lived branch. It is currently unprotected, so push
  access means "can push to `main`" -- please still use a branch and a pull
  request unless you are the maintainer doing a release.
- When a pull request adds a tool to the library, say in the description whether
  it was verified end-to-end and on which platform. "It works on my machine" is
  acceptable if you name the machine.
- Releases are cut by pushing a `v*` tag; PyPI is trusted through OIDC, so there
  is no token to hand anyone. Only maintainers tag.

## Reporting a bug

Open an issue with the template. The three things that make a report actionable:

1. The exact command and the full traceback.
2. What you expected, in one sentence.
3. `python -c "import sys, platform; print(sys.version, platform.platform())"`
   plus the autoforge version (`auto --version` or the `version` field in
   `pyproject.toml`).

## Security

Do not open a public issue for a vulnerability: this agent runs code with real
filesystem and network reach, so a sandbox escape is worth a private report.
Email the maintainer at 2936745@qq.com and give it a week.

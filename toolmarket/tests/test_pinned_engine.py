"""The pin checker's verdicts.

`tools/check_pinned_engine.py` guards the reproduction steps above: if the engine
pip installed is not the release `pyproject.toml` pins, the README's numbers were
measured against something else.

It has three outcomes and the distinction is the whole point. "Different engine"
is a red build. "Same version, no recorded origin" is *not* — that is what a
developer with an editable sibling checkout sees, and reporting it as a mismatch
would cry wolf on the normal path until people learned to ignore the check.
These pin that boundary, because it is the kind of thing that gets flattened by
a later "simplification".
"""
from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _checker():
    spec = importlib.util.spec_from_file_location(
        "check_pinned_engine", ROOT / "tools" / "check_pinned_engine.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pinned() -> str:
    """Read the pin out of pyproject.toml, via the checker's own parser.

    Deliberately not a literal in this file: a copy of the version here would be
    a second source of truth for a number that already has one, and it would go
    stale the first time somebody bumped the pin, at which point this test would
    pass while checking the wrong value.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return _checker().PIN.search(text).group(1)


def test_the_pinned_release_verifies():
    pinned = _pinned()
    code, message = _checker().classify(
        pinned, pinned, {"url": "https://files.pythonhosted.org/packages/..."}
    )
    assert code == 0, message
    assert "ok:" in message


def test_a_different_version_fails_the_build():
    code, message = _checker().classify(
        _pinned(), "0.0.1", {"url": "https://files.pythonhosted.org/packages/..."}
    )
    assert code == 1, message
    assert "FATAL" in message


def test_an_editable_local_install_says_cannot_tell_not_wrong_engine():
    """The developer path. A false mismatch here is worse than no check.

    Same version, unknown commit: the version pin is satisfied, but an editable
    checkout may carry uncommitted edits, so the check must refuse to vouch for
    the code rather than declaring it correct.
    """
    pinned = _pinned()
    code, message = _checker().classify(
        pinned,
        pinned,
        {"url": "file:///home/dev/autoforge", "dir_info": {"editable": True}},
    )
    assert code == 2, message
    assert "cannot tell" in message
    assert "FATAL" not in message


def test_nothing_installed_is_also_cannot_tell():
    """Absence is not a mismatch. There is no engine to be wrong about."""
    code, message = _checker().classify(_pinned(), None, None)
    assert code == 2, message
    assert "cannot tell" in message
    assert "FATAL" not in message


def test_a_checkout_with_no_recorded_origin_is_still_cannot_tell():
    """The signal that survives a metadata-less install.

    pip leaves `direct_url.json` out when an install is interrupted -- a locked
    console script does exactly that, observed here -- and the metadata is then
    silent about an engine demonstrably running out of a working tree. Reading
    that silence as "from the index" is the one failure this check exists to
    prevent, so the import path gets a veto of its own.
    """
    pinned = _pinned()
    code, message = _checker().classify(
        pinned, pinned, None, r"D:\dev\autoforge\autoforge\__init__.py"
    )
    assert code == 2, message
    assert "cannot tell" in message
    assert "FATAL" not in message


def test_an_index_install_with_no_recorded_origin_still_verifies():
    """Silence plus a site-packages import is just a normal pinned install."""
    pinned = _pinned()
    origin = "/usr/lib/python3.12/site-packages/autoforge/__init__.py"
    code, message = _checker().classify(pinned, pinned, None, origin)
    assert code == 0, message
    assert "ok:" in message

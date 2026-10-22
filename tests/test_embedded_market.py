"""The market that ships inside autoforge.

The move from a sibling checkout to a subtree is only worth anything if the
path is computed and the failure is legible, so those are what these tests pin:

  * the source tree is found by walking up, not by a hard coded directory -- the
    old launcher contained D:\\Users\\china\\... which is a path that works until
    a folder moves;
  * importing the market does not leave its directory on sys.path, because that
    would shadow a pip-installed `toolmarket` for the whole process;
  * the store is a file by default, because an in-memory shelf forgets every
    tool on restart and the symptom reads as "nobody published anything";
  * a shelf that is down is a returned error, never an exception, on every path
    that touches it -- a forge must not roll back because the market is asleep.
"""
from __future__ import annotations

import json
import socket
import sys

import pytest

from autoforge import embedded


def test_the_source_tree_is_found_by_computed_path():
    root = embedded.source_root()
    assert (root / "toolmarket" / "__init__.py").is_file()
    assert embedded.available()
    assert embedded.missing_reason() == ""


def test_importing_the_market_does_not_leave_it_on_sys_path():
    before = list(sys.path)
    with embedded.importable() as root:
        assert str(root) in sys.path
        import toolmarket  # noqa: F401  - the thing the block exists to make possible

        assert toolmarket.__file__.startswith(str(root))
    assert sys.path == before


def test_a_missing_tree_is_an_import_error_naming_where_it_looked(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    empty = tmp_path / "nowhere"
    monkeypatch.setattr(embedded, "source_root", lambda: empty)
    # The second way this succeeds has to be closed too, or the test proves
    # nothing on a machine where toolmarket is importable from site-packages.
    monkeypatch.setattr(embedded, "installed_root", lambda: None)
    assert not embedded.available()
    with pytest.raises(ImportError) as exc:
        with embedded.importable():
            pass
    assert str(empty) in str(exc.value)


def test_the_default_store_is_a_file_not_memory(monkeypatch, tmp_path):
    monkeypatch.delenv("TOOLMARKET_STORE", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    url = embedded.store_url()
    assert url.startswith("sqlite:///")
    assert not url.endswith(":memory:")
    # The directory is created, so the first `auto market` does not fail on a
    # missing parent -- which would read as a permissions problem.
    parent = url[len("sqlite:///"):].rsplit("/", 1)[0]
    assert (tmp_path / "autoforge").as_posix() in parent


def test_the_store_url_can_be_overridden(monkeypatch):
    monkeypatch.delenv("TOOLMARKET_STORE", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert embedded.store_url("postgresql://x") == "postgresql://x"
    # `sqlite:///:memory:` is the form a config file tends to carry; the store
    # parses it into the path "/:memory:" and sqlite cannot open that, so the
    # bare form is what the embedded default hands through untouched.
    monkeypatch.setenv("TOOLMARKET_STORE", "sqlite:///:memory:")
    assert embedded.store_url() == "sqlite:///:memory:"


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_publishing_to_a_shelf_that_is_down_returns_an_error(monkeypatch):
    """Not an exception: every caller of this is mid-forge or mid-repair, and
    both of those have to survive the shelf being asleep."""
    from autoforge.tools.spec import ToolSpec

    spec = ToolSpec(name="nobody_is_home", description="d", parameters={},
                    fn=lambda: 1, code="def f(): return 1")
    res = embedded.publish(spec, url=f"http://127.0.0.1:{_closed_port()}")
    assert res["ok"] is False
    assert "error" in res


def test_the_app_builds_and_answers_health(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("TOOLMARKET_STORE", ":memory:")
    client = TestClient(embedded.app())
    health = client.get("/health").json()
    assert health["ok"] is True
    assert client.get("/").status_code == 200


def test_market_status_is_json_and_exits_nonzero_when_down(monkeypatch, capsys):
    from autoforge import cli

    port = _closed_port()
    rc = cli.main(["market", "status", "--port", str(port)])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["listening"] is False
    assert report["embedded"] is True
    assert rc == 1


def test_an_installed_market_needs_no_directory_of_its_own(monkeypatch, tmp_path):
    """A wheel install has no sibling tree. It must not read as "missing"."""
    monkeypatch.setattr(embedded, "source_root", lambda: tmp_path / "nowhere")
    monkeypatch.setattr(embedded, "installed_root", lambda: tmp_path / "site-packages")
    assert embedded.available()
    assert embedded.missing_reason() == ""
    with embedded.importable() as root:
        assert root == tmp_path / "site-packages"

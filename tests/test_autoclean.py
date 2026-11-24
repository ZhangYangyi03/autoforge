"""The crawl hook: it must keep what is worth keeping, and never break a crawl."""
import json
import os

import pytest

from autoforge import autoclean


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = tmp_path / "auto" / "web.graph.json"
    monkeypatch.setenv("ENTITYGRAPH_AUTO_STORE", str(s))
    monkeypatch.setenv("ENTITYGRAPH_AUTO_LOG", str(tmp_path / "auto" / "auto.log"))
    return s


def _page(url, text="x" * 500):
    return {"url": url, "status": 200, "title": "T", "text": text,
            "links": [], "bytes": len(text)}


def _res(rows):
    return {"mode": "fast", "count": len(rows), "elapsed": 0.1, "rps": 9.0,
            "results": rows}


def test_report_says_whether_the_hook_is_on():
    line = autoclean.report()
    assert line.startswith("autoclean ")
    assert "store" in line


def test_a_crawl_worth_keeping_is_kept_without_being_asked(store):
    res = _res([_page("https://a.example/1"), _page("https://a.example/2")])
    verdict = autoclean.after_crawl(res, engine="autoforge.crawler:fast")
    assert verdict["worth"] is True and verdict["cleaned"] is True
    assert res["_autoclean"]["cleaned"] is True          # recorded on the result itself
    assert "kept" in autoclean.one_line(verdict)
    assert store.exists()


def test_a_thin_crawl_is_not_kept_and_the_reason_rides_along(store):
    res = _res([_page("https://a.example/1", text="short")])
    verdict = autoclean.after_crawl(res, engine="autoforge.crawler:fast")
    assert verdict["cleaned"] is False
    assert res["_autoclean"]["worth"] is False
    assert "not kept" in autoclean.one_line(verdict)
    assert not store.exists()                            # nothing was written


def test_the_hook_can_be_switched_off(store, monkeypatch):
    monkeypatch.setenv("AUTOFORGE_AUTOCLEAN", "0")
    res = _res([_page("https://a.example/1"), _page("https://a.example/2")])
    assert autoclean.after_crawl(res) is None
    assert "_autoclean" not in res
    assert "off" in autoclean.report()


def test_a_broken_result_does_not_raise_out_of_the_hook(store):
    verdict = autoclean.after_crawl({"results": None})
    assert verdict is not None and verdict["cleaned"] is False


def test_a_non_dict_result_is_ignored(store):
    assert autoclean.after_crawl(["not", "a", "result"]) is None


def test_one_line_is_readable_in_all_three_states():
    assert autoclean.one_line(None) == "autoclean: skipped"
    assert autoclean.one_line({"cleaned": True, "reason": "r",
                               "store_nodes_after": 3, "dropped_oldest": 0}).startswith(
        "autoclean: kept")
    assert autoclean.one_line({"cleaned": False, "reason": "why"}).endswith("why")


def test_the_crawler_actually_calls_it(store, monkeypatch):
    """End-to-end through crawler.crawl, with the network stubbed out."""
    from autoforge import crawler

    def fake_isolated(mode, urls, concurrency, timeout, with_text):
        return {"mode": mode, "count": len(urls), "elapsed": 0.01, "rps": 9.0,
                "results": [_page(u) for u in urls]}
    monkeypatch.setattr(crawler, "_isolated", fake_isolated)
    monkeypatch.delenv("AUTOFORGE_CRAWLER_INPROCESS", raising=False)
    out = crawler.crawl(["https://a.example/1", "https://a.example/2"], mode="fast")
    assert out["_autoclean"]["cleaned"] is True
    assert "kept" in out["_autoclean_line"]


def test_the_crawler_hook_can_be_disabled(store, monkeypatch):
    from autoforge import crawler

    def fake_isolated(mode, urls, concurrency, timeout, with_text):
        return {"mode": mode, "count": len(urls), "elapsed": 0.01, "rps": 9.0,
                "results": [_page(u) for u in urls]}
    monkeypatch.setattr(crawler, "_isolated", fake_isolated)
    monkeypatch.setenv("AUTOFORGE_AUTOCLEAN", "0")
    out = crawler.crawl(["https://a.example/1", "https://a.example/2"], mode="fast")
    assert "_autoclean" not in out and out["count"] == 2


def test_a_failing_cleaner_cannot_break_a_crawl(store, monkeypatch):
    from autoforge import crawler, autoclean as ac

    def boom(*a, **k):
        raise RuntimeError("entitygraph exploded")
    monkeypatch.setattr(ac, "after_crawl", boom)

    def fake_isolated(mode, urls, concurrency, timeout, with_text):
        return {"mode": mode, "count": len(urls), "elapsed": 0.01, "rps": 9.0,
                "results": [_page(u) for u in urls]}
    monkeypatch.setattr(crawler, "_isolated", fake_isolated)
    out = crawler.crawl(["https://a.example/1"], mode="fast")
    assert out["count"] == 1                      # the fetch result is intact
    assert out["results"][0]["status"] == 200


def test_the_report_the_model_reads_says_what_was_kept(store, monkeypatch):
    """"抓到了但没留" and "根本没提过清洗" must not look the same in the report."""
    from autoforge import crawler

    def fake_isolated(mode, urls, concurrency, timeout, with_text):
        return {"mode": mode, "count": len(urls), "elapsed": 0.01, "rps": 9.0,
                "results": [_page(u) for u in urls]}
    monkeypatch.setattr(crawler, "_isolated", fake_isolated)
    r = crawler.crawl(["https://a.example/1"], mode="fast")
    text = crawler.report(r)
    assert "autoclean:" in text and "not kept" in text


def test_a_report_without_a_verdict_gains_no_line():
    from autoforge import crawler
    text = crawler.report({"mode": "fast", "count": 0, "elapsed": 0.0, "rps": 0.0,
                           "results": []})
    assert "autoclean" not in text

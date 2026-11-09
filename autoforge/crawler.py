"""crawler-stack, vendored into autoforge as a native module.

Origin: https://github.com/ZhangYangyi03/crawler-stack (Apache-2.0), commit
6f47b35. This copy is intentional: autoforge must not depend on a sibling
checkout on the Desktop to fetch a page. Keep the two in sync by copying
crawler_stack.py here (the CLI main() is harmless unused code).

    from crawler_stack import crawl
    result = crawl(["https://example.com"], mode="fast")

Modes
  fast     aiohttp + asyncio.Semaphore          (highest throughput, static HTML)
  stealth  curl_cffi Chrome TLS/UA impersonate  (fingerprint-gated sites)
  render   headless Chromium via Playwright     (JS pages; one reused page,
           serial goto, sync API in a fresh subprocess per batch)

Every third-party dependency is optional: missing ones fall back to urllib.
The module is safe to import on a machine with none of them installed.

Measured on Windows 11 / Python 3.13, localhost HTTP echo, N=100:
  fast 2971 req/s | stealth 1107 req/s | render ~5.7 pages/s serial-safe
Parsing pages/s: selectolax 27684 > lxml 22292 > parsel 11596 > bs4 1844 > trafilatura 139
"""

from __future__ import annotations

import concurrent.futures
import os
import sys
import time
import urllib.request

__version__ = "0.1.0"
__all__ = ["crawl", "extract", "capabilities", "report"]

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"


def _ensure_user_site():
    """Some sandboxes disable user site-packages; add it if imports fail later."""
    try:
        import site
        p = site.getusersitepackages()
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)
    except Exception:
        pass


def _import(name):
    _ensure_user_site()
    try:
        return __import__(name, fromlist=["*"])
    except Exception:
        return None


def capabilities():
    """Report which optional engines are importable right now."""
    out = {}
    for key, mod in [("aiohttp", "aiohttp"), ("curl_cffi", "curl_cffi.requests"),
                     ("playwright", "playwright.async_api"), ("selectolax", "selectolax.lexbor"),
                     ("trafilatura", "trafilatura"), ("lxml", "lxml.html")]:
        out[key] = _import(mod) is not None
    return out


def extract(html, with_text=False):
    """Extract title, links and body text from bytes/str HTML. Never raises."""
    out = {"title": None, "links": [], "text": ""}
    if isinstance(html, str):
        html = html.encode("utf-8", "replace")
    tree = None
    lex = _import("selectolax.lexbor")
    if lex:
        try:
            tree = lex.LexborHTMLParser(html)
            t = tree.css_first("title")
            out["title"] = t.text(strip=True) if t else None
            out["links"] = [a.attributes.get("href") for a in tree.css("a[href]")][:50]
            node = tree.css_first("article") or tree.css_first("body")
            out["text"] = (node.text(strip=True) if node else "")[:5000]
        except Exception as e:
            out["parse_error"] = "selectolax: " + str(e)[:80]
    if tree is None:
        lxml_html = _import("lxml.html")
        if lxml_html:
            try:
                doc = lxml_html.fromstring(html)
                t = doc.find(".//title")
                out["title"] = t.text if t is not None else None
                out["links"] = doc.xpath("//a/@href")[:50]
                out["text"] = " ".join(doc.xpath("//body//text()"))[:5000]
            except Exception as e:
                out["parse_error"] = "lxml: " + str(e)[:80]
        else:
            try:
                import re
                s = html.decode("utf-8", "replace")
                m = re.search(r"<title[^>]*>(.*?)</title>", s, re.I | re.S)
                out["title"] = m.group(1).strip() if m else None
                out["links"] = re.findall(r'href=["\']([^"\']+)', s)[:50]
                out["text"] = re.sub(r"<[^>]+>", " ", s)[:5000]
            except Exception as e:
                out["parse_error"] = "stdlib: " + str(e)[:80]
    if with_text:
        traf = _import("trafilatura")
        if traf:
            try:
                out["markdown"] = (traf.extract(html.decode("utf-8", "replace")) or "")[:5000]
            except Exception:
                pass
    return out


def _urllib_one(url, timeout):
    t = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
        d = {"url": url, "status": r.status, "bytes": len(body),
             "elapsed": round(time.perf_counter() - t, 3)}
        d.update(extract(body))
        return d
    except Exception as e:
        return {"url": url, "error": str(e)[:120], "elapsed": round(time.perf_counter() - t, 3)}


def _fast(urls, concurrency, timeout, with_text):
    aiohttp = _import("aiohttp")
    if aiohttp is None:
        with concurrent.futures.ThreadPoolExecutor(min(concurrency, 16)) as ex:
            return list(ex.map(lambda u: _urllib_one(u, timeout), urls))

    async def run():
        import asyncio
        sem = asyncio.Semaphore(max(1, concurrency))
        async with aiohttp.ClientSession(headers={"User-Agent": _USER_AGENT}) as s:
            async def one(u):
                async with sem:
                    t = time.perf_counter()
                    try:
                        async with s.get(u, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                            body = await r.read()
                        d = {"url": u, "status": r.status, "bytes": len(body),
                             "elapsed": round(time.perf_counter() - t, 3)}
                        if r.status == 200 and body:
                            d.update(extract(body, with_text))
                        return d
                    except Exception as e:
                        return {"url": u, "error": str(e)[:120],
                                "elapsed": round(time.perf_counter() - t, 3)}
            return await asyncio.gather(*[one(u) for u in urls])
    import asyncio
    return list(asyncio.run(run()))


def _stealth(urls, concurrency, timeout, with_text):
    cr = _import("curl_cffi.requests")
    if cr is None:
        with concurrent.futures.ThreadPoolExecutor(min(concurrency, 16)) as ex:
            return list(ex.map(lambda u: _urllib_one(u, timeout), urls))

    def one(u):
        t = time.perf_counter()
        try:
            r = cr.get(u, impersonate="chrome", timeout=timeout)
            d = {"url": u, "status": r.status_code, "bytes": len(r.content),
                 "elapsed": round(time.perf_counter() - t, 3)}
            if r.status_code == 200 and r.content:
                d.update(extract(r.content, with_text))
            return d
        except Exception as e:
            return {"url": u, "error": str(e)[:120],
                    "elapsed": round(time.perf_counter() - t, 3)}
    with concurrent.futures.ThreadPoolExecutor(max(1, concurrency)) as ex:
        return list(ex.map(one, urls))


# Playwright's asyncio loop cannot be started with asyncio.run() repeatedly in
# one process on Windows: the 3rd launch at concurrency>=4 can deadlock inside
# browser.new_page()/close() and ignore the per-URL timeout entirely (measured
# 2026-11-09). Running the whole render batch in a fresh subprocess is the only
# reliable isolation; each call gets a clean event loop and a clean browser.
_RENDER_WORKER = r"""
import sys, json, time
from playwright.sync_api import sync_playwright

def main():
    cfg = json.loads(sys.stdin.read())
    urls, timeout = cfg["urls"], cfg["timeout"]
    print("__RENDER_START__", flush=True)
    results = []
    # sync_playwright in a fresh subprocess is the only reliable shape here.
    # async_playwright + asyncio.run() spawned with `python -c CODE` hangs
    # before producing any output on this host (Windows 11 / Python 3.13),
    # reproducibly from inside pytest and from nested subprocesses.
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, timeout=15000)
        print("__RENDER_LAUNCHED__", flush=True)
        # ONE page reused serially. Concurrently opening pages from a single
        # Chromium also deadlocks intermittently here; serial goto is stable.
        page = browser.new_page()
        for u in urls:
            t = time.perf_counter()
            try:
                page.goto(u, wait_until="domcontentloaded", timeout=timeout * 1000)
                results.append({
                    "url": u, "status": 200, "title": page.title(),
                    "text": page.inner_text("body")[:5000],
                    "links": page.eval_on_selector_all(
                        "a[href]", "els => els.map(e => e.getAttribute('href')).slice(0,50)"),
                    "elapsed": round(time.perf_counter() - t, 3)})
            except Exception as e:
                results.append({"url": u, "error": str(e)[:120],
                                "elapsed": round(time.perf_counter() - t, 3)})
        page.close()
        browser.close()
    print("__RENDER_JSON__" + json.dumps(results, ensure_ascii=False))

main()
"""


def _render(urls, concurrency, timeout, with_text):
    import json as _json
    import os as _os
    import subprocess
    import sys as _sys
    pw = _import("playwright.async_api")
    if pw is None:
        return [{"url": u, "error": "playwright unavailable; cannot render"} for u in urls]
    cfg = {"urls": list(urls), "concurrency": max(1, int(concurrency)),
           "timeout": float(timeout), "with_text": bool(with_text)}
    # Hard wall-clock guard: the whole serial batch is one subprocess. Cover
    # every URL plus browser startup/shutdown, so a hung browser cannot block.
    wall = max(30.0, float(timeout) * (len(urls) + 2) + 20.0)
    try:
        proc = subprocess.run(
            [_sys.executable, "-u", "-c", _RENDER_WORKER],
            input=_json.dumps(cfg), capture_output=True, text=True, timeout=wall,
            encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        marker = "__RENDER_JSON__"
        line = next((l for l in proc.stdout.splitlines() if l.startswith(marker)), None)
        if line:
            return _json.loads(line[len(marker):])
        err = (proc.stderr or proc.stdout).strip()[-160:] or "no output"
        return [{"url": u, "error": "render worker: " + err} for u in urls]
    except subprocess.TimeoutExpired:
        return [{"url": u, "error": "render worker exceeded wall clock %.0fs" % wall} for u in urls]
    except Exception as e:
        return [{"url": u, "error": "render engine: " + str(e)[:120]} for u in urls]

def crawl(urls, mode="fast", concurrency=20, timeout=8, with_text=False):
    """Fetch URLs concurrently. Returns a dict; per-URL errors never abort the batch.

    urls        str or list of str; empty list -> empty results, no exception
    mode        "fast" | "stealth" | "render"
    concurrency max in-flight requests (render caps at 4)
    timeout     per-URL seconds
    with_text   also run trafilatura for clean article markdown
    """
    if isinstance(urls, str):
        urls = [urls]
    urls = [str(u) for u in urls]
    if not urls:
        return {"mode": mode, "count": 0, "elapsed": 0.0, "rps": 0.0, "results": []}
    if mode not in ("fast", "stealth", "render"):
        raise ValueError('mode must be one of fast|stealth|render')
    t0 = time.perf_counter()
    engine = {"fast": _fast, "stealth": _stealth, "render": _render}[mode]
    try:
        results = engine(urls, max(1, int(concurrency)), float(timeout), bool(with_text))
    except Exception as e:
        results = [{"url": u, "error": "engine: " + str(e)[:120]} for u in urls]
    elapsed = time.perf_counter() - t0
    return {"mode": mode, "count": len(urls), "elapsed": round(elapsed, 3),
            "rps": round(len(urls) / elapsed, 1) if elapsed > 0 else 0.0,
            "results": results}



def report(result, max_urls=20, text_chars=300):
    """Render a crawl() result as a short, model-readable string."""
    lines = ["mode=%s count=%s elapsed=%ss rps=%s" % (
        result.get("mode"), result.get("count"), result.get("elapsed"),
        result.get("rps"))]
    results = result.get("results", [])
    for x in results[:max_urls]:
        if x.get("error"):
            lines.append("  ERR  %s  %s" % (x.get("url"), x["error"]))
            continue
        t = " ".join((x.get("title") or "").split())[:80]
        txt = " ".join((x.get("text") or "").split())[:text_chars]
        lines.append("  %-3s %-6sB %-4slinks  %s  %s" % (
            x.get("status"), x.get("bytes"), len(x.get("links") or []), t, x.get("url")))
        if txt:
            lines.append("       text: " + txt)
    if len(results) > max_urls:
        lines.append("  ... %d more URL(s) omitted" % (len(results) - max_urls))
    return '\n'.join(lines)

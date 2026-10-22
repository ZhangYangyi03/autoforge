"""A one-page console over the substrate — `tool-market web`.

This is a *view*, and it stays one. Every panel is a fetch against the JSON API
that already exists; the page never decides a resource's state, never advances a
version, and never writes an event by itself. All of that lives behind the
registry, and duplicating any of it here would create a second, unaudited state
machine — which is the exact failure the API module's docstring warns about.

    GET /                this page
    GET /api/...         the substrate, as in toolmarket.api.main

Serving it is deliberately dull: build the app, seed a registry, hand it to
uvicorn. The interesting behaviour is all in the protocol.
"""
from __future__ import annotations

from typing import Any, Optional

from toolmarket.api.main import create_app
from toolmarket.registry import ResourceRegistry
from toolmarket.store import ResourceStore

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tool-market · resource substrate</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --line: #30363d; --ink: #e6edf3;
    --dim: #8b949e; --accent: #58a6ff; --ok: #3fb950; --warn: #d29922;
    --bad: #f85149; --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  header {
    padding: 20px 28px; border-bottom: 1px solid var(--line);
    display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap;
  }
  h1 { font-size: 17px; margin: 0; font-weight: 600; letter-spacing: -0.01em; }
  h1 span { color: var(--dim); font-weight: 400; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-left: auto; }
  .chip {
    background: var(--panel); border: 1px solid var(--line); border-radius: 999px;
    padding: 3px 11px; font-size: 12px; color: var(--dim);
  }
  .chip b { color: var(--ink); font-weight: 600; }
  .chip.ok b { color: var(--ok); }
  main { display: grid; grid-template-columns: 340px 1fr; gap: 0; min-height: calc(100vh - 62px); }
  .left { border-right: 1px solid var(--line); padding: 18px; }
  .right { padding: 18px 24px; overflow-y: auto; }
  .label {
    font-size: 11px; text-transform: uppercase; letter-spacing: .07em;
    color: var(--dim); margin: 0 0 10px; font-weight: 600;
  }
  .tool {
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
    padding: 11px 13px; margin-bottom: 8px; cursor: pointer;
    transition: border-color .12s, background .12s;
  }
  .tool:hover { border-color: #484f58; }
  .tool.sel { border-color: var(--accent); background: #1c2430; }
  .tool .nm { font-family: var(--mono); font-size: 13px; }
  .tool .ds { color: var(--dim); font-size: 12px; margin-top: 3px; }
  .badge {
    display: inline-block; font-size: 10px; font-weight: 700; letter-spacing: .05em;
    text-transform: uppercase; padding: 2px 7px; border-radius: 4px; margin-top: 8px;
  }
  .s-active      { background: rgba(63,185,80,.15);  color: var(--ok); }
  .s-draft       { background: rgba(139,148,158,.15); color: var(--dim); }
  .s-probation   { background: rgba(210,153,34,.15);  color: var(--warn); }
  .s-quarantined { background: rgba(248,81,73,.15);   color: var(--bad); }
  .s-retired     { background: rgba(139,148,158,.10); color: #6e7681; }
  h2 { font-size: 13px; margin: 22px 0 9px; font-weight: 600; }
  h2:first-child { margin-top: 0; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    text-align: left; color: var(--dim); font-weight: 500; font-size: 11px;
    text-transform: uppercase; letter-spacing: .05em;
    padding: 5px 10px 5px 0; border-bottom: 1px solid var(--line);
  }
  td { padding: 6px 10px 6px 0; border-bottom: 1px solid #21262d; vertical-align: top; }
  code, .mono { font-family: var(--mono); font-size: 12.5px; }
  pre {
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
    padding: 12px 14px; overflow-x: auto; font-family: var(--mono);
    font-size: 12.5px; margin: 0 0 14px; line-height: 1.5;
  }
  .kv { display: flex; gap: 10px; padding: 3px 0; font-size: 13px; }
  .kv .k { color: var(--dim); min-width: 118px; }
  .row { display: flex; gap: 8px; align-items: center; margin: 10px 0; flex-wrap: wrap; }
  input[type=text] {
    flex: 1; min-width: 190px; background: #0d1117; border: 1px solid var(--line);
    border-radius: 6px; color: var(--ink); padding: 7px 10px;
    font-family: var(--mono); font-size: 12.5px;
  }
  input[type=text]:focus { outline: none; border-color: var(--accent); }
  button {
    background: #21262d; border: 1px solid var(--line); color: var(--ink);
    border-radius: 6px; padding: 7px 13px; font-size: 12.5px; cursor: pointer;
    font-weight: 500;
  }
  button:hover { background: #30363d; border-color: #484f58; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  button.p { background: #1f6feb; border-color: #1f6feb; }
  button.p:hover { background: #388bfd; }
  .muted { color: var(--dim); font-size: 12.5px; }
  .admissible { color: var(--ok); font-weight: 600; }
  .vetoed { color: var(--bad); font-weight: 600; }
  .empty { color: var(--dim); font-style: italic; padding: 14px 0; }
  .ev { font-family: var(--mono); font-size: 12px; }
  .ev .sq { color: #6e7681; }
  .ev .kd { color: var(--accent); }
  @media (max-width: 860px) {
    main { grid-template-columns: 1fr; }
    .left { border-right: none; border-bottom: 1px solid var(--line); }
  }
</style>
</head>
<body>
<header>
  <h1>tool-market <span>· resource substrate</span></h1>
  <div class="chips" id="chips"></div>
</header>
<main>
  <div class="left">
    <p class="label">Resources</p>
    <div id="tools"><div class="empty">loading…</div></div>
  </div>
  <div class="right" id="detail">
    <div class="empty">Select a resource.</div>
  </div>
</main>

<script>
const $ = (id) => document.getElementById(id);
// Every string that reaches an innerHTML template below goes through esc()
// first -- tool names, descriptions, event data, veto reasons and error
// messages are all attacker-influenced (a tool's code and its goal string are
// whatever the caller submitted). The one value deliberately NOT escaped is a
// tool's *output*: it is rendered with textContent, which cannot execute.
// Keep that split if you add a panel: dynamic -> esc(), or textContent.
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const rid = (id) => encodeURIComponent(id);

async function api(path, opts) {
  const r = await fetch("/api" + path, opts);
  if (!r.ok) throw new Error((await r.text()).slice(0, 300));
  return r.json();
}

async function chips() {
  const [h, s] = await Promise.all([api("/health"), api("/stats")]);
  const states = Object.entries(s.by_state || {})
    .map(([k, v]) => `${k} <b>${v}</b>`).join(" · ");
  $("chips").innerHTML = [
    `<div class="chip">resources <b>${h.resources}</b></div>`,
    `<div class="chip">${states || "no state"}</div>`,
    `<div class="chip">events <b>${h.events}</b></div>`,
    `<div class="chip">lineage <b>${s.lineage_nodes}</b></div>`,
    `<div class="chip ${h.chain_ok ? "ok" : ""}">
       hash chain <b>${h.chain_ok ? "intact" : "BROKEN"}</b></div>`,
  ].join("");
}

async function list(selected) {
  const { resources } = await api("/resources");
  if (!resources.length) { $("tools").innerHTML = '<div class="empty">none</div>'; return; }
  $("tools").innerHTML = resources.map((r) => `
    <div class="tool ${r.id === selected ? "sel" : ""}" data-id="${esc(r.id)}">
      <div class="nm">${esc(r.name)}</div>
      <div class="ds">${esc(r.description || "—")}</div>
      <span class="badge s-${esc(r.state)}">${esc(r.state)}</span>
    </div>`).join("");
  document.querySelectorAll(".tool").forEach((el) =>
    el.onclick = () => show(el.dataset.id));
}

async function show(id) {
  list(id);
  // The trust view is fetched defensively. The console is also pointed at
  // deployments whose API predates the route, and a 404 there should cost one
  // panel rather than blank the whole page -- which is what an unguarded
  // Promise.all would do, since it rejects the entire render on any member.
  const [r, lin, tr] = await Promise.all([
    api("/resources/" + rid(id)),
    api("/resources/" + rid(id) + "/lineage"),
    api("/resources/" + rid(id) + "/trust").catch(() => null),
  ]);
  const ev = await api("/resources/" + rid(id) + "/events");
  const inv = (r.contract && r.contract.invariances) || [];
  const tcls = tr ? ({ promote: "active", quarantine: "quarantined" }[tr.action] || "draft") : "";
  const tblock = tr ? `
    <h2>Trust <span class="muted">· what the ledger earns</span></h2>
    <div class="kv"><span class="k">decision</span><code><span class="badge s-${tcls}" style="margin-top:0">${esc(tr.action)}</span>${tr.to ? " &rarr; " + esc(tr.to) : ""}</code></div>
    <div class="kv"><span class="k">reason</span><code>${esc(tr.reason)}</code></div>
    <div class="kv"><span class="k">evidence</span><code>calls ${esc(tr.evidence.calls)} · success ${esc(tr.evidence.success_rate)} · consecutive failures ${esc(tr.evidence.consecutive_failures)}</code></div>
    <div class="kv"><span class="k">the bar</span><code>${tr.thresholds ? `calls &ge; ${esc(tr.thresholds.min_calls)} · success &ge; ${esc(tr.thresholds.min_success_rate)} · consecutive failures = 0` : "—"}</code></div>`
    : `<h2>Trust</h2><div class="empty">trust view unavailable on this API</div>`;
  $("detail").innerHTML = `
    <h2>${esc(r.name)} <span class="badge s-${esc(r.state)}">${esc(r.state)}</span></h2>
    <div class="kv"><span class="k">id</span><code>${esc(r.id)}</code></div>
    <div class="kv"><span class="k">version</span><code>${esc(r.version_current.version)}</code></div>
    <div class="kv"><span class="k">effect_signature</span><code>${esc(r.contract.effect_signature || "—")}</code></div>
    <div class="kv"><span class="k">invariances</span><code>${esc(inv.join(", ") || "—")}</code></div>
    <div class="kv"><span class="k">source</span><code>${esc(r.provenance && r.provenance.source || "—")}</code></div>
    <div class="kv"><span class="k">ledger</span><code>calls ${r.ledger ? r.ledger.calls : 0} · success ${r.ledger ? r.ledger.success_rate : "—"}</code></div>
${tblock}

    <h2>Invoke</h2>
    <div class="row">
      <input type="text" id="arg" value="${esc(r.name === "parse_duration" ? "2h30m" : "Hello, World! 2026")}">
      <button class="p" id="go">invoke</button>
    </div>
    <pre id="out">— not called yet —</pre>

    <h2>Evolve <span class="muted">· propose → assess → commit</span></h2>
    <div class="row">
      <input type="text" id="goal" value="handle unicode and punctuation">
      <button id="evo">propose &amp; assess</button>
    </div>
    <div id="evoout"></div>

    <h2>Lineage</h2>
    <table><thead><tr><th>node</th><th>mutation</th><th>parents</th><th>reason</th></tr></thead>
      <tbody>${lin.nodes.map((n) => `<tr>
        <td class="mono">${esc(n.node_id)}</td><td class="mono">${esc(n.mutation)}</td>
        <td class="mono">${esc((n.parents || []).join(", ") || "—")}</td>
        <td class="muted">${esc(n.reason || "")}</td></tr>`).join("")}</tbody></table>

    <h2>Events <span class="muted">· append-only</span></h2>
    <div class="ev">${ev.events.length ? ev.events.map((e) => `
      <div><span class="sq">#${e.seq}</span>
        <span class="kd">${esc(e.kind)}</span>
        ${esc(JSON.stringify(e.data))}</div>`).join("") : '<div class="empty">none</div>'}</div>
  `;
  $("go").onclick = async () => {
    const text = $("arg").value;
    $("out").textContent = "running…";
    try {
      const res = await api(`/resources/${rid(id)}/invoke`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        // force is never sent on the user's behalf. An earlier version set it
        // whenever the resource was not ACTIVE, which silently defeated the
        // only guard that exists here: a QUARANTINED tool would have run anyway,
        // and the console would have shown a clean call. Enforcement you cannot
        // see on the screen that exists to show enforcement is worse than none.
        // A refused call now renders as a refusal, which is the point.
        body: JSON.stringify({ arguments: { text }, force: false }),
      });
      $("out").textContent =
        `${r.name}(${JSON.stringify(text)}) -> ${JSON.stringify(res.output)}\\n` +
        `ok=${res.ok}${res.error ? "  error=" + res.error : ""}`;
    } catch (e) { $("out").textContent = "error: " + e.message; }
    chips(); show(id);
  };
  $("evo").onclick = async () => {
    $("evoout").innerHTML = '<div class="muted">gate running…</div>';
    try {
      const res = await api(`/resources/${rid(id)}/evolve`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ goal: $("goal").value, commit: true }),
      });
      const vd = (res.assessment && res.assessment.verdicts) || [];
      $("evoout").innerHTML = `
        <table><thead><tr><th>#</th><th>verdict</th><th>fitness</th><th>reason</th></tr></thead>
        <tbody>${vd.map((v) => `<tr>
          <td class="mono">${v.index}</td>
          <td class="${v.admissible ? "admissible" : "vetoed"}">${v.admissible ? "ADMISSIBLE" : "VETOED"}</td>
          <td class="mono">${Number(v.fitness).toFixed(3)}</td>
          <td class="muted">${esc(v.reason)}</td></tr>`).join("")}</tbody></table>
        <div class="kv"><span class="k">committed</span>
          <code>${res.committed}${res.reject_reason ? " · " + esc(res.reject_reason) : ""}</code></div>`;
    } catch (e) { $("evoout").innerHTML = '<div class="muted">error: ' + esc(e.message) + "</div>"; }
    chips(); list(id);
  };
}

chips();
list(null);
</script>
</body>
</html>
"""


def build_app(registry: Optional[ResourceRegistry] = None,
              *, seed_demo: bool = True) -> Any:
    """The console: the JSON API, plus this page at the root.

    The API keeps its own routes untouched and lives under `/api` here purely so
    `/` can be the page. Nothing else about it changes — the same app object
    serves both.
    """
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    reg = registry
    if reg is None:
        reg = ResourceRegistry(ResourceStore(":memory:"))
        if seed_demo:
            from toolmarket import seed as _seed
            _seed.seed(reg)

    api = create_app(reg)

    app = FastAPI(title="tool-market console", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "resources": len(reg.list()),
                "events": len(reg.log), "chain_ok": reg.log.verify_chain()}

    @app.get("/metrics")
    def metrics_endpoint() -> Any:
        """The same exposition the API serves, reachable from the console's root.

        Duplicated here rather than proxied through `/api/metrics` so a scraper
        pointed at either process gets the same answer without knowing which is
        which. It reads the process-local registry, so scraping the console
        reports the console's counters — not the API's, which is the correct and
        only meaningful thing it could report.
        """
        from fastapi.responses import Response

        from toolmarket import metrics as _metrics
        return Response(content=_metrics.render(),
                        media_type="text/plain; version=0.0.4; charset=utf-8")

    # Mount the substrate's own app under /api, so its routes stay exactly as
    # they are and the page has a namespace of its own.
    app.mount("/api", api)

    # Wrap last, so the middleware sees every request that reaches either the
    # page or the API — including the ones the page's own fetches make, which is
    # what makes the dashboard's latency panels say anything true.
    from toolmarket.ratelimit import install

    return install(app)

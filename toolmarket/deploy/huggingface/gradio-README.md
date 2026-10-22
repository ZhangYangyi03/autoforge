---
title: toolmarket
emoji: 🧰
colorFrom: blue
colorTo: indigo
# Gradio, and this Space will not run on a free account: HuggingFace requires a
# PRO subscription for both Gradio and Docker Spaces on cpu-basic, and only Static
# Spaces (no server) are free. Verified against the API — `deploy.py` reports the
# 402 verbatim. This file exists for a PRO account or any host that runs a Python
# process; the free deployment story is `docker compose` and the Render blueprint.
sdk: gradio
app_file: app.py
app_port: 7860
pinned: false
license: mit
short_description: Protocol-registered resource platform for evolvable agent tools
---

# toolmarket

A protocol-registered resource platform for evolvable agent tools, with the AGP
resource substrate enforced on [autoforge](https://github.com/ZhangYangyi03/autoforge).

The **API is at the root of this Space**, not behind the UI:

| Path | What it is |
|---|---|
| `/docs` | the interactive schema (FastAPI's own) |
| `/health` | process-local liveness, plus the event hash chain's verdict |
| `/ready` | store and cache probed separately; `503` names the failure |
| `/metrics` | Prometheus exposition, including the chain-intact gauge |
| `/stats` | resource counts by lifecycle state |
| `/resources` | register (`POST`) and list (`GET`) tools |
| `/resources/{id}/evolve` | propose → assess → commit, behind the real gate |
| `/resources/{id}/evolve/async` + `/tasks/{id}` | the queued path |

`/ui` is a small panel that fetches those endpoints over HTTP, so what it shows
is what a client would get.

## What this deployment is, and is not

This Space is the **single-process** configuration: in-memory store, in-process
cache, inline evolution queue. The substrate is lost on rebuild, and that is the
only thing lost — the enforcement, the lifecycle rules, the hash-chained event
log and the lineage DAG are the same code the tests run against.

The durable configuration — Postgres, Redis, a Celery worker, and optionally
Prometheus and Grafana — is `docker-compose.yml` in the source repository:

    docker compose up -d
    docker compose --profile observability up -d

The README there states what each backend changes about behaviour, rather than
just what it adds to a diagram.

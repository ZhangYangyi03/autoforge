---
title: toolmarket
emoji: 🧰
colorFrom: blue
colorTo: indigo
sdk: docker
# Must match the Dockerfile's PORT. A Docker Space routes only the port named
# here, and a mismatch produces a Space that builds, runs, logs nothing, and
# serves 404 on every path — the least diagnosable failure in this repo.
app_port: 8000
pinned: false
license: mit
short_description: Protocol-registered resource platform for evolvable agent tools
---

# toolmarket

A protocol-registered resource platform for evolvable agent tools, with the
AGP resource substrate enforced on [autoforge](https://github.com/).

This Space runs the API — the same image `docker compose up` builds.

## Why this deployment exists

The Space is deliberately the **single-process** configuration: in-memory store,
in-process cache, inline evolution queue, and no monitoring stack. That is not a
cut-down demo of the compose stack; it is the configuration that makes the
substrate's own claims checkable from a URL:

- `GET /health` — process-local liveness, plus the event hash chain's verdict
- `GET /ready` — the store and the cache, each probed separately
- `GET /metrics` — Prometheus exposition, including the chain-intact gauge
- `GET /stats` — resource counts by lifecycle state
- `POST /resources/{id}/evolve` — propose, assess, commit behind the real gate
- `POST /resources/{id}/evolve/async` + `GET /tasks/{id}` — the queued path

## The durable configuration

Postgres, Redis and a Celery worker need services a Space does not provide, so
they are not here. They are in `docker-compose.yml` in the source repo, along
with Prometheus and Grafana:

    docker compose up -d
    docker compose --profile observability up -d

The README in the repo is the source of truth for the deployment story and states
what each backend changes about behaviour. The short version: this Space loses
its substrate on rebuild, and that is the only thing it loses.

## Plan requirement, stated plainly

HuggingFace requires a **PRO subscription** to host a Docker (or Gradio) Space on
free `cpu-basic`; only Static Spaces are free, and a static Space cannot run this
API. Creating this Space on a free account fails with `402`, which
`deploy/huggingface/deploy.py` reports verbatim. For a free deployment, use
`docker compose up` locally or the Render blueprint in `deploy/render/`.

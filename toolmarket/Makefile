# The commands somebody actually runs. Everything here is a one-liner over the
# real tool, so nothing is hidden behind a make target that could drift from what
# the README tells a reader to type.

PY ?= python
COMPOSE ?= docker compose
IMAGE ?= toolmarket:local
API_PORT ?= 8000

# -- kubernetes -------------------------------------------------------------
# One cluster per release name, so two releases cannot fight over node ports or
# over the Postgres PVC. `kind` and `helm` are the real tools; every target
# below is a thin wrapper so nothing here can drift from what the README says
# to type.
#
# KUBECONFIG is repaired before use. Under git-bash it is exported as an MSYS
# path (`/c/Users/.../.kube/config`), and the native kubectl.exe does not
# understand that: it does not fail, it resolves to an empty config, so every
# kubectl call reports "context does not exist" while the file sits right
# there. Converting the one leading segment to the Windows form fixes it, and
# is a no-op on Linux and macOS where KUBECONFIG is already correct.
ifeq ($(OS),Windows_NT)
ifneq (,$(findstring /c/,$(KUBECONFIG)))
KUBECONFIG := $(subst /c/,C:/,$(KUBECONFIG))
export KUBECONFIG
endif
endif
K8S_CLUSTER ?= toolmarket
K8S_NAMESPACE ?= toolmarket
# Note: the chart's fullname resolves to the release name when the release name
# already contains the chart name, so `tool-market` + `tool-market` gives the
# service `tool-market-api`. Changing the release name changes that hostname.
K8S_RELEASE ?= tool-market
K8S_CONTEXT ?= kind-$(K8S_CLUSTER)
# Deliberately not API_PORT. `docker compose up` publishes the API on API_PORT,
# and if the k8s smoke reused it the port-forward would fail to bind while curl
# quietly talked to the *compose* container -- a smoke test reporting on a stack
# it was not asked to test. A separate port lets both run at once, which is also
# how you compare them.
K8S_PORT ?= 18000
# Same reasoning one line up, and for the same failure: `tf-ready` used to curl
# `$(API_PORT)`, which is the *compose* stack's port. With compose up, `make
# tf-ready` answered 200 from a container Terraform never created, so the target
# whose entire job is "prove this stack found Postgres and Redis" reported on a
# different stack. Its own port, so the two can run side by side.
TF_API_PORT ?= 18001

.PHONY: help install test lint check-config grpc-gen compose-up compose-obs compose-down \
        compose-logs smoke verify image psql redis-cli hf-deploy hf-deploy-docker \
        render-init tf-init tf-plan tf-apply tf-ready tf-destroy k8s-lint k8s-up \
        k8s-status k8s-smoke k8s-down clean

help:  ## list targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /  - /'

install:  ## editable install with every backend
	$(PY) -m pip install -e ".[all]"

# Referenced by .gitignore, by tools/grpc_gen.py and by the error message in
# toolmarket/grpc/server.py. It existed in all three places and in none of this
# file until CI went red, which is the drift a doc-cited target is prone to.
grpc-gen:  ## regenerate the gRPC stubs from proto/ (needs grpcio-tools)
	$(PY) tools/grpc_gen.py

test:  ## the whole suite (no server needed; the Postgres cases skip)
	$(PY) -m pytest tests -q

lint:  ## the CI lint selection
	$(PY) -m ruff check --select E9,F63,F7,F82,F821 toolmarket tests tools

check-config:  ## validate the prometheus/grafana files (same script CI runs)
	$(PY) tools/check_dashboards.py

image:  ## build the runtime image
	docker build --target runtime -t $(IMAGE) .

compose-up:  ## core stack: api + worker + postgres + redis (~500 MB)
	$(COMPOSE) up -d --build
	@echo "api      -> http://127.0.0.1:$(API_PORT)/docs"
	@echo "health   -> http://127.0.0.1:$(API_PORT)/health"
	@echo "metrics  -> http://127.0.0.1:$(API_PORT)/metrics"

compose-obs:  ## core stack + prometheus + grafana (needs ~1 GB free)
	$(COMPOSE) --profile observability up -d --build
	@echo "grafana    -> http://127.0.0.1:3000  (dashboard: toolmarket / substrate)"
	@echo "prometheus -> http://127.0.0.1:9090"

compose-down:  ## stop everything, keep volumes
	$(COMPOSE) --profile observability down

compose-logs:  ## follow the api and worker
	$(COMPOSE) logs -f api worker

psql:  ## a psql shell in the database container
	$(COMPOSE) exec postgres psql -U toolmarket -d toolmarket

redis-cli:  ## a redis-cli shell
	$(COMPOSE) exec redis redis-cli

verify:  ## the live Postgres suite against the running stack
	@./deploy/verify.sh

smoke:  ## end-to-end against a running stack: health, ready, metrics, async
	@./deploy/smoke.sh

hf-deploy:  ## publish the API to a HuggingFace Space (gradio mode; free)
	$(PY) deploy/huggingface/deploy.py --mode gradio

hf-deploy-docker:  ## publish the Dockerfile to a Space (needs a PRO account)
	$(PY) deploy/huggingface/deploy.py --mode docker

render-init:  ## copy the Render blueprint to the repo root (Render reads it there)
	cp deploy/render/render.yaml render.yaml
	@echo "render.yaml copied; commit it and point a Blueprint at the repo"

tf-init:  ## terraform: download the docker provider (~25 MB, once)
	cd terraform && terraform init

tf-plan:  ## terraform: show every change before any of it happens
	cd terraform && terraform plan -var api_port=$(TF_API_PORT)

tf-apply:  ## terraform: bring up postgres + redis + api + worker as a graph
	cd terraform && terraform apply -var api_port=$(TF_API_PORT)

tf-ready:  ## terraform: prove the stack found Postgres/Redis, not the fallbacks
	@curl -s "$$(cd terraform && terraform output -raw ready_url)"

tf-destroy:  ## terraform: tear the stack down (base images are kept)
	cd terraform && terraform destroy

.PHONY: k8s-lint k8s-up k8s-status k8s-smoke k8s-down

k8s-lint:  ## helm lint the chart and render every template (no cluster needed)
	helm lint charts/tool-market
	helm template $(K8S_RELEASE) charts/tool-market > /dev/null

k8s-up:  ## kind cluster + helm install, running the same image as every other path
	@docker image inspect $(IMAGE) > /dev/null 2>&1 || { echo "no image $(IMAGE) -- run 'make image' first"; exit 1; }
	@kind get clusters 2>/dev/null | grep -qx '$(K8S_CLUSTER)' || kind create cluster --name $(K8S_CLUSTER)
	kind load docker-image $(IMAGE) --name $(K8S_CLUSTER)
	helm --kube-context $(K8S_CONTEXT) upgrade --install $(K8S_RELEASE) charts/tool-market \
	  --namespace $(K8S_NAMESPACE) --create-namespace --wait --timeout 5m

k8s-status:  ## what the cluster believes is running
	kubectl --context $(K8S_CONTEXT) -n $(K8S_NAMESPACE) get deploy,po,svc,pvc

k8s-smoke:  ## the same deploy/smoke.sh the compose stack passes, through a port-forward
	@kubectl --context $(K8S_CONTEXT) -n $(K8S_NAMESPACE) port-forward svc/$(K8S_RELEASE)-api $(K8S_PORT):$(API_PORT) > /dev/null 2>&1 & \
	  pf=$$!; trap 'kill $$pf 2>/dev/null' EXIT; \
	  for _ in $$(seq 1 40); do curl -sf -o /dev/null http://127.0.0.1:$(K8S_PORT)/health && break; sleep 1; done; \
	  BASE=http://127.0.0.1:$(K8S_PORT) ./deploy/smoke.sh

k8s-down:  ## uninstall the release and delete the cluster; nothing stays resident
	-helm --kube-context $(K8S_CONTEXT) uninstall $(K8S_RELEASE) --namespace $(K8S_NAMESPACE)
	-kind delete cluster --name $(K8S_CLUSTER)

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info build dist render.yaml

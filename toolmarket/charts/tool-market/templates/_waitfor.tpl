{{/* A dependency gate, because Kubernetes has no `depends_on`.

     Compose guarantees ordering with `depends_on: {condition: service_healthy}`:
     the API container is not created until Postgres and Redis answer. Nothing
     in a PodSpec does that. All containers in a Pod start together, and a
     Deployment's replicas start whenever the scheduler places them -- so on a
     cold cluster the API routinely wins the race against its own database.

     That race is not a slow start, it is a permanently broken process. The app
     builds its ASGI object at import time (`toolmarket/api/main.py`), and
     `ResourceRegistry.__init__` constructs the store eagerly, which opens a
     connection pool in `PostgresStore.__init__`. If Postgres is not listening
     yet, the import raises, a bare `except Exception: served_app = None` in the
     module swallows it, and the process serves a `None` application forever:
     every probe answers 500, the liveness probe restarts the container, and the
     loop repeats. The fix belongs here, as ordering, and not as a probe
     relaxation that would hide the same failure in production.

     This runs the app's own image so the gate speaks the same DNS and the same
     protocols as the process it is gating, and it reads the same Secret, so it
     cannot pass by checking a different address than the app will dial. */ -}}
{{- define "tool-market.waitForDeps" -}}
- name: wait-for-deps
  image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
  imagePullPolicy: {{ .Values.image.pullPolicy }}
  command:
    - python
    - -c
    - |
      import os, socket, sys, time

      targets = []
      dsn = os.environ.get("TOOLMARKET_STORE", "")
      if dsn.startswith(("postgres://", "postgresql://")):
          hostport = dsn.split("://", 1)[1].split("@")[-1].split("/")[0]
          host, _, port = hostport.partition(":")
          targets.append((host, int(port or 5432)))
      url = os.environ.get("REDIS_URL", "")
      if url.startswith("redis://"):
          hostport = url.split("://", 1)[1].split("/")[0].split("@")[-1]
          host, _, port = hostport.partition(":")
          targets.append((host, int(port or 6379)))

      if not targets:
          print("no external dependencies configured; nothing to wait for")
          sys.exit(0)

      deadline = time.time() + 300
      for host, port in targets:
          while True:
              try:
                  socket.create_connection((host, port), 3).close()
                  print(f"reachable: {host}:{port}", flush=True)
                  break
              except OSError as exc:
                  if time.time() > deadline:
                      sys.exit(f"gave up waiting for {host}:{port} after 300s: {exc}")
                  print(f"waiting for {host}:{port} ({exc})", flush=True)
                  time.sleep(2)
  env:
    {{- include "tool-market.substrateEnv" . | nindent 4 }}
{{- end }}

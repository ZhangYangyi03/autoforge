# Outputs are what make the stack usable without reading `docker ps`. Each one is
# a value the next command needs, so nothing here is decorative.

output "api_url" {
  description = "Base URL of the API."
  value       = "http://127.0.0.1:${var.api_port}"
}

output "ready_url" {
  description = "Readiness probe. Reports the backend it actually connected to — the call that tells you whether the Terraform stack found Postgres and Redis or silently fell back."
  value       = "http://127.0.0.1:${var.api_port}/ready"
}

output "metrics_url" {
  description = "Prometheus exposition."
  value       = "http://127.0.0.1:${var.api_port}/metrics"
}

output "store_dsn" {
  description = "The DSN the API was given. Reachable from a peer container on the network, not from the host."
  value       = "postgresql://toolmarket:${var.postgres_password}@postgres:5432/toolmarket"
  sensitive   = true
}

output "containers" {
  description = "Container names, for `docker logs` and `docker exec`."
  value = compact([
    docker_container.postgres.name,
    docker_container.redis.name,
    docker_container.api.name,
    var.task_queue == "celery" ? docker_container.worker[0].name : "",
  ])
}

output "task_queue" {
  description = "Which queue backend this apply wired in. `celery` means a worker container exists; `inline` means evolutions run on a thread of the API process."
  value       = var.task_queue
}

{{/* Helpers. Names and labels follow the Helm convention; the DSN helpers
     exist so the *one* place a password is generated is the Secret, and every
     consumer reads it back from there rather than recomputing it. */}}

{{- define "tool-market.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "tool-market.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "tool-market.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "tool-market.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tool-market.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "tool-market.labels" -}}
helm.sh/chart: {{ include "tool-market.chart" . }}
{{ include "tool-market.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "tool-market.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "tool-market.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "tool-market.substrateSecret" -}}
{{- printf "%s-substrate" (include "tool-market.fullname" .) }}
{{- end }}

{{- define "tool-market.postgresHost" -}}
{{- printf "%s-postgres" (include "tool-market.fullname" .) }}
{{- end }}

{{- define "tool-market.redisHost" -}}
{{- printf "%s-redis" (include "tool-market.fullname" .) }}
{{- end }}

{{/*
  The Redis URLs. Bundled: three database indexes, the same split compose
  makes (/0 cache, /1 broker, /2 results) so a task record and a cached read
  can never collide on a key. External: one URL unless the caller separated
  them, since a chart cannot know what indexes somebody else's Redis uses.
*/}}
{{- define "tool-market.redis.url" -}}
{{- if .Values.redis.enabled -}}
{{- printf "redis://%s:6379/0" (include "tool-market.redisHost" .) -}}
{{- else -}}
{{- required "redis.enabled=false requires externalRedis.url" .Values.externalRedis.url -}}
{{- end -}}
{{- end }}

{{- define "tool-market.redis.brokerUrl" -}}
{{- if .Values.redis.enabled -}}
{{- printf "redis://%s:6379/1" (include "tool-market.redisHost" .) -}}
{{- else -}}
{{- default .Values.externalRedis.url .Values.externalRedis.brokerUrl -}}
{{- end -}}
{{- end }}

{{- define "tool-market.redis.resultBackendUrl" -}}
{{- if .Values.redis.enabled -}}
{{- printf "redis://%s:6379/2" (include "tool-market.redisHost" .) -}}
{{- else -}}
{{- default .Values.externalRedis.url .Values.externalRedis.resultBackendUrl -}}
{{- end -}}
{{- end }}

{{/*
  The substrate environment, shared by the API and the worker.

  It is one helper rather than two copies because the failure mode of two
  copies is silent: an API pointed at Postgres and a worker still writing to
  sqlite both look healthy, and the symptom is an async evolution whose result
  never appears.
*/}}
{{- define "tool-market.substrateEnv" -}}
- name: TOOLMARKET_STORE
  valueFrom:
    secretKeyRef:
      name: {{ include "tool-market.substrateSecret" . }}
      key: dsn
- name: REDIS_URL
  valueFrom:
    secretKeyRef:
      name: {{ include "tool-market.substrateSecret" . }}
      key: redisUrl
- name: CELERY_BROKER_URL
  valueFrom:
    secretKeyRef:
      name: {{ include "tool-market.substrateSecret" . }}
      key: brokerUrl
- name: CELERY_RESULT_BACKEND
  valueFrom:
    secretKeyRef:
      name: {{ include "tool-market.substrateSecret" . }}
      key: resultBackendUrl
- name: TASK_QUEUE
  value: {{ .Values.config.taskQueue | quote }}
- name: CACHE_TTL
  value: {{ .Values.config.cacheTtl | quote }}
- name: RATE_LIMIT
  value: {{ .Values.config.rateLimit | quote }}
- name: RATE_LIMIT_WINDOW
  value: {{ .Values.config.rateLimitWindow | quote }}
- name: TRUST_PROXY
  value: {{ .Values.config.trustProxy | quote }}
- name: PYTHONUNBUFFERED
  value: "1"
{{- end }}

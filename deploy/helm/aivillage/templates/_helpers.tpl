{{/*
Expand the name of the chart.
*/}}
{{- define "aivillage.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "aivillage.fullname" -}}
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

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "aivillage.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "aivillage.labels" -}}
helm.sh/chart: {{ include "aivillage.chart" . }}
{{ include "aivillage.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "aivillage.selectorLabels" -}}
app.kubernetes.io/name: {{ include "aivillage.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "aivillage.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "aivillage.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Common labels for a specific service
*/}}
{{- define "aivillage.serviceLabels" -}}
{{- $serviceName := index . 0 -}}
{{- $context := index . 1 -}}
helm.sh/chart: {{ include "aivillage.chart" $context }}
{{ include "aivillage.serviceSelectorLabels" . }}
{{- if $context.Chart.AppVersion }}
app.kubernetes.io/version: {{ $context.Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ $context.Release.Service }}
app.kubernetes.io/component: {{ $serviceName }}
{{- end }}

{{/*
Selector labels for a specific service
*/}}
{{- define "aivillage.serviceSelectorLabels" -}}
{{- $serviceName := index . 0 -}}
{{- $context := index . 1 -}}
app.kubernetes.io/name: {{ include "aivillage.name" $context }}-{{ $serviceName }}
app.kubernetes.io/instance: {{ $context.Release.Name }}
{{- end }}

{{/*
Create PostgreSQL connection string
*/}}
{{- define "aivillage.postgresql.connectionString" -}}
{{- if .Values.postgresql.enabled }}
postgresql://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ include "aivillage.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- else }}
{{ .Values.externalDatabase.connectionString }}
{{- end }}
{{- end }}

{{/*
Create Redis connection string
*/}}
{{- define "aivillage.redis.connectionString" -}}
{{- if .Values.redis.enabled }}
redis://:{{ .Values.redis.auth.password }}@{{ include "aivillage.fullname" . }}-redis-master:6379/0
{{- else }}
{{ .Values.externalRedis.connectionString }}
{{- end }}
{{- end }}

{{/*
Create Neo4j connection string
*/}}
{{- define "aivillage.neo4j.connectionString" -}}
{{- if .Values.neo4j.enabled }}
bolt://neo4j:{{ .Values.neo4j.auth.password }}@{{ include "aivillage.fullname" . }}-neo4j:7687
{{- else }}
{{ .Values.externalNeo4j.connectionString }}
{{- end }}
{{- end }}

{{/*
Create Qdrant connection string
*/}}
{{- define "aivillage.qdrant.connectionString" -}}
{{- if .Values.qdrant.enabled }}
http://{{ include "aivillage.fullname" . }}-qdrant:6333
{{- else }}
{{ .Values.externalQdrant.connectionString }}
{{- end }}
{{- end }}

{{/*
Create image name helper
*/}}
{{- define "aivillage.image" -}}
{{- $serviceName := index . 0 -}}
{{- $serviceConfig := index . 1 -}}
{{- $context := index . 2 -}}
{{- $registry := $context.Values.global.imageRegistry | default "" -}}
{{- $repository := $serviceConfig.image.repository -}}
{{- $tag := $serviceConfig.image.tag | default $context.Chart.AppVersion -}}
{{- if $registry -}}
{{ $registry }}/{{ $repository }}:{{ $tag }}
{{- else -}}
{{ $repository }}:{{ $tag }}
{{- end -}}
{{- end }}

{{/*
Security context helper
*/}}
{{- define "aivillage.securityContext" -}}
{{- if .Values.security.securityContext }}
{{- toYaml .Values.security.securityContext }}
{{- end }}
{{- end }}

{{/*
Pod security context helper
*/}}
{{- define "aivillage.podSecurityContext" -}}
{{- if .Values.security.podSecurityContext }}
{{- toYaml .Values.security.podSecurityContext }}
{{- end }}
{{- end }}

{{/*
Resource helper
*/}}
{{- define "aivillage.resources" -}}
{{- $serviceConfig := . -}}
{{- if $serviceConfig.resources }}
{{- toYaml $serviceConfig.resources }}
{{- end }}
{{- end }}

{{/*
Environment variables helper
*/}}
{{- define "aivillage.commonEnv" -}}
- name: DATABASE_URL
  value: {{ include "aivillage.postgresql.connectionString" . | quote }}
- name: REDIS_URL
  value: {{ include "aivillage.redis.connectionString" . | quote }}
- name: NEO4J_URI
  value: {{ include "aivillage.neo4j.connectionString" . | quote }}
- name: QDRANT_URL
  value: {{ include "aivillage.qdrant.connectionString" . | quote }}
- name: ENVIRONMENT
  value: {{ .Values.environment | default "production" | quote }}
- name: LOG_LEVEL
  value: {{ .Values.logLevel | default "INFO" | quote }}
{{- if .Values.secrets.openaiApiKey }}
- name: OPENAI_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "aivillage.fullname" . }}-secrets
      key: openai-api-key
{{- end }}
{{- if .Values.secrets.anthropicApiKey }}
- name: ANTHROPIC_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "aivillage.fullname" . }}-secrets
      key: anthropic-api-key
{{- end }}
{{- end }}

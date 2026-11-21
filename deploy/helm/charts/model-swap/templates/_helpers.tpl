{{/* Chart name, overridable. */}}
{{- define "model-swap.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Release-qualified name for every object the chart creates. */}}
{{- define "model-swap.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "model-swap.watchNamespace" -}}
{{- default .Release.Namespace .Values.watchNamespace -}}
{{- end -}}

{{- define "model-swap.selectorLabels" -}}
app.kubernetes.io/name: {{ include "model-swap.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "model-swap.labels" -}}
{{ include "model-swap.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{/* The controller code, one key per file; its hash rolls the controller pod on every code change. */}}
{{- define "model-swap.code" -}}
{{ (.Files.Glob "controller/*.py").AsConfig }}
{{- end -}}

{{/* The cache store claim the seed and publish Jobs mount. */}}
{{- define "model-swap.cacheStoreClaim" -}}
{{- default (printf "%s-cache-store" (include "model-swap.fullname" .)) .Values.cacheStore.existingClaim -}}
{{- end -}}

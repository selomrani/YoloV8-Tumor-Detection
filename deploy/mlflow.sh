#!/bin/bash
# MLflow tracking server wrapper.
#
# On a Vast instance this lives at /opt/supervisor-scripts/mlflow.sh and is
# launched by /etc/supervisor/conf.d/mlflow.conf. On a local machine just run it
# directly (or copy it next to a systemd unit).
#
# Overridable via environment:
#   MLFLOW_PORT            (default 5001)
#   MLFLOW_BACKEND_STORE   (default sqlite:///<project>/mlruns/mlflow.db)
#   MLFLOW_ARTIFACTS_DEST  (default <project>/mlruns/artifacts)
#                          set to wasbs://<container>@<account>.blob.core.windows.net/mlflow
#                          to keep artifacts in Azure Blob Storage instead of on disk.
#   AZURE_STORAGE_CONNECTION_STRING  required when MLFLOW_ARTIFACTS_DEST is wasbs://

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLFLOW_HOME="${PROJECT_DIR}/mlruns"

MLFLOW_PORT="${MLFLOW_PORT:-5001}"
MLFLOW_BACKEND_STORE="${MLFLOW_BACKEND_STORE:-sqlite:///${MLFLOW_HOME}/mlflow.db}"
MLFLOW_ARTIFACTS_DEST="${MLFLOW_ARTIFACTS_DEST:-${MLFLOW_HOME}/artifacts}"

mkdir -p "${MLFLOW_HOME}/artifacts"

echo "starting mlflow on 127.0.0.1:${MLFLOW_PORT}"
echo "  backend-store-uri:    ${MLFLOW_BACKEND_STORE}"
echo "  artifacts-destination: ${MLFLOW_ARTIFACTS_DEST}"

exec "${PROJECT_DIR}/.venv/bin/mlflow" server \
    --host 127.0.0.1 \
    --port "${MLFLOW_PORT}" \
    --backend-store-uri "${MLFLOW_BACKEND_STORE}" \
    --artifacts-destination "${MLFLOW_ARTIFACTS_DEST}"

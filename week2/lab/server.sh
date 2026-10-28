#!/usr/bin/env bash
# MLflow 추적 서버를 sqlite backend로 띄운다 (포트 5055, 127.0.0.1 전용).
#   LAB_DIR=/path/to/work ./server.sh [추가 env는 앞에 붙여서]
# 예: MLFLOW_WEBHOOK_REQUEST_TIMEOUT=3 LAB_DIR=... ./server.sh
set -euo pipefail
LAB_DIR="${LAB_DIR:?LAB_DIR (venv·DB·로그 둘 디렉터리) 를 지정할 것}"
BACKEND="${BACKEND:-sqlite:///$LAB_DIR/mlflow.db}"
mkdir -p "$LAB_DIR/artifacts"

# 웹훅 수신기가 127.0.0.1 이라 SSRF 보호(사설 IP 차단)와 https 전용 제한을 로컬 실험용으로 푼다.
export MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS="${MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS:-true}"
export MLFLOW_WEBHOOK_ALLOWED_SCHEMES="${MLFLOW_WEBHOOK_ALLOWED_SCHEMES:-http,https}"
# 고정하지 않으면 프로세스마다 Fernet 키를 새로 만들어, 재시작 후 저장된 secret을 복호화하지 못한다.
export MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY="${MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY:-$(cat "$LAB_DIR/fernet.key" 2>/dev/null || true)}"
export MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1

exec "$LAB_DIR/venv/bin/mlflow" server \
  --backend-store-uri "$BACKEND" \
  --artifacts-destination "$LAB_DIR/artifacts" \
  --host 127.0.0.1 --port 5055 --workers 1

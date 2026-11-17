#!/usr/bin/env bash
# MLflow tracking 서버 (sqlite + 로컬 아티팩트 프록시) 를 5065에 띄운다
source "$(dirname "$0")/env.sh"
cd "$LAB"
nohup mlflow server --host 127.0.0.1 --port $PORT_TRACK \
  --backend-store-uri "sqlite:///$LAB/mlflow.db" --artifacts-destination "$LAB/mlartifacts" \
  > "$LAB/logs/tracking.log" 2>&1 &
echo $! > "$LAB/logs/tracking.pid"
for i in $(seq 60); do curl -sf localhost:$PORT_TRACK/health >/dev/null && { echo tracking up; exit 0; }; sleep 1; done
echo "tracking failed"; tail -20 "$LAB/logs/tracking.log"; exit 1

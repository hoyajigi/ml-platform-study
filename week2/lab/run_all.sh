#!/usr/bin/env bash
# 2주차 웹훅 실험 전체 재현. 포트 5055(MLflow)·5056(수신기)를 쓴다.
#   LAB_DIR=/path/to/work ./run_all.sh
# LAB_DIR 에 venv 를 만들고 mlflow 를 설치한다. 결과는 ./results/ 에 jsonl 로 쌓인다.
set -euo pipefail
cd "$(dirname "$0")"
LAB_DIR="${LAB_DIR:?LAB_DIR 지정}"; export LAB_DIR
R="${RESULTS:-results}"   # 결과 디렉터리 (검증용으로 바꿀 수 있음)
mkdir -p "$LAB_DIR" "$R"; export RESULTS="$R"
[ -x "$LAB_DIR/venv/bin/python" ] || { python3 -m venv "$LAB_DIR/venv"; "$LAB_DIR/venv/bin/pip" install -q mlflow; }
[ -f "$LAB_DIR/fernet.key" ] || "$LAB_DIR/venv/bin/python" -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())" > "$LAB_DIR/fernet.key"
PY="$LAB_DIR/venv/bin/python"
export MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_HTTP_REQUEST_MAX_RETRIES=0
PIDFILE="$LAB_DIR/server.pid"
N=0
start() { N=$((N+1)); "$@" ./server.sh > "$LAB_DIR/server_run_all_$N.log" 2>&1 & echo $! > "$PIDFILE"; ./wait_port.sh 5055 120; }
trap './stop_server.sh "$PIDFILE" 2>/dev/null || true; [ -n "${RX:-}" ] && kill $RX 2>/dev/null || true' EXIT

# 0) 필요 조건: 파일 스토어 / 기본 SSRF 설정
start env MLFLOW_ALLOW_FILE_STORE=true BACKEND="$LAB_DIR/filestore"
timeout 60 "$PY" prereq_check.py filestore_allowed http://127.0.0.1:5056/hook || true
./stop_server.sh "$PIDFILE"
start env MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS=false MLFLOW_WEBHOOK_ALLOWED_SCHEMES=https
for u in http://127.0.0.1:5056/hook https://127.0.0.1:5056/hook; do timeout 60 "$PY" prereq_check.py sqlite_default_env "$u"; done
./stop_server.sh "$PIDFILE"

# 1~2) 시나리오 a~m, 캐시 TTL (기본 TTL 60초)
start env
"$PY" receiver.py --port 5056 --log "$LAB_DIR/recv_main.jsonl" --secret lab-secret-123 2>/dev/null & RX=$!
./wait_port.sh 5056 10
"$PY" cache_ttl_test.py --recv-log "$LAB_DIR/recv_main.jsonl" --out "$R"/cache_ttl.jsonl
sleep 62   # 캐시 만료 대기
for n in 1 2 3; do "$PY" run_scenarios.py --recv-log "$LAB_DIR/recv_main.jsonl" --out "$R/scenarios_run$n.jsonl"; sleep 62; done
kill $RX; RX=; ./stop_server.sh "$PIDFILE"

# 3~4) 전달 실패·재시도·블로킹 (TTL 1초)
start env MLFLOW_WEBHOOK_CACHE_TTL=1
"$PY" failure_test.py --work "$LAB_DIR" --out "$R"/failure.jsonl F1 F2 F3 F4 B0 B1
./stop_server.sh "$PIDFILE"
start env MLFLOW_WEBHOOK_CACHE_TTL=1 MLFLOW_WEBHOOK_REQUEST_TIMEOUT=3
"$PY" failure_test.py --work "$LAB_DIR" --out "$R"/failure_timeout.jsonl T1
./stop_server.sh "$PIDFILE"

# 5) 암호화 키가 바뀐 서버
NEWKEY=$("$PY" -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())")
start env MLFLOW_WEBHOOK_CACHE_TTL=1 MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY="$NEWKEY"
"$PY" key_rotation_test.py --work "$LAB_DIR" --out "$R"/key_rotation.json

#!/usr/bin/env bash
# 포트가 LISTEN 될 때까지 최대 N초 대기: wait_port.sh 5055 60
for _ in $(seq 1 "${2:-60}"); do
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1 && exit 0; sleep 1
done
echo "port $1 not listening" >&2; exit 1

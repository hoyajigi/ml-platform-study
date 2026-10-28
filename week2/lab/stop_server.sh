#!/usr/bin/env bash
# server.sh 로 띄운 서버(및 uvicorn 자식)만 종료: stop_server.sh <pidfile>
pid=$(cat "$1"); pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null
for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || exit 0; sleep 0.5; done
kill -9 "$pid" 2>/dev/null; true

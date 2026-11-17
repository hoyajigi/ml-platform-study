#!/usr/bin/env bash
# 사용: ./serve.sh start|stop mlserve|live|built
# mlserve: mlflow models serve -m models:/NAME@champion (5066)
# live   : bentoml serve service_live:IncomeLive (5067, 시작 시 alias 해석+import)
# built  : bentoml serve <빌드된 bento 최신> (5068)
source "$(dirname "$0")/env.sh"
act=$1; kind=$2; pidf="$LAB/logs/$kind.pid"; log="$LAB/logs/$kind.log"
case $kind in mlserve) port=$PORT_MLSERVE;; live) port=$PORT_BENTO;; built) port=$PORT_BENTO_BUILT;; *) exit 2;; esac
if [ "$act" = stop ]; then
  [ -f "$pidf" ] && pkill -TERM -P "$(cat $pidf)" 2>/dev/null; [ -f "$pidf" ] && kill "$(cat $pidf)" 2>/dev/null
  for i in $(seq 30); do lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null || { echo "$kind stopped"; rm -f $pidf; exit 0; }; sleep 0.5; done
  echo "$kind still listening on $port"; exit 1
fi
cd "$HERE"
echo "=== start $(date +%T)" >> "$log"
case $kind in
  mlserve) nohup mlflow models serve -m "models:/$MODEL_NAME@champion" --env-manager local \
             --host 127.0.0.1 -p $port >> "$log" 2>&1 & ;;
  live)    nohup bentoml serve service_live:IncomeLive --host 127.0.0.1 --port $port >> "$log" 2>&1 & ;;
  built)   nohup bentoml serve "income_built:latest" --host 127.0.0.1 --port $port >> "$log" 2>&1 & ;;
esac
echo $! > "$pidf"
t0=$(date +%s)
for i in $(seq 240); do
  code=$(curl -s -o /dev/null -w '%{http_code}' localhost:$port/readyz 2>/dev/null)
  [ "$kind" = mlserve ] && code=$(curl -s -o /dev/null -w '%{http_code}' localhost:$port/ping 2>/dev/null)
  [ "$code" = 200 ] && { echo "$kind up on $port in $(( $(date +%s)-t0 ))s"; exit 0; }; sleep 0.5
done
echo "$kind failed"; tail -30 "$log"; exit 1

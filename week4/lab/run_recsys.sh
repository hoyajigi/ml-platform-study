#!/usr/bin/env bash
# 9장 영화 추천 모델 재현성 매트릭스. 순차 실행. 사용: LAB=<작업 디렉터리> bash run_recsys.sh
set -euo pipefail
: "${LAB:?LAB 환경변수 필요}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$LAB/.venv/bin/python"
R="$LAB/mlens/ml-latest-small/ratings.csv"
SUM="$LAB/recsys_runs.jsonl"
export MLFLOW_TRACKING_URI="sqlite:///$LAB/mlflow.db" MLFLOW_EXPERIMENT_NAME="recsys-repro"
mkdir -p "$LAB/logs"
: > "$SUM"

run() { # name device seed det workers
  echo "== $1 $(date +%T)"
  "$PY" "$HERE/train_recsys.py" --ratings "$R" --device "$2" --seed "$3" --deterministic "$4" \
    --num-workers "$5" --epochs 3 --name "$1" --outdir "$LAB/recsys" --summary "$SUM" > "$LAB/logs/rec_$1.log" 2>&1
  tail -1 "$LAB/logs/rec_$1.log"
}
run cpu_s0_a cpu 0 0 0
run cpu_s0_b cpu 0 0 0
run mps_s0_a mps 0 0 0
run mps_s0_b mps 0 0 0
run mps_s0_det mps 0 1 0
run cpu_noseed_a cpu -1 0 0   # 원본 코드처럼 seed 미설정
run cpu_noseed_b cpu -1 0 0
run cpu_s0_w2 cpu 0 0 2       # DataLoader num_workers=2
echo "== done $(date +%T)"

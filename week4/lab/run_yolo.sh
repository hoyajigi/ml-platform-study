#!/usr/bin/env bash
# YOLO 재현성 매트릭스 + 도전과제(a)(b). 순차 실행(측정 위생: 동시에 두 run을 돌리지 않는다).
# 사용: LAB=<작업 디렉터리> bash run_yolo.sh
set -euo pipefail
: "${LAB:?LAB 환경변수(venv·data·runs가 있는 디렉터리) 필요}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$LAB/.venv/bin/python"
DATA="$LAB/data/midv_alb/data.yaml"
W="$LAB/dl/yolov8n.pt"
SUM="$LAB/yolo_runs.jsonl"
export YOLO_CONFIG_DIR="$LAB/ycfg" YOLO_OFFLINE=1
export MLFLOW_TRACKING_URI="sqlite:///$LAB/mlflow.db" MLFLOW_EXPERIMENT_NAME="yolo-repro"

run() { # name device seed det epochs
  echo "== $1 $(date +%T)"
  "$PY" "$HERE/train_yolo.py" --data "$DATA" --weights "$W" --device "$2" --seed "$3" \
    --deterministic "$4" --epochs "$5" --project "$LAB/runs" --name "$1" --summary "$SUM" \
    > "$LAB/logs/$1.log" 2>&1
  tail -1 "$LAB/logs/$1.log"
}
mkdir -p "$LAB/logs"
: > "$SUM"

# 1) 재현성: 같은 설정 2회씩 (epochs=3)
run cpu_det_s0_a cpu 0 1 3
run cpu_det_s0_b cpu 0 1 3
run mps_det_s0_a mps 0 1 3
run mps_det_s0_b mps 0 1 3
run mps_det_s0_c mps 0 1 3
for s in d e f g h; do run mps_det_s0_$s mps 0 1 3; done  # MPS 반복 횟수 확대
run cpu_det_s0_c cpu 0 1 3
run cpu_det_s0_d cpu 0 1 3
# 2) 대조군
run cpu_nondet_s0 cpu 0 0 3
run mps_nondet_s0 mps 0 0 3
run cpu_det_s1 cpu 1 1 3
run mps_det_s1 mps 1 1 3
# 3) 도전과제 (a): 에포크 수별 (단독 run, MPS)
for e in 1 5 10 20; do run mps_ep$e mps 0 1 $e; done
echo "== done $(date +%T)"

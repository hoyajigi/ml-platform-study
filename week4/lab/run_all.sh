#!/usr/bin/env bash
# 4주차 재현성 실험 전체. 사용: LAB=<scratch 디렉터리> bash run_all.sh   (맥북 M3 Pro 기준 약 20분)
set -euo pipefail
: "${LAB:?LAB 필요}"
HERE="$(cd "$(dirname "$0")" && pwd)"; RES="$HERE/results"; cd "$LAB"
# 0) 환경 (mlflow 3.0.0 은 sqlalchemy 2.1 과 sqlite 스토어가 깨지므로 <2.1 고정)
[ -x .venv/bin/python ] || { uv venv -q -p 3.12 .venv; uv pip install -q -p .venv/bin/python "ultralytics==8.4.207" "torch==2.15.0" torchvision "mlflow==3.0.0" "sqlalchemy<2.1" pandas scikit-learn; }
for v in 8.3.56 8.0.194; do [ -x .venv-$v/bin/python ] || { uv venv -q -p 3.12 .venv-$v; uv pip install -q -p .venv-$v/bin/python "ultralytics==$v" "torch==2.15.0" torchvision "numpy<2" "pandas<2.3"; }; done
# 1) 데이터 (MIDV-500 문서 1종 657MB, MovieLens small 1MB, yolov8n.pt)
mkdir -p dl data mlens
[ -f dl/01_alb_id.zip ] || curl -s -o dl/01_alb_id.zip ftp://smartengines.com/midv-500/dataset/01_alb_id.zip
[ -f dl/ml-latest-small.zip ] || curl -sL -o dl/ml-latest-small.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
[ -f dl/yolov8n.pt ] || curl -sL -o dl/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
[ -f data/midv_alb/data.yaml ] || .venv/bin/python -I "$HERE/prep_midv.py" --zip dl/01_alb_id.zip --out "$LAB/data/midv_alb"
[ -f mlens/ml-latest-small/ratings.csv ] || unzip -q -o dl/ml-latest-small.zip -d mlens
top -l 1 | head -10 > "$RES/idle_before.txt"
# 2) 실험
LAB="$LAB" bash "$HERE/run_yolo.sh"
LAB="$LAB" bash "$HERE/run_recsys.sh"
# 3) 비교·부가 확인
.venv/bin/python "$HERE/compare_yolo.py" --runs runs --out "$RES/yolo_compare.csv"
.venv/bin/python "$HERE/repeat_groups.py" runs "$RES/yolo_repeat_mps.json" mps_det_s0_{a,b,c,d,e,f,g,h} mps_nondet_s0
.venv/bin/python "$HERE/repeat_groups.py" runs "$RES/yolo_repeat_cpu.json" cpu_det_s0_{a,b,c,d} cpu_nondet_s0
.venv/bin/python "$HERE/compare_recsys.py" --summary recsys_runs.jsonl --outdir recsys --out "$RES/recsys_compare.csv"
export YOLO_CONFIG_DIR="$LAB/ycfg" YOLO_OFFLINE=1
.venv/bin/python "$HERE/val_epochs.py" --data data/midv_alb/data.yaml --pretrained dl/yolov8n.pt --runs runs --out "$RES/epochs_vs_pretrained.csv"
: > val_versions.jsonl
for v in "" -8.3.56 -8.0.194; do
  YOLO_CONFIG_DIR="$LAB/ycfg$v" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MLFLOW_TRACKING_URI= .venv$v/bin/python "$HERE/val_versions.py" \
    --data data/midv_alb/data.yaml --weights "$LAB/runs/mps_ep10/weights/best.pt" "$LAB/runs/cpu_det_s0_a/weights/best.pt" --out val_versions.jsonl
done
cp val_versions.jsonl yolo_runs.jsonl recsys_runs.jsonl "$RES/"
.venv/bin/python -I "$HERE/split_order.py" --zip dl/01_alb_id.zip --out "$RES/split_order.json"
.venv/bin/python "$HERE/mlflow_diff.py" --uri "sqlite:///$LAB/mlflow.db" --out "$RES/mlflow_diff.json" \
  --pairs yolo-repro:mps_det_s0_e:mps_det_s0_h yolo-repro:mps_det_s0_e:mps_det_s0_f yolo-repro:cpu_det_s0_c:cpu_det_s0_d \
  recsys-repro:cpu_s0_a:cpu_s0_b recsys-repro:cpu_s0_a:mps_s0_a recsys-repro:cpu_noseed_a:cpu_noseed_b

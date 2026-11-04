#!/usr/bin/env bash
# Run every drift_lab experiment with one venv.
# Usage: run_all.sh <venv-dir> <adult.data> <out-dir> <tag> [grid_reps] [workers]
set -euo pipefail

VENV=$1
DATA=$2
OUT=$3
TAG=$4
GRID_REPS=${5:-200}
WORKERS=${6:-6}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUT"
start=$(date +%s)
run() {  # exp reps
  "$VENV/bin/python" "$HERE/drift_lab.py" --data "$DATA" --exp "$1" --reps "$2" \
    --workers "$WORKERS" --out "$OUT/$1-$TAG.jsonl" 2>&1 | grep -v -i warn || true
}
run validate 25
run grid "$GRID_REPS"
run refsize 100
run sliding 3
echo "total $(( $(date +%s) - start ))s"

"""같은 가중치를 현재 venv의 ultralytics 버전으로 val (버전별 venv에서 각각 실행). 결과를 jsonl에 1줄씩 추가."""
import argparse
import json

import ultralytics
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for w in args.weights:
        for batch in (1, 8):
            m = YOLO(w).val(data=args.data, imgsz=640, batch=batch, device="cpu", plots=False)
            rec = {"ultralytics": ultralytics.__version__, "weights": w.split("/runs/")[-1], "batch": batch,
                   "mAP50": round(float(m.box.map50), 5), "mAP50-95": round(float(m.box.map), 5),
                   "precision": round(float(m.box.mp), 5), "recall": round(float(m.box.mr), 5)}
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec))


if __name__ == "__main__":
    main()

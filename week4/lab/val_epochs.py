"""도전과제 (a): 기본 사전학습 yolov8n.pt vs 에포크별 단독 훈련 best.pt를 같은 val 30장으로 검증."""
import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


def val(weights, data, single_cls):
    m = YOLO(weights).val(data=data, imgsz=640, batch=8, device="cpu", single_cls=single_cls,
                          plots=False, verbose=False, project=None)
    return {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for sc in (False, True):
        rows.append({"model": "pretrained yolov8n.pt (COCO 80cls)", "single_cls": sc,
                     **val(args.pretrained, args.data, sc)})
    for e in (1, 5, 10, 20):
        w = Path(args.runs) / f"mps_ep{e}" / "weights" / "best.pt"
        if w.exists():
            rows.append({"model": f"trained {e} epoch(s) best.pt", "single_cls": False, **val(str(w), args.data, False)})
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()

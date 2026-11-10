"""YOLO 훈련 1회 (노션 8장 설정: yolov8n.pt, imgsz=640). 프로세스 1개 = run 1개.

훈련이 끝나면 fp32 EMA·raw 가중치 state_dict를 따로 저장한다(ultralytics의 last.pt/best.pt는 fp16 + 날짜 메타데이터).
결과 요약은 --summary(jsonl)에 1줄 추가한다.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", choices=["cpu", "mps"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    model = YOLO(args.weights)
    saved = {}

    def on_train_end(trainer):
        out = Path(trainer.save_dir) / "fp32"
        out.mkdir(exist_ok=True)
        torch.save({k: v.detach().cpu().float() for k, v in trainer.ema.ema.state_dict().items()}, out / "ema.pt")
        torch.save({k: v.detach().cpu().float() for k, v in trainer.model.state_dict().items()}, out / "raw.pt")
        saved["dir"] = str(trainer.save_dir)
        saved["deterministic_flag_at_end"] = torch.are_deterministic_algorithms_enabled()

    model.add_callback("on_train_end", on_train_end)
    t0 = time.perf_counter()
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        deterministic=bool(args.deterministic),
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    wall = time.perf_counter() - t0
    save_dir = Path(saved["dir"])
    rec = {
        "name": args.name,
        "device": args.device,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "wall_s": round(wall, 2),
        "last_pt_sha256": file_sha256(save_dir / "weights" / "last.pt"),
        "best_pt_sha256": file_sha256(save_dir / "weights" / "best.pt"),
        "save_dir": str(save_dir),
    }
    with open(args.summary, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()

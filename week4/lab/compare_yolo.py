"""YOLO run 쌍별 비교: 파일 해시, fp16 저장 가중치 해시, fp32 EMA 최대 절대차, 에포크별 지표 일치."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

PAIRS = [
    ("cpu_det_s0_a", "cpu_det_s0_b", "CPU 같은 설정 2회"),
    ("mps_det_s0_a", "mps_det_s0_b", "MPS 같은 설정 2회"),
    ("mps_det_s0_a", "mps_det_s0_c", "MPS 같은 설정 (a vs c)"),
    ("mps_det_s0_b", "mps_det_s0_c", "MPS 같은 설정 (b vs c)"),
    ("cpu_det_s0_a", "mps_det_s0_a", "CPU vs MPS"),
    ("cpu_det_s0_a", "cpu_nondet_s0", "CPU deterministic True vs False"),
    ("mps_det_s0_a", "mps_nondet_s0", "MPS deterministic True vs False"),
    ("cpu_det_s0_a", "cpu_det_s1", "CPU seed 0 vs 1"),
    ("mps_det_s0_a", "mps_det_s1", "MPS seed 0 vs 1"),
]
METRICS = ["train/box_loss", "train/cls_loss", "train/dfl_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]


def tensor_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_saved_fp16(run_dir):
    ckpt = torch.load(run_dir / "weights" / "last.pt", map_location="cpu", weights_only=False)
    return ckpt["model"].state_dict()


def max_abs_diff(a, b):
    worst, key, n_diff = 0.0, None, 0
    for k in a:
        if not torch.is_floating_point(a[k]):
            continue
        d = (a[k].float() - b[k].float()).abs().max().item() if a[k].numel() else 0.0
        if d > 0:
            n_diff += 1
        if d > worst:
            worst, key = d, k
    return worst, key, n_diff


def read_results(run_dir):
    with open(run_dir / "results.csv") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = Path(args.runs)
    rows = []
    for a, b, label in PAIRS:
        ra, rb = runs / a, runs / b
        if not (ra / "results.csv").exists() or not (rb / "results.csv").exists():
            continue
        fa, fb = (ra / "weights/last.pt").read_bytes(), (rb / "weights/last.pt").read_bytes()
        sa, sb = load_saved_fp16(ra), load_saved_fp16(rb)
        ea = torch.load(ra / "fp32/ema.pt")
        eb = torch.load(rb / "fp32/ema.pt")
        ma, mb = torch.load(ra / "fp32/raw.pt"), torch.load(rb / "fp32/raw.pt")
        da, db = read_results(ra), read_results(rb)
        n_float = sum(1 for v in ea.values() if torch.is_floating_point(v))
        ema_d, ema_k, ema_n = max_abs_diff(ea, eb)
        raw_d, _, _ = max_abs_diff(ma, mb)
        fp16_d, _, _ = max_abs_diff(sa, sb)
        metric_eq = all(x[m] == y[m] for x, y in zip(da, db) for m in METRICS)
        last_a, last_b = da[-1], db[-1]
        rows.append({
            "pair": label, "a": a, "b": b,
            "last_pt_file_sha_equal": hashlib.sha256(fa).digest() == hashlib.sha256(fb).digest(),
            "saved_fp16_tensor_sha_equal": tensor_hash(sa) == tensor_hash(sb),
            "fp32_ema_tensor_sha_equal": tensor_hash(ea) == tensor_hash(eb),
            "fp32_ema_max_abs_diff": ema_d, "fp32_ema_worst_key": ema_k,
            "fp32_ema_tensors_differing": f"{ema_n}/{n_float}",
            "fp32_raw_max_abs_diff": raw_d, "saved_fp16_max_abs_diff": fp16_d,
            "per_epoch_metrics_identical": metric_eq,
            "a_mAP50-95_last": float(last_a["metrics/mAP50-95(B)"]),
            "b_mAP50-95_last": float(last_b["metrics/mAP50-95(B)"]),
            "a_box_loss_last": float(last_a["train/box_loss"]),
            "b_box_loss_last": float(last_b["train/box_loss"]),
        })
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()

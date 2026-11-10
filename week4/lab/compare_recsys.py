"""영화 추천 run 쌍별 비교: 초기 가중치 해시, 최종 텐서 해시, 최대 절대차, 에포크별 loss 차."""
import argparse
import csv
import json
from pathlib import Path

import torch

PAIRS = [
    ("cpu_s0_a", "cpu_s0_b", "CPU seed0 2회"),
    ("mps_s0_a", "mps_s0_b", "MPS seed0 2회"),
    ("mps_s0_a", "mps_s0_c", "MPS seed0 (a vs c)"),
    ("cpu_s0_a", "mps_s0_a", "CPU vs MPS"),
    ("mps_s0_a", "mps_s0_det", "MPS use_deterministic False vs True"),
    ("cpu_noseed_a", "cpu_noseed_b", "CPU seed 미설정(원본 코드) 2회"),
    ("cpu_s0_a", "cpu_s0_w2", "CPU num_workers 0 vs 2"),
    ("cpu_s0_a", "cpu_s1", "CPU seed 0 vs 1"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = {r["name"]: r for r in map(json.loads, Path(args.summary).read_text().splitlines())}
    rows = []
    for a, b, label in PAIRS:
        if a not in runs or b not in runs:
            continue
        sa = torch.load(Path(args.outdir) / a / "state_dict.pt")
        sb = torch.load(Path(args.outdir) / b / "state_dict.pt")
        diffs = {k: (sa[k] - sb[k]).abs().max().item() for k in sa}
        ea, eb = runs[a]["epochs"], runs[b]["epochs"]
        rows.append({
            "pair": label, "a": a, "b": b,
            "init_equal": runs[a]["init_sha256"] == runs[b]["init_sha256"],
            "final_tensor_sha_equal": runs[a]["final_tensor_sha256"] == runs[b]["final_tensor_sha256"],
            "max_abs_diff": max(diffs.values()),
            "worst_tensor": max(diffs, key=diffs.get),
            "train_l1_seq_identical": [x["train_l1"] for x in ea] == [x["train_l1"] for x in eb],
            "max_epoch_train_l1_diff": max(abs(x["train_l1"] - y["train_l1"]) for x, y in zip(ea, eb)),
            "a_test_rmse_last": ea[-1]["test_rmse"], "b_test_rmse_last": eb[-1]["test_rmse"],
            "test_rmse_diff": abs(ea[-1]["test_rmse"] - eb[-1]["test_rmse"]),
            "train_l1_6dp_equal": [f"{x['train_l1']:3f}" for x in ea] == [f"{x['train_l1']:3f}" for x in eb],
        })
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()

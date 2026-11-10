"""9장 영화 추천 모델(practical-mlops/movie-recommender-project의 model_training.py) 축소 재현.

모델·손실·옵티마이저·DataLoader 구성은 원본 그대로:
  MatrixFactorization(Embedding×2 → mul → Linear(20,256) → ReLU → Dropout(0.2) → Linear(256,1)),
  SGD(lr=0.001), StepLR(step_size=10, gamma=0.1), L1Loss, batch 64, shuffle=True, iloc 기반 Dataset.
바꾼 점: 데이터(ml-25m → ml-latest-small, 무작위 80/20 분할), 에포크(30 → 기본 5), device 인자, 선택적 seed.
원본은 seed를 전혀 설정하지 않는다(--seed -1 이 원본과 같은 상태).
MLflow 기록은 원본처럼 metric을 f"{x:3f}" 문자열로 넣은 값과 full-precision 값을 함께 남긴다.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class DatasetReader(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sd = self.df.iloc[idx]
        return (torch.tensor(sd["userId"] - 1).long(), torch.tensor(sd["movieId"] - 1).long(),
                torch.tensor(sd["rating"]).float())


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors, hidden_dim, dropout_rate):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users + 1, n_factors, sparse=False)
        self.item_factors = torch.nn.Embedding(n_items + 1, n_factors, sparse=False)
        self.linear = torch.nn.Linear(in_features=n_factors, out_features=hidden_dim)
        self.linear2 = torch.nn.Linear(in_features=hidden_dim, out_features=1)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()

    def forward(self, user, item):
        x = torch.mul(self.user_factors(user), self.item_factors(item))
        x = self.dropout(self.relu(self.linear(x)))
        return self.linear2(x)


def split(ratings_csv, split_seed=42):
    df = pd.read_csv(ratings_csv)
    rng = np.random.default_rng(split_seed)
    mask = rng.random(len(df)) < 0.8
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True), int(df.userId.max()), int(df.movieId.max())


def evaluate(model, loader, device, loss_func):
    model.eval()
    l1, sq, n, batches = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for u, i, r in loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            p = model(u, i)
            l1 += loss_func(p, r.unsqueeze(1)).item()
            sq += ((p.squeeze(1) - r) ** 2).sum().item()
            n += r.numel()
            batches += 1
    return l1 / batches, (sq / n) ** 0.5


def train_epoch(model, loader, device, loss_func, optimizer):
    model.train()
    t_loss, t_count = 0.0, 0
    for u, i, r in loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        loss = loss_func(model(u, i), r.unsqueeze(1))
        t_loss += loss.item()
        t_count += 1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return t_loss / t_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--device", choices=["cpu", "mps"], required=True)
    ap.add_argument("--seed", type=int, default=0, help="-1: seed 미설정(원본 코드와 동일)")
    ap.add_argument("--deterministic", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    if args.seed >= 0:
        torch.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    train_df, test_df, n_users, n_items = split(args.ratings)
    model = MatrixFactorization(n_users, n_items, n_factors=20, hidden_dim=256, dropout_rate=0.2).to(device)
    init_hash = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_func = torch.nn.L1Loss()
    train_dl = DataLoader(DatasetReader(train_df), batch_size=64, shuffle=True, num_workers=args.num_workers)
    test_dl = DataLoader(DatasetReader(test_df), batch_size=64, shuffle=True, num_workers=args.num_workers)

    epochs_log = []
    with mlflow.start_run(run_name=args.name):
        mlflow.log_params({"device": args.device, "seed": args.seed, "deterministic": args.deterministic,
                           "num_workers": args.num_workers, "epochs": args.epochs, "lr": 0.001, "batch": 64,
                           "n_factors": 20, "hidden_dim": 256, "dropout": 0.2, "n_users": n_users,
                           "n_items": n_items, "n_train": len(train_df), "n_test": len(test_df)})
        for ep in range(args.epochs):
            t0 = time.perf_counter()
            tr = train_epoch(model, train_dl, device, loss_func, optimizer)
            scheduler.step()
            t_train = time.perf_counter() - t0
            te_l1, te_rmse = evaluate(model, test_dl, device, loss_func)
            rec = {"epoch": ep, "train_l1": tr, "test_l1": te_l1, "test_rmse": te_rmse,
                   "train_s": round(t_train, 3), "epoch_s": round(time.perf_counter() - t0, 3)}
            epochs_log.append(rec)
            mlflow.log_metric("avg_training_loss", float(f"{tr:3f}"), step=ep)  # 원본 포맷(소수 6자리)
            mlflow.log_metric("avg_testing_loss", float(f"{te_l1:3f}"), step=ep)
            mlflow.log_metric("avg_training_loss_full", tr, step=ep)
            mlflow.log_metric("test_rmse_full", te_rmse, step=ep)
            print(json.dumps(rec))
        run_id = mlflow.active_run().info.run_id

    out = Path(args.outdir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    torch.save(sd, out / "state_dict.pt")
    final_hash = hashlib.sha256(b"".join(sd[k].numpy().tobytes() for k in sorted(sd))).hexdigest()
    summary = {"name": args.name, "device": args.device, "seed": args.seed, "deterministic": args.deterministic,
               "num_workers": args.num_workers, "init_sha256": init_hash, "final_tensor_sha256": final_hash,
               "mlflow_run_id": run_id, "epochs": epochs_log}
    with open(args.summary, "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()

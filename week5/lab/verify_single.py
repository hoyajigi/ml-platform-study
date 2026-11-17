"""1000행을 1행짜리 요청 1000개로 동시 32개씩 보내 (배칭이 실제로 행을 묶는 조건) 기준선과 비트 비교."""
import asyncio
import json
import os

import httpx
import mlflow
import numpy as np

from data import probe_rows

rows = probe_rows(1000)
recs = rows.to_dict(orient="records")
base = np.asarray(mlflow.pyfunc.load_model(f"models:/{os.environ['MODEL_NAME']}@champion").predict(rows))
TARGETS = {
    "mlserve": ("http://127.0.0.1:5066/invocations", "mlflow"),
    "bento_nobatch": ("http://127.0.0.1:5067/predict", "bento"),
    "bento_batch1s": ("http://127.0.0.1:5067/predict_batched_1s", "bento"),
}


async def run(url, kind):
    sem = asyncio.Semaphore(32)
    async with httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=32)) as cli:
        async def one(r):
            async with sem:
                b = {"dataframe_records": [r]} if kind == "mlflow" else {"records": [r]}
                j = (await cli.post(url, json=b)).json()
                return (j["predictions"] if kind == "mlflow" else j)[0]
        return np.asarray(await asyncio.gather(*[one(r) for r in recs]), dtype=np.float64)

out = []
for name, (url, kind) in TARGETS.items():
    p = asyncio.run(run(url, kind))
    rec = {"target": name, "rows": len(p), "bit_exact": bool(np.array_equal(p, base)),
           "n_diff_rows": int((p != base).any(1).sum()), "maxabs": float(np.max(np.abs(p - base)))}
    out.append(rec); print(rec)
with open(os.path.join(os.environ["HERE"], "results", "verify_single.jsonl"), "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in out)

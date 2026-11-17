"""단일 행 요청 지연/처리량. 사용: python bench.py <label> <url> <mlflow|bento> <n> <c1,c8,...>
in-process 기준선은 label=inproc, url=- 로 호출 (동시성 1만)."""
import asyncio
import json
import os
import sys
import time

import httpx
import numpy as np

from data import probe_rows


def body(kind, rec):
    return {"dataframe_records": [rec]} if kind == "mlflow" else {"records": [rec]}


async def run_level(url, kind, recs, conc, n):
    lat, errs = [], {}
    idx = iter(range(n))
    limits = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
    async with httpx.AsyncClient(limits=limits, timeout=60) as cli:
        async def worker():
            for i in idx:
                t = time.perf_counter()
                r = await cli.post(url, json=body(kind, recs[i % len(recs)]))
                if r.status_code != 200:  # 503(배칭 지연 상한 초과) 등은 세고 지연 통계에서 뺀다
                    errs[r.status_code] = errs.get(r.status_code, 0) + 1
                    continue
                lat.append(time.perf_counter() - t)
        for i in range(50):  # warmup
            await cli.post(url, json=body(kind, recs[i]))
        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(conc)])
        wall = time.perf_counter() - t0
    return lat, wall, errs


def inproc(recs, n):
    import mlflow
    import pandas as pd
    m = mlflow.pyfunc.load_model(f"models:/{os.environ['MODEL_NAME']}@champion")
    frames = [pd.DataFrame([r]) for r in recs]
    for f in frames[:50]:
        m.predict(f)
    lat = []
    t0 = time.perf_counter()
    for i in range(n):
        t = time.perf_counter(); m.predict(frames[i % len(frames)]); lat.append(time.perf_counter() - t)
    return lat, time.perf_counter() - t0, {}


def main(label, url, kind, n, levels):
    recs = probe_rows(1000).to_dict(orient="records")
    out = open(os.path.join(os.environ["HERE"], "results", "latency.jsonl"), "a")
    for c in [int(x) for x in levels.split(",")]:
        lat, wall, errs = inproc(recs, n) if label == "inproc" else asyncio.run(run_level(url, kind, recs, c, n))
        a = np.array(lat) * 1000
        row = {"label": label, "conc": c, "n": len(a), "p50_ms": round(float(np.percentile(a, 50)), 2),
               "p99_ms": round(float(np.percentile(a, 99)), 2), "mean_ms": round(float(a.mean()), 2),
               "rps_ok": round(len(a) / wall, 1), "errors": errs}
        out.write(json.dumps(row) + "\n"); out.flush()
        print(row, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])

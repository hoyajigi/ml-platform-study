"""각 서버에 같은 1000행을 보내 (1) 어느 버전이 응답했는지 예측값으로 판정, (2) 비트 일치,
(3) 응답 헤더·메타 엔드포인트에서 버전/run_id/alias 문자열이 보이는지 확인. 사용: python probe.py <phase>"""
import json
import os
import re
import sys

import httpx
import mlflow
import numpy as np

from data import probe_rows

LAB = os.environ["LAB"]
NAME = os.environ["MODEL_NAME"]
VERS = json.load(open(os.path.join(LAB, "versions.json")))
SERVERS = {
    "mlserve": (int(os.environ["PORT_MLSERVE"]), "/invocations", ["/version", "/ping", "/health"]),
    "live": (int(os.environ["PORT_BENTO"]), "/predict", ["/docs.json", "/schema.json", "/metrics", "/readyz", "/livez"]),
    "built": (int(os.environ["PORT_BENTO_BUILT"]), "/predict", ["/docs.json", "/schema.json", "/metrics", "/readyz", "/livez"]),
}


def baseline(rows):
    """(c) 같은 프로세스 pyfunc.predict — 버전 번호로 직접 지정해서 v1, v2 둘 다."""
    return {t: np.asarray(mlflow.pyfunc.load_model(f"models:/{NAME}/{v['version']}").predict(rows))
            for t, v in VERS.items()}


def call(kind, rows, path=None):
    port, default_path, _ = SERVERS[kind]
    path = path or default_path
    recs = rows.to_dict(orient="records")
    body = {"dataframe_records": recs} if kind == "mlserve" else {"records": recs}
    r = httpx.post(f"http://127.0.0.1:{port}{path}", json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    data = data["predictions"] if kind == "mlserve" else data
    return np.asarray(data, dtype=np.float64), dict(r.headers)


def needles():
    out = {"alias": ["champion"]}
    for t, v in VERS.items():
        out[t] = [v["run_id"], v["model_id"], f'"version": "{v["version"]}"']
    return out


def scan(text):
    hits = []
    for key, ns in needles().items():
        hits += [f"{key}:{n[:12]}" for n in ns if n and n in text]
    return hits


def main(phase):
    rows = probe_rows(1000)
    base = baseline(rows)
    res = []
    for kind in SERVERS:
        rec = {"phase": phase, "server": kind}
        try:
            pred, headers = call(kind, rows)
        except Exception as e:  # 서버가 내려가 있으면 기록만
            rec["error"] = repr(e)[:200]
            res.append(rec)
            continue
        rec["answered_by"] = [t for t, b in base.items() if np.array_equal(pred, b)] or ["none-bitexact"]
        rec["maxabs_vs"] = {t: float(np.max(np.abs(pred - b))) for t, b in base.items()}
        rec["label_agree_vs"] = {t: float(np.mean(pred.argmax(1) == b.argmax(1))) for t, b in base.items()}
        rec["headers"] = headers
        rec["header_hits"] = scan(json.dumps(headers))
        meta = {}
        for p in SERVERS[kind][2]:
            try:
                t = httpx.get(f"http://127.0.0.1:{SERVERS[kind][0]}{p}", timeout=10).text
                meta[p] = {"len": len(t), "hits": scan(t), "head": t[:120]}
            except Exception as e:
                meta[p] = {"error": repr(e)[:100]}
        rec["meta"] = meta
        res.append(rec)
    v1v2 = {"v1_vs_v2_label_agree": float(np.mean(base["v1"].argmax(1) == base["v2"].argmax(1))),
            "v1_vs_v2_maxabs": float(np.max(np.abs(base["v1"] - base["v2"])))}
    with open(os.path.join(os.environ["HERE"], "results", "probe.jsonl"), "a") as f:
        for r in res:
            f.write(json.dumps({**r, **v1v2}) + "\n")
    for r in res:
        print(phase, r["server"], r.get("answered_by", r.get("error")), "hdr", r.get("header_hits"),
              {p: m.get("hits") for p, m in r.get("meta", {}).items()})
    print(v1v2)


if __name__ == "__main__":
    main(sys.argv[1])

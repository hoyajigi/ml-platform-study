"""웹훅 목록 캐시(MLFLOW_WEBHOOK_CACHE_TTL, 기본 60초)가 전달 대상에 미치는 영향 측정.

서버는 기본 TTL 로 떠 있어야 한다. 수신기는 :5056 에서 경로별로 기록 중이어야 한다.

    python cache_ttl_test.py --recv-log <receiver jsonl> --out results/cache_ttl.jsonl
"""

import argparse
import json
import time
import uuid
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from run_scenarios import URI, load_receipts, log_dummy_model

EVENT = "model_version_tag.set"
BASE = "http://127.0.0.1:5056/hook/"
SETTLE_SEC = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recv-log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mlflow.set_tracking_uri(URI)
    mlflow.set_registry_uri(URI)
    mlflow.set_experiment("webhook-lab-cache")
    client = MlflowClient()
    tag = uuid.uuid4().hex[:6]
    name = f"cache-model-{tag}"
    client.create_registered_model(name)
    run_id, uri, model_id = log_dummy_model("cache-v1")
    client.create_model_version(name, uri, run_id, model_id=model_id)

    path_a, path_b = f"cacheA-{tag}", f"cacheB-{tag}"
    steps = []

    def trigger(label):
        t = time.time_ns()
        client.set_model_version_tag(name, "1", "step", label)
        time.sleep(SETTLE_SEC)
        steps.append({"step": label, "t_ns": t})

    wh_a = client.create_webhook(name=path_a, url=BASE + path_a, events=[EVENT])
    trigger("1_after_create_A")  # 캐시가 [A] 로 채워짐 (이 이벤트에 대해 처음 조회라면)
    t_cache_fill = steps[-1]["t_ns"]

    client.create_webhook(name=path_b, url=BASE + path_b, events=[EVENT])
    trigger("2_after_create_B_immediately")

    client.delete_webhook(wh_a.webhook_id)
    trigger("3_after_delete_A_immediately")

    remaining = 61 - (time.time_ns() - t_cache_fill) / 1e9
    if remaining > 0:
        time.sleep(remaining)
    trigger("4_after_61s_from_cache_fill")

    receipts = load_receipts(Path(args.recv_log))
    rows = []
    for i, s in enumerate(steps):
        end = steps[i + 1]["t_ns"] if i + 1 < len(steps) else float("inf")
        got = [r["path"].rsplit("/", 1)[-1] for r in receipts if s["t_ns"] <= r["recv_ns"] < end]
        rows.append({
            "step": s["step"],
            "sec_since_cache_fill": round((s["t_ns"] - t_cache_fill) / 1e9, 1),
            "A_received": got.count(path_a),
            "B_received": got.count(path_b),
            "A_exists_in_db": s["step"].startswith(("1", "2")),
            "B_exists_in_db": not s["step"].startswith("1"),
        })
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()

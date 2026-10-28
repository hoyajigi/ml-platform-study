"""시나리오 a~m 을 순서대로 실행하고, 각 API 호출 시각과 수신기 로그를 맞춰 지연을 계산한다.

전제: server.sh 로 MLflow 서버(:5055), receiver.py 로 수신기(:5056)가 떠 있을 것.

    python run_scenarios.py --recv-log <receiver jsonl> --out results/scenarios.jsonl
"""

import argparse
import json
import time
import uuid
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

URI = "http://127.0.0.1:5055"
SECRET = "lab-secret-123"
GAP_SEC = 2.0  # 시나리오 사이 대기: 비동기 전달이 다음 시나리오 창으로 넘어가지 않게
MODEL_EVENTS = [
    "registered_model.created",
    "model_version.created",
    "model_version_tag.set",
    "model_version_tag.deleted",
    "model_version_alias.created",
    "model_version_alias.deleted",
]


class Echo(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return model_input


def log_dummy_model(run_name: str) -> tuple[str, str, str]:
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metric("acc", 0.9)
        info = mlflow.pyfunc.log_model(name="model", python_model=Echo())
    return run.info.run_id, info.model_uri, info.model_id


def timed(fn):
    t0 = time.time_ns()
    result = fn()
    return t0, time.time_ns(), result


def build_scenarios(client: MlflowClient, name: str) -> list[tuple[str, str, object]]:
    """(id, 설명, 실행 함수). 모델 아티팩트 준비(log_model)는 시간 측정 밖에서 미리 한다."""
    prep = {}

    def prep_versions():
        prep["v1"] = log_dummy_model("v1-train")
        prep["v2"] = log_dummy_model("v2-train")

    def run_only():
        with mlflow.start_run(run_name="unregistered"):
            mlflow.log_metric("loss", 0.1)
            mlflow.pyfunc.log_model(name="model", python_model=Echo())

    def log_input():
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        ds = mlflow.data.from_pandas(df, name="train-v2", targets="y")
        with mlflow.start_run(run_name="dataset-only"):
            mlflow.log_input(ds, context="training")

    def mv(key):
        run_id, uri, model_id = prep[key]
        return lambda: client.create_model_version(name, uri, run_id, model_id=model_id)

    return [
        ("prep", "모델 아티팩트 사전 기록(측정 제외)", prep_versions),
        ("a", "등록 모델 생성", lambda: client.create_registered_model(name)),
        ("b", "모델 버전 v1 등록", lambda: mv("v1")()),
        ("c", "alias champion -> v1", lambda: client.set_registered_model_alias(name, "champion", "1")),
        ("d", "모델 버전 v2 등록", lambda: mv("v2")()),
        ("e", "alias champion -> v2 (재지정)", lambda: client.set_registered_model_alias(name, "champion", "2")),
        ("e2", "alias champion -> v2 (같은 값 재설정)", lambda: client.set_registered_model_alias(name, "champion", "2")),
        ("f", "alias champion 삭제", lambda: client.delete_registered_model_alias(name, "champion")),
        ("g", "모델 버전 태그 설정", lambda: client.set_model_version_tag(name, "2", "validated", "true")),
        ("g2", "모델 버전 태그 삭제", lambda: client.delete_model_version_tag(name, "2", "validated")),
        ("h", "레지스트리 미등록 run (log_metric + log_model)", run_only),
        ("i", "run 에 log_input 데이터셋 기록", log_input),
        ("j", "등록 모델 태그 설정", lambda: client.set_registered_model_tag(name, "team", "ml")),
        ("k", "모델 버전 description 수정", lambda: client.update_model_version(name, "2", description="desc")),
        ("l", "stage 전이(deprecated API) v2 -> Production",
         lambda: client.transition_model_version_stage(name, "2", "Production")),
        ("m", "모델 버전 v1 삭제", lambda: client.delete_model_version(name, "1")),
    ]


def load_receipts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recv-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--webhook-base", default="http://127.0.0.1:5056/hook")
    args = ap.parse_args()

    mlflow.set_tracking_uri(URI)
    mlflow.set_registry_uri(URI)
    mlflow.set_experiment("webhook-lab")
    client = MlflowClient()
    # 웹훅마다 고유 경로를 줘서, 이전 실행에서 남은 웹훅의 수신과 섞이지 않게 한다
    wh_name = f"lab-{uuid.uuid4().hex[:6]}"
    path = f"/hook/{wh_name}"
    wh = client.create_webhook(
        name=wh_name, url=args.webhook_base.rsplit("/hook", 1)[0] + path, events=MODEL_EVENTS,
        secret=SECRET,
    )
    name = f"lab-model-{uuid.uuid4().hex[:6]}"

    calls = []
    for sid, desc, fn in build_scenarios(client, name):
        t0, t1, _ = timed(fn)
        calls.append({"id": sid, "desc": desc, "t_start_ns": t0, "t_end_ns": t1})
        time.sleep(GAP_SEC)
    time.sleep(3)

    receipts = [r for r in load_receipts(Path(args.recv_log)) if r["path"] == path]
    rows = []
    for idx, c in enumerate(calls):
        if c["id"] == "prep":
            continue
        window_end = calls[idx + 1]["t_start_ns"] if idx + 1 < len(calls) else float("inf")
        hits = [r for r in receipts if c["t_start_ns"] <= r["recv_ns"] < window_end]
        rows.append({
            **c,
            "api_ms": round((c["t_end_ns"] - c["t_start_ns"]) / 1e6, 1),
            "events": [
                {
                    "event": f'{r["json"]["entity"]}.{r["json"]["action"]}',
                    "delivery_id": r["headers"].get("X-MLflow-Delivery-Id"),
                    "data": r["json"]["data"],
                    "lat_from_call_start_ms": round((r["recv_ns"] - c["t_start_ns"]) / 1e6, 1),
                    "lat_from_call_end_ms": round((r["recv_ns"] - c["t_end_ns"]) / 1e6, 1),
                    "sig_valid": r.get("signature", {}).get("valid"),
                    "sig_valid_wrong_secret": r.get("signature_with_wrong_secret", {}).get("valid"),
                }
                for r in hits
            ],
        })
    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"webhook_id={wh.webhook_id} model={name}")
    for r in rows:
        evs = ", ".join(f'{e["event"]}({e["lat_from_call_start_ms"]}ms/{e["lat_from_call_end_ms"]}ms sig={e["sig_valid"]})'
                        for e in r["events"]) or "-"
        print(f'{r["id"]:>3} api={r["api_ms"]:>7}ms  {r["desc"]}: {evs}')


if __name__ == "__main__":
    main()

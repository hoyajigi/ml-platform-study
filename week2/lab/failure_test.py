"""전달 실패·재시도·블로킹 측정. 수신기를 이 스크립트가 직접 띄우고 내린다.

서버는 MLFLOW_WEBHOOK_CACHE_TTL=1 로 띄울 것 (새 웹훅이 캐시 때문에 누락되지 않게).
타임아웃 시나리오(T1)는 MLFLOW_WEBHOOK_REQUEST_TIMEOUT=3 으로 서버를 따로 띄워 실행한다.

    python failure_test.py --work <dir> --out results/failure.jsonl F1 F2 F3 B0 B1
    python failure_test.py --work <dir> --out results/failure_timeout.jsonl T1
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from run_scenarios import URI, load_receipts, log_dummy_model

HERE = Path(__file__).parent
SECRET = "lab-secret-123"
EVENT = "model_version_alias.created"
HOOK_PATH = "/hook/fail"


class Receiver:
    def __init__(self, log: Path, status=200, delay=0.0):
        self.log, self.status, self.delay, self.proc = log, status, delay, None

    def start(self):
        cmd = [sys.executable, str(HERE / "receiver.py"), "--port", "5056", "--log", str(self.log),
               "--secret", SECRET, "--status", str(self.status), "--delay", str(self.delay)]
        self.proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
        for _ in range(50):  # 포트 열릴 때까지
            if subprocess.run(["lsof", "-nP", "-iTCP:5056", "-sTCP:LISTEN"],
                              capture_output=True).returncode == 0:
                return time.time_ns()
            time.sleep(0.1)
        raise RuntimeError("receiver did not start")

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=10)
            self.proc = None


class Lab:
    def __init__(self, work: Path):
        mlflow.set_tracking_uri(URI)
        mlflow.set_registry_uri(URI)
        mlflow.set_experiment("webhook-lab-failure")
        self.client = MlflowClient()
        self.work = work
        # 이전 실험에서 남은 웹훅이 같은 이벤트를 구독하면 수신 횟수가 섞이므로 전부 지운다 (실험 전용 DB)
        for old in self.client.list_webhooks():
            self.client.delete_webhook(old.webhook_id)
        self.name = f"fail-model-{uuid.uuid4().hex[:6]}"
        self.client.create_registered_model(self.name)
        run_id, uri, model_id = log_dummy_model("fail-v1")
        self.client.create_model_version(self.name, uri, run_id, model_id=model_id)
        self.wh = self.client.create_webhook(
            name=f"fail-{uuid.uuid4().hex[:6]}", url="http://127.0.0.1:5056" + HOOK_PATH,
            events=[EVENT], secret=SECRET,
        )
        time.sleep(1.5)  # 캐시 TTL(1초) 경과 대기

    def fire(self, alias):
        t0 = time.time_ns()
        self.client.set_registered_model_alias(self.name, alias, "1")
        return t0, time.time_ns()


def summarize(label, log: Path, t_fire, t_api_end, extra=None):
    recs = [r for r in load_receipts(log) if r["recv_ns"] >= t_fire and r["path"] == HOOK_PATH]
    attempts = [round((r["recv_ns"] - t_fire) / 1e6, 1) for r in recs]
    return {
        "test": label,
        "api_ms": round((t_api_end - t_fire) / 1e6, 1),
        "attempts_received": len(recs),
        "attempt_ms_from_fire": attempts,
        "intervals_ms": [round(b - a, 1) for a, b in zip(attempts, attempts[1:])],
        "distinct_delivery_ids": len({r["headers"].get("X-MLflow-Delivery-Id") for r in recs}),
        "sig_valid": [r.get("signature", {}).get("valid") for r in recs],
        **(extra or {}),
    }


def f1_down_long(lab: Lab, log: Path):
    """수신기 내린 상태에서 이벤트 → 20초 뒤 수신기 기동 → 15초 관찰."""
    t0, t1 = lab.fire("f1")
    time.sleep(20)
    rx = Receiver(log)
    t_up = rx.start()
    time.sleep(15)
    rx.stop()
    return summarize("F1_receiver_down_20s_then_up", log, t0, t1,
                     {"receiver_up_ms_after_fire": round((t_up - t0) / 1e6, 1)})


def f2_down_short(lab: Lab, log: Path):
    """수신기 내린 상태에서 이벤트 → 1.5초 뒤 기동: 재시도가 늦게 뜬 수신기에 닿는가."""
    t0, t1 = lab.fire("f2")
    time.sleep(1.5)
    rx = Receiver(log)
    t_up = rx.start()
    time.sleep(15)
    rx.stop()
    return summarize("F2_receiver_down_1.5s_then_up", log, t0, t1,
                     {"receiver_up_ms_after_fire": round((t_up - t0) / 1e6, 1)})


def f3_5xx(lab: Lab, log: Path, status=500):
    rx = Receiver(log, status=status)
    rx.start()
    t0, t1 = lab.fire(f"f3s{status}")
    time.sleep(20)
    rx.stop()
    return summarize(f"F3_receiver_returns_{status}", log, t0, t1)


def blocking(lab: Lab, log: Path, delay, label, observe_sec=None):
    rx = Receiver(log, delay=delay)
    rx.start()
    t0, t1 = lab.fire(label.lower())
    time.sleep(observe_sec if observe_sec is not None else delay + 3)
    rx.stop()
    return summarize(label, log, t0, t1, {"receiver_delay_sec": delay})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("tests", nargs="+")
    args = ap.parse_args()
    work = Path(args.work)
    lab = Lab(work)
    tests = {
        "F1": lambda log: f1_down_long(lab, log),
        "F2": lambda log: f2_down_short(lab, log),
        "F3": lambda log: f3_5xx(lab, log, 500),
        "F4": lambda log: f3_5xx(lab, log, 404),
        "B0": lambda log: blocking(lab, log, 0.0, "B0_receiver_delay_0s"),
        "B1": lambda log: blocking(lab, log, 5.0, "B1_receiver_delay_5s"),
        "T1": lambda log: blocking(lab, log, 5.0, "T1_delay_5s_with_server_timeout_3s",
                                     observe_sec=40),
    }
    out = Path(args.out)
    for t in args.tests:
        log = work / f"recv_{t}.jsonl"
        log.unlink(missing_ok=True)
        row = {"server_log_marker_ns": time.time_ns(), **tests[t](log)}
        with out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY 가 바뀐 서버(재시작 시 키 미고정과 같은 상황)에서
secret 있는 웹훅이 어떻게 되는지 확인. failure_test.py 가 남긴 /hook/fail 웹훅(secret 있음)을 그대로 쓴다.

    (서버를 다른 키로 띄운 뒤) python key_rotation_test.py --work <dir> --out results/key_rotation.json
"""
import argparse
import json
import time
from pathlib import Path

from mlflow import MlflowClient

from failure_test import Receiver
from run_scenarios import URI, load_receipts

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
client = MlflowClient(URI, URI)
res = {}
try:
    res["list_webhooks"] = [w.name for w in client.list_webhooks()]
except Exception as e:
    res["list_webhooks_error"] = str(e)[:300]
name = sorted(m.name for m in client.search_registered_models() if m.name.startswith("fail-model-"))[-1]
log = Path(args.work) / "recv_keyrot.jsonl"
log.unlink(missing_ok=True)
rx = Receiver(log)
rx.start()
t0 = time.time_ns()
try:
    client.set_registered_model_alias(name, "keyrot", "1")
    res["set_alias"] = f"ok {round((time.time_ns() - t0) / 1e6, 1)}ms"
except Exception as e:
    res["set_alias_error"] = str(e)[:300]
time.sleep(10)
rx.stop()
res["received"] = len([r for r in load_receipts(log) if r["recv_ns"] >= t0])
Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps(res, ensure_ascii=False, indent=1))

"""웹훅 생성 가능 여부만 확인: 결과를 results/prereq.jsonl 에 한 줄 추가한다.

    python prereq_check.py <label> <webhook_url>
"""
import json
import os
import sys
import time
from pathlib import Path

from mlflow import MlflowClient

label, url = sys.argv[1], sys.argv[2]
client = MlflowClient(tracking_uri="http://127.0.0.1:5055", registry_uri="http://127.0.0.1:5055")
rec = {"label": label, "url": url, "ts": time.time()}
try:
    wh = client.create_webhook(
        name=f"prereq-{int(time.time())}", url=url, events=["model_version.created"]
    )
    rec.update(ok=True, webhook_id=wh.webhook_id)
    client.delete_webhook(wh.webhook_id)
except Exception as e:  # 실패 메시지 자체가 측정 결과
    rec.update(ok=False, error_type=type(e).__name__, error=str(e)[:600])
out = Path(os.environ.get("RESULTS", Path(__file__).parent / "results")) / "prereq.jsonl"
with out.open("a") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(json.dumps(rec, ensure_ascii=False, indent=1))

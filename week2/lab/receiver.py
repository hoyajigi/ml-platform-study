"""MLflow 웹훅 수신기 (표준 라이브러리만 사용).

받은 요청마다 수신 시각(ns), 헤더, 바디, HMAC 검증 결과를 jsonl 한 줄로 기록한다.

    python receiver.py --port 5056 --log recv.jsonl --secret s3cr3t [--status 500] [--delay 5]

--status: 응답 상태 코드 (기본 200). 5xx로 두면 MLflow 쪽 재시도를 관찰할 수 있다.
--delay : 응답 전에 기다릴 초. 전달이 API 호출을 블로킹하는지 볼 때 쓴다.
"""

import argparse
import base64
import hashlib
import hmac
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_TIMESTAMP_SKEW_SEC = 300


def verify_signature(secret: str, headers: dict, body: bytes) -> dict:
    """MLflow 서명(X-MLflow-Signature: v1,<base64>)을 수신 측에서 독립적으로 검증한다.

    서명 대상 문자열: f"{delivery_id}.{timestamp}.{raw_body}" (HMAC-SHA256)
    """
    sig_header = headers.get("X-MLflow-Signature")
    delivery_id = headers.get("X-MLflow-Delivery-Id")
    timestamp = headers.get("X-MLflow-Timestamp")
    if not sig_header:
        return {"present": False, "valid": False, "reason": "no signature header"}
    if not (delivery_id and timestamp):
        return {"present": True, "valid": False, "reason": "missing id/timestamp header"}
    version, _, received_b64 = sig_header.partition(",")
    if version != "v1":
        return {"present": True, "valid": False, "reason": f"unknown version {version}"}
    signed = f"{delivery_id}.{timestamp}.".encode() + body
    expected = base64.b64encode(hmac.new(secret.encode(), signed, hashlib.sha256).digest()).decode()
    skew = abs(time.time() - int(timestamp))
    valid = hmac.compare_digest(expected, received_b64) and skew <= MAX_TIMESTAMP_SKEW_SEC
    return {"present": True, "valid": valid, "timestamp_skew_sec": round(skew, 3)}


def make_handler(args, lock):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            recv_ns = time.time_ns()
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            headers = dict(self.headers.items())
            record = {
                "recv_ns": recv_ns,
                "path": self.path,
                "headers": headers,
                "body": body.decode("utf-8", errors="replace"),
                "respond_status": args.status,
                "respond_delay_sec": args.delay,
            }
            try:
                record["json"] = json.loads(body)
            except ValueError:
                record["json"] = None
            if args.secret:
                record["signature"] = verify_signature(args.secret, headers, body)
                # 일부러 틀린 secret으로도 검증해 실패가 실제로 감지되는지 확인
                record["signature_with_wrong_secret"] = verify_signature(
                    args.secret + "-wrong", headers, body
                )
            with lock, open(args.log, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.delay:
                time.sleep(args.delay)
            self.send_response(args.status)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok" if args.status < 400 else b"error")

        def log_message(self, fmt, *a):
            sys.stderr.write("[receiver] " + fmt % a + "\n")

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5056)
    p.add_argument("--log", required=True)
    p.add_argument("--secret", default="")
    p.add_argument("--status", type=int, default=200)
    p.add_argument("--delay", type=float, default=0.0)
    args = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args, threading.Lock()))
    sys.stderr.write(f"[receiver] listening :{args.port} status={args.status} delay={args.delay}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

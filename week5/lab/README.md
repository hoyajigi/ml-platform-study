# 5주차 실측: 서빙 중인 모델 버전과 alias 이동

venv: `uv venv -p 3.12 && uv pip install mlflow bentoml scikit-learn pandas httpx`
(env.sh 가 venv·mlruns·BENTOML_HOME 경로와 포트 5065~5068 을 정한다)

```bash
source env.sh
./start_tracking.sh && $PY train_register.py           # v1, v2 등록, champion -> v1
$PY import_fixed.py && bentoml build -f bentofile.yaml # (b') 고정 Bento
for k in mlserve live built; do ./serve.sh start $k; done
$PY probe.py 0_start
$PY move_alias.py 2 && $PY probe.py 1_no_restart
for k in mlserve live built; do ./serve.sh stop $k; ./serve.sh start $k; done && $PY probe.py 2_restart
./serve.sh stop built; bentoml build -f bentofile.yaml; ./serve.sh start built; $PY probe.py 3a
./serve.sh stop built; $PY import_fixed.py; bentoml build -f bentofile.yaml; ./serve.sh start built; $PY probe.py 3b
$PY bench.py mlserve http://127.0.0.1:5066/invocations mlflow 1000 1,8,32
$PY verify_single.py
```

결과는 `results/` (probe.jsonl, latency.jsonl, verify_single.jsonl, store_listing.txt, versions.json).
`latency.jsonl` 에는 처음 돌린 배칭 1회분(503에서 멈춘 `bento_batch` 행)이 남아 있다.

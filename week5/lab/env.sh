# source 해서 쓰는 공통 환경. LAB 아래에 venv, mlruns, bento store, 로그가 생긴다.
export LAB="${LAB:-/private/tmp/claude-501/-Users-macbook-personal/ed139640-917e-4192-92c9-3dc847b1b9e2/scratchpad/serving-lab}"
export HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PY="$LAB/venv/bin/python"
export PATH="$LAB/venv/bin:$PATH"
export BENTOML_HOME="$LAB/bentoml_home"
export MLFLOW_TRACKING_URI="http://127.0.0.1:5065"
export MLFLOW_DISABLE_AGENT_HINT=1
export MODEL_NAME="income-rf"
export ADULT_CSV="$LAB/data/adult.data"
export PORT_TRACK=5065 PORT_MLSERVE=5066 PORT_BENTO=5067 PORT_BENTO_BUILT=5068
mkdir -p "$LAB/logs" "$BENTOML_HOME"

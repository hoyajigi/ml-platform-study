# 레지스트리 웹훅은 7초 안에 못 받으면 사라진다

*Machine Learning Platform Engineering 스터디 2주차 정리 (CH4~CH5)*

---

2주차 범위는 두 장이다. 4장은 MLflow로 실험을 추적하고 모델을 레지스트리에 올리는 법, Feast로 피처를 등록하고 꺼내 쓰는 법이다. 5장은 그 모델과 피처로 배치 추론을 Kubeflow Pipelines(KFP) 파이프라인으로 묶는다.

1주차 글은 "3장 CI/CD는 git push로만 돌기 때문에 코드 변경 없이 모델만 바뀌면 배포를 깨울 트리거가 없다"로 끝났다. 4장에서 모델 레지스트리가 나왔으니 이번 주에는 그 질문을 레지스트리 쪽에서 다시 봤다. MLflow 3.x에는 레지스트리 웹훅이 있다. 맥북 로컬에서 MLflow 3.17.2로 어떤 이벤트가 오는지, 받는 쪽이 죽어 있으면 어떻게 되는지 측정했다.

## CH4 요약 — 실험 추적과 피처 스토어

예제는 미국 성인 소득 데이터(Adult/Census Income)로 연소득이 50K 이하인지 초과인지 맞히는 이진 분류기다. 데이터는 30,162행, 입력은 범주형 8개(Workclass, Education, Marital-Status, Occupation, Relationship, Race, Sex, Native_country), 정답은 `Target`이고 >50K 비율은 약 24.9%다.

1. EDA: `describe`/`info`로 기술 통계를 보고 plotnine으로 범주형 변수마다 타깃 비율 막대그래프를 그린다
2. 전처리: `get_dummies(drop_first=True)`로 원-핫 인코딩하면 특성이 90개가 된다. 컬럼 이름·순서를 `column_list.pkl`로 저장해 두고 추론 때 같은 순서로 맞춘다. 학습/평가는 80:20(24,129 / 6,033행)
3. 모델 3종 학습: 결정 트리, 랜덤 포레스트, XGBoost
4. MLflow에서 AUC 1등을 골라 모델 레지스트리에 등록
5. 피처를 Feast에 등록하고 오프라인·온라인으로 조회

### MLflow 추적

실험(`income-classifier`) 아래 run이 쌓인다. run 하나가 학습 1회다.

| 종류 | API | 이 예제에서 기록한 것 |
|---|---|---|
| 데이터셋 | `mlflow.data.from_pandas` + `log_input` | train/test/reference 3개. 컬럼·타입·행 수 메타데이터만 MLflow에 남고 CSV 본체는 MinIO `mlflow-datasets` 버킷 |
| 지표 | `log_metric` | `roc_auc_score_train/test`, `training_accuracy`, `test_accuracy` |
| 파라미터 | `log_params(tree.get_params())` | 모델 하이퍼파라미터 전체 |
| 모델 | `mlflow.sklearn.log_model(tree, "income-classifier")` | MLflow 3.0부터는 run 아티팩트가 아니라 별도 "logged model"로 저장 |
| 아티팩트 | `log_artifacts` | EDA 플롯 |

XGBoost는 `mlflow.xgboost.autolog()` 한 줄로 파라미터, 학습 중 지표, 모델, 특성 중요도 플롯을 기록한다. 직접 계산한 AUC·정확도는 autolog가 남기지 않아서 `log_metric`을 따로 불러야 한다.

스터디 서버의 test AUC는 결정 트리 0.8200, 랜덤 포레스트 0.8571, 원본 코드 XGBoost 0.7248이었다. XGBoost를 objective별 run으로 나눠 다시 돌리면 `binary:logistic`이 0.8633으로 1등이 된다.

### 모델 레지스트리

`MlflowClient().search_runs`에 `metrics.roc_auc_score_test > 0.8` 필터와 `order_by=["metrics.roc_auc_score_test DESC"]`를 줘서 1등 run을 고르고, `random-forest-classifier`라는 이름으로 등록한다. 같은 이름으로 다시 등록하면 v2, v3으로 쌓인다.

어느 버전을 쓸지는 alias가 아니라 stage로 정한다. 5장 추론 코드가 `current_stage == "Production"`인 버전을 찾는다.

```python
mv = mlflow.register_model(f"models:/{model_id}", "random-forest-classifier")
MlflowClient().transition_model_version_stage(
    "random-forest-classifier", mv.version,
    stage="Production", archive_existing_versions=True)
```

MLflow 3.0.0에서 `transition_model_version_stage`는 deprecated이고, 권장 방식은 alias(`champion` 같은 이름표를 버전에 붙이는 것)다. 아래 실험에서 이 차이가 중요해진다.

### Feast

원래 한 파일인 피처를 일부러 parquet 3개로 쪼개서(각각 `user_id`, 타임스탬프 포함) 피처가 여러 테이블에 흩어진 상황을 만든다.

| 개념 | 이 예제에서 |
|---|---|
| 엔티티 | `user_id` |
| 피처 뷰 | `demographic`(Native_country, Sex, Race), `relationship`(Relationship, Marital-Status), `occupation`(Workclass, Education, Occupation). TTL 365일 |
| 데이터 소스 | `FileSource`(Parquet), `s3://feature-data-sets/*.parquet`, `s3_endpoint_override`로 MinIO 지정 |
| 레지스트리 | `s3://feature-registry/registry_local.db`. 엔티티·피처 뷰 정의 저장 |
| 오프라인 스토어 | `type: file`. MinIO의 parquet |
| 온라인 스토어 | `type: redis` |
| 구체화 | `feast materialize START END`. 구간 안의 오프라인 데이터를 Redis에 기록 |

`feast apply` 직후 Redis는 비어 있고(dbsize 0), materialize 후에 키가 30,162개가 된다. 뷰마다 키가 생기지 않고 사용자당 hash 키 하나에 세 뷰의 피처와 뷰별 타임스탬프(`_ts:<view>`)가 같이 들어간다(Feast 0.40.1). TTL은 과거 데이터셋을 만들 때 기준 시점에서 얼마나 이전 값까지 볼지 정하는 룩백 윈도다.

```python
store = FeatureStore(repo_path=".")
# 학습·배치: entity_df(user_id, event_timestamp) 시점 이전의 가장 최근 값, MinIO parquet에서
store.get_historical_features(entity_df=entity_df, features=["demographic:Sex"]).to_df()
# 실시간: materialize로 Redis에 올라간 최신 값, 시점 인자 없음
store.get_online_features(features=["demographic:Sex"],
                          entity_rows=[{"user_id": "9f2ac416-..."}]).to_df()
```

온라인 조회는 레지스트리만 S3에서 읽고 값은 Redis에서 가져온다. 피처 파일이 바뀌면 materialize를 다시 해야 온라인 값에 반영된다. 이외에 `feast serve`(HTTP 피처 서버, 6566 포트, `POST /get-online-features`)와 `feast ui`(엔티티·피처 뷰·데이터 소스 조회)가 있다.

## CH5 요약 — Kubeflow Pipelines

5장은 4장의 모델과 피처로 배치 추론을 자동화한다. 실시간 추론은 6장 BentoML 몫이다.

- 컴포넌트: 파이프라인 단계 하나. 별도 파드로 실행된다. 파이썬 스크립트(argparse로 인자 받음) + 도커 이미지 + `component.yaml`(이름, 입출력과 타입, 이미지, 실행 명령)로 만든다
- 파이프라인: 컴포넌트를 이은 DAG. `@dsl.pipeline` 함수의 인자가 실행 때 바꿀 수 있는 파라미터다
- 실험(Experiment): 실행을 묶어 비교하는 단위
- 실행(Run): 파이프라인을 파라미터와 함께 한 번 돌린 것

| 단계 | 하는 일 | 입력 → 출력 |
|---|---|---|
| read_data | MinIO에서 사용자 ID 목록 parquet 다운로드 | 버킷/파일 → `Dataset` |
| retrieve_features | MinIO에서 `feature_store.yaml`을 받아 `get_historical_features` | entity_df + 피처 목록 → `Dataset` |
| run_inference | Production stage 모델과 그 run의 `column_list.pkl`을 받아 원-핫 → `reindex` → `predict_proba` | `Dataset` → `Dataset`(`Predicted_Income_Class` 추가) |
| write_data | 결과 parquet을 MinIO에 업로드 | `Dataset` → `inference-datasets` 버킷 |

`component.yaml`에서 값 인자는 `{inputValue: ...}`, 파일로 주고받는 아티팩트는 `{inputPath: ...}`/`{outputPath: ...}`로 넘긴다. 한 태스크의 출력을 다음 태스크 입력에 넣으면 KFP가 그걸로 의존 관계를 잡는다. `Compiler().compile(...)`이 만드는 `income_classifier_pipeline.yaml`은 태스크, 컴포넌트별 이미지·명령·입출력, 파라미터 14개를 KFP v2 형식으로 직렬화한 파일이고, 이걸 UI에 올려 Experiment를 만들고 Run을 시작한다.

스터디 실행 결과는 4개 태스크 모두 Succeeded, 출력 30,162행 × 91컬럼, 고소득(≥0.5) 예측 22.1%였다.

## 실습 환경

| 항목 | 스터디 구성 | 비고 |
|---|---|---|
| 클러스터 | K3s 단일 노드 | 부록A 책 기준 K3s `v1.24.17+k3s1` 또는 kind `v1.24.17`, 최신 경로 kind v1.36 / K3s v1.36.5 |
| Kubeflow | v1.7.0 (KFP v2.4.0) | 최신 경로는 KFP 2.17.2 |
| MinIO | Kubeflow 내장, NodePort 30010(API) / 30009(콘솔) | 버킷 `mlflow-artifacts`, `mlflow-datasets`, `feature-data-sets`, `feature-registry`, `inference-datasets` |
| MLflow | 3.0.0, 호스트 프로세스, 백엔드 SQLite, 포트 5000 | 책 본문은 2.6.0. 부록A는 K8s Deployment + Postgres |
| Feast | `feast[redis]==0.40.1` | |
| Redis | `redis:8.0.6`, NodePort 30006 | 온라인 스토어 |
| Python | 3.11, `sqlalchemy>=2.0,<2.1` 고정 | 2.1.x에서 MLflow 3.0.0 서버 기동 실패 |
| KFP SDK | 저장소 2.9.0 → 2.4.0으로 재컴파일 | 백엔드 버전에 맞춤 |
| 컴포넌트 이미지 | `python:3.10-slim-buster` → `-bookworm` | buster 저장소 종료 |

실습하면서 걸린 것도 적어 둔다.

- 책은 AUC 1등을 무조건 `random-forest-classifier`라는 이름으로 등록한다. XGBoost run을 objective별로 나누면 1등이 XGBoost가 되어 이름과 내용물이 어긋나고, 5장 `run_inference`가 쓰는 `predict_proba`가 `xgb.train` Booster에는 없어서 파이프라인이 실패한다
- `run_inference`는 배치마다 `get_dummies(drop_first=True)`를 다시 한다. 배치에 어떤 범주가 빠지면 기준 범주가 달라지고 `reindex`가 그 컬럼을 0으로 채워 예외 없이 다른 예측이 나올 수 있다. 전체 30,162행으로만 돌려서 실제로 어긋나는지는 확인하지 않았다
- Feast 레지스트리에는 데이터 소스의 `s3_endpoint_override`가 그대로 저장된다. 호스트에서 `localhost:30010`으로 만든 레지스트리는 파드 안에서 쓸 수 없어서 클러스터 안 Job으로 따로 만들었다
- MinIO `access_key`/`secret_key`가 일반 파이프라인 파라미터라 KFP 실행 이력에 평문으로 남는다

## 레지스트리가 바뀌면 누가 아나

1주차의 질문을 5장 파이프라인에 대 보면 이렇다. 추론 파이프라인은 실행할 때 `Production` stage 버전을 찾아 쓴다. 그런데 이 파이프라인을 실행하는 건 사람이다(UI의 Start Run이나 REST 호출). 새 버전이 Production으로 올라가도 그 사실을 받아서 파이프라인을 다시 돌리거나 실시간 서빙을 교체하는 주체가 없다.

MLflow는 3.x에서 모델 레지스트리 웹훅을 추가했다. 설치된 패키지의 웹훅 테이블 마이그레이션 날짜는 2025-07-07이고, 스터디가 쓰는 3.0.0에는 없다. 이게 "모델만 바뀌었을 때"의 트리거가 되는지 보려고 다음을 측정했다.

### 환경

- 맥북 로컬, MLflow 3.17.2, Python 3.14.7, 백엔드 sqlite
- `mlflow server` 5055 포트, 수신기(표준 라이브러리 `http.server`, 받은 요청을 시각·헤더·바디와 함께 jsonl로 기록) 5056 포트
- 시나리오는 3회 반복했고, 빈 디렉터리에서 venv 설치부터 다시 한 번 돌려 같은 결과를 확인했다

웹훅을 쓰기 전에 걸리는 조건이 세 가지 있었다.

- 파일 스토어(`./mlruns`)에서는 안 된다. `create_webhook`이 서버 500을 내고 로그에 `NotImplementedError: FileStore does not support create_webhook`이 찍힌다. DB 백엔드가 필요하다
- 기본 설정은 `http://`와 사설 IP를 수신 주소로 받지 않는다. `MLFLOW_WEBHOOK_ALLOWED_SCHEMES=http,https`, `MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS=true`로 풀었다. 클러스터 안의 배포 서비스를 받는 쪽으로 쓰려면 이 설정이 필요하다
- 웹훅 목록은 서버가 60초 캐시한다(`MLFLOW_WEBHOOK_CACHE_TTL`). 새로 만든 웹훅은 최대 60초 동안 이벤트를 못 받고, 지운 웹훅은 최대 60초 동안 계속 받는다. 첫 실행에서 실제로 지운 웹훅 쪽으로 이벤트가 갔다

### 어떤 동작에 이벤트가 오나

| 동작 | 이벤트 | API 응답 → 수신 |
|---|---|---|
| 등록 모델 생성 | `registered_model.created` | −0.8 ~ +0.6ms |
| 버전 등록 (v1, v2) | `model_version.created` | −0.5 ~ +1.1ms |
| alias `champion` → v1 | `model_version_alias.created` | 0.7 ~ 1.1ms |
| `champion`을 v2로 옮김 | `model_version_alias.created` | 0.5 ~ 1.7ms |
| 같은 값(v2)으로 다시 지정 | `model_version_alias.created` | 0.3 ~ 1.5ms |
| alias 삭제 | `model_version_alias.deleted` | 0.3 ~ 1.1ms |
| 버전 태그 설정 / 삭제 | `model_version_tag.set` / `.deleted` | 0.9 ~ 1.7ms |
| 미등록 run (`log_metric`, `log_model`) | 없음 | |
| `log_input` 데이터셋 기록 | 없음 | |
| **stage 전이 (`transition_model_version_stage`)** | **없음** | |
| 등록 모델 태그 / description 수정 / 버전 삭제 | 없음 | |

음수는 웹훅이 HTTP 응답보다 먼저 도착했다는 뜻이다. 전달은 스레드풀에서 비동기로 돌아서 API를 막지 않는다. 수신기가 5초 늦게 응답하게 해도 alias 설정 API는 15.6ms, 18.8ms로 지연 없을 때(12.7ms, 17.9ms)와 같았다.

표에서 두 가지를 봤다.

- **stage 전이에는 이벤트가 없다.** 이벤트 종류는 소스(`mlflow/entities/webhook.py`)에서 고정돼 있고 stage 관련 항목이 없다. 책과 스터디 코드는 stage로 Production을 지정하므로 웹훅을 켜도 그 변경은 전달되지 않는다. 웹훅을 트리거로 쓰려면 버전 지정을 alias로 바꿔야 한다
- alias를 옮겨도 `updated`가 아니라 `created`가 오고, 페이로드는 `{name, alias, version}`뿐이라 이전 버전이 없다. 같은 값으로 다시 지정해도 이벤트가 온다. 받는 쪽은 지금 배포된 버전과 비교해서 같으면 무시해야 한다

secret을 주면 `X-MLflow-Signature: v1,<base64>`, `X-MLflow-Delivery-Id`, `X-MLflow-Timestamp` 세 헤더가 온다. `"{id}.{timestamp}.{body}"`의 HMAC-SHA256이고, 수신기에서 따로 구현한 검증이 전부 통과했다(틀린 secret으로는 전부 실패).

```python
def verify(secret, headers, body):
    version, _, received = headers["X-MLflow-Signature"].partition(",")
    signed = f'{headers["X-MLflow-Delivery-Id"]}.{headers["X-MLflow-Timestamp"]}.'.encode() + body
    expected = base64.b64encode(hmac.new(secret.encode(), signed, hashlib.sha256).digest()).decode()
    return version == "v1" and hmac.compare_digest(expected, received)
```

### 받는 쪽이 죽어 있으면

재시도는 urllib3 `Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])`이다(`mlflow/webhooks/delivery.py`). 전달 이력을 저장하는 테이블은 없다.

| 경우 | 결과 |
|---|---|
| 수신기를 내린 상태에서 이벤트, 20초 뒤 기동 | 수신 0건. 약 7초 뒤 서버에 `Failed to send webhook ... NewConnectionError` 로그 한 줄 남기고 끝 |
| 1.5초 뒤 기동 | 2.1~2.8초 시점에 1건 수신. 7초 안쪽의 중단은 재시도로 메워짐 |
| 수신기가 500 반환 | 같은 Delivery-Id로 4회 시도(0, ~0, ~2.5, ~7초) 후 유실. **서버 로그가 남지 않음** |
| 수신기가 404 반환 | 1회로 끝, 재시도·로그 없음 |
| 서버 타임아웃 3초, 수신기가 5초 뒤 응답 | 같은 Delivery-Id로 4번 모두 수신기에 도착. 서버는 실패로 처리 |

배포 서비스가 롤링 업데이트로 10초만 내려가 있어도 그 사이 alias 변경은 사라진다. 5xx로 끝난 경우엔 서버 로그에도 흔적이 없다. 반대로 받는 쪽이 느리면 같은 이벤트가 네 번 온다.

`MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`를 지정하지 않으면 서버가 프로세스마다 키를 새로 만든다. 다른 키로 재기동하자 `list_webhooks`가 500(InvalidToken)을 내고, alias 설정 API는 정상 응답하는데 웹훅은 한 건도 나가지 않았다. 컨테이너로 띄울 때 이 키를 고정하지 않으면 재시작할 때마다 웹훅이 끊긴다.

### 정리

웹훅은 1~2ms 안에 서명까지 붙어서 오지만 한 번 놓치면 다시 받을 방법이 없다. 그래서 웹훅만으로 1주차의 빈자리를 채울 수는 없고, 받는 쪽이 주기적으로 레지스트리의 `alias → version`을 읽어 지금 배포된 버전과 비교하는 루프를 같이 둬야 한다. 웹훅은 그 루프를 기다리지 않고 바로 돌게 하는 신호로 쓴다. 1주차의 Argo CD가 git 웹훅과 별개로 기본 3분 주기로 저장소를 다시 읽는 것과 같은 구성이다.

받는 쪽에 필요한 처리를 모으면 다음과 같다.

1. 버전 지정은 stage 대신 alias
2. 서명 검증, `X-MLflow-Delivery-Id`로 중복 제거
3. 이벤트의 version을 현재 배포 버전과 비교해 같으면 무시
4. 주기적으로 레지스트리 상태와 대조(웹훅 유실 대비)
5. 서버 쪽 `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY` 고정

측정 코드와 원본 로그는 [`lab/`](lab/)에 있다. `LAB_DIR=<작업 디렉터리> ./run_all.sh`로 venv 설치부터 전체가 약 10분에 재현된다.

## 아직 안 해본 것

- sqlite 외 백엔드(Postgres)와 서버 워커가 여러 개일 때의 캐시 동작. 워커마다 캐시를 따로 가진다면 웹훅 생성·삭제 반영이 워커별로 다를 수 있다
- 위 루프를 실제로 만들어서 alias 변경 → KFP 추론 실행(또는 BentoML 교체)까지 이어 보는 것. 6장 BentoML을 보면서 해 볼 생각이다
- 도전과제 중 K8s에 Feast를 올려 4.2절을 다시 하는 것은 하지 않았다
- 책이 랜덤 포레스트를 고른 이유가 `predict_proba` 때문인지는 책에 적혀 있지 않아 확인하지 못했다

---

*Study: Machine Learning Platform Engineering (CloudNet@)*
*CH4. 신뢰할 수 있는 ML 시스템 설계 / CH5. ML 파이프라인 오케스트레이션*

# alias를 옮겨도 서빙 중인 모델은 바뀌지 않았다

*Machine Learning Platform Engineering 스터디 5주차 정리 (CH9.2~CH10)*

---

5주차 범위는 9장 후반(TensorBoard, 영화 추천기 훈련·MLflow 추적·레지스트리)과 10장(BentoML, MLflow, KServe로 모델 서빙)이다. 11장 모니터링은 6주차로 넘어갔다.

지금까지 글은 "코드 변경 없이 모델이 바뀔 때"를 따라갔다. 1주차는 그 변경을 깨울 트리거가 CI/CD에 없다는 것, 2주차는 레지스트리 웹훅이 그 역할을 하다가 놓치면 끝이라는 것, 4주차는 같은 입력으로 다시 훈련해도 같은 모델이 나온다는 보장이 없다는 것이었다. 이번 주는 마지막 단계인 서빙을 봤다. 레지스트리의 alias를 옮기면 서빙 중인 모델이 따라가는지, 그리고 요청을 보내는 쪽에서 지금 어느 버전이 응답하는지 알 수 있는지를 맥북에서 측정했다.

## CH9 후반 요약 — TensorBoard와 영화 추천기

### 9.2 TensorBoard

TensorBoard는 run 하나의 훈련 과정을 자세히 보는 도구이고, MLflow는 여러 실험의 파라미터·지표·모델을 모으는 도구로 나눠 쓴다. YOLOv8은 `project` 디렉터리에 TensorBoard 이벤트 파일을 쓰므로, 훈련 컴포넌트에 PVC를 마운트하고 그 경로를 `project`로 넘긴 뒤 Kubeflow의 TensorBoards에서 소스를 PVC로 고른다. 같은 디렉터리에 `args.yaml`(학습 인자 전체)과 `results.csv`도 남는다.

### 9.3 영화 추천기

| 단계 | 컴포넌트 | 입력 → 출력 |
|---|---|---|
| QA | `qa_data` | MinIO parquet → 컬럼·dtype·null·rating 범위 assert |
| 메타 | `get_dataset_metadata` | → `{n_users: max(userId), n_items: max(movieId)}` |
| 네거티브 샘플링 | `negative_sampling` | 사용자마다 본 적 없는 영화 10개를 rating 0.0으로 추가 |
| 훈련 | `train_model` | 데이터 + 하이퍼파라미터 → MLflow run ID |
| 검증 | `validate_model` | run ID + val → 같은 run에 지표 추가 |
| 승격 | `promote_model_to_staging` | run ID + 임계값 → 레지스트리 등록·`staging` alias |

모델은 PyTorch 행렬 분해다. 사용자·영화 임베딩의 원소별 곱을 Linear → ReLU → Dropout → Linear에 넣어 평점 하나를 낸다. 지표는 RMSE, precision@50, recall@50(평점 3 이상을 관련 있음으로 봄)이다.

MLflow에는 함수 인자 중 `mlflow_`가 아닌 것을 params로, epoch마다 `avg_training_loss`/`avg_testing_loss`를 metrics로, `mlflow.pytorch.log_model`로 모델을 남긴다. 검증 컴포넌트는 새 run을 만들지 않고 같은 run에 `precision_50`, `recall_50`, `rms`를 덧붙인다.

레지스트리는 stage가 아니라 alias를 쓴다(2주차의 소득 분류기는 stage였다).

```python
current_staging = client.get_model_version_by_alias(name, "staging")
# rms_diff > rms_th, prec_diff < prec_th, rec_diff < rec_th 중 하나라도 걸리면 return
result = mlflow.register_model(f"runs:/{new_model_run_id}/model", "recommender_production")
client.set_registered_model_alias("recommender_production", "staging", result.version)
```

- 승격 컴포넌트는 `staging`까지만 옮기고, 서빙이 쓰는 `prod`는 사람이 지정한다는 전제다
- 기본 임계값은 rms 0.0001, precision −0.3, recall −0.2다. rms는 조금만 나빠져도 거부하고 precision·recall은 많이 떨어져도 통과시킨다
- 거부돼도 `return`만 하므로 KFP 태스크는 SUCCEEDED로 끝난다. 거부 여부는 MLflow 레지스트리를 봐야 안다

스터디는 MovieLens 25M을 사용자 81명(train 6,400행)으로 줄여 돌렸다. 메타가 최댓값을 쓰기 때문에 n_users 162,502, n_items 208,715가 되고 파라미터 1,856,544개 대부분이 학습되지 않는 임베딩이다. 3에포크에서 test loss는 2.566 → 1.843 → 1.474였다. 네거티브 샘플링에 시드가 없어서 실행마다 샘플이 달라진다.

## CH10 요약 — BentoML, MLflow, KServe 서빙

### BentoML 서비스와 러너 (1.1.x, YOLO 예제)

Service는 API 서버(HTTP)와 Runner를 묶는다. API 서버가 입력을 검증해 Runner에 넘기고, Runner가 별도 워커 프로세스에서 추론한다. `--api-workers 2`로 띄우면 컨테이너 안에 runner 1개와 http_api_server 2개가 유닉스 소켓으로 연결돼 있다. YOLO 예제는 `/invocation`(탐지 JSON)과 `/render`(박스 그린 이미지) 두 API이고, 메서드는 `batchable=False`다. `bentofile.yaml`에 가중치 파일을 include로 넣어 빌드하므로 모델이 빌드 시점에 Bento 안으로 복사된다(5.99MiB, 컨테이너 이미지 6.66GB). Bento에는 `서비스명:버전레이블` 태그가 자동으로 붙는다.

### MLflow 모델을 BentoML로 (10.6, 1.4 문법)

```python
current_prod = MlflowClient().get_model_version_by_alias(registered_model_name, "prod")
bentoml.mlflow.import_model("recommender", f"runs:/{current_prod.run_id}/model")
bento_model = bentoml.mlflow.get("recommender:latest")
self.model = mlflow.pytorch.load_model(bento_model.path_of(bentoml.mlflow.MLFLOW_MODEL_FOLDER))
```

이 코드는 `@bentoml.service` 클래스의 `__init__`에 있어서 alias 조회와 다운로드가 빌드가 아니라 서비스 프로세스 시작 때 일어난다. 빌드된 Bento는 27.49KiB에 모델 0B였고, 컨테이너를 띄울 때마다 다시 받아 `/readyz`까지 256초가 걸렸다. 책 코드에는 alias가 바뀌었을 때 다시 읽는 경로가 없다.

### MLflow만으로, KServe로

`mlflow models serve -m "models:/recommender_production@prod"`는 `/invocations`와 `/health`를 연다. 모델 `forward(user, item)`이 인자 두 개라 스터디에서는 `PythonModel` 래퍼로 감쌌다. KServe는 `InferenceService`에 `modelFormat`과 `storageUri`를 적으면 `storage-initializer`가 모델을 `/mnt/models`로 받고 서버가 읽는다. 스터디에서는 sklearn iris 스모크만 했다.

| 항목 | BentoML | MLflow | KServe |
|---|---|---|---|
| 초점 | 배포·서빙 단순화 | ML 라이프사이클 전체 | 서빙 오케스트레이션 |
| K8s 중심 | 아니오 | 아니오 | 예 |
| 실험 추적 | 아니오 | 예 | 아니오 |

책의 결론은 빨리 띄우려면 BentoML이나 MLflow, K8s에서 세밀하게 제어하려면 KServe다.

## 실습 환경

| 항목 | 버전 | 비고 |
|---|---|---|
| 클러스터 | k3s | `local-path`, 객체 스토리지 SeaweedFS |
| BentoML | 1.1.6 (YOLO) / 1.4 계열 (추천) | 두 문법이 달라 venv 분리 |
| ultralytics | 8.0.194 | Bento base image `8.0.203-cpu` |
| MLflow 클라이언트 | 3.17.2 | |
| KServe | v0.19.0 | |
| KFP SDK | 2.16.1 | GitOps 도전과제 제출용 |

실습하면서 걸린 것:

- BentoML 1.4 build가 base image의 CPU torch를 무시하고 의존성을 PyPI 기준으로 잠가 CUDA torch를 깔았고, 이미지가 13.2GB가 됐다
- 추천 상위 10개 중 학습 데이터에 나온 영화가 0개였다. 후보가 1..n_items 전체라서다
- SeaweedFS와 최신 boto3 조합에는 `AWS_*_CHECKSUM_*=when_required`가 필요했다
- GitOps 도전과제는 `runs/*.yaml`을 push하면 Gitea Actions가 KFP run을 제출하는 데까지 했다. 승격 거부가 KFP에서는 성공으로 보이는 점은 그대로 남는다

## 서빙 중인 모델은 무엇인가

### 설정

- 맥북 로컬, python 3.12.11, mlflow 3.18.0, bentoml 1.4.40, scikit-learn 1.9.1
- 모델: Adult 데이터 30,162행으로 학습한 OneHotEncoder + RandomForest. 같은 등록 모델에 v1(나무 50개, 깊이 8)과 v2(나무 200개, 깊이 제한 없음)를 올렸다. 1000행 기준 두 버전의 라벨 일치율은 94.8%, 확률 최대 차이는 0.256이라 응답을 보면 어느 버전인지 구분된다
- alias `champion` → v1에서 시작
- 서빙 3종
  - (a) `mlflow models serve -m models:/<name>@champion`
  - (b) BentoML, 노션 10.6과 같은 방식(`@bentoml.service`의 `__init__`에서 `import_model`)
  - (b') BentoML, 미리 `import_model`해 둔 모델을 `bentoml build`로 고정한 Bento
  - 기준선으로 같은 프로세스 안 `pyfunc.predict`

mlflow 3.18은 sklearn 모델을 기본으로 skops 형식으로 저장하는데, RandomForest의 `sklearn.tree._tree.Tree`를 신뢰할 수 없는 타입으로 보고 `log_model`을 거부했다. `skops_trusted_types=["sklearn.tree._tree.Tree"]`를 넘겨서 저장했다.

### alias를 v2로 옮긴 뒤

| 서빙 방식 | 재시작 없음 | 재시작 | build만 다시 | import 다시 + build |
|---|---|---|---|---|
| (a) mlflow serve | v1 | v2 | | |
| (b) BentoML, 시작 시 import | v1 | v2 | | |
| (b') BentoML, build로 고정 | v1 | v1 | v1 | v2 |

판정은 응답 확률 1000행을 v1·v2를 로컬에서 직접 돌린 결과와 비트 단위로 비교해서 했다.

- (a)와 (b)는 alias를 프로세스 시작 때 한 번 해석한다. mlflow는 `pyfunc/backend.py:249`의 `serve()`에서 모델을 한 번 받고, 이후 다시 로드하는 경로가 없다. alias 해석은 `store/artifact/utils/models.py:150`의 `get_model_version_by_alias` 한 번이다
- `import_model`은 MLflow 아티팩트를 BentoML store로 실제로 복사한다(`_bentoml_impl/frameworks/mlflow.py:174-193`, 원본과 inode가 다르다). (b)는 재시작할 때마다 새 태그로 사본이 쌓였다. 이번 실험에서만 v1 2.1MiB 1개, v2 32MiB 2개가 생겼고 지우는 경로는 없다
- (b')는 `bentoml build`를 다시 해도 v1이었다. store에 이미 있는 모델을 묶을 뿐이라 `import_model`부터 다시 해야 v2가 된다
- 재시작에서 `/readyz`까지 3~4초였다. 로컬 tracking 서버라서 빠른 것이고, 스터디 환경(원격 MinIO 경유)은 256초였다

### 요청 쪽에서 버전을 알 수 있나

| 서빙 방식 | 응답 바디 | 응답 헤더 | 메타 엔드포인트 |
|---|---|---|---|
| (a) mlflow serve | `{"predictions": ...}` | `server: uvicorn` | `/version`이 `3.18.0`을 돌려주지만 mlflow 패키지 버전이다 |
| (b) BentoML | 확률 리스트 | `server: BentoML Service/...`, `x-bentoml-request-id` | `/metrics`의 `service_version="not available"`. `/docs.json`, `/readyz`에도 없음 |
| (b') BentoML 빌드본 | 확률 리스트 | 위와 같음 | `/metrics`의 `service_version`에 Bento 태그가 나온다 |

세 방식 모두 응답만으로는 레지스트리 버전, run_id, alias를 알 수 없었다. (b')는 Bento 태그 → store의 `bento.yaml` → 모델 태그 → `MLmodel`의 run_id로 따라갈 수 있지만 store에 접근할 수 있어야 하고, 레지스트리 버전 번호와 alias는 그 경로에도 없다. `import_model`은 labels와 metadata를 인자로 넘길 때만 기록한다(같은 파일 155-165행). 노션의 C7 자체 서버처럼 응답 헤더에 직접 넣는 엔드포인트를 만들어 보니 `x-model-version: 2`, `x-model-run-id`가 정상으로 실렸다. 버전 노출은 서빙 코드가 직접 해야 한다.

### 예측값과 지연

1000행을 요청 한 번으로 보내든, 1행 요청 1000개를 동시성 32로 보내든, (a)(b)(b') 모두 기준선 `predict_proba`와 비트 단위로 같았다. 배칭을 켜도 같았다.

1행 요청 1000건(모델 v2, BentoML 워커 1개)의 지연이다. 부하 생성기가 같은 맥에서 돌았고 WindowServer가 CPU 47%를 쓰고 있어서 절대값보다 방식끼리 비교용으로 봐야 한다.

| 방식 | 동시성 | p50 ms | p99 ms | 성공 rps | 503 |
|---|---|---|---|---|---|
| in-process | 1 | 5.25 | 5.95 | 187.2 | |
| (a) mlflow serve | 1 / 8 / 32 | 7.70 / 61.46 / 255.69 | 8.39 / 93.86 / 404.60 | 132.0 / 129.6 / 126.0 | 0 |
| (b) 배칭 꺼짐 | 1 / 8 / 32 | 7.01 / 64.60 / 292.09 | 9.01 / 82.71 / 395.66 | 138.6 / 121.0 / 107.5 | 0 |
| (b) 배칭, `max_latency_ms=20` | 1 / 8 / 32 | 7.42 / 20.18 / 39.34 | 10.59 / 26.75 / 648.13 | 128.1 / 282.6 / 241.6 | 0 / 378 / 189 |
| (b) 배칭, `max_latency_ms=1000` | 1 / 8 / 32 | 8.03 / 15.29 / 36.26 | 11.14 / 41.37 / 549.10 | 122.8 / 437.8 / 412.3 | 0 |

노션 예제는 배칭을 켜지 않았다(`batchable=False`). 켜면 동시성 8에서 처리량이 121에서 438 rps로 올랐다. 다만 `max_latency_ms`를 20으로 짧게 주면 느려지는 게 아니라 요청을 버린다. 동시성 8에서 1000건 중 378건이 503 "process is overloaded"였다. 첫 요청의 대기가 `max_latency`를 넘으면 배치를 취소하고(`bentoml/_internal/marshal/dispatcher.py:227-229, 305-310`) 서버가 `ServiceUnavailable`을 낸다(`_bentoml_impl/server/app.py:121-122`).

### 정리

- 레지스트리에서 alias를 옮겨도 서빙 중인 프로세스는 따라가지 않는다. mlflow serve와 BentoML(시작 시 import)은 재시작해야, build로 고정한 Bento는 import부터 다시 해야 바뀐다. alias 이동 → 배포 반영 사이에는 재시작이라는 별도 단계가 필요하고, 2주차의 웹훅이나 주기적 대조가 그 단계를 깨워야 한다
- 요청 쪽에서는 지금 어느 버전이 응답하는지 알 수 없다. 응답 헤더나 `/info` 같은 엔드포인트에 버전·run_id를 넣는 건 서빙 코드가 직접 해야 하고, 그래야 롤아웃 중 두 버전이 섞여 있는지 확인할 수 있다
- 서빙 방식이 예측값을 바꾸지는 않았다(전부 비트 일치). 4주차처럼 차이가 생긴다면 서빙이 아니라 모델을 만드는 쪽이다
- 시작 시 import 방식은 재시작마다 모델 사본이 store에 쌓인다. 장기 운영이면 정리 절차가 필요하다
- adaptive batching은 동시 요청이 몰릴 때 처리량을 3배 이상 올리지만 `max_latency_ms`가 짧으면 503으로 요청을 버린다

측정 코드와 결과는 [`lab/`](lab/)에 있고, 재현 순서는 [`lab/README.md`](lab/README.md)에 적었다. 전체 약 16분 걸렸다.

## 아직 안 해본 것

- 원격 tracking 서버·MinIO를 거칠 때 재시작 비용(로컬은 3~4초, 스터디는 256초)과, 그 사이 요청이 어떻게 되는지
- 두 BentoML 서버가 같은 `BENTOML_HOME`을 쓰면 `/metrics`가 다른 서버의 카운터를 보여 줬다. 원인은 소스로 확인하지 않았다
- mlflow serve의 실제 워커 수와 동시성별 배치 크기
- YOLO 예제와 KServe(`model_version: null`)는 측정하지 않았다. 11장 모니터링에서 서빙 버전을 지표 레이블로 붙이는 방법을 보고 다시 정리할 생각이다

---

*Study: Machine Learning Platform Engineering (CloudNet@)*
*CH9. 모델 훈련과 검증 2부 / CH10. 모델 추론과 서빙*

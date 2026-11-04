# 학습 데이터를 그대로 넣어도 드리프트로 나온다

*Machine Learning Platform Engineering 스터디 3주차 정리 (CH6~CH7)*

---

3주차 범위는 두 장이다. 6장은 소득 분류기를 BentoML로 실시간 API로 올리고(Yatai 배포), Evidently로 입력 데이터 드리프트를 본다. 7장부터는 캡스톤 프로젝트로 넘어가서 Kubeflow Notebooks와 KFP 컴포넌트 사이의 데이터 전달을 다룬다.

1·2주차에 이어 "모델만 바뀌는 상황을 누가 알아채나"를 따라가고 있다. 2장의 폐쇄 루프에서 재훈련을 깨우는 신호는 모니터링이고, 6장의 Evidently가 그 신호를 만든다. 그런데 6.2.3에서 책은 학습 데이터에서 뽑은 요청 50건을 넣었는데도 드리프트 컬럼 비율이 0.875로 나왔다고 쓰고, "윈도가 작아서일 수 있다"고만 한다. 이 숫자가 왜 나오는지, 몇 건부터 믿을 수 있는지를 측정했다.

## CH6 요약 — BentoML 서빙과 Evidently 드리프트

### 벤토 구성

벤토(Bento)는 모델, 서빙 코드, 의존성, 설정 파일을 묶은 배포 단위다. 컨테이너 이미지는 별도 단계에서 만든다.

| 구성 요소 | 이 예제에서 |
|---|---|
| 모델 등록 | `bentoml.mlflow.import_model`로 MLflow `models:/random-forest-classifier/Production`을 BentoML 로컬 스토어로 가져온다 |
| 러너 | `bentoml.mlflow.get(...).to_runner()`. 추론 담당이고 API 서버와 따로 확장된다 |
| 서비스 (`service.py`) | `@svc.on_startup`에서 Production 버전 run의 `column_list.pkl`을 받고 Feast 스토어를 만들어 `context.state`에 둔다 |
| API | `/predict`: `{"user_id": ...}` → Feast 온라인 조회 → 더미 컬럼 매핑 → `runner.predict.run` → `{"income_category": "<=50K"}` |
| `bentofile.yaml` | 서비스 진입점, 포함 파일, `requirements.txt`, docker env |

```python
income_clf_runner = bentoml.mlflow.get("random-forest-classifier:latest").to_runner()
svc = bentoml.Service("income_classifier_service", runners=[income_clf_runner])

@svc.api(input=JSON(pydantic_model=IncomeClassifierUsers), output=JSON(), route="/predict")
def predict(inputs, ctx):
    feature_df = ctx.state["store"].get_online_features(
        features=ctx.state["feature_list"], entity_rows=[inputs.dict()]).to_df()
    ...
```

### 빌드, 푸시, Yatai 배포

1. `bentoml build -f bento/bentofile.yaml` → `income_classifier_service:<태그>` (앱 22KiB + 모델 37MiB)
2. `bentoml yatai login` → `bentoml push`. 모델과 벤토가 Yatai(S3는 MinIO `yatai` 버킷)로 올라간다
3. Yatai UI에서 Deployment 생성

Yatai 1.1.x는 Yatai(레지스트리·대시보드), yatai-image-builder(kaniko로 벤토를 이미지로 빌드), yatai-deployment(API 서버·러너 Deployment 생성) 세 부분이다. UI에서 배포를 만들면 `BentoRequest` → 빌더 파드 → `Bento` → `BentoDeployment` 순서로 CR이 생긴다. 결과는 API 서버 파드(main + metrics-transformer + envoy)와 러너 파드이고 HPA(min 1 / max 2, CPU 80%)가 붙는다.

빌더 파드(kaniko)는 기본 limits 3Gi로 떠서 `OOMKilled`가 났다. UI에서 지정한 리소스는 `BentoDeployment`에만 들어가서, `BentoRequest`를 직접 패치해야 했다.

```bash
kubectl -n yatai patch bentorequest <이름> --type merge -p \
  '{"spec":{"imageBuilderContainerResources":{"requests":{"cpu":"500m","memory":"2Gi"},"limits":{"cpu":"2","memory":"6Gi"}}}}'
```

### Evidently

버전은 0.5.0, `Report` + metric 클래스를 쓰는 API다.

- reference: 학습 때 MLflow run에 `log_input`으로 남긴 `reference_features`. run의 `dataset_inputs` 태그에서 찾아 MinIO에서 CSV로 받는다
- current: 배치에서는 Feast로 조회한 피처, API에서는 최근 요청 윈도
- metric: `DatasetDriftMetric`, `DatasetMissingValuesMetric`, 컬럼 8개마다 `ColumnDriftMetric` + `ColumnSummaryMetric`
- 판정: 컬럼마다 검정으로 드리프트 여부를 정하고, 드리프트 컬럼 비율이 0.5 이상이면 데이터셋 드리프트
- 대시보드: `evidently ui`에 `RemoteWorkspace`로 붙어 프로젝트·패널을 만들고 `workspace.add_report`로 스냅샷을 쌓는다

6.2.2는 `detect_drift` 컴포넌트를 5장 파이프라인의 피처 조회와 추론 사이에 넣는다(`fetch_data → retrieve_features → detect_drift → run_inference → write_data`). 본문은 "드리프트가 나오면 파이프라인을 멈춘다"고 하지만 코드는 `run_inference.after(detect_drift)`로 순서만 정해서, 드리프트가 나와도 추론은 진행된다. 멈추려면 드리프트 여부를 출력으로 받아 `dsl.If`로 분기해야 한다.

6.2.3은 API로 들어온 요청의 드리프트를 본다. `/predict`마다 Feast 피처 1행을 서비스 메모리의 DataFrame에 붙이고, N행이 찬 뒤부터는 요청이 올 때마다 가장 오래된 행을 버리고 보고서를 다시 만든다.

```python
self.current = pandas.concat([self.current, new_rows], ignore_index=True)
if self.current.shape[0] > self.window_size:
    self.current = self.current.iloc[-self.window_size:]   # 최근 N행만 유지
if self.current.shape[0] < self.window_size:
    return                                                # N행이 찰 때까지 대기
self.report.run(reference_data=self.reference, current_data=self.current)
self.workspace.add_report(project_id=self.project_id, report=self.report)
```

N은 `EVIDENTLY_REPORT_WINDOW_SIZE`이고 책 `.env`는 50이다. 윈도가 프로세스 메모리에 있어서 `api_server.workers=1`로 띄워야 한다. 워커가 여럿이면 워커마다 윈도가 따로 쌓인다.

## CH7 요약 — 노트북과 KFP 데이터 전달

7장부터 캡스톤이다. 프로젝트 1은 신분증 탐지(YOLO 변형), 프로젝트 2는 MovieLens 영화 추천이다. 그 전에 농담 생성 파이프라인(pyjokes)으로 노트북과 데이터 전달을 먼저 다룬다. 노션 기준으로 7.2.2 이후와 7.3은 4주차 범위다.

### Kubeflow Notebooks

| 항목 | 내용 |
|---|---|
| 노트북 서버 | `notebooks.kubeflow.org` CR. 이미지, CPU/메모리, GPU 수를 고르면 컨트롤러가 StatefulSet·Service·VirtualService를 만든다 |
| 볼륨 | 워크스페이스 볼륨은 `/home/jovyan`에 새로 생성, 데이터 볼륨은 기존 PVC를 추가 마운트 |
| PodDefault | 라벨이 맞는 파드에 env 등을 주입하는 네임스페이스 리소스. `MLFLOW_TRACKING_URI`나 `KF_PIPELINES_ENDPOINT`를 넣는 데 쓴다 |
| 어피니티/톨러레이션, 메뉴 | 관리자가 `jupyter-web-app-config` ConfigMap에 선택지와 기본값을 둔다 |
| 커스텀 이미지 | 베이스 이미지 위에 의존성, Jupyter 런타임, Kubeflow 호환 설정을 얹는다 |

### KFP v2 데이터 전달

컴포넌트마다 별도 파드에서 돌아서 데이터를 넘길 방법이 따로 필요하다. 컴파일된 YAML의 각 executor는 `pip install kfp` → 함수 코드를 `ephemeral_component.py`로 저장 → 실행이다. 그래서 import는 함수 안에 쓰고, 추가 패키지는 `packages_to_install`이나 `base_image`로 넣는다.

| 방식 | 예 | 저장 위치 |
|---|---|---|
| 값 | `-> str/int` 반환, 다음 태스크에 `.output` | 파라미터 |
| 아티팩트 | `Output[Artifact]`의 `.path`에 쓰고 `Input[Artifact]`로 받음 | MinIO `mlpipeline` 버킷 `v2/artifacts/...` |
| 타입 있는 아티팩트 | `Dataset`, `Model`, `Metrics`, `HTML` 등 | `Metrics`류는 파일 없이 MLMD(MySQL)에만 |

멤버가 같은 문자열을 값과 아티팩트로 크기별로 넘겨 본 결과다.

| 크기 | 값 전달 | 아티팩트 전달 |
|---|---|---|
| 10KB, 60KB | 성공 | |
| 100KB | 실패: `argument list too long` (값이 파드 명령 인자로 들어감) | |
| 1MB | 실패: annotations 262144바이트 초과 | |
| 10MB | 실패: MLMD gRPC 4MB 한도 | 성공 (200MB도 성공) |

## 실습 환경

| 항목 | 버전·구성 | 비고 |
|---|---|---|
| 클러스터 | kind (`kindest/node:v1.31.6`) | |
| BentoML | 1.1.10, Python 3.10 | `setuptools<81` 고정 |
| Yatai | 1.1.13 (세 차트 모두) | 2023-10-09가 마지막 릴리스 |
| 빌더 | kaniko 1.9.1, 클러스터 내 `registry:2` + socat 프록시 | |
| Evidently | 0.5.0 (+ `litestar==2.12.1`, `sniffio`) | |
| MLflow / Feast | 3.0.0 / 0.40.1 | |
| KFP | 백엔드 2.4.1, SDK 2.9.0 | |

실습하면서 걸린 것:

- Yatai 차트 기본 values의 `s3.accessKeyExistingSecretKey`가 오타(`aceess…`)라 직접 지정해야 한다
- ingress-nginx가 없는 kind에서는 `yatai-deployment`에 `automaticDomainSuffixGeneration: false`와 고정 `domainSuffix`를 줘야 뜬다
- Yatai로 띄운 벤토는 `production.env`의 MLflow/Feast/Redis 주소가 로컬 포워딩 기준이라 파드 안에서 `/predict`가 실패할 수 있다. 드리프트 서비스는 `docker run`으로만 확인했다
- 책 listing 7.5는 함수 이름을 같은 이름 변수로 덮어써서 NameError가 난다

## 윈도 50건의 드리프트 판정은 대부분 잡음이다

### 설정

책 설정에 맞췄다. 데이터는 UCI Adult(32,561행)이고 70%(22,793행)를 reference, 나머지 30%를 current 추출 풀로 썼다. 컬럼은 책과 같은 범주형 8개이고, 참고용으로 수치형 2개(Age, Hours-per-week)를 따로 측정했다. 데이터셋 판정은 범주형 8개로만 냈다. 맥북 로컬에서 Evidently 0.5.0(노션 버전)과 0.7.23(최신)을 venv로 나눠 같은 시드로 돌렸는데, 7,200회 × 10컬럼에서 판정이 한 건도 다르지 않았다. 0.7의 새 API도 내부에서 같은 legacy 코드를 부른다. 아래 수치는 0.5.0 기준이다.

### 검정은 reference 크기로 정해진다

기본 검정 선택은 `evidently/calculations/stattests/registry.py:137-160`의 `_get_default_stattest`에 있다. 분기 기준은 current가 아니라 **reference 행 수**다.

| reference 행 수 | 범주형 | 수치형 | 드리프트 조건 |
|---|---|---|---|
| ≤ 1000 | 카이제곱(범주 3개 이상) / Z-test(2개) | KS | p < 0.05 |
| > 1000 | Jensen-Shannon 거리 | Wasserstein 거리(reference 표준편차로 나눈 값) | 거리 ≥ 0.1 |

학습 데이터를 reference로 쓰면 거의 항상 1000행을 넘으므로 거리 검정이 선택된다. 거리 검정은 표본 크기를 고려하지 않는다. 같은 분포에서 50건을 뽑아도 경험 분포가 reference와 어긋난 만큼 거리가 나오고, 그 값이 0.1을 넘으면 드리프트다.

### 변화가 없을 때 (거짓 경보, n당 200회)

| n | 평균 드리프트 컬럼 비율 | 데이터셋 드리프트 비율 | Native_country | Education | Sex |
|---|---|---|---|---|---|
| 30 | 0.829 | 1.000 | 1.000 | 1.000 | 0.145 |
| 50 | 0.709 | 0.985 | 1.000 | 1.000 | 0.060 |
| 100 | 0.471 | 0.565 | 1.000 | 0.975 | 0.025 |
| 200 | 0.244 | 0.010 | 1.000 | 0.505 | 0 |
| 500 | 0.100 | 0 | 0.795 | 0.005 | 0 |
| 1000 | 0.001 | 0 | 0.010 | 0 | 0 |
| 2000, 5000 | 0 | 0 | 0 | 0 | 0 |

n=50 한 번 실행에서 정확히 8개 중 7개(0.875)가 나왔다. 책의 숫자는 재현되고, 원인은 데이터가 아니라 표본 크기와 거리 검정의 조합이다. 범주가 많은 컬럼일수록 오래 남는다. Native_country(41개 범주, United-States 비율 0.895)는 n=500에서도 79.5%가 드리프트로 찍혔다. 범주 2개인 Sex는 n=50에서도 6%였다.

6.2.3처럼 요청마다 판정하는 슬라이딩 윈도로도 돌렸다(윈도 크기마다 스트림 3개 × 판정 200회).

| N | 데이터셋 경보 비율 | 비고 |
|---|---|---|
| 50 | 0.985 / 1.000 / 1.000 | 첫 판정부터 경보 |
| 100 | 0.665 / 0.670 / 0.525 | 스트림마다 경보가 5~6번 켜졌다 꺼짐 |
| 200 | 0 / 0 / 0 | |
| 500 | 0 / 0 / 0 | Native_country 컬럼은 200회 내내 드리프트 |

### 변화가 있을 때 (탐지력)

Sex의 Female 비율(reference 0.330)을 올리고, Age에 상수를 더해 변화를 넣었다. 무한 표본에서의 거리는 Sex +5%p가 JS 0.037, +10%p가 0.073, +20%p가 0.143이다.

| n | Sex +5%p | Sex +10%p | Sex +20%p | 데이터셋 판정 (+20%p) |
|---|---|---|---|---|
| 50 | 0.080 | 0.310 | 0.810 | 1.000 |
| 100 | 0.045 | 0.285 | 0.915 | 0.915 |
| 200 | 0.010 | 0.170 | 0.960 | 0.140 |
| 500 | 0 | 0.065 | 0.990 | 0 |
| 1000 | 0 | 0.010 | 1.000 | 0 |
| 5000 | 0 | 0 | 1.000 | 0 |

거리 검정은 효과 크기 문턱으로 동작한다. 무한 표본 거리가 0.1보다 작은 변화(+5, +10%p)는 n을 키울수록 탐지율이 0으로 간다. n≤100의 높은 탐지율은 앞 표의 잡음 경보가 대부분이다. 그리고 한 컬럼만 바뀌면 데이터셋 판정(비율 0.5 이상)은 n≥500에서 0이다. 8개 중 1개라서다.

### reference를 1000행으로 줄이면

reference 크기가 검정을 바꾸는 기준이라, reference 1000행과 1001행으로 같은 무변화 실험을 돌렸다(100회).

| 데이터셋 드리프트 비율 | n=50 | n=100 | n=500 | n=1000 |
|---|---|---|---|---|
| reference 1000행 (p-value 검정) | 0.03 | 0.03 | 0.32 | 0.68 |
| reference 1001행 (거리 검정) | 0.98 | 0.67 | 0 | 0 |

reference 한 행 차이로 경향이 반대가 된다. p-value 쪽은 n이 커질수록 거짓 경보가 늘었다. Evidently의 카이제곱은 두 표본을 비교하는 검정이 아니라 reference 비율을 기댓값으로 고정한 적합도 검정이라서(`chisquare_stattest.py:36-46`), reference 1000행의 추정 오차까지 차이로 잡는 것으로 보인다. 원인을 분리하는 실험은 하지 않았다.

### 정리

- 책 설정(reference 2만여 행, 범주형 8개, 윈도 50)에서 드리프트 판정은 변화가 없어도 98.5% 확률로 난다. 0.875는 정상 상태의 값이다
- 이 설정이면 윈도 200건부터 데이터셋 거짓 경보가 1% 아래로 내려가고, 범주가 많은 컬럼까지 보려면 1000건은 모아야 한다
- 기본 거리 검정은 거리 0.1 아래의 변화는 표본을 늘려도 못 잡는다. Female 비율 +10%p는 n=1000에서 1%만 잡혔다
- 데이터셋 판정(비율 0.5)은 한 컬럼만 바뀐 드리프트를 거의 놓친다. 재훈련 트리거로 쓰려면 컬럼 단위 결과를 따로 봐야 한다
- `stattest`와 `stattest_threshold`를 명시하지 않으면 reference 크기에 따라 검정이 조용히 바뀐다

1·2주차의 질문에 대 보면, 이 판정을 그대로 재훈련 트리거로 연결하면 윈도 50에서는 요청마다 재훈련을 요청하게 된다. 6.2.2 코드가 드리프트에도 파이프라인을 멈추지 않는 것은 결과적으로 이 경보가 아무 동작도 일으키지 않는다는 뜻이기도 하다.

측정 코드와 결과 CSV는 [`lab/`](lab/)에 있다. 재현은 `lab/run_all.sh <venv> <adult.data> <out> <tag> <reps> <workers>` 후 `analyze.py`로, 두 버전 합쳐 약 7분 걸렸다.

## 아직 안 해본 것

- 책 실습의 실제 `reference_features` CSV(MinIO)와 Feast를 거친 current의 전처리(앞 공백, `?` 처리)가 내 설정과 같은지 확인하지 못했다
- 범주형 컬럼 여러 개가 같이 바뀌는 상관된 변화, 수치형의 분포 모양 변화는 측정하지 않았다
- 거짓 경보를 줄이는 쪽(p-value 검정 + 다중 비교 보정, 컬럼별 임곗값)으로 실제 운영 설정을 정해 보는 것은 다음에 해 볼 생각이다
- Yatai 대신 KServe로 서빙하는 멤버 가이드는 읽기만 했다

---

*Study: Machine Learning Platform Engineering (CloudNet@)*
*CH6. ML 모델 프로덕션화 / CH7. 데이터 분석 및 준비*

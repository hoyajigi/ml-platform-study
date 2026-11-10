# 같은 시드로 다시 훈련했는데 MPS에서는 다른 모델이 나왔다

*Machine Learning Platform Engineering 스터디 4주차 정리 (CH7 후반~CH9.1)*

---

4주차 범위는 7장 후반(두 캡스톤의 데이터 준비 파이프라인), 8장(YOLO 훈련·검증 컴포넌트), 9장 앞부분(PVC로 데이터 공유)이다. 9.2 TensorBoard와 9.3 영화 추천 + MLflow는 5주차로 넘어갔다.

1~3주차는 "모델만 바뀌는 상황을 누가 알아채나"를 따라갔다. 이번 주 책은 그 반대쪽, 바뀐 모델을 다시 만들 수 있느냐를 다룬다. 8장은 재현 가능한 훈련 프로세스가 필요하다고 하고, 8.2 요약은 데이터를 어떻게 가져오든 버전 관리와 계보(lineage)를 지원해야 한다고 한다. 그래서 책 코드 그대로 같은 데이터로 두 번 훈련하면 같은 모델이 나오는지, MLflow 기록만 보고 두 run이 같은지 판단할 수 있는지를 맥북에서 측정했다. 도전과제 두 개(에포크별 성능 비교, MPS 훈련 속도 비교)도 같이 했다.

## CH7 후반 요약 — 캡스톤 데이터 준비

3주차에 7.2.1(값 전달)까지 정리했다. 7.2.2는 큰 데이터를 값 대신 파일 경로로 넘기는 방식이다. `Output[Artifact].path`에 파일·디렉터리를 쓰고(디렉터리는 컴포넌트가 직접 `os.makedirs`), 받는 쪽은 `Input[Artifact]`로 읽는다. 아티팩트 출력은 `task.outputs["이름"]`, 단순 반환값은 `task.output`으로 꺼낸다.

### 7.3.1 객체 탐지 (MIDV-500)

| 컴포넌트 | 하는 일 |
|---|---|
| `download_dataset` | tar.gz를 받아 풀고 `Output[Dataset]`으로 내보냄 |
| `split_dataset` | `train_test_split` 두 번(`random_state=42`)으로 75/15/10, YOLO 디렉터리 구조로 저장 |
| `output_file_contents` | 분할 결과 일부를 출력해 확인 |

멤버 실습은 전체 35분(다운로드 약 25분, split 약 9분)이었다. tar 10.28GB, 이미지·라벨 쌍 5,609개를 train 4,206 / val 841 / test 562로 나눴다. 원본 코드는 이미지와 라벨을 따로 glob해서 정렬 없이 분할에 넘기는데, 멤버는 파일명 교집합 + 정렬로 바꿨다. 이 차이는 아래 실험에서 다시 나온다.

### 7.3.2 영화 추천 (MovieLens 25M)

```
download ─▶ unzip ─┬▶ csv_to_parquet(ratings) ─▶ split_dataset ─▶ put_to_minio ─▶ qa_data
                   └▶ csv_to_parquet(movies)  ─▶ put_to_minio
```

큰 표 데이터를 Parquet로 저장하고, QA 컴포넌트가 MinIO의 train 파일을 열어 컬럼 수와 행 수(≥ 0.75 × 25M)를 assert한다. 실습 결과는 train 18,750,071 / val 3,750,014 / test 2,500,010으로 QA 통과, 전체 11분이었다.

같은 코드를 컴포넌트별 경량 이미지(A)와 패키지를 합친 단일 이미지(B)로 돌렸더니 parquet 산출물이 달랐다. A는 fastparquet로 인덱스 열 `index`가 맨 앞, B는 pandas `engine="auto"`가 pyarrow를 골라 `__index_level_0__`가 맨 뒤에 붙었다. 이미지에 어떤 패키지가 같이 들어가느냐에 따라 같은 코드의 출력 바이트가 바뀐다. B의 split이 344초로 느렸던 것은 `compression_level=1`을 주자 32초가 됐다.

## CH8 요약 — YOLO 훈련과 검증 컴포넌트

모델은 YOLOv8n이다. 연산 자원을 아끼고 큰 모델에 투자하기 전에 가능성을 빨리 확인하려는 선택이다. `train.py`는 `data.yaml`을 쓰고(nc 1, `id_card`) `YOLO('yolov8n.pt').train(data, imgsz=640, epochs, batch, project, name)`을 부른 뒤 `weights/best.pt`를 출력으로 복사한다.

`@dsl.component`는 pip 패키지만 설치해서 `libGL.so.1`용 apt 패키지를 넣을 수 없다. 그래서 훈련 컴포넌트는 `@dsl.container_component` + `dsl.ContainerSpec`으로 apt/pip 설치, heredoc으로 `/train.py` 작성, 실행을 셸 명령 하나에 넣는다. KFP 입력은 bash 위치 인자(`$0`~`$7`)로 넘어간다.

```python
@dsl.container_component
def train_model(epochs: int, batch: int, yolo_model_name: str, train_dataset: Input[Dataset],
                validation_dataset: Input[Dataset], test_dataset: Input[Dataset],
                model_output: Output[Model], data_yaml: Output[Artifact]):
    return dsl.ContainerSpec(image='python:3.11-slim', command=['bash', '-c'],
        args=[f'''apt-get ... && pip install ... && cat << 'EOF' > /train.py
{TRAINING_SCRIPT}
EOF
python3 /train.py --train-path "$0" ... --data-yaml "$7"''', train_dataset.path, ...])
```

검증 컴포넌트는 `ultralytics/ultralytics:8.0.194-cpu` 이미지에서 `model.val(...)`을 돌려 mAP50-95, mAP50, mAP75를 `Output[Metrics]`에 기록한다. 8.1.7은 기본 `yolov8n.pt`와 학습한 `best.pt`의 추론을 같은 이미지로 비교한다. 기본 모델은 신분증을 book, 얼굴을 person으로 잡는다.

멤버 환경(CPU 16코어, GPU 없음) 결과:

| run | 구성 | 시간 | 지표 |
|---|---|---|---|
| 전체 1에포크 | download → split → train → validate | 53분 33초 | val mAP50-95 0.8317, test 0.8097 |
| importer + 3에포크 + 게이트 | 분할 산출물 재사용 | 79분 47초 | test mAP50-95 0.9853 |

반복 실습에서는 분할 산출물을 MinIO 고정 경로에 두고 `dsl.importer`로 가져와 download·split 약 23분을 뺐다. 책 마지막 단락의 "프로그램으로 확인하라"는 `validate_model_gate`로 구현했다. mAP50 ≥ 0.90, mAP50-95 ≥ 0.60, precision ≥ 0.90 등 조건 하나라도 어기면 태스크가 실패한다.

## CH9.1 요약 — PVC로 데이터 공유

download에서 받은 데이터를 train과 validate가 MinIO에서 다시 받는 대신, `kfp-kubernetes`로 PVC를 만들어 각 태스크에 `/data`로 마운트한다.

```python
from kfp import kubernetes
pvc = kubernetes.CreatePVC(pvc_name='yolo-pipeline-pvc', access_modes=['ReadWriteOnce'],
                           size='1Gi', storage_class_name='local-path')
download_op = download_dataset()
kubernetes.mount_pvc(download_op, pvc_name=pvc_name, mount_path='/data')
split_op = split_dataset(random_state=random_state).after(download_op)
kubernetes.mount_pvc(split_op, pvc_name=pvc_name, mount_path='/data')
```

아티팩트 의존성이 없어지므로 실행 순서는 `.after()`로 직접 정한다. 멤버의 비교로는 importer/아티팩트 방식은 MinIO 경로로 계보가 남지만 단계마다 업로드·다운로드가 들고, PVC 재사용은 복사가 없는 대신 데이터 버전이 파이프라인 밖에 있다.

## 실습 환경

| 항목 | 버전·구성 | 비고 |
|---|---|---|
| 클러스터 | kind | CPU 16코어, 메모리 14GiB, GPU 없음 |
| KFP | 백엔드 2.4.1, SDK 2.9.0 | `kfp-kubernetes` 1.5.0 |
| ultralytics | 학습 8.4.207 (`torch 2.15.0+cpu`), 검증 이미지 8.0.194 / 8.3.56 | |
| 객체 스토리지 | SeaweedFS 3.85 | `-volume.max` 100 → 1000 |
| MLflow | 서버 3.0.0 | 9.3은 5주차 |
| StorageClass | `standard` (`rancher.io/local-path`) | 책은 `local-path` |

실습하면서 걸린 것:

- SeaweedFS 기본 인자로는 저장 한도가 약 6.4GB라 10GB 데이터셋 업로드가 500으로 실패했다
- 책의 `libgl1-mesa-glx`는 Debian 13에 없어서 `libgl1`로 바꿨다
- `DeletePVC`는 완료된 파드가 참조하는 동안 PVC를 `Terminating`에 묶어 둔다
- 경량 컴포넌트는 함수 본문만 직렬화해서, 기본 인자가 모듈 상수를 참조하면 `NameError`가 난다

## 같은 입력으로 다시 훈련하면 같은 모델이 나오나

### 설정

- MacBook M3 Pro 36GB, python 3.12.11, torch 2.15.0, ultralytics 8.4.207(노션 학습 버전), mlflow 3.0.0
- 데이터: MIDV-500 문서 1종(`01_alb_id`)을 촬영 조건 단위로 나눠 train 120장 / val 30장, 긴 변 640px
- 설정: `yolov8n.pt`, imgsz 640, batch 8, 3에포크
- 책과 멤버 코드는 `seed`, `deterministic` 인자를 주지 않는다. ultralytics 기본값이 `seed: 0`, `deterministic: True`(`ultralytics/cfg/default.yaml:29-30`)라서 이 값이 곧 "책 코드 그대로" 조건이다

ultralytics가 시드를 다루는 방식은 이렇다(8.4.207 소스).

| 위치 | 내용 |
|---|---|
| `engine/trainer.py:159` | `init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)` |
| `utils/torch_utils.py:791` | `torch.manual_seed(seed)`. MPS 장치도 같이 시드된다 |
| `utils/torch_utils.py:795` | `torch.use_deterministic_algorithms(True, warn_only=True)`. 결정적 구현이 없는 연산은 **경고만** 낸다 |
| `utils/torch_utils.py:796-800` | cuDNN 플래그, `CUBLAS_WORKSPACE_CONFIG`. CUDA 전용이고 MPS에 해당하는 설정은 없다 |
| `utils/checks.py:1036-1037` | CPU·MPS에서는 AMP를 끈다(fp32 훈련) |
| `engine/trainer.py:179-180` | CPU·MPS에서는 DataLoader workers를 0으로 강제 |

### 결과

| 비교 | 가중치(fp16 텐서 해시) | fp32 EMA 최대 절대차 | 마지막 mAP50-95 |
|---|---|---|---|
| CPU 같은 설정 5회 (`deterministic=False` 1회 포함) | 5/5 일치 | 0 | 0.90307 × 5 |
| MPS 같은 설정 9회 (`deterministic=False` 1회 포함) | 6/9 일치 | 0 / 1.38e-5 / 3.57e-3 / 4.95e-2 | 0.8785 × 7, 0.87526, 0.87402 |
| CPU vs MPS | 다름 | 0.0215 (텐서 355개 중 289개) | 0.90307 vs 0.8785 |
| CPU seed 0 vs 1 | 다름 | 1.74 | 0.90307 vs 0.89709 |

- **CPU는 비트 단위로 재현된다.** 5번 모두 텐서 355개가 같았다
- **MPS는 같은 설정으로 9번 중 3번 다른 모델이 나왔다.** 가장 크게 벗어난 run은 mAP50-95가 0.8785에서 0.8740으로 내려갔다. MPS backward에서 `scatter_reduce_mps`, `index_put_with_accumulate_mps`에 결정적 구현이 없다는 경고가 run마다 나왔고, `warn_only=True`라 훈련은 그대로 진행된다
- `deterministic` 플래그를 켜든 끄든 CPU·MPS 어느 쪽에서도 결과가 바뀌지 않았다
- **CPU와 MPS는 같은 시드로도 다른 모델을 만든다.** mAP50-95 차이 0.025는 seed를 바꿨을 때(0.006)보다 크다
- **체크포인트 파일 해시는 매번 다르다.** `date`와 `train_args`(name, save_dir)가 파일에 들어가기 때문이다(`utils/torch_utils.py`의 `strip_optimizer`). 같은 모델인지 보려면 파일이 아니라 텐서를 비교해야 한다
- 1.38e-5 차이는 `results.csv`에 보이지 않았다. 지표를 소수 5자리까지만 적어서 에포크별 값이 똑같이 찍힌다

### MLflow 기록만으로 구분되나

ultralytics의 MLflow 콜백으로 sqlite 스토어에 기록했다.

- MPS에서 가중치가 다른 두 run: 파라미터 115개 중 다른 것은 원래 run마다 다른 `name`, `save_dir`뿐이고 태그와 `mlflow.source.git.commit`도 같았다. 다른 것은 지표(mAP50-95 0.8785 vs 0.8741)뿐이다. 기록만 보면 "같은 입력, 다른 결과"이고 이유는 기록에 없다. 실행 장치(cpu/mps)가 파라미터로는 남지만 비결정성 경고는 어디에도 남지 않는다
- MLflow가 기록한 git commit은 레포 HEAD였는데, 실제로 실행한 스크립트는 아직 커밋 전이었다. MLflow는 working tree 변경 여부를 경고하지 않는다
- `pip install mlflow==3.0.0`을 그냥 하면 sqlalchemy 2.1.x가 같이 깔리고 sqlite 스토어가 ImportError로 실패한다. ultralytics는 `MLflow: Not tracking this run` 경고 한 줄만 남기고 훈련을 계속한다. 처음 9개 run은 그래서 MLflow에 없다. 스터디 노션이 `sqlalchemy<2.1`을 고정한 것도 같은 문제다

### 분할 단계: 같은 random_state, 다른 분할

7.3.1 원본 코드처럼 이미지와 라벨을 따로 glob하고 정렬 없이 `train_test_split(random_state=42)`에 넣어 봤다(MIDV 파일 300개, macOS APFS).

- glob 순서는 이름순도 생성순도 아니었다. 확장자가 다른 이미지 목록과 라벨 목록은 300개 중 300개 위치가 어긋났다
- 원본처럼 이미지와 라벨을 각각 분할해 복사하면 자기 라벨이 같은 split에 있는 이미지가 train 166/225, val 6/45, test 4/30뿐이다
- 입력 순서만 바꿔도 같은 `random_state=42`로 val은 45개 중 8개, test는 30개 중 2개만 겹쳤다

`random_state`는 입력 순서가 같을 때만 같은 분할을 보장한다. 멤버가 정렬을 넣은 수정은 재현성에 필요한 수정이었다. Linux ext4(멤버 환경)에서 원본 순서가 어떻게 되는지는 확인하지 못했다.

### 정리

- 책 코드 그대로면 CPU 훈련은 재현되고 MPS 훈련은 가끔 재현되지 않는다. 이 조건에서 "seed 0, deterministic True"라는 기록은 MPS에서 같은 모델을 보장하지 않는다
- 같은 모델인지는 체크포인트 파일 해시가 아니라 텐서 해시로 비교해야 한다. 파일에는 날짜가 들어간다
- MLflow 기록에는 장치와 파라미터만 남고 비결정성은 남지 않는다. 재현성을 리니지로 주장하려면 실행 장치를 같게 고정하거나, 같은 입력으로 두 번 돌려 텐서가 같은지 확인하는 단계를 파이프라인에 두는 편이 낫다
- 리니지가 깨지는 지점은 훈련보다 앞에도 있었다. 정렬하지 않은 glob, 이미지에 같이 깔린 패키지(parquet 엔진), 의존성 충돌로 꺼진 MLflow 추적은 모두 오류 없이 지나간다

### 도전과제

**(a) 기본 가중치 vs 훈련 에포크** (val 30장)

| 모델 | mAP50 | mAP50-95 |
|---|---|---|
| 기본 `yolov8n.pt` (클래스 무시) | 0.5121 | 0.4887 |
| 1에포크 단독 run | 0.0283 | 0.0064 |
| 3에포크 run의 1에포크 시점 | | 0.872 |
| 5에포크 | 0.9940 | 0.9429 |
| 10에포크 | 0.9950 | 0.9911 |
| 20에포크 | 0.9950 | 0.9950 |

1에포크 단독 run이 기본 가중치보다도 낮다. `engine/trainer.py:277-280`이 `warmup_epochs = min(warmup_epochs, epochs - 1)`이라 epochs=1이면 워밍업이 0이 되고, 첫 반복부터 lr 0.002로 시작한다(3에포크 run의 1에포크 lr은 0.00093). 멤버가 본 "단독 1에포크가 3에포크 run의 1에포크보다 낮다"도 같은 원인으로 보인다. 멤버 환경에서 직접 확인하지는 않았다.

검증 버전 차이(멤버의 0.809 vs 0.8317)도 확인했다. 같은 `best.pt`를 ultralytics 8.0.194, 8.3.56, 8.4.207로 검증하니 mAP50-95 차이는 최대 4e-5였다. 반면 같은 8.4.207에서도 훈련 중 val과 훈련 뒤 `best.pt`(fp16 저장) val은 0.90307 vs 0.89928로 0.0038 차이가 났다. 멤버의 차이는 버전보다 "훈련 중 val vs 별도 val" 쪽일 가능성이 크다.

**(b) CPU vs MPS 속도**

| 작업 | CPU | MPS |
|---|---|---|
| YOLO 1에포크 (train 120 + val 30장) | 12.5~14.0초 | 4.4~5.3초 (약 2.6배 빠름) |
| 작은 추천 모델 1에포크 (약 1,260 반복) | 2.29~2.51초 | 3.38~4.29초 (더 느림) |

추천 모델은 책 저장소의 `model_training.py` 구조(임베딩 20차원 × 2 → Linear 256 → Linear 1)를 MovieLens small로 옮긴 것이다. 원본은 seed를 아예 설정하지 않아서 CPU에서 두 번 돌리면 test RMSE가 1.2399와 1.2009로 갈렸고, seed 한 줄을 넣자 CPU에서 비트 단위로 같아졌다. 추천 모델 속도 측정 중에는 다른 프로세스가 CPU를 쓰고 있어서 수치가 정확하지 않다.

측정 코드와 결과는 [`lab/`](lab/)에 있다. `LAB=<작업 디렉터리> bash lab/run_all.sh`로 돌리면 되고, 각 단계는 따로 확인했지만 `run_all.sh` 한 번에 끝까지 돌려 보지는 않았다. YOLO 훈련은 합쳐서 약 12분 걸렸다.

## 아직 안 해본 것

- MPS 비결정성의 빈도(9회 중 3회)를 제대로 추정하려면 반복이 더 필요하다. 추천 모델 MPS는 2회만 돌렸다
- MIDV-500 전체(4,206장)와 GPU(CUDA)에서의 재현성은 측정하지 않았다. CUDA는 ultralytics가 cuDNN 플래그를 따로 설정하므로 결과가 다를 수 있다
- MPS 경고를 낸 연산이 ultralytics 코드 어느 줄에서 불리는지 특정하지 못했다
- 9.2 TensorBoard, 9.3 영화 추천 + MLflow는 5주차에 정리한다

---

*Study: Machine Learning Platform Engineering (CloudNet@)*
*CH7. 데이터 분석 및 준비 / CH8. 모델 훈련과 검증 1부 / CH9. 모델 훈련과 검증 2부*

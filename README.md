# 딥러닝 기반 얼굴 유형 분류를 활용한 소개팅 자동 매칭 시스템

[딥러닝] 팀 프로젝트 · 팀장 김주환 · 팀원 박정원, 김정유, 이동윤, 김채하

참가자의 얼굴 사진을 입력받아 ResNet18로 얼굴 유형(1차 후보: **강아지상 / 고양이상 / 토끼상 / 여우상**)별 확률을 출력하고,
그 확률과 사용자가 입력한 선호 유형·조건을 결합해 **Top-N 상대를 추천**합니다.

```
카메라(getUserMedia) → MediaPipe Face Landmarker 얼굴 검출 → Canvas로 얼굴 영역 추출·정렬
  → ResNet18 (ImageNet 사전학습 → Fine-tuning) → Softmax 유형별 확률 [강아지 0.55, 고양이 0.15, 토끼 0.25, 여우 0.05]
  → 선호 유형 벡터와 코사인 유사도 (+ 나이·취미) → 매칭 점수 → Top-N 추천
```

| 제안서 항목 | 구현 위치 |
|---|---|
| 3-1 얼굴 유형 분류 모델 (ResNet18 Transfer Learning / Fine-tuning, Softmax 확률) | `faceapp/models.py`, `faceapp/train.py` |
| 3-2 데이터 규모별 성능 비교 (Loss, Accuracy, Precision, Recall, F1, Confusion Matrix) | `scripts/data_scaling.py`, `faceapp/evaluate.py` |
| 3-3 얼굴 입력 및 전처리 (getUserMedia, Video, Canvas, MediaPipe Face Landmarker) | `faceapp/webapp.py`, `faceapp/web/` |
| 3-4 매칭 시스템 (확률 벡터 × 선호 벡터 유사도, 나이·취미, Top-N) | `faceapp/matching.py` |
| 4-1 FairFace East Asian subset + 자체 라벨링 | `scripts/sample_fairface_candidates.py`, `scripts/label_tool.py`, `scripts/merge_labels.py` |
| 4-2 Train/Val/Test 분리, 동일·유사 이미지 중복 확인 | `faceapp/dedup.py`, `faceapp/data.py` |
| 2) 라벨링 파일럿으로 최종 클래스·기준 확정 | `configs/classes.yaml`, `merge_labels`의 클래스별 일치도 |

설계 상세와 미정 사항은 [CLAUDE.md](CLAUDE.md)를 참고하세요.

---

## 1. 설치 (Windows / PowerShell)

```powershell
# 1) 가상환경
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# 실행 정책 오류 시: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip

# 2) PyTorch CUDA 빌드 먼저 설치 (GTX 1650 Ti)
#    https://pytorch.org/get-started/locally/ 에서 Stable / Windows / Pip / CUDA 를 골라 나온 명령을 쓰세요. 예:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3) 나머지 의존성
pip install -r requirements.txt

# 4) GPU 확인 → True, NVIDIA GeForce GTX 1650 Ti 가 나오면 성공
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 5) 웹캠 데모용 MediaPipe 받기 (처음 한 번, 약 40MB, 버전·해시 고정)
python -m scripts.fetch_mediapipe
```

- `nvidia-smi` 우측 상단의 `CUDA Version`이 드라이버가 지원하는 최대 버전입니다. 그 이하의 CUDA 빌드를 고르세요.
- `torch.cuda.is_available()`가 False면 CPU 빌드가 깔린 것입니다: `pip uninstall torch torchvision` 후 2)를 다시 실행.
- `pip install -r requirements.txt`를 먼저 하면 CPU 빌드가 깔릴 수 있으니 **순서를 지켜주세요**.

## 2. 빠른 확인 (데이터 없이)

```powershell
python -m pytest -q                        # 단위 테스트
python -m scripts.smoke_test               # 합성 데이터로 전체 파이프라인 1회 (CPU 약 15초)
python -m scripts.smoke_test --pretrained  # ImageNet 사전학습 가중치 다운로드까지 확인 (처음 한 번)
python -m scripts.data_scaling --smoke     # 데이터 규모별 비교표·그래프 생성 확인
python -m scripts.compare_models --smoke   # ResNet18 vs SimpleCNN 비교표 생성 확인
```

스모크 테스트는 합성 도형 이미지와 가상 라벨러 5명으로 **다수결 병합 → 학습 → 확률 보정 → 평가 → 예측 → Grad-CAM → Top-N 매칭 → HTML 리포트**를 돌립니다.
결과는 `outputs/smoke/run/`에 저장되고, `report.html`을 브라우저로 열면 볼 수 있습니다. 합성 데이터라 수치에는 의미가 없습니다.

웹캠 데모도 스모크 테스트 모델로 바로 띄워 볼 수 있습니다 (분류 결과는 의미 없음, 화면·흐름 확인용):
```powershell
python -m faceapp.webapp --set output_dir=outputs/smoke/run `
    --seed-participants outputs/smoke/participants.json --seed-predictions outputs/smoke/run/predictions.csv
```

## 3. 실제 데이터 워크플로

### 3-1. FairFace 받기 (각자 PC, git에 올리지 않음)
[FairFace GitHub](https://github.com/joojs/fairface)의 링크에서 아래 파일을 받아 `data/fairface/`에 둡니다.

| 받을 파일 | 놓을 위치 |
|---|---|
| Padding=0.25 이미지 `fairface-img-margin025-trainval.zip` | 압축을 풀어 `data/fairface/fairface-img-margin025-trainval/train/1.jpg …`, `…/val/1.jpg …` |
| 라벨 `fairface_label_train.csv`, `fairface_label_val.csv` | `data/fairface/` |

- `configs/default.yaml`의 기본 경로와 같으므로 설정을 바꿀 필요가 없습니다.
- `data/`는 `.gitignore`로 제외되어 있습니다. 팀 내 공유는 git이 아닌 비공개 드라이브로 합니다.

### 3-2. 라벨링 후보 샘플링 (한 명이 실행해서 배포)
```powershell
python -m scripts.sample_fairface_candidates --fairface-root data/fairface --n 1500 `
    --annotators 김주환 박정원 김정유 이동윤 김채하 --copy-to data/labeling_images
```
- `race == "East Asian"`만 골라 1,500장을 뽑고, 그 안의 동일·유사 이미지는 하나만 남깁니다.
- `data/labels/candidates.csv`와 팀원별 시트 `data/labels/annotators/<이름>.csv`를 만듭니다.
- `--copy-to` 폴더(약 1,500장)만 나눠 주면 각자 FairFace 전체를 받을 필요가 없습니다.
- 나이대·성별 필터: `--ages 20-29 30-39`, `--genders Female`, `--balance-gender`. 어떤 필터를 쓸지는 아직 미정입니다.
- 같은 `--seed`(기본 42)면 누가 실행해도 같은 결과가 나옵니다.

### 3-3. 라벨링 파일럿 → 클래스·기준 확정
제안서에 따라, 먼저 소량(예: 100~200장)으로 파일럿 라벨링을 해서 최종 클래스와 기준을 정합니다.

```powershell
python -m scripts.label_tool --sheet data/labels/annotators/김주환.csv --images data/labeling_images
```
- 브라우저에서 숫자키로 라벨을 답니다: `1` 강아지상 · `2` 고양이상 · `3` 토끼상 · `4` 여우상 · `0` 모름(기권). 누르면 다음 사진으로 넘어갑니다.
- 이동 `←`/`→` · 다음 미라벨 `N` · 지우기 `Backspace`. 누를 때마다 CSV에 바로 저장되고, 다시 켜면 이어서 합니다.
- 다른 사람의 라벨을 보지 않고 **독립적으로** 라벨링합니다. 기준 문서가 있으면 `--guide 기준.txt`로 화면에 띄울 수 있습니다.

모은 시트를 병합하면 파일럿 분석이 함께 나옵니다.
```powershell
python -m scripts.merge_labels --input-dir data/labels/annotators --out-dir data/labels
```
- `final_labels.csv`: 다수결 최종 라벨(`filename, identity, label` + 투표 분포 `votes_*`)
- `review.csv`: 최다 득표 **동률**이거나 유효 투표가 `--min-votes`(기본 3) 미만인 항목
- `agreement.json`과 터미널 출력:
  - 전체·**클래스별 Fleiss' kappa**: 낮은 클래스는 기준이 모호하다는 뜻
  - **가장 많이 엇갈린 클래스 쌍**: 예) 강아지상↔토끼상
  - **라벨러별 다수결 일치율**: 기준을 다르게 이해한 사람이 있는지 확인

파일럿 결과로 클래스를 바꾸려면 `configs/classes.yaml`만 수정하면 됩니다. 기존 모델은 클래스가 다르면 로드할 때 에러가 나므로 다시 학습하세요.

### 3-4. 학습 / 평가
```powershell
python -m faceapp.train      # configs/default.yaml 사용
python -m faceapp.evaluate   # test: Accuracy, Macro Precision/Recall/F1, Confusion Matrix, ECE
```
- **중복 확인**: 학습 전에 모든 이미지의 dHash를 비교해 동일·매우 유사한 이미지(크기 변경, 좌우 반전 포함)를 한 그룹으로 묶고, 그룹 단위로 분리합니다. 그룹이 여러 분할에 걸치면 `assert`로 실패합니다. 찾은 쌍은 `duplicates.csv`에 저장됩니다.
- 결과(`outputs/run1/`): `best.pt`, `history.csv`·`learning_curve.png`(Training/Validation Loss), `metrics_test.json`, `confusion_matrix_test.{csv,png}`, `splits.csv`.
- OOM(CUDA out of memory) 시: `--set train.batch_size=16 train.accum_steps=2`
- Windows에서 DataLoader 오류 시: `--set data.num_workers=0`

### 3-5. 데이터 규모별 성능 비교 (제안서 3-2)
```powershell
python -m scripts.data_scaling                                    # 학습 데이터 25/50/75/100%
python -m scripts.data_scaling --fractions 0.1 0.25 0.5 1.0 --seeds 1 2 3
```
- val/test는 고정하고, train만 **클래스 균형을 맞춰** 단계적으로 늘립니다. 작은 단계는 큰 단계에 포함됩니다.
- `--seeds`를 여러 개 주면 평균±표준편차를 냅니다. 데이터가 적을 때 권장합니다.
- 결과(`outputs/data_scaling/`):
  - `scaling_summary.md`: 보고서에 붙일 표
  - `scaling.png`: 데이터 수 vs Accuracy/Precision/Recall/F1
  - `loss_curves.png`: 단계별 Training/Validation Loss
  - 단계별 폴더의 Confusion Matrix

학습 방법 비교(Transfer vs 처음부터 학습 등)는 `compare_models`로 합니다.
```powershell
python -m scripts.compare_models   # 기본: resnet18_transfer vs cnn_baseline
python -m scripts.compare_models --runs "transfer:model.name=resnet18" "scratch:model.name=resnet18 model.pretrained=false" `
    "linear_probe:train.freeze_backbone_epochs=100"
```

### 3-6. 웹캠 데모 (제안서 3-3, 3-4)
```powershell
python -m faceapp.webapp     # → 브라우저에서 http://127.0.0.1:8000
```
1. **카메라 켜기**: `getUserMedia`로 카메라를 열고, MediaPipe Face Landmarker가 얼굴을 실시간으로 찾습니다. 촬영 상태(얼굴 크기, 정면 여부, 기울기)도 안내합니다.
2. **촬영 후 분석**: Canvas로 프레임을 캡처하고, 눈이 수평이 되도록 회전해 FairFace와 비슷한 구도의 224×224 얼굴 영역을 잘라 ResNet18로 보냅니다. 화면에 유형별 확률과 조명 상태가 표시됩니다. 카메라가 없으면 **사진 올리기**를 쓰면 됩니다.
3. **내 정보와 선호 조건**: 선호 유형 비율, 나이, 취미, 선호 성별을 입력하고 등록하면 **추천 상대 Top-N**이 나옵니다.

참고:
- **사진은 저장하지 않습니다.** 확률값과 입력한 프로필만 `data/webapp/participants.json`에 남고, "내 정보 삭제"로 지울 수 있습니다.
- 시연 전에 상대 풀을 채우려면 `--seed-participants <participants.json> --seed-predictions <predictions.csv>`를 붙입니다(3-7 참고).
- 카메라는 보안상 `localhost`(127.0.0.1)에서만 열립니다. 다른 기기에서 접속하려면 HTTPS가 필요합니다.
- `scripts.fetch_mediapipe`를 안 했다면 CDN에서 불러옵니다. 그것도 실패하면 얼굴 검출 없이 가운데 영역으로 분석합니다.

### 3-7. 사진 폴더 일괄 처리 (리포트)
```powershell
python -m faceapp.pipeline --participants data/participants.json --photos data/participants/photos --skip-train
```
참가자 사진 폴더를 한 번에 예측·매칭하고 `report.html`을 만듭니다. `participants.json` 형식은 `examples/participants.example.json`을 참고하세요(잠정).

## 4. 매칭 점수 (제안서 3-4)
방향 점수 `S(a→b) = Σ wᵢ·sᵢ / Σ wᵢ` (정보가 없는 항목은 빼고 가중치를 재정규화)
- `s_face = cos(b의 얼굴 유형 확률, a의 선호 유형 벡터)`. 예) A 얼굴 [0.55, 0.15, 0.25, 0.05] · B 선호 [0.70, 0.00, 0.30, 0.00]
- `s_age`: 상대 나이가 선호 범위 안이면 1, 벗어나면 `age_tolerance`년에 걸쳐 0까지 감소
- `s_hobby`: 취미 Jaccard 유사도
- 최종 점수 = `(S(a→b) + S(b→a)) / 2` (양방향). 선호 성별은 필터로 적용
- 가중치(`matching.weights`: 얼굴 0.6, 나이 0.25, 취미 0.15)는 임시값입니다.

## 5. 딥러닝 기법 (config 키로 on/off → 비교 실험)
| 기법 | 목적 | 키 |
|---|---|---|
| ImageNet 사전학습 ResNet18 + 분류층 교체 | 적은 데이터로 학습 (Transfer Learning) | `model.name`, `model.pretrained` |
| 처음 N epoch backbone 고정 → 전체 Fine-tuning, discriminative LR | 사전학습 특징 보존 + 얼굴 도메인 적응 | `train.freeze_backbone_epochs`, `train.backbone_lr_factor` |
| AdamW + warmup + cosine, early stopping | 안정적 수렴, 과적합 방지 | `train.*` |
| AMP, gradient accumulation | GTX 1650 Ti 4GB 대응 | `train.amp`, `train.accum_steps` |
| Augmentation, label smoothing, Mixup/CutMix | 과적합 방지 | `augment.*`, `train.*` |
| 라벨러 투표 분포 soft label | 주관적 라벨의 불확실성 반영 | `train.soft_labels` |
| Temperature scaling (ECE) | 매칭에 쓰는 확률 보정 | `calibration.enabled` |
| TTA, Grad-CAM | 추론 안정성, 판단 근거 시각화 | `predict.tta`, `faceapp.gradcam` |

## 6. 폴더 구조
```
configs/default.yaml, classes.yaml   설정, 클래스 정의(파일럿 후 확정)
faceapp/       config, data, dedup, models, losses, engine, train, calibrate, evaluate, predict, gradcam,
               matching, report, pipeline, webapp(+ web/: index.html, app.js, style.css)
scripts/       sample_fairface_candidates, label_tool, merge_labels, data_scaling, compare_models,
               fetch_mediapipe, smoke_test, compare_matching_time(선택)
examples/      participants.json, 시간 측정 CSV 예시
tests/         pytest
data/, outputs/, faceapp/web/vendor/   git 제외
```

## 참고문헌
1. Kärkkäinen, K. & Joo, J., "FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation," WACV 2021. https://arxiv.org/abs/1908.04913
2. He, K., Zhang, X., Ren, S., & Sun, J., "Deep Residual Learning for Image Recognition," CVPR 2016. https://arxiv.org/abs/1512.03385
3. PyTorch Transfer Learning Tutorial. https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
4. MediaPipe Face Landmarker. https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
5. Guo et al., "On Calibration of Modern Neural Networks," ICML 2017 / Selvaraju et al., "Grad-CAM," ICCV 2017

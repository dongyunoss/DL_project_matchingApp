# 딥러닝 얼굴 유형 분류 기반 대학생 소개팅 자동 매칭 (프로토타입)

참가자 사진을 **강아지상 / 고양이상 / 토끼상 / 여우상**으로 분류하고(ResNet18 transfer learning),
보정된 유형별 확률을 선호 조건(선호 유형, 나이, 취미)과 결합해 **Top-N 상대를 추천**합니다.

```
사진 → 전처리/증강 → ResNet18 (ImageNet 사전학습 → fine-tune) → Softmax(logits/T) 확률 + 임베딩
     → 코사인 유사도 + 나이/취미 가중합 (양방향) → Top-N
```

설계 상세, 데이터 규칙, 미정 사항(TODO)은 [CLAUDE.md](CLAUDE.md)를 참고하세요.

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
```

- `nvidia-smi` 우측 상단의 `CUDA Version`이 드라이버가 지원하는 최대 버전입니다. 그 이하의 CUDA 빌드를 고르세요.
- `torch.cuda.is_available()`가 False면 CPU 빌드가 깔린 것입니다: `pip uninstall torch torchvision` 후 2)를 다시 실행.
- `pip install -r requirements.txt`를 먼저 하면 CPU 빌드가 깔릴 수 있으니 **순서를 지켜주세요**.

## 2. 빠른 확인 (라벨 없이)

```powershell
python -m pytest -q                       # 단위 테스트 (매칭, 라벨 병합, split, loss/지표/보정, 리포트, 시간 비교)
python -m scripts.smoke_test              # 전체 파이프라인 1회 (CPU 약 15초)
python -m scripts.smoke_test --pretrained # ImageNet 가중치 다운로드까지 확인 (처음 한 번, 인터넷 필요)
python -m scripts.compare_models --smoke  # ResNet18 vs SimpleCNN 비교표 생성 동작 확인
```

스모크 테스트는 합성 도형 이미지와 가상 라벨러 5명을 만들어 **다수결 병합 → 학습(soft label, Mixup, freeze→finetune)
→ temperature scaling → 평가 → 참가자 예측(TTA) → Grad-CAM → Top-N 매칭 → HTML 리포트 → 시간 비교**까지 돌립니다.
결과는 `outputs/smoke/run/`에 저장되며, **`report.html`을 브라우저로 열면 매칭 결과를 볼 수 있습니다.**
합성 데이터라 성능 수치에는 의미가 없습니다.
`--model simple_cnn`(베이스라인), `--image-size 224`(실제 해상도)도 확인할 수 있습니다.

## 3. 실제 데이터 워크플로

### 3-1. CelebA 받기 (각자 PC, git에 올리지 않음)
[CelebA 공식 페이지](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)의 Google Drive 링크에서 아래 파일을 받습니다.

| 받을 파일 (Drive 안 위치) | 놓을 위치 |
|---|---|
| `Img/img_align_celeba.zip` (약 1.4GB, 정렬된 얼굴 178×218) | 압축을 풀어 `data/celeba/img_align_celeba/000001.jpg …` |
| `Anno/identity_CelebA.txt` (identity 분리에 **필수**) | `data/celeba/identity_CelebA.txt` |
| `Anno/list_attr_celeba.txt` (속성 필터용, 선택) | `data/celeba/list_attr_celeba.txt` |

- Kaggle 등의 미러판에는 `identity_CelebA.txt`가 빠져 있는 경우가 많으니 공식 링크에서 받으세요.
- 비상업적 연구용 데이터입니다. `data/`는 `.gitignore`로 제외되어 있고, 팀 내 공유는 git이 아닌 비공개 드라이브로 합니다.
- `configs/default.yaml`의 기본 경로(`data/celeba/img_align_celeba`)와 같으므로 설정을 바꿀 필요가 없습니다.

### 3-2. 라벨링 후보 샘플링 (한 명만 실행해서 시트 배포)
```powershell
python -m scripts.sample_celeba_candidates --celeba-root data/celeba --n 1500 `
    --annotators member1 member2 member3 member4 member5 --copy-to data/labeling_images
```
- identity당 1장씩 1,500명을 뽑아 `data/labels/candidates.csv`와 팀원별 시트 `data/labels/annotators/<이름>.csv`를 만듭니다.
- `--copy-to` 폴더(`data/labeling_images`, 약 1,500장)만 팀원에게 나눠 주면 각자 CelebA 전체를 받을 필요가 없습니다.
- `--exclude-attr Eyeglasses Blurry`, `--require-attr Young` 등 CelebA 속성 필터를 쓸 수 있습니다. 어떤 필터를 쓸지는 아직 미정입니다.
- 같은 `--seed`(기본 42)로 실행하면 누가 돌려도 같은 결과가 나옵니다.

### 3-2b. 라벨링 (각자 독립적으로)
```powershell
python -m scripts.label_tool --sheet data/labels/annotators/member1.csv --images data/labeling_images
```
- 브라우저가 열리면 숫자키로 라벨을 답니다: `1` 강아지상 · `2` 고양이상 · `3` 토끼상 · `4` 여우상 · `0` 모름(기권). 누르면 다음 사진으로 넘어갑니다.
- 이동은 `←`/`→`, 다음 미라벨로 건너뛰기는 `N`, 라벨 지우기는 `Backspace`입니다.
- 누를 때마다 시트 CSV에 바로 저장됩니다. 언제 꺼도 되고, 다시 켜면 이어서 합니다. 종료는 터미널에서 `Ctrl+C`.
- 다른 사람의 라벨을 보지 않고 **독립적으로** 라벨링해야 다수결이 의미 있습니다.
- 팀 라벨링 기준이 정해지면 텍스트 파일로 만들어 `--guide 기준.txt`로 화면에 띄울 수 있습니다 (기준은 아직 미정).
- 끝나면 각자의 `<이름>.csv`를 한 사람에게 모아 `data/labels/annotators/`에 넣습니다. CSV에는 사진이 없어 메신저로 공유해도 됩니다.

### 3-3. 다수결 병합
```powershell
python -m scripts.merge_labels --input-dir data/labels/annotators --out-dir data/labels
```
- `final_labels.csv`: `filename, identity, label` 형식이며, 투표 분포 `votes_*`도 함께 저장됩니다 (soft label 학습에 사용).
- `review.csv`: 최다 득표 **동률**이거나 유효 투표가 `--min-votes`(기본 3) 미만인 항목 → 재검토 목록.
- `agreement.json`: Fleiss' kappa(라벨러 간 일치도)와 라벨 분포. 보고서에 쓸 수 있습니다.

### 3-4. 학습 / 평가
```powershell
python -m faceapp.train                                   # configs/default.yaml 사용
python -m faceapp.evaluate                                # test split: Accuracy, Macro F1, Confusion Matrix, ECE
```
- identity 기준으로 split하고, 같은 인물이 두 split에 들어가면 `assert`로 실패합니다. 사용한 split은 `outputs/run1/splits.csv`에 저장됩니다.
- 학습이 끝나면 val set으로 temperature scaling을 수행하고, 그 T를 체크포인트에 저장합니다.
- 결과물(`outputs/run1/`): `best.pt`, `history.csv`, `learning_curve.png`, `metrics_test.json`,
  `confusion_matrix_test.{csv,png}`, `predictions_test.csv`.

설정 덮어쓰기 예시:
```powershell
# OOM (CUDA out of memory) 시: batch 16 + gradient accumulation 2 = 유효 batch 32
python -m faceapp.train --set train.batch_size=16 train.accum_steps=2

# CNN 베이스라인 (처음부터 학습) — ResNet18 transfer learning과 비교
python -m faceapp.train  --set model.name=simple_cnn output_dir=outputs/cnn_baseline
python -m faceapp.evaluate --set model.name=simple_cnn output_dir=outputs/cnn_baseline

# 실험 예: 투표 분포 soft label + Mixup
python -m faceapp.train --set train.soft_labels=true train.mixup_alpha=0.2 output_dir=outputs/soft_mixup
```
Windows에서 DataLoader 관련 오류가 나면 `--set data.num_workers=0`을 붙이세요.

### 3-5. 한 번에 실행: 학습 → 평가 → 예측 → Grad-CAM → 매칭 → 리포트
```powershell
python -m faceapp.pipeline --participants data/participants.json --photos data/participants/photos
python -m faceapp.pipeline --participants data/participants.json --photos data/participants/photos --skip-train  # 기존 모델로 매칭만
```
결과(`outputs/run1/`):
- `report.html`: 참가자별 얼굴 유형 확률과 Top-N 상대, 세부 점수 (발표·시연용)
- `matches.json`: 매칭 결과
- `gradcam_participants.png`: 모델이 얼굴의 어느 부분을 보고 분류했는지
- `pipeline_summary.json`: 테스트 지표와 단계별 소요 시간, AI 매칭(예측+매칭) 시간

단계별로 따로 실행할 수도 있습니다.
```powershell
python -m faceapp.predict  --images data/participants/photos           # → predictions.csv (+ embeddings.npy)
python -m faceapp.gradcam  --images data/participants/photos --max 8   # → gradcam.png
python -m faceapp.matching --participants data/participants.json --top-n 5   # → matches.json
```
`participants.json` 형식 (**잠정 — 입력 형식은 아직 미정**, 예시: `examples/participants.example.json`):
```json
[
  {"id": "p01", "gender": "F", "age": 22, "photo": "p01.jpg", "hobbies": ["영화", "러닝"],
   "preference": {"face_type": {"dog": 0.7, "cat": 0.3}, "age_min": 21, "age_max": 26, "gender": "M"}}
]
```

### 3-6. 모델 비교 실험 (ResNet18 transfer vs CNN 베이스라인 등)
```powershell
python -m scripts.compare_models          # 기본: resnet18_transfer vs cnn_baseline
python -m scripts.compare_models --runs "resnet18_transfer:model.name=resnet18" `
    "resnet18_scratch:model.name=resnet18 model.pretrained=false" "cnn_baseline:model.name=simple_cnn"
python -m scripts.compare_models --runs "hard:train.soft_labels=false" "soft:train.soft_labels=true" `
    --set train.epochs=15
```
모든 run이 **같은 split**으로 학습·평가됩니다 (assert로 확인).
`outputs/compare/comparison.md`에 Accuracy, Macro F1, ECE, 파라미터 수, 학습 시간 표가 생성되며, 보고서에 바로 붙일 수 있습니다.

### 3-7. 수작업 vs AI 매칭 시간 비교
1. `examples/matching_time.example.csv`를 `data/timing/matching_time.csv`로 복사합니다.
2. 같은 참가자 그룹에 대해 수작업 매칭 시간(`manual`)과 AI 추천 검토 시간(`ai_assisted`)을 분 단위로 기록합니다.
3. 아래 명령으로 비교합니다.
```powershell
python -m scripts.compare_matching_time --timing data/timing/matching_time.csv `
    --pipeline-summary outputs/run1/pipeline_summary.json
```
- 모델 계산 시간(참가자당 초)은 `ai_assisted` 시간에 자동으로 더해집니다.
- 결과로 참가자당 소요 시간과 절감률이 `time_comparison.json`에 저장됩니다.
- 측정 프로토콜(누가, 몇 명, 어떤 조건으로 측정할지)은 아직 미정입니다.

## 4. 매칭 점수
방향 점수 `S(a→b) = Σ wᵢ·sᵢ / Σ wᵢ` (정보가 없는 항목은 빼고 가중치를 재정규화)
- `s_face = cos(b의 보정된 유형 확률, a의 선호 유형 벡터)`
- `s_age` = b의 나이가 a의 선호 범위 안이면 1, 범위 밖이면 `age_tolerance`년에 걸쳐 0까지 선형 감소
- `s_hobby` = 취미 Jaccard 유사도
- 양방향: `(S(a→b) + S(b→a)) / 2`. 선호 성별이 지정되면 하드 필터로 적용합니다.
- 가중치(`matching.weights`)는 임시값입니다 (TODO).

## 5. 딥러닝 기법 요약 (config 키로 on/off → 실험 비교)
| 기법 | 목적 | 키 |
|---|---|---|
| ImageNet 사전학습 ResNet18 transfer learning | 적은 데이터(~1.5k)로 학습 | `model.name`, `model.pretrained` |
| Backbone freeze → 전체 fine-tune, discriminative LR | 사전학습 특징 보존 + 도메인 적응 | `train.freeze_backbone_epochs`, `train.backbone_lr_factor` |
| AdamW + warmup + cosine | 안정적 수렴 | `train.lr`, `train.warmup_epochs` |
| AMP, gradient accumulation | 4GB VRAM 대응 | `train.amp`, `train.accum_steps` |
| Augmentation, label smoothing, Mixup/CutMix, dropout | 과적합 방지 | `augment.*`, `train.*` |
| 라벨러 투표 분포 soft label | 주관적 라벨의 불확실성 반영 | `train.soft_labels` |
| Class-weighted loss | 클래스 불균형 | `train.class_weighted_loss` |
| Temperature scaling, ECE | 매칭에 쓰는 확률 보정 | `calibration.enabled` |
| TTA (좌우 반전) | 추론 안정성 | `predict.tta` |
| Grad-CAM | 판단 근거 시각화 (배경/안경 의존 점검) | `faceapp.gradcam` |
| SimpleCNN 베이스라인 | transfer learning 효과 비교 | `model.name=simple_cnn` |

## 6. 폴더 구조
```
configs/default.yaml   설정 (경로, 하이퍼파라미터, 매칭 가중치)
faceapp/               config, data, models, losses, engine, train, calibrate, evaluate, predict, gradcam,
                       matching, report(HTML), pipeline(end-to-end)
scripts/               sample_celeba_candidates, merge_labels, smoke_test, compare_models, compare_matching_time
examples/              participants.json, 시간 측정 CSV 예시 (사진 없음)
tests/                 pytest
data/, outputs/        git 제외
```

## 참고문헌
- He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
- Liu et al., "Deep Learning Face Attributes in the Wild," ICCV 2015
- Guo et al., "On Calibration of Modern Neural Networks," ICML 2017
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017
- Zhang et al., "mixup," ICLR 2018 / Yun et al., "CutMix," ICCV 2019
- PyTorch Transfer Learning Tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

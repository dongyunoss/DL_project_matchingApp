# 딥러닝 얼굴 유형 분류 기반 대학생 소개팅 자동 매칭 (1주차 프로토타입)

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
python -m pytest -q                 # 단위 테스트 (매칭, 라벨 병합, split, loss/지표/보정)
python -m scripts.smoke_test        # 전체 파이프라인 1회: 합성 데이터 → 학습 → 평가 → 예측 → Grad-CAM → Top-N
```

스모크 테스트는 합성 도형 이미지와 가상 라벨러 5명을 만들어 **다수결 병합 → 학습(soft label, Mixup, freeze→finetune)
→ temperature scaling → 평가 → 참가자 예측(TTA) → Grad-CAM → 매칭**까지 수십 초 안에 돌립니다.
결과는 `outputs/smoke/run/`에 저장됩니다. 합성 데이터라 성능 수치에는 의미가 없습니다.
`--model simple_cnn`으로 베이스라인 모델도, `--pretrained`로 ImageNet 가중치 다운로드도 확인할 수 있습니다.

## 3. 실제 데이터 워크플로

### 3-1. CelebA 준비 (git에 올리지 않음)
[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)에서 `img_align_celeba/`, `identity_CelebA.txt`,
(선택) `list_attr_celeba.txt`를 받아 `data/celeba/`에 둡니다. `data/`는 `.gitignore`로 제외돼 있습니다.

### 3-2. 라벨링 후보 샘플링 (identity당 1장)
```powershell
python -m scripts.sample_celeba_candidates --celeba-root data/celeba --n 1500 `
    --annotators member1 member2 member3 member4 member5 --copy-to data/labeling_images
```
- `data/labels/candidates.csv`와 팀원별 시트 `data/labels/annotators/<이름>.csv`가 생성됩니다.
- 각자 **독립적으로** `label` 열에 `dog / cat / rabbit / fox`(또는 `강아지상` 등 한글)를 입력합니다. 모르겠으면 빈 칸(기권).
- `--exclude-attr Eyeglasses Blurry`, `--require-attr Young` 등 CelebA 속성 필터를 쓸 수 있습니다. 어떤 필터를 쓸지는 아직 미정입니다.

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

### 3-5. 예측 → Grad-CAM → 매칭
```powershell
python -m faceapp.predict  --images data/participants/photos           # → outputs/run1/predictions.csv (+ embeddings.npy)
python -m faceapp.gradcam  --images data/participants/photos --max 8   # → outputs/run1/gradcam.png
python -m faceapp.matching --participants data/participants.json --top-n 5   # → outputs/run1/matches.json
```
`participants.json` 형식 (**잠정 — 입력 형식은 아직 미정**):
```json
[
  {"id": "p01", "gender": "F", "age": 22, "photo": "p01.jpg", "hobbies": ["영화", "러닝"],
   "preference": {"face_type": {"dog": 0.7, "cat": 0.3}, "age_min": 21, "age_max": 26, "gender": "M"}}
]
```

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
faceapp/               config, data, models, losses, engine, train, calibrate, evaluate, predict, gradcam, matching
scripts/               sample_celeba_candidates, merge_labels, smoke_test
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

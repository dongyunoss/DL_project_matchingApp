# CLAUDE.md — 프로젝트 작업 가이드

[딥러닝] 팀 프로젝트 「딥러닝 기반 얼굴 유형 분류를 활용한 소개팅 자동 매칭 시스템」 (방식: 개발).
팀장 김주환 · 팀원 박정원, 김정유, 이동윤, 김채하. 이 문서는 사람과 AI 어시스턴트가 공유하는 설계/규칙 문서이며,
**제안서(주제제안서 최종본)가 기준**이다. 제안서와 다른 방향으로 바꾸려면 팀 합의가 먼저다.

## 제안서 요약
- 핵심 목표: 얼굴 유형 분류 모델 개발 및 성능 평가, **데이터 규모·학습 방법에 따른 성능 비교·분석**, 매칭 시스템 적용
- 클래스: 1차 후보 강아지상·고양이상·토끼상·여우상 → **라벨링 파일럿 결과로 최종 클래스와 기준 확정**
- 모델: PyTorch/TorchVision, ImageNet 사전학습 ResNet18 Transfer Learning/Fine-tuning, 분류층을 클래스 수에 맞게 교체, Softmax 확률
- 평가: Training/Validation Loss, Accuracy, Precision, Recall, F1-score, Confusion Matrix
- 입력: MediaDevices getUserMedia → HTMLVideoElement → Canvas 캡처 → MediaPipe Face Landmarker 얼굴 검출·영역 추출 → ResNet18
- 매칭: 얼굴 유형 확률 벡터 × 선호 유형 벡터 유사도 (+ 나이·취미) → Top-N
- 데이터: **FairFace East Asian subset** + 팀 자체 라벨링. Train/Val/Test 분리 시 동일·매우 유사한 이미지 중복 확인

## 파이프라인

```
[학습]  FairFace(East Asian) → 샘플링·중복 제거 → 팀원 5명 독립 라벨링 → 다수결 (+파일럿 분석)
        → (identity + 근접 중복) 그룹 단위 split → ResNet18 transfer/fine-tune → temperature scaling → 평가
[서비스] 카메라 → MediaPipe 얼굴 검출 → Canvas 크롭·정렬(224) → ResNet18 → 보정 Softmax 확률
        → 선호 조건 결합 → 양방향 매칭 점수 → Top-N 추천
```

## 폴더 구조

```
configs/default.yaml       모든 경로·하이퍼파라미터·매칭 가중치 (코드에 하드코딩 금지)
configs/classes.yaml       클래스 정의의 유일한 기준 (파일럿 후 여기만 수정, 체크포인트에 저장·검증)
faceapp/
  config.py                YAML 로드 + CLI override (--set train.epochs=5)
  data.py                  라벨 CSV 로드, 그룹 기준 split(+누수 assert), Dataset, transforms
  dedup.py                 dHash(+좌우반전)로 동일·유사 이미지 그룹화 (제안서 4-2)
  models.py                resnet18 / resnet34 / efficientnet_b0 / mobilenet_v3_small / simple_cnn
  losses.py                soft-label CE(+label smoothing), Mixup/CutMix
  train.py                 AMP, grad accumulation, discriminative LR, warmup+cosine, 단계적 unfreeze, early stopping
  calibrate.py             temperature scaling (val NLL 최소화), ECE
  evaluate.py              Accuracy, Macro Precision/Recall/F1, per-class P/R/F1, Confusion Matrix, ECE
  predict.py               이미지 → 확률 CSV + 임베딩 .npy (TTA: 좌우 반전)
  gradcam.py               Grad-CAM 시각화
  matching.py              코사인 유사도 + 가중치 결합 → 양방향 Top-N
  engine.py                공용 추론 루프
  report.py, pipeline.py   사진 폴더 일괄 예측·매칭 → HTML 리포트
  webapp.py + web/         웹캠 데모 서버(표준 라이브러리) + 프론트(getUserMedia, Canvas, MediaPipe)
  utils.py
scripts/
  sample_fairface_candidates.py  FairFace East Asian 샘플링(+중복 제거) → 팀원별 라벨링 시트
  label_tool.py                  브라우저 라벨링 도구 (숫자키 1~4/0, 즉시 저장)
  merge_labels.py                다수결(동률 → review.csv), 투표 분포, 파일럿 분석(클래스별 kappa, 엇갈린 쌍, 라벨러 일치율)
  data_scaling.py                데이터 규모별 성능 비교 (val/test 고정, 클래스 균형 nested subset, seed 반복)
  compare_models.py              학습 방법 비교 (transfer vs scratch vs linear probe 등, 같은 split)
  fetch_mediapipe.py             MediaPipe tasks-vision 0.10.35 + face_landmarker 모델 받기 (해시 고정)
  smoke_test.py                  합성 데이터로 전체 파이프라인 확인
  compare_matching_time.py       (선택) 수작업 vs AI 매칭 시간 비교 — 제안서 최종본에는 없는 항목
examples/                  participants.json / 시간 측정 CSV 예시
tests/                     pytest
data/, outputs/, faceapp/web/vendor/   git 제외 (.gitignore)
```

## 딥러닝 설계 포인트 (비교/실험할 축)

| 영역 | 기법 | config 키 / 스크립트 |
|---|---|---|
| Transfer learning | ImageNet 사전학습 ResNet18, 분류층 교체 | `model.name`, `model.pretrained` |
| Fine-tuning 전략 | 처음 N epoch backbone freeze → 전체 unfreeze / linear probe | `train.freeze_backbone_epochs` |
| | Discriminative LR (backbone lr = head lr × factor) | `train.backbone_lr_factor` |
| 데이터 규모 | 학습 데이터 비율별 성능 곡선 (제안서 3-2) | `scripts.data_scaling` |
| 학습 방법 비교 | transfer vs scratch vs SimpleCNN 등 | `scripts.compare_models` |
| 최적화 | AdamW + linear warmup + cosine decay, early stopping | `train.*` |
| 메모리(4GB VRAM) | AMP(fp16), gradient accumulation | `train.amp`, `train.accum_steps` |
| 정규화 | Augmentation, label smoothing, Mixup/CutMix, dropout, weight decay | `augment.*`, `train.*` |
| 라벨 노이즈 | 5명 투표 분포를 soft label로 학습 | `train.soft_labels` |
| 불균형 | class-weighted loss | `train.class_weighted_loss` |
| 확률 보정 | Temperature scaling — 매칭에 확률을 직접 쓰므로 | `calibration.enabled` |
| 추론/해석 | TTA, Grad-CAM | `predict.tta`, `faceapp.gradcam` |

### 확장 후보 (아직 구현 안 함)
- 라벨 없는 FairFace East Asian 이미지로 pseudo-labeling / self-training
- 임베딩 t-SNE/UMAP 시각화, 이상형 참고 사진 → 선호 벡터 (`matching.preference_from_reference_probs` 골격만 있음)
- 인물 단위 교차검증 (소규모 데이터에서 수치 신뢰도 향상)

## 데이터 규칙 (반드시 지킬 것)
- **원본 이미지·얼굴 사진은 절대 git에 커밋하지 않는다** (`data/`, `outputs/`는 .gitignore). 웹앱은 사진을 저장하지 않는다.
- FairFace에는 동물상 라벨이 없다 → 팀원 5명이 독립 라벨링 후 **다수결**. 동률이면 최종 라벨에서 제외하고 `review.csv`로 보낸다.
- 라벨 CSV 형식: `filename, identity, label`. filename은 `data.image_dir` 기준 상대 경로(FairFace: `train/123.jpg`).
  FairFace에는 인물 ID가 없으므로 identity = filename. 추가 열 `votes_<class>`는 선택(soft label용).
- Train/Val/Test는 **(identity + 동일·유사 이미지) 그룹 기준**으로 분리한다. 그룹이 두 split에 걸치면 `assert`로 실패해야 한다
  (`data.assert_no_identity_leak`). 유사 판정은 dHash 해밍 거리 `data.dedup_hamming`(기본 5).
- 목표 규모 약 1,200~1,500장. 데이터셋 경로는 항상 config(`data.image_dir`, `data.labels_csv`)로 받는다.

## 환경
- 팀원 환경: Windows + PowerShell, GTX 1650 Ti (4GB). 기본 batch 32, 224×224, AMP. OOM 시 `train.batch_size=16 train.accum_steps=2`.
- Python venv + requirements.txt. torch는 CUDA 빌드 별도 설치 (README 참고).
- 학습·추론: PyTorch / TorchVision / Python (+ numpy, pyyaml, pillow, matplotlib, pytest). 외부 추론 API 사용 금지.
- 웹 데모: 브라우저 API(MediaDevices, Canvas) + MediaPipe Tasks Vision(브라우저에서 로컬 실행, `scripts.fetch_mediapipe`로 받아 둠).

## 아직 미정 (임의로 확정하지 말 것 — 코드에는 `TODO(미정)`으로 표시)
- [ ] 최종 클래스와 동물상 라벨링 기준 → 라벨링 파일럿(`merge_labels`의 클래스별 kappa·엇갈린 쌍)으로 확정 (`configs/classes.yaml`)
- [ ] FairFace 필터 기준: 나이대(`--ages`), 성별 균형, 얼굴 각도·안경·조도 (FairFace 20장 테스트에서 문제 확인됨)
- [ ] "East Asian"에는 한국인 외 동아시아인도 포함됨 → 한계로 둘지, 추가 필터/데이터를 쓸지
- [ ] 웹캠 크롭 구도(`webapp.crop_scale`)를 FairFace 이미지와 맞추는 값 — 실제 사진으로 조정 필요
- [ ] 매칭 가중치(`matching.weights`)와 선호 조건 입력 형식(`participants.json` 스키마는 잠정)
- [ ] 데이터 규모 비교의 단계(비율)와 반복 seed 수

## 진행 기록
- 1주차(~11/3): 골격(config/데이터/모델/학습/평가/예측/매칭), 다수결 병합, 스모크 테스트, matching pytest, README
- 이후: pipeline·HTML 리포트, compare_models, 라벨링 도구(`label_tool`, '모름'은 `skip` 저장 → 병합 시 기권)
- 제안서 최종본 반영: 데이터셋 CelebA → **FairFace East Asian** 교체(CelebA 샘플러 삭제), 클래스 설정 분리(`classes.yaml`),
  동일·유사 이미지 그룹 split(`dedup.py`), 데이터 규모별 비교(`data_scaling.py`), Macro Precision/Recall 추가,
  파일럿 분석, **웹캠 데모**(`webapp.py`: getUserMedia + MediaPipe Face Landmarker + Canvas)

## 작업 규칙
- 실행은 저장소 루트에서 `python -m faceapp.<module>` / `python -m scripts.<script>`.
- 변경 후 `python -m pytest -q`와 `python -m scripts.smoke_test`가 통과해야 한다.
- 클래스 순서는 `configs/classes.yaml` 하나만 기준으로 쓴다 (`faceapp.CLASSES`로 읽힘).

## 참고문헌
- Kärkkäinen & Joo, "FairFace," WACV 2021 — https://arxiv.org/abs/1908.04913
- He et al., "Deep Residual Learning for Image Recognition," CVPR 2016 — https://arxiv.org/abs/1512.03385
- PyTorch Transfer Learning Tutorial — https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- MediaPipe Face Landmarker — https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- Guo et al., "On Calibration of Modern Neural Networks," ICML 2017 / Selvaraju et al., "Grad-CAM," ICCV 2017
- Zhang et al., "mixup," ICLR 2018 / Yun et al., "CutMix," ICCV 2019

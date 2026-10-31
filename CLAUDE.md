# CLAUDE.md — 프로젝트 작업 가이드

딥러닝 기반 얼굴 유형 분류(강아지상/고양이상/토끼상/여우상)를 활용한 대학생 소개팅 자동 매칭 시스템.
딥러닝 수업 팀 프로젝트(연구 및 응용 개발). 이 문서는 사람과 AI 어시스턴트가 공유하는 설계/규칙 문서다.

## 파이프라인

```
참가자 사진 ─▶ 전처리/증강 ─▶ ResNet18 (ImageNet 사전학습 → fine-tune)
           ─▶ logits / T (temperature scaling 보정) ─▶ Softmax 4클래스 확률 (+ 512-d 임베딩)
           ─▶ 선호 조건(선호 유형 벡터, 나이, 취미) 결합 ─▶ 양방향 매칭 점수 ─▶ Top-N 추천
```

## 폴더 구조

```
configs/default.yaml       모든 경로·하이퍼파라미터·매칭 가중치 (코드에 하드코딩 금지)
faceapp/
  config.py                YAML 로드 + CLI override (--set train.epochs=5)
  data.py                  라벨 CSV 로드, identity 기준 split(+누수 assert), Dataset, transforms
  models.py                resnet18 / resnet34 / efficientnet_b0 / mobilenet_v3_small / simple_cnn
  losses.py                soft-label CE(+label smoothing), Mixup/CutMix
  train.py                 AMP, grad accumulation, discriminative LR, warmup+cosine, 단계적 unfreeze, early stopping
  calibrate.py             temperature scaling (val NLL 최소화), ECE
  evaluate.py              Accuracy, Macro F1, per-class P/R/F1, Confusion Matrix, ECE
  predict.py               이미지 → 확률 CSV + 임베딩 .npy (TTA: 좌우 반전)
  gradcam.py               Grad-CAM 시각화 (모델이 얼굴을 보는지 검증)
  matching.py              코사인 유사도 + 가중치 결합 → 양방향 Top-N
  engine.py                공용 추론 루프 (평가/보정/예측 재사용, TTA)
  report.py                매칭 결과 HTML 리포트 (사진은 상대 경로 참조만)
  pipeline.py              end-to-end: 학습→평가→예측→Grad-CAM→매칭→리포트, 단계별 시간 기록
  utils.py                 seed, device, 공용 헬퍼
scripts/
  sample_celeba_candidates.py   CelebA identity당 1장 샘플링 → 라벨링 시트
  merge_labels.py               팀원 5명 라벨 다수결 (동률 → review.csv), 투표 분포(soft label) 저장
  smoke_test.py                 합성 데이터로 pipeline 전체 + 시간 비교 일괄 실행
  compare_models.py             여러 설정(ResNet18 transfer vs SimpleCNN 등)을 같은 split으로 학습·평가 → 비교표
  compare_matching_time.py      수작업 vs AI 보조 매칭 시간 비교 (제안서 평가 항목)
examples/                  participants.json / 시간 측정 CSV 예시
tests/                     pytest (matching, merge_labels, split, losses/metrics/calibration, sampling, pipeline tools)
data/, outputs/            git 제외 (.gitignore)
```

## 딥러닝 설계 포인트 (수업 관점에서 비교/실험할 축)

| 영역 | 기법 | config 키 |
|---|---|---|
| Transfer learning | ImageNet 사전학습 ResNet18, head 교체 | `model.name`, `model.pretrained` |
| Fine-tuning 전략 | 처음 N epoch backbone freeze → 전체 unfreeze | `train.freeze_backbone_epochs` |
| | Discriminative LR (backbone lr = head lr × factor) | `train.backbone_lr_factor` |
| 최적화 | AdamW + linear warmup + cosine decay | `train.lr`, `train.warmup_epochs` |
| 메모리(4GB VRAM) | AMP(fp16), gradient accumulation (batch 16 × 2 = 32) | `train.amp`, `train.accum_steps` |
| 정규화 | Data augmentation, label smoothing, Mixup/CutMix, dropout, weight decay | `augment.*`, `train.label_smoothing`, `train.mixup_alpha` |
| 라벨 노이즈 | 5명 투표 분포를 soft label로 학습 (주관적 라벨에 적합) | `train.soft_labels` |
| 불균형 | class-weighted loss | `train.class_weighted_loss` |
| 확률 보정 | Temperature scaling — 매칭에 확률을 직접 쓰므로 보정 필수 | `calibration.enabled` |
| 추론 | TTA (horizontal flip 평균) | `predict.tta` |
| 해석 | Grad-CAM으로 근거 영역 시각화 | `python -m faceapp.gradcam` |
| 베이스라인 | 처음부터 학습하는 SimpleCNN vs ResNet18 transfer | `model.name=simple_cnn`, `scripts.compare_models` |
| 표현 | penultimate 512-d 임베딩 저장 (향후 이상형 사진 기반 매칭/t-SNE) | `predict.save_embeddings` |

### 2주차 이후 확장 후보 (아직 구현 안 함)
- 라벨 없는 CelebA 대량 이미지로 pseudo-labeling / self-training (semi-supervised)
- 얼굴 검출·정렬 (현재는 CelebA aligned crop 가정)
- 임베딩 t-SNE/UMAP 시각화, 이상형 참고 사진 → 선호 벡터 자동 생성 (`matching.preference_from_reference_probs` 골격만 있음)
- 하이퍼파라미터 sweep, 백본 비교 실험표

## 데이터 규칙 (반드시 지킬 것)
- CelebA는 비상업적 연구용. **원본 이미지·얼굴 사진은 절대 git에 커밋하지 않는다** (`data/`, `outputs/`는 .gitignore).
- CelebA에는 동물상 라벨이 없다 → 팀원 5명이 독립 라벨링 후 **다수결**. 동률이면 최종 라벨에서 제외하고 `review.csv`로 보낸다.
- 라벨 CSV 형식: `filename, identity, label` (label ∈ dog, cat, rabbit, fox). 추가 열 `votes_dog, votes_cat, votes_rabbit, votes_fox`는 선택(soft label용).
- Train/Val/Test는 **identity 기준**으로 분리한다. 같은 인물이 두 split에 들어가면 `assert`로 실패해야 한다 (`data.assert_no_identity_leak`).
- 목표 규모 약 1,200~1,500장. 데이터셋 경로는 항상 config(`data.image_dir`, `data.labels_csv`)로 받는다.

## 환경
- 팀원 환경: Windows + PowerShell, GTX 1650 Ti (4GB). 기본 batch 32, 224×224, AMP. OOM 시 `train.batch_size=16 train.accum_steps=2`.
- Python venv + requirements.txt. torch는 CUDA 빌드 별도 설치 (README 참고).
- 외부 API 사용 금지. PyTorch / TorchVision / Python (+ numpy, pyyaml, pillow, matplotlib, pytest).

## 아직 미정 (임의로 확정하지 말 것 — 코드에는 `TODO(미정)`으로 표시)
- [ ] 최종 데이터셋: CelebA vs K-FACE(AI Hub 사용 신청 필요) vs 연예인 이미지
- [ ] CelebA엔 인종 속성이 없어 "Asian 필터"를 속성만으로는 걸 수 없음 → 방법 미정
- [ ] FairFace 20장 테스트에서 나온 문제(나이대, 얼굴 각도, 안경, 조도, 비한국인) → 전처리/필터링 기준
- [ ] 동물상 라벨링 기준 (어떤 얼굴 특징으로 나눌지)
- [ ] 매칭 가중치(`matching.weights`)와 선호 조건 입력 형식(`participants.json` 스키마는 잠정)
- [ ] 수작업 vs AI 매칭 시간 측정 프로토콜 (`compare_matching_time.py` CSV 형식은 잠정)
- [ ] 실제 참가자 사진(비정렬 셀카)의 얼굴 검출·정렬 방법 — 현재는 CelebA aligned crop과 같은 resize+center crop만 적용

## 1주차 범위 (~11/3) — 끝까지 한 번 돌아가는 최소 프로토타입
1. 프로젝트 골격 (config, 데이터 로더, 모델, 학습, 평가, 예측, 매칭)
2. CelebA identity당 1장 샘플링 스크립트
3. 팀원별 라벨 CSV 다수결 병합 스크립트
4. 더미 라벨 스모크 테스트 (학습 → 평가 → Top-N 매칭)
5. matching 로직 pytest
6. README 설치/실행 방법

### 1주차 이후 추가된 것 (프로토타입 완성)
- `faceapp.pipeline` 한 줄 실행 + `report.html`, `scripts.compare_models` 비교표, `scripts.compare_matching_time` 시간 비교
- 평가/예측 transform은 체크포인트의 학습 당시 `image_size`를 따른다 (`data.build_eval_transforms`)

## 작업 규칙
- 실행은 저장소 루트에서 `python -m faceapp.<module>` / `python -m scripts.<script>`.
- 변경 후 `python -m pytest -q`와 `python -m scripts.smoke_test`가 통과해야 한다.
- 클래스 순서는 `faceapp/__init__.py`의 `CLASSES` 하나만 기준으로 쓴다.

## 참고문헌
- He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
- Liu et al., "Deep Learning Face Attributes in the Wild," ICCV 2015
- Guo et al., "On Calibration of Modern Neural Networks," ICML 2017 (temperature scaling)
- Selvaraju et al., "Grad-CAM," ICCV 2017
- Zhang et al., "mixup," ICLR 2018 / Yun et al., "CutMix," ICCV 2019
- CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- PyTorch Transfer Learning Tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

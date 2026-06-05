# Wafer Open-Set Defect Classification

ImageNet-pretrained **ResNet18**로 known 3-class 웨이퍼 결함을 학습하고, **post-hoc OOD rejection**으로 train에 없던 unknown 결함(DIE_CRACK, DIE_INK)을 거부하는 open-set 분류 프로젝트입니다.

**Deep Learning Basics — Final Project**

---

## Task

| Split | Classes | Train? |
|-------|---------|--------|
| `Data/train` | DIE_BROKEN, NORMAL, NO_DIE | ✓ (known only) |
| `Data/val`, `Data/test` | 위 3 + DIE_CRACK, DIE_INK | eval only |

| Class | Label |
|-------|-------|
| DIE_BROKEN | 0 |
| NORMAL | 1 |
| NO_DIE | 2 |
| DIE_CRACK, DIE_INK | **3 (Unknown)** — evaluation only |

---

## Method summary

1. **Classifier:** ResNet18 + linear head, **exactly 3 outputs** (Unknown은 출력 클래스가 아님)
2. **Training:** Cross-entropy on known classes only → `model.pth`
3. **OOD baselines:** MSP, Energy, Mahalanobis (pinv cov), Energy + Mahalanobis (hard OR)
4. **Proposed:** Diagonal covariance Mahalanobis + z-scored Energy–Mahalanobis **soft fusion**
   ```text
   score = α · zscore(energy) + (1−α) · zscore(maha)
   ```
   `α=0.2` (paper config), rejection threshold는 **validation**에서 튜닝 → test 1회 평가

---

## Final results (test)

출처: `outputs/final_ranking.csv` (2025-06-05 실행)

| Method | Known Acc | Unknown Prec | Unknown Recall | Macro F1 |
|--------|----------:|-------------:|---------------:|---------:|
| Argmax | 0.971 | 0.000 | 0.000 | 0.517 |
| MSP | 0.914 | 0.727 | 0.200 | 0.607 |
| Energy | 0.800 | 0.696 | 0.400 | 0.632 |
| Mahalanobis | 0.971 | 1.000 | 0.600 | 0.790 |
| Energy + Mahalanobis | 0.971 | 1.000 | 0.625 | 0.801 |
| **Proposed: Diagonal + Soft Fusion** | 0.857 | 0.878 | 0.900 | **0.870** |

- **Proposed**가 test macro F1 **최고** (unknown recall 0.90 + balanced open-set F1)
- Argmax는 unknown recall 0 → open-set에서 rejection 불가
- Mahalanobis / Energy+Maha는 unknown precision은 높지만 recall이 낮음

---

## Configuration (notebook)

```python
RETRAIN = False       # True: scratch 학습
PROPOSED_ALPHA = 0.2  # None: val macro-F1로 α grid search
USE_WANDB = WANDB_AVAILABLE
```

| 목적 | 설정 |
|------|------|
| 논문/보고서 숫자 재현 | `RETRAIN=False` + `outputs/best_model.pth` |
| 처음부터 학습 | `RETRAIN=True` → `best_model.pth` 자동 저장 |

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ipykernel nbconvert
```

---

## How to run

1. `Data.zip` 압축 해제 → `./Data`
2. `jupyter notebook notebook.ipynb`
3. **Kernel → Restart and Run All**

---

## Project files

```text
final_project/
├── notebook.ipynb          # 제출용 전체 파이프라인
├── model.pth               # checkpoint + Mahalanobis stats (실행 후 생성)
├── report.pdf              # 보고서 
├── requirements.txt
├── README.md
└── outputs/                # 실행 결과
    ├── final_ranking.csv
    ├── metrics.csv
    ├── final_summary.json
    ├── confusion_matrix_final.png
    ├── confusion_matrix.png
    ├── gradcam_summary.png
    ├── dataset_visualization.png
    ├── training_history.csv
    ├── training_curves.png
    └── open_set_scores.png
```

---

## Open-set design

- Classifier는 **3-class closed-set**만 학습
- Unknown(label 3)은 **rejection rule**으로만 생성
- Threshold는 val에서 고정 후 test에 적용 (leakage 방지)
- Grad-CAM: ResNet argmax-class logit 기준, 2×2 summary (`gradcam_summary.png`)

---

## W&B

- Project: `wafer-open-set-classification`
- `USE_WANDB = WANDB_AVAILABLE` — wandb 설치·로그인 시 자동 활성

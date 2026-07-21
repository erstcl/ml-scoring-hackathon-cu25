[🇷🇺 Русская версия](README.ru.md) | [🇬🇧 English version](README.md)

---

# T-Bank Credit Attrition Prediction

[![Notebook quality](https://github.com/erstcl/ml-scoring-hackathon-cu25/actions/workflows/notebook-quality.yml/badge.svg)](https://github.com/erstcl/ml-scoring-hackathon-cu25/actions/workflows/notebook-quality.yml)

**Competition**: CU 2025 Scoring
**Platform**: Kaggle  
**Organizer**: Central University & T-Bank  
**Timeline**: November 2025

---

## Challenge

Build a machine learning model to predict early loan repayment (attrition) using T-Bank credit product data. When customers repay loans early, banks earn less interest revenue — predicting this at the application stage is crucial for profitability optimization.

---

## Data Overview

### Key Characteristics
- **Target variable**: `a6_flg` (early repayment flag)
- **Products**: 4 credit products (product_1 — product_4)
- **Time period**: data split by months (`month_dt`)
- **Features**: ~100+ features (feature_0 — feature_N)

### Main Challenge
**Temporal distribution shift**: test set significantly differs from training set in temporal distribution. Requires model stability monitoring across months and overfitting control.

---

## Solution

### Exploratory Data Analysis (EDA)

**Missing values**:
- Identified many features with high missing rates (50%+, 70%+)
- Analyzed feature importance for high-missing features using RandomForest
- Removed features with >70% missing values
- Applied median imputation (`SimpleImputer`) for remaining features

**Class imbalance**:
- Target variable shows imbalance (attrition is a rarer event)
- Used stratified validation to preserve class proportions

### Modeling

**Model choice**: CatBoostClassifier

**Rationale**:
- Native handling of categorical features
- Resistant to overfitting (Ordered Boosting)
- High performance on tabular data
- Built-in missing value handling

**Hyperparameters**:
```python
CatBoostClassifier(
    iterations=700,
    learning_rate=0.03,
    depth=6,
    eval_metric='AUC',
    random_state=42
)
```

**Validation strategy**:
- Train-test split with `stratify=y` (80/20)
- ROC-AUC monitoring on validation set
- Early stopping with `use_best_model=True`

This random stratified split was used for fast model iteration. It does not reproduce
the competition's temporal test distribution; the Kaggle leaderboard score is the
external evaluation under that shift.

---

## Results

### Metrics
- **Baseline**: 0.73707 ROC-AUC
- **Kaggle leaderboard**: 45th place, score: 0.75046

### Key Findings
- Removing highly sparse features (>70% missing) improved model stability
- The public leaderboard score remained above the local baseline despite the
  train/test distribution difference
- Median imputation proved effective for numerical features with moderate missing rates

---

## Reproduce the analysis

```bash
git clone https://github.com/erstcl/ml-scoring-hackathon-cu25.git
cd ml-scoring-hackathon-cu25
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab kaggle_hackathon.ipynb
```

Authenticate with Kaggle when prompted and attach the `cu-2025-scoring` competition
data. CI validates the committed notebook structure and saved outputs without
redistributing competition files.

## Tech Stack

### ML Framework
- **Data processing**: `pandas`, `numpy`
- **Visualization**: `matplotlib`
- **Modeling**: `CatBoost`, `RandomForestClassifier` (for feature importance)
- **Metrics**: `roc_auc_score`

### Environment
- **Platform**: Kaggle Notebooks (NVIDIA Tesla T4 GPU)
- **Language**: Python 3.11

---

## About the Competition

**Competition link**: [Kaggle — CU 2025 Scoring](https://www.kaggle.com/competitions/cu-2025-scoring)

The competition data is not redistributed in this repository. Reproduction requires
Kaggle access to the competition files.

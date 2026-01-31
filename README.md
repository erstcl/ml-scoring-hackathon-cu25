[🇷🇺 Русская версия](README.ru.md) | [🇬🇧 English version](README.md)

---

# T-Bank Credit Attrition Prediction

**University competition**: Machine Learning Course (Central University)  
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

---

## Results

### Metrics
- **Baseline**: 0.73707 ROC-AUC
- **Kaggle leaderboard**: 45th place, score: 0.75046

### Key Findings
- Removing highly sparse features (>70% missing) improved model stability
- CatBoost demonstrated robustness to temporal shift through overfitting control
- Median imputation proved effective for numerical features with moderate missing rates

---

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

This competition was part of the "Machine Learning" course at Central University. Top performers in the main task received bonus points, while the best solutions to the additional task (building separate models per product) offered internship opportunities in T-Bank's credit scoring team.

**Competition link**: [Kaggle — CU 2025 Scoring](https://www.kaggle.com/competitions/cu-2025-scoring)

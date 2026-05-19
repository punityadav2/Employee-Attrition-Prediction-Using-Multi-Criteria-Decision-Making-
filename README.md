# Employee Attrition Prediction Using Multi-Criteria Decision Making (MCDM)

> **Capstone Project — 4th Semester**  
> Predicting employee attrition from the IBM HR Analytics dataset using six machine learning classifiers, selecting the best model through a scientific MCDM framework (WSM, TOPSIS, VIKOR + RPM with subjective weights), and explaining predictions via SHAP and LIME.

---

## Table of Contents

- [Overview](#overview)
- [Research Papers](#research-papers)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Pipeline Steps](#pipeline-steps)
- [Feature Engineering](#feature-engineering)
- [Models & Hyperparameter Tuning](#models--hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [MCDM Methods](#mcdm-methods)
- [Results](#results)
- [Explainability](#explainability)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Output Files](#output-files)

---

## Overview

This project builds a complete end-to-end ML pipeline to predict whether an IBM employee will leave the company (**Attrition = Yes/No**). Rather than simply picking the model with the highest single metric, it applies **Multi-Criteria Decision Making (MCDM)** methods to evaluate six classifiers across eight performance criteria simultaneously and select the best model using the **Rank Position Method (RPM)**.

Key highlights:

- **6 ML models** trained and compared
- **Optuna** (TPE sampler, **60 trials**) for automated hyperparameter optimisation
- **Random Oversampling** to handle class imbalance (training data only)
- **SHAP** for global feature importance and top-15 feature selection
- **LIME** for individual prediction explanations
- **3 MCDM methods**: WSM, TOPSIS, VIKOR → aggregated by RPM with **subjective weights**
- **🏆 Final selected model: Logistic Regression** (RPM = 3, unanimous #1 across all MCDM methods)
- **Streamlit dashboard** with 5 interactive pages

---

## Research Papers

| Paper | Purpose |
|-------|---------|
| `computers-15-00185.pdf` | ML model design, feature engineering strategy, evaluation framework |
| `MCDM for MLTs.pdf` | WSM, TOPSIS, VIKOR methodology and RPM aggregation |

---

## Dataset

**IBM HR Employee Attrition** (publicly available on Kaggle)

| Property | Value |
|----------|-------|
| File | `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv` |
| Rows | 1,470 employees |
| Columns | 35 original features |
| Target | `Attrition` (Yes = 1 / No = 0) |
| Class distribution | No: 1,233 (83.9%) / Yes: 237 (16.1%) |
| Missing values | None |

### Dropped Columns (constants in IBM dataset)
`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`

---

## Project Architecture

```
IBM HR CSV
    │
    ▼
Feature Engineering  ──────────────────────────────────────────────────┐
(workload_ratio, tenure_level)                                          │
    │                                                                   │
    ▼                                                                   │
Preprocessing                                                           │
(Encode → Drop constants → Impute → Label Encode)                      │
(Stratified 80:20 split → StandardScaler → RandomOverSampler)         │
    │                                                                   │
    ▼                                                                   │
Baseline XGBoost  ──► SHAP / Feature Importance ──► Top 15 Features   │
    │                                                                   │
    ▼                                                                   │
6 Models × Optuna (60 trials, TPE, F1 objective)                       │
    │                                                                   │
    ▼                                                                   │
Evaluation (8 metrics) ──► Decision Matrix                             │
    │                                                                   │
    ▼                                                                   │
MCDM: WSM + TOPSIS + VIKOR ──► RPM ──► Final Model                    │
    │                                                                   │
    ▼                                                                   │
SHAP Global + LIME Local ──► Explainability Plots                      │
    │                                                                   │
    ▼                                                                   │
Streamlit Dashboard  ◄──────────────────────────────────────────────────┘
```

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Load IBM HR dataset |
| 2 | Feature engineering (`workload_ratio`, `tenure_level`) |
| 3 | Preprocessing: encode, scale, stratified split, Random Oversampling |
| 4 | Baseline XGBoost training |
| 5 | SHAP/native feature selection — top K=15 features |
| 6 | Train 6 ML models |
| 7 | Optuna hyperparameter optimisation (60 trials per model, maximise F1) |
| 8 | Evaluate all models — 8 metrics, build decision matrix |
| 9 | MCDM ranking: WSM, TOPSIS, VIKOR |
| 10 | RPM aggregation — final model selection |
| 11 | SHAP global explanation + LIME local explanations |
| 12 | Generate all visualisations |
| 13 | Save results CSV, models, and encoders |

---

## Feature Engineering

Two new features are created **before** encoding to ensure they use original numeric values:

### 1. `workload_ratio`
```
workload_ratio = MonthlyRate / (TotalWorkingYears + 1)
```
Captures workload intensity relative to career experience. The IBM dataset does not include `AverageMonthlyHours` or `NumberOfProjects` (as in some Kaggle variants), so `MonthlyRate / experience` serves as the equivalent workload-per-year proxy.

### 2. `tenure_level`
```
YearsAtCompany  0–3  → Junior (0)
YearsAtCompany  3–6  → Mid    (1)
YearsAtCompany  > 6  → Senior (2)
```

---

## Models & Hyperparameter Tuning

All six models are tuned with **Optuna TPE sampler** (60 trials each) using 5-fold stratified cross-validation, maximising **F1-score** on the oversampled training set.

| # | Model | Key Tuned Parameters |
|---|-------|---------------------|
| 1 | Logistic Regression | `C` |
| 2 | Decision Tree | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| 3 | Random Forest | `n_estimators`, `max_depth`, `min_samples_split` |
| 4 | SVM | `C`, `kernel` |
| 5 | Gradient Boosting | `n_estimators`, `max_depth`, `learning_rate`, `subsample` |
| 6 | XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `scale_pos_weight`, `gamma`, `min_child_weight`, `reg_alpha`, `reg_lambda` |

> **Reproducibility:** All experiments use `random_state=42`.

---

## Evaluation Metrics

Eight metrics form the **MCDM decision matrix**:

| Metric | Type | Formula |
|--------|------|---------|
| Accuracy | Benefit ↑ | (TP + TN) / Total |
| Precision | Benefit ↑ | TP / (TP + FP) |
| Recall (Sensitivity) | Benefit ↑ | TP / (TP + FN) |
| F1-Score | Benefit ↑ | 2 × Precision × Recall / (Precision + Recall) |
| ROC-AUC | Benefit ↑ | Area under ROC curve |
| Specificity | Benefit ↑ | TN / (TN + FP) |
| FPR | Cost ↓ | FP / (FP + TN) |
| FNR | Cost ↓ | FN / (FN + TP) |

Weights are **subjective** — assigned based on business priority (catching flight-risk employees is the primary goal):

| Metric | Weight | Rationale |
|--------|--------|-----------|
| Recall | **0.20** | Catch as many at-risk employees as possible |
| FNR | **0.20** | Missing attrition is the costliest error |
| F1-Score | **0.20** | Balance recall with precision |
| ROC-AUC | 0.15 | Threshold-robust evaluation |
| Precision | 0.10 | Avoid false retention spend |
| Accuracy | 0.05 | Secondary in imbalanced data |
| Specificity | 0.05 | Correctly identify employees who stay |
| FPR | 0.05 | Limit unnecessary interventions |

> **60% of total weight** is assigned to Recall, FNR, and F1-Score — reflecting that missing an at-risk employee is far more costly than a false alarm.

---

## MCDM Methods

### WSM — Weighted Sum Method
1. Min-max normalise each criterion (cost criteria inverted so higher = better)
2. Multiply by equal weights
3. Sum → WSM score (higher = better)

### TOPSIS
1. Vector normalise the decision matrix
2. Apply weights to obtain weighted normalised matrix
3. Compute ideal (A⁺) and negative-ideal (A⁻) solutions
4. Calculate Euclidean distances D⁺ and D⁻
5. Relative closeness C = D⁻ / (D⁺ + D⁻) — rank descending

### VIKOR
1. Determine best f* and worst f⁻ for each criterion
2. Compute utility S_i and regret R_i
3. Q_i = v·(S − S*)/(S⁻ − S*) + (1 − v)·(R − R*)/(R⁻ − R*) with v = 0.5
4. Rank ascending by Q (lower = better)

### RPM — Rank Position Method
```
RPM_score = WSM_rank + TOPSIS_rank + VIKOR_rank
Final_rank = ascending(RPM_score)   # lowest sum = best model
```

---

## Results

### Decision Matrix (Test Set — 294 employees)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | FPR | FNR |
|-------|---------|-----------|--------|-----|---------|------------|-----|-----|
| **Logistic Regression** | 0.7041 | 0.3214 | **0.7660** | **0.4528** | **0.7873** | 0.6923 | 0.3077 | **0.2340** |
| Decision Tree | 0.7585 | 0.2273 | 0.2128 | 0.2198 | 0.5376 | 0.8623 | 0.1377 | 0.7872 |
| Random Forest | 0.8265 | 0.4286 | 0.2553 | 0.3200 | 0.7622 | 0.9352 | 0.0648 | 0.7447 |
| SVM | 0.8095 | 0.3902 | 0.3404 | 0.3636 | 0.7467 | 0.8988 | 0.1012 | 0.6596 |
| Gradient Boosting | 0.8265 | 0.3750 | 0.1277 | 0.1905 | 0.7333 | **0.9595** | **0.0405** | 0.8723 |
| XGBoost | **0.8469** | **0.5294** | 0.3830 | 0.4444 | 0.7706 | 0.9352 | 0.0648 | 0.6170 |

### MCDM Final Ranking

| Model | WSM Score | WSM Rank | TOPSIS Score | TOPSIS Rank | VIKOR Q | VIKOR Rank | RPM Score | **Final Rank** |
|-------|-----------|----------|-------------|------------|---------|-----------|----------|----------------|
| **Logistic Regression** | **0.781** | **1** | **0.792** | **1** | **0.116** | **1** | **3** | **🏆 1** |
| XGBoost | 0.734 | 2 | 0.522 | 2 | 0.195 | 2 | 6 | 2 |
| SVM | 0.559 | 3 | 0.419 | 3 | 0.399 | 3 | 9 | 3 |
| Random Forest | 0.514 | 4 | 0.330 | 4 | 0.506 | 4 | 12 | 4 |
| Gradient Boosting | 0.309 | 5 | 0.211 | 5 | 0.810 | 5 | 15 | 5 |
| Decision Tree | 0.158 | 6 | 0.178 | 6 | 0.915 | 6 | 18 | 6 |

### 🏆 Final Selected Model: **Logistic Regression**

> Logistic Regression achieves **RPM = 3** — the minimum possible score — by ranking **#1 unanimously** across all three MCDM methods (WSM, TOPSIS, and VIKOR). Under the business-aligned subjective weights (60% on Recall, FNR, F1-Score), it catches **36 of 47** at-risk employees (Recall = 0.766), compared to only 18 caught by XGBoost despite its higher accuracy (0.847).

### Top 5 Predictive Features (SHAP — Logistic Regression)

| Rank | Feature | Mean |SHAP| |
|------|---------|----------------|
| 1 | OverTime | 0.0653 |
| 2 | StockOptionLevel | 0.0352 |
| 3 | EnvironmentSatisfaction | 0.0288 |
| 4 | JobSatisfaction | 0.0239 |
| 5 | YearsWithCurrManager | 0.0232 |

---

## Explainability

### Global — SHAP Summary Plot
A beeswarm SHAP plot is generated for the **final model (Logistic Regression)**, showing the direction and magnitude of every feature's contribution across all test samples.  
Saved to: `outputs/figures/shap_summary.png`

### Local — LIME Explanations
Three individual predictions are explained with LIME:
- An employee who **left** (Attrition = Yes)
- An employee who **stayed** (Attrition = No)
- A **borderline** case (predicted probability ≈ 0.5)

Saved to: `outputs/figures/lime_*.png`

---

## Project Structure

```
implementation/
├── README.md                        ← This file
├── main.py                          ← Full 13-step pipeline orchestrator
├── app.py                           ← Streamlit dashboard (5 pages)
├── config.py                        ← Central configuration (paths, params)
├── requirements.txt                 ← Python dependencies
├── plan.txt                         ← Original project specification
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-HR-Employee-Attrition.csv   ← IBM HR dataset
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               ← Load dataset + EDA summary
│   ├── feature_engineering.py       ← workload_ratio, tenure_level
│   ├── preprocessing.py             ← Encode, scale, split, ROS
│   ├── model_training.py            ← Baseline XGB, SHAP selection, 6 models + Optuna
│   ├── evaluation.py                ← 8 metrics + decision matrix
│   ├── mcdm.py                      ← WSM, TOPSIS, VIKOR, RPM
│   ├── explainability.py            ← SHAP global + LIME local
│   └── visualization.py             ← All plots
│
├── models/
│   ├── best_model.pkl               ← Trained Logistic Regression model
│   ├── encoders.pkl                 ← Label encoders (for inference)
│   ├── scaler.pkl                   ← StandardScaler (for inference)
│   └── top_features.pkl             ← List of top-15 features
│
├── outputs/
│   ├── figures/                     ← All generated plots (20 files)
│   │   ├── model_comparison.png
│   │   ├── metrics_heatmap.png
│   │   ├── mcdm_rankings.png
│   │   ├── roc_curves_all_models.png
│   │   ├── confusion_matrices.png
│   │   ├── feature_importance_comparison.png
│   │   ├── green_efficiency.png
│   │   ├── shap_summary.png
│   │   └── lime_*.png (3 files)
│   └── reports/
│       └── results.csv              ← Full results: metrics + MCDM ranks + params
│
└── notebooks/                       ← Jupyter notebooks (optional)
```

---

## Installation

### Prerequisites
- Python **3.10** (recommended — tested on 3.10.x)
- pip

### 1. Clone / navigate to the project folder
```bash
cd implementation
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Key packages installed:**

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | ≥ 1.2.0 | Core ML models |
| xgboost | ≥ 1.7.0 | XGBoost classifier |
| optuna | ≥ 3.0.0 | Hyperparameter optimisation |
| shap | ≥ 0.41.0 | Global explainability |
| lime | ≥ 0.2.0.1 | Local explainability |
| imbalanced-learn | ≥ 0.10.0 | RandomOverSampler |
| streamlit | ≥ 1.31.0 | Interactive dashboard |
| psutil | ≥ 5.9.0 | Memory tracking (green efficiency) |
| pandas / numpy | ≥ 1.5 / 1.23 | Data manipulation |
| matplotlib / seaborn | ≥ 3.6 / 0.12 | Visualisations |

---

## How to Run

### Step 1 — Place the Dataset
Make sure the IBM HR CSV file is at:
```
data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

### Step 2 — Run the Full Pipeline
```bash
python main.py
```

This executes all 13 steps and saves:
- `models/best_model.pkl` — trained **Logistic Regression** model
- `outputs/figures/` — 20 visualisation files
- `outputs/reports/results.csv` — full metrics + MCDM ranking table

Expected runtime: **~8–12 minutes** (Optuna runs 60 trials × 6 models with 5-fold CV).

### Step 3 — Launch the Dashboard
```bash
python -m streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

> **Note:** If you have multiple Python versions, use the explicit path:
> ```bash
> C:\Users\<you>\AppData\Local\Programs\Python\Python310\python.exe -m streamlit run app.py
> ```

---

## Streamlit Dashboard

The dashboard has **5 pages** accessible via the left sidebar:

| Page | Description |
|------|-------------|
| 🏠 **Home** | Project overview, methodology, SHAP importance chart |
| 👤 **Single Prediction** | Enter one employee's attributes → get attrition risk % |
| 📊 **Batch Prediction** | Upload a CSV → download predictions for all employees |
| 🏆 **MCDM Rankings** | WSM / TOPSIS / VIKOR scores + final RPM ranking table & chart |
| 📈 **Model Performance** | Metrics table, ROC curves, confusion matrices, feature importance plots |

---

## Output Files

### `outputs/figures/`

| File | Description |
|------|-------------|
| `model_comparison.png` | Grouped bar chart — 6 models × 6 benefit metrics |
| `metrics_heatmap.png` | Heatmap — 8 metrics × 6 models |
| `mcdm_rankings.png` | Bar chart — WSM vs TOPSIS vs VIKOR vs Final rank |
| `roc_curves_all_models.png` | Overlay ROC curves for all 6 models |
| `confusion_matrices.png` | 2×3 grid of confusion matrices |
| `feature_importance_comparison.png` | SHAP vs Logistic Regression coefficient magnitudes |
| `green_efficiency.png` | F1 / (training_time × peak_memory) |
| `shap_summary.png` | SHAP beeswarm plot (**Logistic Regression** on test set) |
| `lime_AttritionYes_left.png` | LIME — employee who left |
| `lime_AttritionNo_stayed.png` | LIME — employee who stayed |
| `lime_Borderline_prob0.5.png` | LIME — borderline prediction |

### `outputs/reports/results.csv`
Full table with all 8 metrics + WSM/TOPSIS/VIKOR scores + ranks + training time + memory + best hyperparameters for every model.

---

## Configuration

All key constants live in `config.py`. You can change them without touching any other file:

```python
RANDOM_STATE    = 42          # reproducibility seed
TEST_SIZE       = 0.20        # 80:20 train-test split
TOP_K_FEATURES  = 15          # number of features after SHAP selection
OPTUNA_N_TRIALS = 60          # Optuna trials per model
CV_FOLDS        = 5           # cross-validation folds
VIKOR_V         = 0.5         # VIKOR compromise parameter
```

---

## Reproducibility

All random operations use `random_state=42`:
- Train/test split (`stratify=y`)
- RandomOverSampler
- Optuna TPE sampler seed
- All scikit-learn and XGBoost models

Running `python main.py` twice on the same machine will produce identical results.

---

## References

1. IBM HR Analytics Employee Attrition & Performance — [Kaggle Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
2. *"A Comparative Study of Machine Learning Classifiers for Employee Attrition Prediction"* — `computers-15-00185.pdf`
3. *"MCDM-Based Model Selection for Machine Learning Tasks"* — `MCDM for MLTs.pdf`
4. Optuna: A Next-Generation Hyperparameter Optimization Framework — [optuna.org](https://optuna.org)
5. SHAP: A Unified Approach to Explain Machine Learning Models — [shap.readthedocs.io](https://shap.readthedocs.io)
6. LIME: Local Interpretable Model-Agnostic Explanations — [github.com/marcotcr/lime](https://github.com/marcotcr/lime)

---

<div align="center">

**Capstone Project — 4th Semester | IBM HR Attrition | MCDM Model Selection**  
**🏆 Final Model: Logistic Regression | RPM = 3 | Recall = 0.766 | AUC = 0.787**

</div>

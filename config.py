"""
Configuration file for Capstone Project:
"Model Selection for Employee Attrition Prediction Using Multi-Criteria Decision Methods"
"""

import os

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')

# IBM HR Employee Attrition dataset
ATTRITION_FILE = os.path.join(RAW_DATA_DIR, 'WA_Fn-UseC_-HR-Employee-Attrition.csv')

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_STATE = 42

# ============================================================================
# PREPROCESSING
# ============================================================================
TARGET_COLUMN = 'Attrition'  # Will be encoded: Yes=1, No=0

# Columns to drop (constants in IBM dataset)
DROP_COLUMNS = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']

# Train/test split
TEST_SIZE = 0.20

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
# tenure_level based on YearsAtCompany
TENURE_BINS   = [0, 3, 6, float('inf')]
TENURE_LABELS = ['Junior', 'Mid', 'Senior']
TENURE_ENCODE = {'Junior': 0, 'Mid': 1, 'Senior': 2}

# ============================================================================
# CLASS IMBALANCE
# ============================================================================
IMBALANCE_STRATEGY = 'ROS'   # Random Oversampling (no SMOTE)

# ============================================================================
# SHAP FEATURE SELECTION
# ============================================================================
TOP_K_FEATURES = 15   # Select top 15 features by mean |SHAP|

# ============================================================================
# BASELINE XGBOOST (for SHAP feature importance)
# ============================================================================
BASELINE_XGB_PARAMS = {
    'n_estimators'  : 100,
    'max_depth'     : 5,
    'learning_rate' : 0.1,
    'random_state'  : RANDOM_STATE,
    'eval_metric'   : 'logloss',
    'use_label_encoder': False,
}

# ============================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================================
OPTUNA_N_TRIALS = 60
OPTUNA_METRIC   = 'f1'   # maximize F1-score
CV_FOLDS        = 5

# ============================================================================
# MODELS TO TRAIN
# ============================================================================
MODEL_NAMES = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'SVM',
    'Gradient Boosting',
    'XGBoost',
]

# ============================================================================
# EVALUATION — MCDM DECISION MATRIX CRITERIA
# ============================================================================
BENEFIT_CRITERIA = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'Specificity']
COST_CRITERIA    = ['FPR', 'FNR']
ALL_CRITERIA     = BENEFIT_CRITERIA + COST_CRITERIA

# Subjective weights for MCDM prioritizing Attrition detection (sum = 1.0)
MCDM_WEIGHTS = {
    'Accuracy'    : 0.05,
    'Precision'   : 0.10,
    'Recall'      : 0.20,
    'F1_Score'    : 0.20,
    'ROC_AUC'     : 0.15,
    'Specificity' : 0.05,
    'FPR'         : 0.05,
    'FNR'         : 0.20,
}

# VIKOR v parameter
VIKOR_V = 0.5

# ============================================================================
# VISUALIZATION
# ============================================================================
FIGURE_SIZE    = (12, 7)
FIGURE_DPI     = 150
PLOT_STYLE     = 'seaborn-v0_8-whitegrid'
COLOR_PALETTE  = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']

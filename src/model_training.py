"""
Model Training Module  6 classifiers + Optuna hyperparameter tuning.

Steps covered:
  Step 4  : XGBoost baseline (for SHAP feature selection)
  Step 5  : SHAP-based top-K feature selection
  Step 6  : Train 6 models on selected features
  Step 7  : Optuna (TPE) hyperparameter optimization  maximize F1
"""

import time
import psutil
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import shap
import optuna
import joblib

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ===========================================================================
# STEP 4  Baseline XGBoost
# ===========================================================================
def train_baseline_xgboost(X_train, y_train):
    """Train XGBoost with fixed params (Step 4) for SHAP feature selection."""
    print("\n" + "=" * 70)
    print("  STEP 4: BASELINE XGBOOST (for SHAP feature importance)")
    print("=" * 70)

    params = config.BASELINE_XGB_PARAMS.copy()
    params.pop('use_label_encoder', None)   # removed in newer xgb
    model = xgb.XGBClassifier(**params, verbosity=0)
    model.fit(X_train, y_train)
    print("[OK] Baseline XGBoost trained")
    return model


# ===========================================================================
# STEP 5  SHAP Feature Selection
# ===========================================================================
def select_top_k_features_shap(model, X_train, k=None):
    """
    Use SHAP TreeExplainer to rank features by mean |SHAP value|.
    Returns list of top-K feature names.
    """
    if k is None:
        k = config.TOP_K_FEATURES

    print("\n" + "=" * 70)
    print(f"  STEP 5: SHAP FEATURE SELECTION  (top K={k})")
    print("=" * 70)

    try:
        # feature_perturbation='tree_path_dependent' avoids base_score parsing bug
        explainer   = shap.TreeExplainer(model,
                                         feature_perturbation='tree_path_dependent')
        shap_values = explainer.shap_values(X_train)
    except Exception as shap_err:
        print(f"  [!] SHAP TreeExplainer failed ({shap_err}), using native feature importances")
        # Fallback: use XGBoost native feature importances
        fi = pd.Series(model.feature_importances_, index=X_train.columns)
        top_k = fi.nlargest(k)
        print(f"  Top {k} features (native XGBoost importances):")
        for i, (feat, val) in enumerate(top_k.items(), 1):
            print(f"  {i:>2}. {feat:<35} score={val:.4f}")
        return top_k.index.tolist(), fi.sort_values(ascending=False)

    # For binary classification, shap_values may be 2D (n_samples, n_features)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # class 1

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_train.columns
    ).sort_values(ascending=False)

    top_features = mean_abs_shap.head(k).index.tolist()

    print(f"[OK] Top {k} features selected by SHAP:")
    for i, (feat, val) in enumerate(mean_abs_shap.head(k).items(), 1):
        print(f"  {i:>2}. {feat:<35} SHAP={val:.4f}")

    return top_features, mean_abs_shap


# ===========================================================================
# STEP 6+7  Train 6 Models with Optuna
# ===========================================================================

def _cv_f1(estimator, X, y):
    """Helper: 5-fold stratified CV -> mean F1."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    scores = cross_val_score(estimator, X, y, cv=cv,
                             scoring='f1', n_jobs=-1)
    return scores.mean()


# ---------- Logistic Regression ----------
def _tune_logistic(trial, X, y):
    C   = trial.suggest_float('C', 1e-3, 10.0, log=True)
    clf = LogisticRegression(C=C, max_iter=1000,
                             random_state=config.RANDOM_STATE,
                             class_weight='balanced', solver='lbfgs')
    return _cv_f1(clf, X, y)


# ---------- Decision Tree ----------
def _tune_dt(trial, X, y):
    max_depth         = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 10)
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=config.RANDOM_STATE,
        class_weight='balanced'
    )
    return _cv_f1(clf, X, y)


# ---------- Random Forest ----------
def _tune_rf(trial, X, y):
    n_estimators      = trial.suggest_int('n_estimators', 50, 300)
    max_depth         = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=config.RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )
    return _cv_f1(clf, X, y)


# ---------- SVM ----------
def _tune_svm(trial, X, y):
    C      = trial.suggest_float('C', 1e-2, 100.0, log=True)
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
    clf = SVC(C=C, kernel=kernel, probability=True,
              random_state=config.RANDOM_STATE,
              class_weight='balanced')
    return _cv_f1(clf, X, y)


# ---------- Gradient Boosting ----------
def _tune_gb(trial, X, y):
    n_estimators  = trial.suggest_int('n_estimators', 50, 300)
    max_depth     = trial.suggest_int('max_depth', 2, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    subsample     = trial.suggest_float('subsample', 0.5, 1.0)
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=config.RANDOM_STATE
    )
    return _cv_f1(clf, X, y)


# ---------- XGBoost ----------
def _tune_xgb(trial, X, y):
    n_estimators      = trial.suggest_int('n_estimators', 50, 300)
    max_depth         = trial.suggest_int('max_depth', 2, 12)
    learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.3, log=True)
    subsample         = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    scale_pos_weight  = trial.suggest_float('scale_pos_weight', 1.0, 10.0)
    gamma             = trial.suggest_float('gamma', 0.0, 5.0)
    min_child_weight  = trial.suggest_int('min_child_weight', 1, 10)
    reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True)
    reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True)
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        gamma=gamma,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=config.RANDOM_STATE,
        eval_metric='logloss',
        verbosity=0
    )
    return _cv_f1(clf, X, y)


_TUNE_FNS = {
    'Logistic Regression': _tune_logistic,
    'Decision Tree'      : _tune_dt,
    'Random Forest'      : _tune_rf,
    'SVM'                : _tune_svm,
    'Gradient Boosting'  : _tune_gb,
    'XGBoost'            : _tune_xgb,
}

_BUILD_FNS = {
    'Logistic Regression': lambda p: LogisticRegression(
        C=p.get('C', 1.0), max_iter=1000,
        random_state=config.RANDOM_STATE, class_weight='balanced', solver='lbfgs'),
    'Decision Tree'      : lambda p: DecisionTreeClassifier(
        max_depth=p.get('max_depth', 5),
        min_samples_split=p.get('min_samples_split', 2),
        min_samples_leaf=p.get('min_samples_leaf', 1),
        random_state=config.RANDOM_STATE, class_weight='balanced'),
    'Random Forest'      : lambda p: RandomForestClassifier(
        n_estimators=p.get('n_estimators', 100),
        max_depth=p.get('max_depth', 10),
        min_samples_split=p.get('min_samples_split', 2),
        random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=-1),
    'SVM'                : lambda p: SVC(
        C=p.get('C', 1.0), kernel=p.get('kernel', 'rbf'),
        probability=True, random_state=config.RANDOM_STATE, class_weight='balanced'),
    'Gradient Boosting'  : lambda p: GradientBoostingClassifier(
        n_estimators=p.get('n_estimators', 100),
        max_depth=p.get('max_depth', 3),
        learning_rate=p.get('learning_rate', 0.1),
        subsample=p.get('subsample', 0.8),
        random_state=config.RANDOM_STATE),
    'XGBoost'            : lambda p: xgb.XGBClassifier(
        n_estimators=p.get('n_estimators', 100),
        max_depth=p.get('max_depth', 5),
        learning_rate=p.get('learning_rate', 0.1),
        subsample=p.get('subsample', 0.8),
        colsample_bytree=p.get('colsample_bytree', 0.8),
        scale_pos_weight=p.get('scale_pos_weight', 1.0),
        gamma=p.get('gamma', 0.0),
        min_child_weight=p.get('min_child_weight', 1),
        reg_alpha=p.get('reg_alpha', 0.0),
        reg_lambda=p.get('reg_lambda', 1.0),
        random_state=config.RANDOM_STATE,
        eval_metric='logloss', verbosity=0),
}


def train_model_with_optuna(name, X_train, y_train, n_trials=None):
    """
    Run Optuna (TPE sampler) to find best hyperparameters, then
    retrain the model on full training data with those params.

    Returns fitted model, best_params, training_time, peak_memory_mb
    """
    if n_trials is None:
        n_trials = config.OPTUNA_N_TRIALS

    print(f"\n  -> Optimising {name} ({n_trials} trials)...", end=' ', flush=True)

    tune_fn = _TUNE_FNS[name]
    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(lambda trial: tune_fn(trial, X_train, y_train),
                   n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score  = study.best_value
    print(f"Best CV F1={best_score:.4f}  params={best_params}")

    # Retrain on full training data with best params
    proc   = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / 1024 / 1024
    t0 = time.time()

    model = _BUILD_FNS[name](best_params)
    model.fit(X_train, y_train)

    train_time = time.time() - t0
    mem_after  = proc.memory_info().rss / 1024 / 1024
    peak_mem   = mem_after - mem_before

    return model, best_params, train_time, peak_mem


def train_all_models(X_train, y_train):
    """
    Train all 6 models with Optuna tuning.

    Returns
    -------
    models      : dict  {name: fitted_model}
    train_times : dict  {name: seconds}
    peak_mems   : dict  {name: MB}
    best_params : dict  {name: params_dict}
    """
    print("\n" + "=" * 70)
    print("  STEPS 6-7: MODEL TRAINING WITH OPTUNA HYPERPARAMETER TUNING")
    print("=" * 70)

    models      = {}
    train_times = {}
    peak_mems   = {}
    best_params = {}

    for name in config.MODEL_NAMES:
        model, params, t, mem = train_model_with_optuna(name, X_train, y_train)
        models[name]      = model
        best_params[name] = params
        train_times[name] = round(t, 3)
        peak_mems[name]   = round(mem, 2)

    print(f"\n[OK] All {len(models)} models trained successfully.")
    return models, train_times, peak_mems, best_params


def save_model(model, filename):
    """Save a trained model with joblib."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    path = os.path.join(config.MODEL_DIR, filename)
    joblib.dump(model, path)
    print(f"[OK] Model saved -> {path}")


def load_model(filename):
    """Load a saved model."""
    path = os.path.join(config.MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

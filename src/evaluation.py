"""
Model Evaluation Module  Step 8

Computes 8 metrics per model and builds the MCDM decision matrix.

Benefit criteria (higher = better):
    Accuracy, Precision, Recall (Sensitivity), F1_Score, ROC_AUC, Specificity

Cost criteria (lower = better):
    FPR (False Positive Rate), FNR (False Negative Rate)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Compute all 8 evaluation metrics.

    Returns
    -------
    dict with keys matching config.ALL_CRITERIA
    """
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred, zero_division=0)
    recall      = recall_score(y_true, y_pred, zero_division=0)   # Sensitivity / TPR
    f1          = f1_score(y_true, y_pred, zero_division=0)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0       # TNR
    fpr         = FP / (FP + TN) if (FP + TN) > 0 else 0.0       # cost
    fnr         = FN / (FN + TP) if (FN + TP) > 0 else 0.0       # cost

    roc_auc = (roc_auc_score(y_true, y_proba)
               if y_proba is not None else np.nan)

    return {
        'Accuracy'   : round(accuracy,    4),
        'Precision'  : round(precision,   4),
        'Recall'     : round(recall,      4),
        'F1_Score'   : round(f1,          4),
        'ROC_AUC'    : round(roc_auc,     4),
        'Specificity': round(specificity, 4),
        'FPR'        : round(fpr,         4),
        'FNR'        : round(fnr,         4),
        # extras for display
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
    }


def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate every model and return a clean decision matrix DataFrame.

    Parameters
    ----------
    models : dict  {model_name: fitted_model}

    Returns
    -------
    decision_matrix : pd.DataFrame  rows=models, cols=ALL_CRITERIA
    full_results    : pd.DataFrame  rows=models, cols=ALL_CRITERIA + confusion matrix cells
    """
    print("\n" + "=" * 70)
    print("  STEP 8: MODEL EVALUATION")
    print("=" * 70)

    rows = []
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = (model.predict_proba(X_test)[:, 1]
                   if hasattr(model, 'predict_proba') else None)

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics['Model'] = name
        rows.append(metrics)

        print(f"\n  {name}")
        print(f"    Acc={metrics['Accuracy']:.4f} | Prec={metrics['Precision']:.4f} | "
              f"Rec={metrics['Recall']:.4f} | F1={metrics['F1_Score']:.4f} | "
              f"AUC={metrics['ROC_AUC']:.4f}")
        print(f"    Spec={metrics['Specificity']:.4f} | FPR={metrics['FPR']:.4f} | "
              f"FNR={metrics['FNR']:.4f}")

    full_results = pd.DataFrame(rows).set_index('Model')

    # Decision matrix = only the 8 MCDM criteria
    decision_matrix = full_results[config.ALL_CRITERIA].copy()

    print("\n[OK] Decision Matrix (used for MCDM):")
    print(decision_matrix.round(4).to_string())

    return decision_matrix, full_results


def print_summary_table(decision_matrix):
    """Pretty-print the evaluation results."""
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY TABLE")
    print("=" * 70)
    print(decision_matrix.round(4).to_string())
    print("\n   Benefit criteria ((up) higher is better):", config.BENEFIT_CRITERIA)
    print("   Cost criteria    ((down) lower  is better):", config.COST_CRITERIA)

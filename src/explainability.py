"""
Explainability Module  Step 11

  Global: SHAP summary plot (beeswarm) for the final selected model
  Local : LIME explanations for 3 individual predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import shap
from lime.lime_tabular import LimeTabularExplainer


# ---------------------------------------------------------------------------
# SHAP  Global Explanation
# ---------------------------------------------------------------------------
def shap_global_explanation(model, X_train, X_test, feature_names, model_name,
                             save_dir=None):
    """
    Generate SHAP summary (beeswarm) plot for the final model.

    Works with tree-based models (TreeExplainer) and falls back to
    KernelExplainer for others (e.g. SVM, Logistic Regression).

    Returns
    -------
    shap_values : np.ndarray
    mean_abs_shap : pd.Series (feature importance)
    """
    print(f"\n  Computing SHAP values for {model_name}...")

    try:
        explainer   = shap.TreeExplainer(model,
                                         feature_perturbation='tree_path_dependent')
        shap_values = explainer.shap_values(X_test)
        X_plot      = X_test
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception as e:
        print(f"  TreeExplainer failed ({e}), falling back to KernelExplainer...")
        background  = shap.sample(X_train, 100, random_state=config.RANDOM_STATE)
        X_sample    = X_test.iloc[:200] if hasattr(X_test, 'iloc') else X_test[:200]
        predict_fn  = (model.predict_proba if hasattr(model, 'predict_proba')
                       else model.predict)
        explainer   = shap.KernelExplainer(
            lambda x: predict_fn(x)[:, 1], background
        )
        shap_values = explainer.shap_values(X_sample.values, nsamples=50)
        X_plot      = X_sample

    # Flatten shap_values if needed (3D -> 2D)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names
    ).sort_values(ascending=False)

    # --- Plot ---
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_plot,
                      feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title(f"SHAP Summary Plot  {model_name}", fontweight='bold', pad=15)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'shap_summary.png')
        plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  [OK] SHAP summary saved -> {path}")
    plt.close()

    print(f"  Top 5 features by SHAP importance:")
    for feat, val in mean_abs_shap.head(5).items():
        print(f"    {feat:<35} {val:.4f}")

    return shap_values, mean_abs_shap



# ---------------------------------------------------------------------------
# LIME  Local Explanation
# ---------------------------------------------------------------------------
def lime_local_explanation(model, X_train, X_test, y_test, feature_names,
                            model_name, save_dir=None, n_samples=3):
    """
    Explain n_samples individual predictions with LIME.

    Selects:
      - 1 employee who left     (y_test == 1)
      - 1 employee who stayed   (y_test == 0)
      - 1 borderline case       (probability closest to 0.5)
    """
    print(f"\n  Computing LIME explanations for {model_name}...")

    # Build LIME explainer
    explainer = LimeTabularExplainer(
        training_data    = X_train.values,
        feature_names    = feature_names,
        class_names      = ['Stayed', 'Left'],
        mode             = 'classification',
        random_state     = config.RANDOM_STATE
    )

    predict_fn = (model.predict_proba if hasattr(model, 'predict_proba')
                  else lambda x: np.column_stack([1 - model.predict(x),
                                                   model.predict(x)]))

    # Choose representative samples
    y_arr   = np.array(y_test)
    probas  = predict_fn(X_test.values)[:, 1]
    idx_pos = np.where(y_arr == 1)[0]
    idx_neg = np.where(y_arr == 0)[0]
    idx_borderline = np.argmin(np.abs(probas - 0.5))

    samples = {
        'Attrition=Yes (left)'     : idx_pos[0]  if len(idx_pos) > 0  else 0,
        'Attrition=No (stayed)'    : idx_neg[0]  if len(idx_neg) > 0  else 1,
        'Borderline (prob~0.5)'    : idx_borderline,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for label, idx in samples.items():
        exp = explainer.explain_instance(
            data_row       = X_test.values[idx],
            predict_fn     = predict_fn,
            num_features   = 10,
            num_samples    = 1000
        )
        fig = exp.as_pyplot_figure()
        fig.suptitle(f"LIME  {label}\n({model_name})", fontweight='bold')
        plt.tight_layout()

        if save_dir:
            safe_label = label.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '').replace('~', '')
            path = os.path.join(save_dir, f'lime_{safe_label}.png')
            plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            print(f"  [OK] LIME plot saved -> {path}")
        plt.close()

    print(f"  [OK] LIME explanations complete for {n_samples} samples.")

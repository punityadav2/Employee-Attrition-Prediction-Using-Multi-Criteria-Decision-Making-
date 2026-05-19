"""
Visualization Module  Step 12

Generates and saves all required plots:
  1.  Model performance bar chart (all metrics, all models)
  2.  Metrics heatmap
  3.  MCDM ranking comparison (WSM vs TOPSIS vs VIKOR vs Final)
  4.  SHAP summary plot        -> handled in explainability.py
  5.  ROC curves overlay (all 6 models)
  6.  Confusion matrices (all models)
  7.  Feature importance: SHAP vs final model's native importance
  [Bonus]
  8.  Green efficiency chart (F1 / time x memory)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.makedirs(config.FIGURES_DIR, exist_ok=True)


def _save(fig, filename):
    path = os.path.join(config.FIGURES_DIR, filename)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Bar chart  model performance comparison
# ---------------------------------------------------------------------------
def plot_model_comparison(decision_matrix: pd.DataFrame):
    """Grouped bar chart: each group = one metric, bars = models."""
    metrics  = config.BENEFIT_CRITERIA   # only show benefit metrics
    models   = decision_matrix.index.tolist()
    n_models = len(models)
    n_metrics = len(metrics)

    x     = np.arange(n_metrics)
    width = 0.12

    fig, ax = plt.subplots(figsize=(14, 6))
    colors  = config.COLOR_PALETTE

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [decision_matrix.loc[model, m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metrics, rotation=20, ha='right', fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison  All Metrics', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.4)
    fig.tight_layout()
    return _save(fig, 'model_comparison.png')


# ---------------------------------------------------------------------------
# 2. Heatmap of all metrics
# ---------------------------------------------------------------------------
def plot_metrics_heatmap(decision_matrix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    dm_T = decision_matrix[config.ALL_CRITERIA].T

    sns.heatmap(dm_T, annot=True, fmt='.4f', cmap='YlOrRd',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title('Performance Metrics Heatmap  All Models', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    plt.xticks(rotation=25, ha='right')
    fig.tight_layout()
    return _save(fig, 'metrics_heatmap.png')


# ---------------------------------------------------------------------------
# 3. MCDM ranking comparison
# ---------------------------------------------------------------------------
def plot_mcdm_rankings(rankings_df: pd.DataFrame):
    models = rankings_df.index.tolist()
    rank_cols = ['WSM_Rank', 'TOPSIS_Rank', 'VIKOR_Rank', 'Final_Rank']
    labels    = ['WSM', 'TOPSIS', 'VIKOR', 'Final (RPM)']

    x     = np.arange(len(models))
    width = 0.2
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (col, label, color) in enumerate(zip(rank_cols, labels, colors)):
        ax.bar(x + i * width, rankings_df[col], width, label=label,
               color=color, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=11)
    ax.set_ylabel('Rank (1 = Best)', fontsize=12)
    ax.set_title('MCDM Ranking Comparison  WSM vs TOPSIS vs VIKOR vs Final',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.invert_yaxis()    # rank 1 at top
    ax.grid(axis='y', alpha=0.4)
    fig.tight_layout()
    return _save(fig, 'mcdm_rankings.png')


# ---------------------------------------------------------------------------
# 4. ROC curves overlay
# ---------------------------------------------------------------------------
def plot_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = config.COLOR_PALETTE

    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name}  (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves  All 6 Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, 'roc_curves_all_models.png')


# ---------------------------------------------------------------------------
# 5. Confusion matrices (2x3 grid)
# ---------------------------------------------------------------------------
def plot_confusion_matrices(models, X_test, y_test):
    n = len(models)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4 + 1))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Stayed', 'Left'],
                    yticklabels=['Stayed', 'Left'],
                    ax=axes[i], cbar=False)
        axes[i].set_title(name, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Confusion Matrices  All Models', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    return _save(fig, 'confusion_matrices.png')


# ---------------------------------------------------------------------------
# 6. Feature importance: SHAP vs final model's native importance
# ---------------------------------------------------------------------------
def plot_feature_importance_comparison(mean_abs_shap: pd.Series,
                                       final_model, feature_names: list,
                                       model_name: str = 'Final Model'):
    """Side-by-side: SHAP importance vs the final model's native importance."""
    shap_top = mean_abs_shap.head(15)

    # Extract native importance depending on model type
    try:
        if hasattr(final_model, 'coef_'):
            # Linear model (Logistic Regression, SVM with linear kernel)
            native_fi = pd.Series(
                np.abs(final_model.coef_[0]),
                index=feature_names
            ).sort_values(ascending=False).head(15)
            native_label = f'{model_name} Coefficients\n(|coef| magnitude)'
            native_xlabel = '|Coefficient|'
            bar_color = '#2ecc71'
        elif hasattr(final_model, 'feature_importances_'):
            # Tree-based model
            native_fi = pd.Series(
                final_model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False).head(15)
            native_label = f'{model_name} Feature Importance\n(Gain)'
            native_xlabel = 'Importance Score'
            bar_color = '#e74c3c'
        else:
            native_fi = shap_top
            native_label = f'{model_name} (SHAP fallback)'
            native_xlabel = 'Mean |SHAP|'
            bar_color = '#9b59b6'
    except Exception:
        native_fi = shap_top
        native_label = f'{model_name} (SHAP fallback)'
        native_xlabel = 'Mean |SHAP|'
        bar_color = '#9b59b6'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    shap_top.sort_values().plot.barh(ax=ax1, color='#3498db', alpha=0.85)
    ax1.set_title('SHAP Feature Importance\n(mean |SHAP value|)',
                  fontweight='bold')
    ax1.set_xlabel('Mean |SHAP|')

    native_fi.sort_values().plot.barh(ax=ax2, color=bar_color, alpha=0.85)
    ax2.set_title(native_label, fontweight='bold')
    ax2.set_xlabel(native_xlabel)

    fig.suptitle(f'Feature Importance Comparison  (Final Model: {model_name})',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _save(fig, 'feature_importance_comparison.png')


# ---------------------------------------------------------------------------
# BONUS: Green Efficiency chart
# ---------------------------------------------------------------------------
def plot_green_efficiency(decision_matrix: pd.DataFrame,
                          train_times: dict, peak_mems: dict):
    """
    Green Efficiency = F1 / (training_time x peak_memory)
    """
    models = decision_matrix.index.tolist()
    eff    = []
    for m in models:
        f1   = decision_matrix.loc[m, 'F1_Score']
        t    = train_times.get(m, 1e-3)
        mem  = peak_mems.get(m, 1.0)
        eff.append(f1 / (t * mem + 1e-9))

    eff_series = pd.Series(eff, index=models).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [config.COLOR_PALETTE[i % len(config.COLOR_PALETTE)]
              for i in range(len(eff_series))]
    eff_series.plot.barh(ax=ax, color=colors, alpha=0.85)
    ax.set_xlabel('Green Efficiency  (F1 / time x memory)', fontsize=11)
    ax.set_title('Green Efficiency  Model Efficiency Comparison',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.4)
    fig.tight_layout()
    return _save(fig, 'green_efficiency.png')


# ---------------------------------------------------------------------------
# Generate ALL visualizations
# ---------------------------------------------------------------------------
def generate_all_visualizations(decision_matrix, rankings_df, models,
                                 X_test, y_test, mean_abs_shap,
                                 final_model, final_model_name,
                                 feature_names,
                                 train_times=None, peak_mems=None):
    print("\n" + "=" * 70)
    print("  STEP 12: GENERATING ALL VISUALIZATIONS")
    print("=" * 70)

    plot_model_comparison(decision_matrix)
    plot_metrics_heatmap(decision_matrix)
    plot_mcdm_rankings(rankings_df)
    plot_roc_curves(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_feature_importance_comparison(
        mean_abs_shap, final_model, feature_names, model_name=final_model_name
    )

    if train_times and peak_mems:
        plot_green_efficiency(decision_matrix, train_times, peak_mems)

    print(f"\n[OK] All visualizations saved to {config.FIGURES_DIR}")

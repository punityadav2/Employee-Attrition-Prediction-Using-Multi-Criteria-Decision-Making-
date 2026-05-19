"""
main.py  Complete End-to-End Pipeline
"Model Selection for Employee Attrition Prediction Using Multi-Criteria Decision Methods"

Run:
    python main.py

Steps executed:
  Step 1   Load IBM HR Attrition dataset
  Step 2   Feature Engineering (workload_ratio, tenure_level)
  Step 3   Preprocessing (encode, scale, split, ROS)
  Step 4   Baseline XGBoost
  Step 5   SHAP feature selection (top 15)
  Step 6   Train 6 models
  Step 7   Optuna hyperparameter optimisation
  Step 8   Evaluate all models (8 metrics, decision matrix)
  Step 9   MCDM: WSM + TOPSIS + VIKOR
  Step 10  RPM final model selection
  Step 11  SHAP global + LIME local explanations
  Step 12  Generate all visualizations
  Step 13  Save results CSV + print conclusions
"""

import os
import sys
import warnings
import joblib
import pandas as pd

warnings.filterwarnings('ignore')

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config
from src.data_loader        import load_attrition_data
from src.feature_engineering import create_all_features
from src.preprocessing      import full_preprocessing_pipeline
from src.model_training     import (train_baseline_xgboost,
                                    select_top_k_features_shap,
                                    train_all_models, save_model)
from src.evaluation         import evaluate_all_models, print_summary_table
from src.mcdm               import rank_position_method
from src.explainability     import shap_global_explanation, lime_local_explanation
from src.visualization      import generate_all_visualizations


def main():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  MODEL SELECTION FOR EMPLOYEE ATTRITION PREDICTION")
    print("  Using Multi-Criteria Decision Making (MCDM) Methods")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 1: Load Data
    # ------------------------------------------------------------------
    df_raw = load_attrition_data()

    # ------------------------------------------------------------------
    # STEP 2: Feature Engineering (BEFORE encoding)
    # ------------------------------------------------------------------
    df = create_all_features(df_raw)

    # ------------------------------------------------------------------
    # STEP 3: Preprocessing (encode, split, scale, ROS)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, encoders, scaler = \
        full_preprocessing_pipeline(df)

    feature_names = X_train.columns.tolist()
    print(f"\n  Features after preprocessing: {len(feature_names)}")

    # ------------------------------------------------------------------
    # STEP 4: Baseline XGBoost
    # ------------------------------------------------------------------
    baseline_xgb = train_baseline_xgboost(X_train, y_train)

    # ------------------------------------------------------------------
    # STEP 5: SHAP Feature Selection  top K=15
    # ------------------------------------------------------------------
    top_features, mean_abs_shap_baseline = select_top_k_features_shap(
        baseline_xgb, X_train
    )

    # Reduce to top-K features
    X_train_sel = X_train[top_features]
    X_test_sel  = X_test[top_features]
    print(f"\n  Dataset reduced to top {len(top_features)} SHAP features.")

    # ------------------------------------------------------------------
    # STEPS 6+7: Train 6 models with Optuna
    # ------------------------------------------------------------------
    models, train_times, peak_mems, best_params = train_all_models(
        X_train_sel, y_train
    )

    # ------------------------------------------------------------------
    # STEP 8: Evaluate all models -> decision matrix
    # ------------------------------------------------------------------
    decision_matrix, full_results = evaluate_all_models(
        models, X_test_sel, y_test
    )
    print_summary_table(decision_matrix)

    # ------------------------------------------------------------------
    # STEP 9+10: MCDM  WSM, TOPSIS, VIKOR -> RPM final ranking
    # ------------------------------------------------------------------
    rankings_df, final_model_name = rank_position_method(decision_matrix)

    # ------------------------------------------------------------------
    # STEP 11: Explainability (SHAP global + LIME local)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 11: EXPLAINABLE AI")
    print("=" * 70)

    best_model = models[final_model_name]

    shap_values, mean_abs_shap = shap_global_explanation(
        model        = best_model,
        X_train      = X_train_sel,
        X_test       = X_test_sel,
        feature_names= top_features,
        model_name   = final_model_name,
        save_dir     = config.FIGURES_DIR
    )

    lime_local_explanation(
        model         = best_model,
        X_train       = X_train_sel,
        X_test        = X_test_sel,
        y_test        = y_test,
        feature_names = top_features,
        model_name    = final_model_name,
        save_dir      = config.FIGURES_DIR
    )

    # ------------------------------------------------------------------
    # STEP 12: Visualizations
    # ------------------------------------------------------------------
    generate_all_visualizations(
        decision_matrix  = decision_matrix,
        rankings_df      = rankings_df,
        models           = models,
        X_test           = X_test_sel,
        y_test           = y_test,
        mean_abs_shap    = mean_abs_shap,
        final_model      = best_model,
        final_model_name = final_model_name,
        feature_names    = top_features,
        train_times      = train_times,
        peak_mems        = peak_mems,
    )

    # ------------------------------------------------------------------
    # STEP 13: Save Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 13: SAVING RESULTS")
    print("=" * 70)

    # Merge decision matrix + MCDM rankings
    results_csv = pd.concat([
        decision_matrix,
        rankings_df[['WSM_Score', 'TOPSIS_Score', 'VIKOR_Q',
                     'WSM_Rank', 'TOPSIS_Rank', 'VIKOR_Rank',
                     'RPM_Score', 'Final_Rank']]
    ], axis=1)
    results_csv['Train_Time_sec'] = pd.Series(train_times)
    results_csv['Peak_Mem_MB']    = pd.Series(peak_mems)
    results_csv['Best_Params']    = pd.Series(
        {k: str(v) for k, v in best_params.items()}
    )

    csv_path = os.path.join(config.REPORTS_DIR, 'results.csv')
    results_csv.to_csv(csv_path)
    print(f"[OK] Results saved -> {csv_path}")

    # Save best model + artifacts
    save_model(best_model, 'best_model.pkl')
    joblib.dump(encoders, os.path.join(config.MODEL_DIR, 'encoders.pkl'))
    joblib.dump(scaler,   os.path.join(config.MODEL_DIR, 'scaler.pkl'))
    joblib.dump(top_features, os.path.join(config.MODEL_DIR, 'top_features.pkl'))

    # ------------------------------------------------------------------
    # CONCLUSIONS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONCLUSIONS")
    print("=" * 70)
    print(f"\n  Dataset    : IBM HR Employee Attrition ({len(df_raw)} employees)")
    print(f"  Target     : Attrition  (Yes={int(y_test.sum())} left, "
          f"No={(y_test==0).sum()} stayed in test set)")
    print(f"  Features   : Top {len(top_features)} selected by SHAP out of {len(feature_names)}")

    print(f"\n  Model Rankings:")
    display_cols = ['WSM_Rank', 'TOPSIS_Rank', 'VIKOR_Rank', 'Final_Rank']
    print(rankings_df[display_cols].sort_values('Final_Rank').to_string())

    print(f"\n   BEST MODEL (by MCDM/RPM): {final_model_name}")
    bm_metrics = decision_matrix.loc[final_model_name]
    print(f"     Accuracy   : {bm_metrics['Accuracy']:.4f}")
    print(f"     Precision  : {bm_metrics['Precision']:.4f}")
    print(f"     Recall     : {bm_metrics['Recall']:.4f}")
    print(f"     F1-Score   : {bm_metrics['F1_Score']:.4f}")
    print(f"     ROC-AUC    : {bm_metrics['ROC_AUC']:.4f}")
    print(f"     Specificity: {bm_metrics['Specificity']:.4f}")
    print(f"     FPR        : {bm_metrics['FPR']:.4f}")
    print(f"     FNR        : {bm_metrics['FNR']:.4f}")

    print(f"\n  Top 5 Predictive Features (by SHAP):")
    for i, (feat, val) in enumerate(mean_abs_shap.head(5).items(), 1):
        print(f"    {i}. {feat:<35} SHAP={val:.4f}")

    print("\n  Pipeline complete. Run 'streamlit run app.py' for the dashboard.")
    print("=" * 70)


if __name__ == '__main__':
    main()

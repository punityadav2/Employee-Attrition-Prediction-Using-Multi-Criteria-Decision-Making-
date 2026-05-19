"""
Streamlit Dashboard — IBM HR Employee Attrition Predictor
"Model Selection for Employee Attrition Prediction Using Multi-Criteria Decision Methods"

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).parent)
sys.path.insert(0, ROOT)

import config
from src.feature_engineering import create_all_features
from src.preprocessing import encode_categorical

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition — MCDM Model Selector",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(30,58,95,0.3);
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.4rem 0; }
.hero p  { font-size: 1rem; opacity: 0.85; margin: 0; }

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 4px solid #2d6a9f;
    margin-bottom: 1rem;
}
.metric-card .label { font-size: 0.78rem; color: #888; text-transform: uppercase; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1e3a5f; }

.badge-attrition {
    background: #ff4444; color: white;
    padding: 0.4rem 1rem; border-radius: 20px;
    font-weight: 600; font-size: 1rem;
}
.badge-stay {
    background: #00b894; color: white;
    padding: 0.4rem 1rem; border-radius: 20px;
    font-weight: 600; font-size: 1rem;
}
.rank-table th { background: #1e3a5f; color: white; }
</style>
""", unsafe_allow_html=True)


# ── Auto-run pipeline if artifacts are missing ───────────────────────────────
def _artifacts_exist():
    model_path = os.path.join(config.MODEL_DIR, 'best_model.pkl')
    results_path = os.path.join(config.REPORTS_DIR, 'results.csv')
    return os.path.exists(model_path) and os.path.exists(results_path)


def _run_pipeline():
    """Run main.py in-process so Streamlit Cloud can build artifacts on first boot."""
    import subprocess
    main_path = os.path.join(ROOT, 'main.py')
    result = subprocess.run(
        [sys.executable, main_path],
        capture_output=True, text=True
    )
    return result.returncode == 0, result.stdout, result.stderr


if not _artifacts_exist():
    st.info("⚙️ **First-time setup detected.** Running the ML pipeline to build models and results. This takes ~8–12 minutes on first launch. Please wait…")
    with st.spinner("Running pipeline (training 6 models with Optuna + MCDM ranking)…"):
        success, stdout, stderr = _run_pipeline()
    if success:
        st.success("✅ Pipeline complete! Refreshing dashboard…")
        st.rerun()
    else:
        st.error("❌ Pipeline failed. Check the error below.")
        st.code(stderr[-3000:] if stderr else "No error output captured.")
        st.stop()


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model        = joblib.load(os.path.join(config.MODEL_DIR, 'best_model.pkl'))
        encoders     = joblib.load(os.path.join(config.MODEL_DIR, 'encoders.pkl'))
        scaler       = joblib.load(os.path.join(config.MODEL_DIR, 'scaler.pkl'))
        top_features = joblib.load(os.path.join(config.MODEL_DIR, 'top_features.pkl'))
        return model, encoders, scaler, top_features, True
    except Exception as e:
        return None, None, None, None, False


@st.cache_data
def load_results():
    path = os.path.join(config.REPORTS_DIR, 'results.csv')
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None



def preprocess_input(df, encoders, scaler, top_features):
    """Run the same preprocessing chain on new input."""
    from src.preprocessing import encode_target, drop_constant_columns, handle_missing_values
    df = df.copy()
    df = create_all_features(df)
    # Encode target if present
    if config.TARGET_COLUMN in df.columns:
        df = encode_target(df)
    df = drop_constant_columns(df)
    df = handle_missing_values(df)
    df, _ = encode_categorical(df, fit=False, encoders=encoders)

    # Drop target if still present
    df.drop(columns=[config.TARGET_COLUMN], errors='ignore', inplace=True)

    # Scale numerics
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[num_cols] = scaler.transform(df[num_cols])

    # Select top features (handle missing cols gracefully)
    missing_cols = [c for c in top_features if c not in df.columns]
    for c in missing_cols:
        df[c] = 0
    return df[top_features]


model, encoders, scaler, top_features, model_loaded = load_artifacts()
results_df = load_results()

# ── Sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/IBM_logo.svg/800px-IBM_logo.svg.png",
                 width=120)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "🏠  Home",
    "👤  Single Prediction",
    "📊  Batch Prediction",
    "🏆  MCDM Rankings",
    "📈  Model Performance",
])

if not model_loaded:
    st.error("⚠️ Models not found. Please run `python main.py` first to train all models.")

# ── HOME ──────────────────────────────────────────────────────────────────────
if page == "🏠  Home":
    st.markdown("""
    <div class="hero">
      <h1>🏆 HR Attrition Prediction</h1>
      <p>Model Selection Using Multi-Criteria Decision Making (MCDM) Methods<br>
         <em>WSM · TOPSIS · VIKOR · Rank Position Method</em></p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="label">Dataset</div>'
                    '<div class="value">IBM HR</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="label">Employees</div>'
                    '<div class="value">1,470</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="label">Models Compared</div>'
                    '<div class="value">6</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="label">MCDM Methods</div>'
                    '<div class="value">3 + RPM</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🔬 Methodology")
        st.markdown("""
        1. **Data Preprocessing** — encoding, scaling, stratified split
        2. **Feature Engineering** — `workload_ratio`, `tenure_level`
        3. **Class Balancing** — Random Oversampling (ROS)
        4. **Baseline XGBoost** — for SHAP importance
        5. **SHAP Feature Selection** — top 15 features
        6. **6 ML Models** — trained with Optuna (30 trials)
        7. **8-Metric Evaluation** — decision matrix
        8. **MCDM Ranking** — WSM, TOPSIS, VIKOR
        9. **RPM Selection** — best model chosen
        10. **SHAP + LIME** — explainability
        """)
    with col_r:
        st.markdown("### 🎯 Models Evaluated")
        st.markdown("""
        | # | Model |
        |---|-------|
        | 1 | Logistic Regression |
        | 2 | Decision Tree |
        | 3 | Random Forest |
        | 4 | SVM |
        | 5 | Gradient Boosting |
        | 6 | **XGBoost** |
        """)

    if model_loaded:
        model_name = type(model).__name__
        st.success(f"✅ Pipeline complete. Final MCDM-selected model: **{model_name}**")

    # Show SHAP image if exists
    shap_path = os.path.join(config.FIGURES_DIR, 'shap_summary.png')
    if os.path.exists(shap_path):
        st.markdown("### 🔍 SHAP Global Feature Importance")
        st.image(shap_path, use_container_width=True)


# ── SINGLE PREDICTION ─────────────────────────────────────────────────────────
elif page == "👤  Single Prediction":
    st.markdown('<div class="hero"><h1>👤 Single Employee Attrition Predictor</h1>'
                '<p>Enter employee details to predict attrition risk</p></div>',
                unsafe_allow_html=True)

    if not model_loaded:
        st.stop()

    with st.form("single_pred_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### 👤 Personal")
            age            = st.slider("Age", 18, 65, 35)
            gender         = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            distance       = st.slider("Distance From Home (km)", 1, 30, 5)
            education      = st.selectbox("Education Level",
                                          [1, 2, 3, 4, 5],
                                          format_func=lambda x: {1:"Below College",2:"College",
                                                                   3:"Bachelor",4:"Master",5:"Doctor"}[x])
            education_field = st.selectbox("Education Field",
                                           ["Life Sciences","Medical","Marketing",
                                            "Technical Degree","Human Resources","Other"])

        with c2:
            st.markdown("#### 💼 Job")
            department     = st.selectbox("Department", ["Sales","Research & Development","Human Resources"])
            job_role       = st.selectbox("Job Role",
                                          ["Sales Executive","Research Scientist","Laboratory Technician",
                                           "Manufacturing Director","Healthcare Representative",
                                           "Manager","Sales Representative","Research Director","Human Resources"])
            job_level      = st.selectbox("Job Level", [1, 2, 3, 4, 5])
            job_involve    = st.selectbox("Job Involvement", [1,2,3,4],
                                          format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
            job_satisf     = st.selectbox("Job Satisfaction", [1,2,3,4],
                                          format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
            business_travel= st.selectbox("Business Travel", ["Non-Travel","Travel_Rarely","Travel_Frequently"])
            over_time      = st.selectbox("OverTime", ["Yes", "No"])

        with c3:
            st.markdown("#### 📊 Compensation & Experience")
            monthly_income  = st.number_input("Monthly Income ($)", 1000, 20000, 5000, 500)
            monthly_rate    = st.number_input("Monthly Rate ($)", 2000, 27000, 14000, 500)
            daily_rate      = st.number_input("Daily Rate ($)", 100, 1500, 800, 50)
            hourly_rate     = st.number_input("Hourly Rate ($)", 30, 100, 65)
            percent_hike    = st.slider("Percent Salary Hike (%)", 11, 25, 15)
            stock_option    = st.selectbox("Stock Option Level", [0, 1, 2, 3])
            years_at_co     = st.slider("Years at Company", 0, 40, 5)
            total_working   = st.slider("Total Working Years", 0, 40, 10)
            years_role      = st.slider("Years in Current Role", 0, 18, 3)
            years_promo     = st.slider("Years Since Last Promotion", 0, 15, 2)
            years_manager   = st.slider("Years With Current Manager", 0, 17, 3)
            num_companies   = st.slider("Num Companies Worked", 0, 9, 2)
            training_times  = st.slider("Training Times Last Year", 0, 6, 3)
            env_satisf      = st.selectbox("Environment Satisfaction", [1,2,3,4],
                                           format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
            rel_satisf      = st.selectbox("Relationship Satisfaction", [1,2,3,4],
                                           format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
            worklife        = st.selectbox("Work Life Balance", [1,2,3,4],
                                           format_func=lambda x:{1:"Bad",2:"Good",3:"Better",4:"Best"}[x])
            perf_rating     = st.selectbox("Performance Rating", [1,2,3,4],
                                           format_func=lambda x:{1:"Low",2:"Good",3:"Excellent",4:"Outstanding"}[x])

        submitted = st.form_submit_button("🔮 Predict Attrition Risk", type="primary",
                                          use_container_width=True)

    if submitted:
        input_dict = {
            'Age': age, 'BusinessTravel': business_travel, 'DailyRate': daily_rate,
            'Department': department, 'DistanceFromHome': distance,
            'Education': education, 'EducationField': education_field,
            'EmployeeCount': 1, 'EmployeeNumber': 1, 'EnvironmentSatisfaction': env_satisf,
            'Gender': gender, 'HourlyRate': hourly_rate, 'JobInvolvement': job_involve,
            'JobLevel': job_level, 'JobRole': job_role, 'JobSatisfaction': job_satisf,
            'MaritalStatus': marital_status, 'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_rate, 'NumCompaniesWorked': num_companies,
            'Over18': 'Y', 'OverTime': over_time, 'PercentSalaryHike': percent_hike,
            'PerformanceRating': perf_rating, 'RelationshipSatisfaction': rel_satisf,
            'StandardHours': 80, 'StockOptionLevel': stock_option,
            'TotalWorkingYears': total_working, 'TrainingTimesLastYear': training_times,
            'WorkLifeBalance': worklife, 'YearsAtCompany': years_at_co,
            'YearsInCurrentRole': years_role, 'YearsSinceLastPromotion': years_promo,
            'YearsWithCurrManager': years_manager,
        }
        input_df = pd.DataFrame([input_dict])

        try:
            X_inp     = preprocess_input(input_df, encoders, scaler, top_features)
            pred      = model.predict(X_inp)[0]
            proba     = model.predict_proba(X_inp)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
            risk_pct  = proba[1] * 100

            st.markdown("---")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                if pred == 1:
                    st.markdown(f'<div style="text-align:center;padding:1.5rem;">'
                                f'<div class="badge-attrition">⚠️ HIGH ATTRITION RISK</div>'
                                f'<p style="font-size:2.5rem;font-weight:700;color:#ff4444;margin-top:1rem">'
                                f'{risk_pct:.1f}%</p>'
                                f'<p style="color:#888">Probability of leaving</p></div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="text-align:center;padding:1.5rem;">'
                                f'<div class="badge-stay">✅ LOW ATTRITION RISK</div>'
                                f'<p style="font-size:2.5rem;font-weight:700;color:#00b894;margin-top:1rem">'
                                f'{100-risk_pct:.1f}%</p>'
                                f'<p style="color:#888">Probability of staying</p></div>',
                                unsafe_allow_html=True)
            with col_b:
                st.metric("Staying Probability",   f"{proba[0]*100:.1f}%")
                st.metric("Leaving Probability",   f"{proba[1]*100:.1f}%")
                st.progress(float(proba[1]))
        except Exception as e:
            st.error(f"Prediction error: {e}")


# ── BATCH PREDICTION ──────────────────────────────────────────────────────────
elif page == "📊  Batch Prediction":
    st.markdown('<div class="hero"><h1>📊 Batch Attrition Prediction</h1>'
                '<p>Upload a CSV file to predict attrition for multiple employees</p></div>',
                unsafe_allow_html=True)

    if not model_loaded:
        st.stop()

    uploaded = st.file_uploader("Upload CSV (same format as IBM HR dataset)", type=['csv'])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.success(f"✅ {len(df_up)} employees loaded")
        with st.expander("Preview data"):
            st.dataframe(df_up.head())

        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Processing..."):
                try:
                    X_batch  = preprocess_input(df_up, encoders, scaler, top_features)
                    preds    = model.predict(X_batch)
                    probas   = model.predict_proba(X_batch) if hasattr(model, 'predict_proba') else None

                    out = df_up.copy() if 'EmployeeNumber' not in df_up else df_up[['EmployeeNumber']].copy()
                    out = pd.DataFrame()
                    if 'EmployeeNumber' in df_up.columns:
                        out['EmployeeNumber'] = df_up['EmployeeNumber'].values
                    out['Predicted_Attrition'] = ['Yes' if p == 1 else 'No' for p in preds]
                    if probas is not None:
                        out['Attrition_Probability'] = (probas[:, 1] * 100).round(1)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Employees", len(preds))
                    col2.metric("Likely to Leave",  int(preds.sum()),
                                f"{preds.mean()*100:.1f}%")
                    col3.metric("Likely to Stay",   int((preds==0).sum()),
                                f"{(preds==0).mean()*100:.1f}%")

                    st.dataframe(out, use_container_width=True, height=350)
                    csv_data = out.to_csv(index=False)
                    st.download_button("⬇️ Download Results", csv_data,
                                       "attrition_predictions.csv", "text/csv",
                                       type="primary")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("👆 Upload a CSV file to get started.")


# ── MCDM RANKINGS ─────────────────────────────────────────────────────────────
elif page == "🏆  MCDM Rankings":
    st.markdown('<div class="hero"><h1>🏆 MCDM Model Selection Rankings</h1>'
                '<p>WSM · TOPSIS · VIKOR · Rank Position Method</p></div>',
                unsafe_allow_html=True)

    if results_df is None:
        st.warning("⚠️ Results not found. Please run `python main.py` first.")
        st.stop()

    rank_cols = ['WSM_Rank', 'TOPSIS_Rank', 'VIKOR_Rank', 'Final_Rank']
    score_cols = ['WSM_Score', 'TOPSIS_Score', 'VIKOR_Q']

    # Summary ranking table
    st.markdown("### 📋 MCDM Ranking Summary")
    ranking_table = results_df[rank_cols + score_cols].sort_values('Final_Rank')
    st.dataframe(ranking_table.style.highlight_min(subset=['Final_Rank'], color='#d4efdf')
                                     .format("{:.4f}", subset=score_cols),
                 use_container_width=True)

    # Final winner
    winner = ranking_table.index[0]
    st.success(f"🏆 **Final Selected Model (RPM):** {winner}")

    st.markdown("---")

    # MCDM plot
    mcdm_img = os.path.join(config.FIGURES_DIR, 'mcdm_rankings.png')
    if os.path.exists(mcdm_img):
        st.markdown("### 📊 MCDM Ranking Comparison")
        st.image(mcdm_img, use_container_width=True)

    st.markdown("### ℹ️ MCDM Method Descriptions")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("**WSM** (Weighted Sum Method)\nNormalises each criterion and computes weighted sum. Higher = better.")
    with col_b:
        st.info("**TOPSIS**\nRanks by closeness to ideal solution. Higher closeness = better.")
    with col_c:
        st.info("**VIKOR**\nCompromise ranking using utility (S) and regret (R). Lower Q = better.")


# ── MODEL PERFORMANCE ─────────────────────────────────────────────────────────
elif page == "📈  Model Performance":
    st.markdown('<div class="hero"><h1>📈 Model Performance Dashboard</h1>'
                '<p>8-Metric Evaluation across all 6 classifiers</p></div>',
                unsafe_allow_html=True)

    if results_df is None:
        st.warning("⚠️ Results not found. Please run `python main.py` first.")
        st.stop()

    metric_cols = config.ALL_CRITERIA
    perf_table  = results_df[[c for c in metric_cols if c in results_df.columns]]

    st.markdown("### 📋 Performance Metrics Table")
    st.dataframe(
        perf_table.style.background_gradient(cmap='Blues',
                                              subset=config.BENEFIT_CRITERIA)
                        .background_gradient(cmap='Reds_r',
                                              subset=[c for c in config.COST_CRITERIA
                                                      if c in perf_table.columns])
                        .format("{:.4f}"),
        use_container_width=True
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Model Comparison", "Heatmap",
                                       "ROC Curves", "Feature Importance"])
    imgs = {
        'Model Comparison'    : 'model_comparison.png',
        'Heatmap'             : 'metrics_heatmap.png',
        'ROC Curves'          : 'roc_curves_all_models.png',
        'Feature Importance'  : 'feature_importance_comparison.png',
    }
    for tab, (label, fname) in zip([tab1, tab2, tab3, tab4], imgs.items()):
        with tab:
            path = os.path.join(config.FIGURES_DIR, fname)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.info(f"{label} chart not generated yet. Run `python main.py`.")

    # Confusion matrices
    cm_path = os.path.join(config.FIGURES_DIR, 'confusion_matrices.png')
    if os.path.exists(cm_path):
        st.markdown("### 🔢 Confusion Matrices — All Models")
        st.image(cm_path, use_container_width=True)


if __name__ == '__main__':
    pass

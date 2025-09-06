import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import split_features_target, create_preprocessor

@st.cache_resource
def load_model():
    model = joblib.load(r"C:\Users\User\OneDrive\Desktop\Employee_Attrition_Prediction\XGBoost_best.joblib")
    return model

model = load_model()

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

st.title("👩‍💼 Employee Attrition Prediction Dashboard")
st.markdown("Upload employee dataset and view attrition risk + explanations.")

uploaded_file = st.file_uploader("Upload HR CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Uploaded Data")
    st.dataframe(df.head())

    # Split features/target
    X, y = split_features_target(df) if "Attrition" in df.columns else (df, None)

    # Predictions
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    df_results = df.copy()
    df_results["Attrition_Prob"] = probs
    df_results["Attrition_Pred"] = preds

    st.subheader("🧾 Predictions")
    st.dataframe(df_results[["Attrition_Prob", "Attrition_Pred"]].head(20))

    st.subheader("🌍 Global Feature Importance (SHAP)")

    # Extract preprocessor + final classifier from pipeline
    preprocessor = model.named_steps["preproc"]
    clf = model.named_steps["clf"]

    # Transform features for SHAP
    X_transformed = preprocessor.transform(X)

    # Get feature names after preprocessing
    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(
        preprocessor.named_transformers_["cat"].feature_names_in_
    )
    num_features = preprocessor.named_transformers_["num"].feature_names_in_
    all_feature_names = list(num_features) + list(cat_features)

    # Use TreeExplainer for XGBoost/LightGBM/RandomForest
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_transformed)

    # Global SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_transformed, feature_names=all_feature_names, show=False)
    st.pyplot(fig)
    plt.close(fig)

    # Local explanation for one employee
    st.subheader("🔍 Local Explanation for One Employee")
    emp_idx = st.number_input(
        "Enter Employee Row Index",
        min_value=0,
        max_value=len(df_results) - 1,
        value=0
    )
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[emp_idx], show=False)
    st.pyplot(fig2)
    plt.close(fig2)

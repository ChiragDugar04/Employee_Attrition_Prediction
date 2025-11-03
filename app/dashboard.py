import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import split_features_target, create_preprocessor

@st.cache_resource
def load_model():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "XGBoost_best.joblib")
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        st.error("Please run `python -m src.models.tune` to train and save the model.")
        return None
        
    model = joblib.load(MODEL_PATH)
    return model


def get_feature_description(feature_name):
    descriptions = {
        "OverTime_No": "Working Overtime",
        "MonthlyIncome": "Monthly Income",
        "StockOptionLevel": "Stock Option Level",
        "NumCompaniesWorked": "Number of Past Companies",
        "DistanceFromHome": "Distance from Home",
        "BusinessTravel_Travel_Frequently": "Frequent Business Travel",
        "JobRole_Research Scientist": "Job Role (Research Scientist)",
        "Age": "Age",
    }
    return descriptions.get(feature_name, feature_name.replace("_", " "))


model = load_model()

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("👩‍💼 Employee Attrition Prediction Dashboard")
st.markdown("Upload employee dataset and view attrition risk + explanations.")

uploaded_file = st.file_uploader("Upload HR CSV file", type=["csv"])

if not model:
    st.stop()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Uploaded Data")
    st.dataframe(df.head())

    X, y = split_features_target(df) if "Attrition" in df.columns else (df, None)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    df_results = df.copy()
    df_results["Attrition_Prob"] = probs
    df_results["Attrition_Pred"] = preds

   
    st.subheader("📈 High-Level Summary")
    total_employees = len(df_results)
    at_risk_count = df_results['Attrition_Pred'].sum()
    at_risk_percent = (at_risk_count / total_employees) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", f"{total_employees}")
    col2.metric("High Attrition Risk", f"{at_risk_count} employees")
    col3.metric("Overall Attrition Rate", f"{at_risk_percent:.1f}%")
    
    
    st.subheader("🚨 High-Risk Employees (Top 20)")
    high_risk_df = df_results[df_results['Attrition_Pred'] == 1].sort_values(
        by="Attrition_Prob", ascending=False
    )
    cols_to_show = ["Department", "JobRole", "MonthlyIncome", "Age", "OverTime", "Attrition_Prob"]
    display_cols = [col for col in cols_to_show if col in high_risk_df.columns]
    st.dataframe(high_risk_df[display_cols].head(20))

    
    st.header("🔬 Model Explanations (SHAP)")
    preprocessor = model.named_steps["preproc"]
    clf = model.named_steps["clf"]
    X_transformed = preprocessor.transform(X)
    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out()
    num_features = preprocessor.named_transformers_["num"].feature_names_in_
    all_feature_names = list(num_features) + list(cat_features)
    X_transformed_df = pd.DataFrame(
        X_transformed, 
        columns=all_feature_names,
        index=X.index 
    )
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_transformed_df) 

    
    st.subheader("🌍 Key Drivers of Attrition (Company-Wide)")
    st.info("""
    **What this is:** This section shows the **Top 5 factors** that have the biggest
    impact on employee attrition across the *entire company*, according to the AI model.
    """)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_impact = pd.DataFrame(
        {'Feature': all_feature_names, 'Impact': mean_abs_shap}
    ).sort_values(by="Impact", ascending=False)

    st.markdown("#### Top 5 Most Impactful Factors:")
    
    
    cols = st.columns(5)
    top_5_features = feature_impact['Feature'].head(5).tolist()
    
    for i, feature in enumerate(top_5_features):
        with cols[i]:
            st.markdown(f"**#{i+1} Driver**")
            st.markdown(get_feature_description(feature))

    
    with st.expander("How to Read the Global SHAP Plot"):
        st.markdown("""
        **How to Read This Chart:**

        This chart shows the biggest factors that impact attrition risk across the company.

        * **Factors:** Each row is a factor (like `MonthlyIncome`).
        * **Colors:** Red means a *high value* for that factor (e.g., high income). Blue means a *low value* (e.g., low income).
        * **Impact on Risk:**
            * Dots on the **right** = **Increases Risk** (pushes employee to leave).
            * Dots on the **left** = **Decreases Risk** (helps retain employee).

        ---

        **Simple Example (looking at `MonthlyIncome`):**

        1.  We see a big cluster of **blue dots** (low income) on the **right side**.
        2.  This means: **Low income strongly increases the risk** of an employee leaving.

        **Another Example (looking at `OverTime_No`):**

        1.  The feature `OverTime_No` means "Is the employee *not* working overtime?"
        2.  The **red dots** (meaning "Yes, they are *not* working overtime") are far to the **left**.
        3.  This means: **Not working overtime strongly decreases the risk** of an employee leaving.
        """)

    
    with st.expander("View Global Feature Importance Plot"):
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X_transformed_df, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

   


    st.subheader("🔍 Individual Employee Risk Analysis")
    st.info("""
    **What this is:** This section explains *why* a specific employee is at risk.
    It highlights the top factors pushing their risk score up or down.
    """)

    high_risk_options = {}
    for idx in high_risk_df.index:
        emp_id = high_risk_df.loc[idx].get('EmployeeNumber', f'Row {idx}') 
        job_role = high_risk_df.loc[idx].get('JobRole', 'N/A')
        
        
        label = f"EmpID: {emp_id} - {job_role}"
        
       
        high_risk_options[label] = idx

    
    if not high_risk_options:
        st.warning("No high-risk employees were predicted.")
    else:
        selected_employee_label = st.selectbox(
            "Select a High-Risk Employee to Analyze",
            options=high_risk_options.keys()
        )
        
        
        emp_idx = high_risk_options[selected_employee_label]

        
        selected_emp_number = df.loc[emp_idx].get('EmployeeNumber', emp_idx)
        
        st.markdown(f"**Showing analysis for EmployeeID: {selected_emp_number} (Row {emp_idx})**")
    

        emp_shap_values = shap_values.values[emp_idx]
        emp_risk_factors = pd.DataFrame({
            'Feature': all_feature_names,
            'SHAP_Value': emp_shap_values
        }).sort_values(by="SHAP_Value", ascending=False)

       
        top_risk_factors = emp_risk_factors[emp_risk_factors['SHAP_Value'] > 0].head(3)
        top_retention_factors = emp_risk_factors[emp_risk_factors['SHAP_Value'] < 0].tail(3).iloc[::-1]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔴 Top 3 Risk Factors")
            for _, row in top_risk_factors.iterrows():
                st.error(f"**{get_feature_description(row['Feature'])}**")

        with col2:
            st.markdown("#### 🟢 Top 3 Retention Factors")
            for _, row in top_retention_factors.iterrows():
                st.success(f"**{get_feature_description(row['Feature'])}**")

        
        with st.expander("View Technical Waterfall Plot"):
            st.markdown("""
            **How to read this plot:**
            * Starts from the average prediction (`E[f(X)]`).
            * **Red bars** (like `OverTime_No = -0.0`) are factors *increasing* this employee's risk.
            * **Blue bars** (like `MonthlyIncome = 2.136`) are factors *decreasing* their risk.
            * The final value (`f(x)`) is their total risk score.
            """)
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[emp_idx], show=False)
            st.pyplot(fig2, use_container_width=True) 
            plt.close(fig2)
        
        st.markdown("**Original Data for this Employee:**")
        st.dataframe(df.iloc[emp_idx:emp_idx+1])
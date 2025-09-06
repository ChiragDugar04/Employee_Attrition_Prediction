# Employee Attrition Prediction & HR Analytics Dashboard


An end-to-end machine learning system to predict employee attrition and provide actionable HR insights using interpretable models, interactive visualizations, and a Streamlit dashboard.

---

## 🚀 Project Overview

Employee attrition is a critical challenge for organizations. This project predicts which employees are at risk of leaving and provides HR actionable insights using:

- **Machine Learning Models:** XGBoost, LightGBM, RandomForest, Logistic Regression
- **Explainability:** SHAP for global and local feature importance
- **Hyperparameter Optimization:** Optuna for tuning model performance
- **Interactive Dashboard:** Streamlit for uploading datasets, predicting attrition, and visualizing explanations

---

## 📊 Features

- **Predict Attrition:** Upload CSV employee data and predict the probability of leaving.
- **Global Feature Insights:** Identify key drivers of attrition (overtime, salary, promotion history, work-life balance).
- **Local Explanations:** Understand why the model predicts an individual employee as at risk.
- **Hyperparameter Tuning:** Optimized ML models for better recall and precision.
- **Threshold Calibration:** Adjustable probability threshold to balance precision and recall according to HR needs.

---

## 🛠️ Technologies Used

- **Python:** Core language for data processing and ML.
- **scikit-learn:** Data preprocessing, baseline ML models.
- **XGBoost / LightGBM:** Gradient boosting models for high accuracy.
- **Optuna:** Hyperparameter optimization.
- **SHAP:** Model explainability (global & local).
- **Streamlit:** Interactive dashboard for end users.
- **Pandas & NumPy:** Data handling.
- **Matplotlib & Seaborn:** Visualizations.

---

## 📁 Project Structure
Employee_Attrition_Prediction/
│
├─ app/
│ └─ dashboard.py # Streamlit dashboard
│
├─ src/
│ ├─ data/
│ │ └─ preprocess.py # Data preprocessing scripts
│ ├─ models/
│ │ └─ tune.py # Model training & hyperparameter tuning
│
├─ models/
│ └─ XGBoost_best.joblib # Trained model artifact
│
├─ requirements.txt
└─ README.md


---

## 🔧 Installation

1. Clone the repository:

```bash
git clone https://github.com/ChiragDugar04/Employee_Attrition_Prediction.git
cd Employee_Attrition_Prediction
```
2.Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```
3.Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚡ Running the Dashboard

Start the Streamlit dashboard from the project's root directory:

```bash
streamlit run -m app.dashboard
```
The dashboard will open in your browser, where you can:
   a.Upload an employee CSV dataset.
   b.View predicted attrition probabilities.
   c.Explore global and local SHAP explanations.
   d.Adjust the threshold for precision-recall tradeoff analysis.

---

### 🧩 Usage Example

1. **Prepare your data**: Ensure you have a CSV file with relevant employee features (e.g., `Age`, `JobRole`, `MonthlyIncome`, `Overtime`, etc.).  

2. **Upload the CSV**: Use the file uploader in the dashboard.  

3. **See predictions and explanations**:

   **Prediction Table Example:**

   | EmployeeNumber | Attrition_Prob | Attrition_Pred |
   |----------------|----------------|----------------|
   | 101            | 0.75           | 1              |
   | 102            | 0.12           | 0              |

   - **Global SHAP:** View summary plots to understand the overall importance of features.  
   - **Local SHAP:** Select a specific employee's row index to see a detailed waterfall plot explaining their individual attrition risk.  

---

### 📈 Model Performance

The following table summarizes the performance of the baseline models on a test set:

| Model         | Precision (Attrition=1) | Recall (Attrition=1) | F1-score | ROC-AUC |
|---------------|-------------------------|---------------------|----------|---------|
| LogisticReg   | 0.35                    | 0.64                | 0.45     | 0.80    |
| DecisionTree  | 0.31                    | 0.32                | 0.31     | 0.59    |
| RandomForest  | 0.67                    | 0.09                | 0.15     | 0.78    |
| XGBoost       | 0.62                    | 0.32                | 0.42     | 0.78    |
| LightGBM      | 0.59                    | 0.36                | 0.45     | 0.77    |

*Metrics reflect results after hyperparameter tuning and threshold calibration.*

---

### 💡 Future Enhancements

- Add **interactive filtering** by department, role, or tenure.  
- Deploy as a **cloud-hosted web application** for HR teams.  
- Integrate **real-time data updates** from HR databases.  
- Add **automated alerts or notifications** for high-risk employees.

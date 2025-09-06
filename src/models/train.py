import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Gradient boosting libs
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data.preprocess import load_data, split_features_target, create_preprocessor


def train_and_evaluate(model, X_train, X_test, y_train, y_test, name: str):
    """Train model, evaluate, and print results"""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n==== {name} ====")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    return model


def main():
    # Load data
    df = load_data("data/raw/HR-Employee-Attrition.csv")
    X, y = split_features_target(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Preprocessor
    preproc = create_preprocessor(X)

    # Models
    models = {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=(y.value_counts()[0] / y.value_counts()[1])  # handle imbalance
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            random_state=42,
            class_weight="balanced"
        ),
    }

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preproc", preproc), ("clf", clf)])
        trained_model = train_and_evaluate(pipe, X_train, X_test, y_train, y_test, name)
        joblib.dump(trained_model, f"{name}_pipeline.joblib")


if __name__ == "__main__":
    main()

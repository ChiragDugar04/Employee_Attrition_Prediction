import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

from src.data.preprocess import load_data, split_features_target, create_preprocessor


def tune_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    params = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [5, 10, None],
        "clf__min_samples_split": [2, 5, 10],
    }
    pipe = Pipeline(steps=[("preproc", create_preprocessor(X_train)), ("clf", rf)])
    search = RandomizedSearchCV(pipe, params, n_iter=5, scoring="roc_auc", cv=3, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def tune_xgboost(X_train, y_train):
    xgb = XGBClassifier(random_state=42, scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]))
    params = {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
    }
    pipe = Pipeline(steps=[("preproc", create_preprocessor(X_train)), ("clf", xgb)])
    search = RandomizedSearchCV(pipe, params, n_iter=5, scoring="roc_auc", cv=3, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def main():
    df = load_data("data/raw/HR-Employee-Attrition.csv")
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    print("🔹 Tuning Random Forest...")
    rf_best, rf_params = tune_random_forest(X_train, y_train)
    print("Best RF params:", rf_params)
    rf_preds = rf_best.predict(X_test)
    print(classification_report(y_test, rf_preds))
    print("ROC-AUC:", roc_auc_score(y_test, rf_best.predict_proba(X_test)[:, 1]))
    joblib.dump(rf_best, "RandomForest_best.joblib")

    print("\n🔹 Tuning XGBoost...")
    xgb_best, xgb_params = tune_xgboost(X_train, y_train)
    print("Best XGB params:", xgb_params)
    xgb_preds = xgb_best.predict(X_test)
    print(classification_report(y_test, xgb_preds))
    print("ROC-AUC:", roc_auc_score(y_test, xgb_best.predict_proba(X_test)[:, 1]))
    joblib.dump(xgb_best, "XGBoost_best.joblib")


if __name__ == "__main__":
    main()

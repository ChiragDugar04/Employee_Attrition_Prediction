import optuna
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data.preprocess import load_data, split_features_target, create_preprocessor


def objective_xgb(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        # Add min_child_weight
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10), 
        "scale_pos_weight": (y.value_counts()[0] / y.value_counts()[1]),
        "random_state": 42,
    }
    clf = XGBClassifier(**params)
    pipe = Pipeline([("preproc", create_preprocessor(X)), ("clf", clf)])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv)
    return scores.mean()


def objective_lgbm(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        # Add min_child_samples
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 20),
        "class_weight": "balanced",
        "random_state": 42,
    }
    clf = LGBMClassifier(**params)
    pipe = Pipeline([("preproc", create_preprocessor(X)), ("clf", clf)])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv)
    return scores.mean()



def main():
    df = load_data("data/raw/HR-Employee-Attrition.csv")
    X, y = split_features_target(df)

    # XGBoost tuning
    print("🔹 Tuning XGBoost with Optuna...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=30)
    print("Best XGB params:", study_xgb.best_params)

    best_xgb = XGBClassifier(**study_xgb.best_params, random_state=42)
    pipe_xgb = Pipeline([("preproc", create_preprocessor(X)), ("clf", best_xgb)])
    pipe_xgb.fit(X, y)
    joblib.dump(pipe_xgb, "XGBoost_optuna.joblib")

    # LightGBM tuning
    print("\n🔹 Tuning LightGBM with Optuna...")
    study_lgbm = optuna.create_study(direction="maximize")
    study_lgbm.optimize(lambda trial: objective_lgbm(trial, X, y), n_trials=30)
    print("Best LGBM params:", study_lgbm.best_params)

    best_lgbm = LGBMClassifier(**study_lgbm.best_params, random_state=42)
    pipe_lgbm = Pipeline([("preproc", create_preprocessor(X)), ("clf", best_lgbm)])
    pipe_lgbm.fit(X, y)
    joblib.dump(pipe_lgbm, "LightGBM_optuna.joblib")


if __name__ == "__main__":
    main()
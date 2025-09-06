import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from src.data.preprocess import load_data, split_features_target, create_preprocessor

df = load_data("data/raw/HR-Employee-Attrition.csv")
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load("XGBoost_optuna.joblib")

preproc = model.named_steps["preproc"]
clf = model.named_steps["clf"]

calibrated_clf = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
calibrated_clf.fit(preproc.transform(X_train), y_train)

preprocessed_X_test = preproc.transform(X_test)
calibrated_probs = calibrated_clf.predict_proba(preprocessed_X_test)[:, 1]

prec, rec, thresholds = precision_recall_curve(y_test, calibrated_probs)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, prec[:-1], label="Precision")
plt.plot(thresholds, rec[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall Tradeoff")
plt.legend()
plt.grid(True)
plt.show()
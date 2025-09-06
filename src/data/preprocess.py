import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

def load_data(path: str):
    df=pd.read_csv(path)
    if "EmployeeNumber" in df.columns:
        df = df.drop(columns=["EmployeeNumber"])
    return df
def split_features_target(df: pd.DataFrame, target_col="Attrition"):
    X = df.drop(columns=[target_col])
    y = df[target_col].map({"Yes": 1, "No": 0})  # encode target
    return X, y
def create_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_tf = Pipeline(steps=[("scaler", StandardScaler())])
    cat_tf = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_tf, num_cols),
            ("cat", cat_tf, cat_cols),
        ]
    )
    return preprocessor
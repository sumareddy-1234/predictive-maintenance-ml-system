import pandas as pd

from src.data_pipeline.ingestion import load_raw_data
from src.data_pipeline.feature_store import run_full_feature_pipeline

from src.data_pipeline.preprocessing import (
    split_data,
    scale_features
)

from src.models.baseline import (
    train_logistic_regression,
    train_decision_tree,
    evaluate_model
)


print("Loading dataset...")
df = load_raw_data("data/raw/ai4i2020.csv")

print("Running feature engineering...")
df = run_full_feature_pipeline(df)

print("Creating features and target...")

X = df.drop(
    columns=[
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]
)

y = df["Machine failure"]

print("\nFeatures used:")
print(X.columns.tolist())

print("\nSplitting data...")

X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    pd.concat([X, y], axis=1),
    "Machine failure"
)

print("Scaling features...")

X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
    X_train,
    X_val,
    X_test
)

print("\nTraining Logistic Regression...")

lr_model = train_logistic_regression(
    X_train_scaled,
    y_train
)

lr_results = evaluate_model(
    lr_model,
    X_val_scaled,
    y_val
)

print("\n===== Logistic Regression =====")
print(lr_results)

print("\nTraining Decision Tree...")

dt_model = train_decision_tree(
    X_train_scaled,
    y_train
)

dt_results = evaluate_model(
    dt_model,
    X_val_scaled,
    y_val
)

print("\n===== Decision Tree =====")
print(dt_results)
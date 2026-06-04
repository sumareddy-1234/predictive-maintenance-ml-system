import pandas as pd

from src.data_pipeline.ingestion import (
    load_raw_data
)

from src.data_pipeline.feature_store import (
    run_full_feature_pipeline
)

from src.data_pipeline.preprocessing import (
    split_data,
    scale_features
)

from src.models.ensemble import (
    train_voting_classifier,
    evaluate_ensemble_model
)


print("Loading dataset...")

df = load_raw_data(
    "data/raw/ai4i2020.csv"
)

print("Running feature engineering...")

df = run_full_feature_pipeline(df)

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

print("Splitting data...")

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

print("Training Ensemble Model...")

model = train_voting_classifier(
    X_train_scaled,
    y_train
)

results = evaluate_ensemble_model(
    model,
    X_val_scaled,
    y_val
)

print()
print("===== Ensemble Results =====")
print(results)
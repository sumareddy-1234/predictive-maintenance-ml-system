import pandas as pd

from src.data_pipeline.ingestion import load_raw_data

from src.data_pipeline.feature_store import (
    run_full_feature_pipeline
)

from src.data_pipeline.preprocessing import (
    split_data,
    scale_features
)

from src.models.advanced import (
    train_random_forest
)

from src.utils.model_io import (
    save_model,
    load_model,
    save_scaler,
    load_scaler,
    save_feature_list,
    load_feature_list
)


print("Loading dataset...")

df = load_raw_data(
    "data/raw/ai4i2020.csv"
)

print("Feature engineering...")

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

features = X.columns.tolist()

print("Splitting data...")

X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    pd.concat([X, y], axis=1),
    "Machine failure"
)

print("Scaling data...")

X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
    X_train,
    X_val,
    X_test
)

print("Training model...")

model = train_random_forest(
    X_train_scaled,
    y_train
)

print("Saving artifacts...")

save_model(
    model,
    "artifacts/random_forest.pkl"
)

save_scaler(
    scaler,
    "artifacts/scaler.pkl"
)

save_feature_list(
    features,
    "artifacts/features.pkl"
)

print("Loading artifacts...")

loaded_model = load_model(
    "artifacts/random_forest.pkl"
)

loaded_scaler = load_scaler(
    "artifacts/scaler.pkl"
)

loaded_features = load_feature_list(
    "artifacts/features.pkl"
)

print()
print("Model Type:")
print(type(loaded_model))

print()
print("Scaler Type:")
print(type(loaded_scaler))

print()
print("Number of Features:")
print(len(loaded_features))
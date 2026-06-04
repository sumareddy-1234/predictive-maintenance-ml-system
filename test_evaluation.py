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

from src.models.advanced import (
    tune_random_forest
)

from src.evaluation.metrics import (
    calculate_metrics,
    print_metrics,
    get_confusion_matrix
)


print("Loading dataset...")

df = load_raw_data(
    "data/raw/ai4i2020.csv"
)

print("Feature engineering...")

df = run_full_feature_pipeline(df)

feature_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "power_watts",
    "temp_delta_K",
    "tool_wear_torque_interaction",
    "tool_wear_rpm_interaction",
    "strain_index"
]

X = df[feature_columns]

y = df["Machine failure"]

print("Splitting data...")

X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    pd.concat([X, y], axis=1),
    "Machine failure"
)

print("Scaling data...")

X_train_scaled, X_val_scaled, X_test_scaled, scaler = (
    scale_features(
        X_train,
        X_val,
        X_test
    )
)

print("Training model...")

model, best_params, best_score = (
    tune_random_forest(
        X_train_scaled,
        y_train
    )
)

y_pred = model.predict(
    X_val_scaled
)

y_proba = model.predict_proba(
    X_val_scaled
)[:, 1]

results = calculate_metrics(
    y_val,
    y_pred,
    y_proba
)

print_metrics(results)

print("\nConfusion Matrix")

print(
    get_confusion_matrix(
        y_val,
        y_pred
    )
)
import numpy as np

from src.data_pipeline.ingestion import load_raw_data, get_data_summary
from src.data_pipeline.preprocessing import (
    encode_product_type,
    remove_identifier_columns,
    split_data,
    scale_features
)

from src.models.advanced import train_random_forest
from src.evaluation.metrics import compute_classification_report


DATA_PATH = "data/raw/ai4i2020.csv"
TARGET = "Machine failure"


def run_pipeline():

    print("\n🚀 Loading data...")
    df = load_raw_data(DATA_PATH)

    print("📊 Data summary:")
    summary = get_data_summary(df)
    print(summary)

    print("\n⚙️ Preprocessing data...")

    df = encode_product_type(df)
    df = remove_identifier_columns(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target_column=TARGET
    )

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )

    print("\n🤖 Training model...")

    model = train_random_forest(X_train_scaled, y_train)

    print("\n📊 Evaluating model...")

    val_preds = model.predict(X_val_scaled)
    val_proba = model.predict_proba(X_val_scaled)[:, 1]

    metrics = compute_classification_report(
        y_val,
        val_preds,
        val_proba
    )

    print("\n===== VALIDATION METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
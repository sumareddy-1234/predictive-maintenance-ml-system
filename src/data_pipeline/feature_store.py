import numpy as np
import pandas as pd

from src.data_pipeline.preprocessing import (
    encode_product_type,
    remove_identifier_columns
)

# -----------------------------
# 1. Power Feature
# -----------------------------
def compute_power_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rpm = df["Rotational speed [rpm]"]
    torque = df["Torque [Nm]"]

    df["power_watts"] = torque * ((2 * np.pi * rpm) / 60)
    df["power_watts"] = df["power_watts"].round(4)

    return df


# -----------------------------
# 2. Temperature Delta
# -----------------------------
def compute_temperature_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["temp_delta_K"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    ).round(4)

    return df


# -----------------------------
# 3. Tool Wear Interactions
# -----------------------------
def compute_tool_wear_interaction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tool_wear_torque_interaction"] = (
        df["Tool wear [min]"] * df["Torque [Nm]"]
    )

    df["tool_wear_rpm_interaction"] = (
        df["Tool wear [min]"] * df["Rotational speed [rpm]"]
    )

    return df


# -----------------------------
# 4. Strain Index
# -----------------------------
def compute_strain_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["strain_index"] = (
        df["Torque [Nm]"] * df["Tool wear [min]"]
    ) / (df["Rotational speed [rpm]"] + 1)

    df["strain_index"] = df["strain_index"].round(4)

    return df


# -----------------------------
# 5. Feature Matrix Builder
# -----------------------------
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_power_feature(df)
    df = compute_temperature_delta(df)
    df = compute_tool_wear_interaction(df)
    df = compute_strain_index(df)

    return df


# -----------------------------
# 6. Feature Selection
# -----------------------------
def select_features_by_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int = 10
) -> list:

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    importances = model.feature_importances_

    sorted_features = sorted(
        zip(X_train.columns, importances),
        key=lambda x: x[1],
        reverse=True
    )

    return [f[0] for f in sorted_features[:n_features]]


# -----------------------------
# 7. Full Feature Pipeline (IMPORTANT)
# -----------------------------
def run_full_feature_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df = encode_product_type(df)
    df = remove_identifier_columns(df)
    df = build_feature_matrix(df)

    return df
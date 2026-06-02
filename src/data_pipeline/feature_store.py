import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.data_pipeline.preprocessing import (
    encode_product_type,
    remove_identifier_columns
)


def compute_power_feature(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["power_watts"] = (
        df["Torque [Nm]"]
        * (2 * np.pi * df["Rotational speed [rpm]"] / 60)
    ).round(4)

    return df


def compute_temperature_delta(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["temp_delta_K"] = (
        df["Process temperature [K]"]
        - df["Air temperature [K]"]
    )

    return df


def compute_tool_wear_interaction(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["tool_wear_torque_interaction"] = (
        df["Tool wear [min]"]
        * df["Torque [Nm]"]
    )

    df["tool_wear_rpm_interaction"] = (
        df["Tool wear [min]"]
        * df["Rotational speed [rpm]"]
    )

    return df


def compute_strain_index(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["strain_index"] = (
        (
            df["Torque [Nm]"]
            * df["Tool wear [min]"]
        )
        /
        (
            df["Rotational speed [rpm]"] + 1
        )
    ).round(4)

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:

    df = compute_power_feature(df)

    df = compute_temperature_delta(df)

    df = compute_tool_wear_interaction(df)

    df = compute_strain_index(df)

    return df


def select_features_by_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int = 10
) -> list:

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    return (
        importance_df["feature"]
        .head(n_features)
        .tolist()
    )


def run_full_feature_pipeline(
    raw_df: pd.DataFrame
) -> pd.DataFrame:

    df = encode_product_type(raw_df)

    df = remove_identifier_columns(df)

    df = build_feature_matrix(df)

    return df
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1. Encoding
# -----------------------------
def encode_product_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {"L": 0, "M": 1, "H": 2}

    if "Type" in df.columns:
        df["Type"] = df["Type"].map(mapping)

    return df


# -----------------------------
# 2. REMOVE IDENTIFIERS (FIXED FOR ALL IMPORT STYLES)
# -----------------------------
def remove_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = ["UDI", "Product ID"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


# 🔥 ALIAS for pipeline compatibility (IMPORTANT FIX)
remove_identifier_and_leakage_columns = remove_identifier_columns


# -----------------------------
# 3. CLASS IMBALANCE INFO
# -----------------------------
def handle_class_imbalance_info(df: pd.DataFrame) -> dict:
    counts = df["Machine failure"].value_counts()

    majority = int(counts.max())
    minority = int(counts.min())

    ratio = round(majority / minority, 4)

    return {
        "majority_class_count": majority,
        "minority_class_count": minority,
        "imbalance_ratio": ratio,
        "recommended_strategy": "SMOTE"
    }


# -----------------------------
# 4. TRAIN / VAL / TEST SPLIT
# -----------------------------
def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(test_size + val_size),
        stratify=y,
        random_state=random_state
    )

    val_ratio = val_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# -----------------------------
# 5. SCALING
# -----------------------------
def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
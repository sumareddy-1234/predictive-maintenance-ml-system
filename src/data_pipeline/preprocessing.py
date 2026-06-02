import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def encode_product_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode Type column:
    L -> 0
    M -> 1
    H -> 2
    """

    df = df.copy()

    mapping = {
        "L": 0,
        "M": 1,
        "H": 2
    }

    df["Type"] = df["Type"].map(mapping)

    return df


def remove_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identifier columns.
    """

    df = df.copy()

    df = df.drop(
        columns=["UDI", "Product ID"],
        errors="ignore"
    )

    return df


def handle_class_imbalance_info(df: pd.DataFrame) -> dict:
    """
    Return imbalance statistics.
    """

    class_counts = df["Machine failure"].value_counts()

    majority_count = int(class_counts.max())
    minority_count = int(class_counts.min())

    imbalance_ratio = round(
        majority_count / minority_count,
        4
    )

    return {
        "majority_class_count": majority_count,
        "minority_class_count": minority_count,
        "imbalance_ratio": imbalance_ratio,
        "recommended_strategy": "SMOTE"
    }


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> tuple:
    """
    Train / Validation / Test split
    """

    X = df.drop(columns=[target_column])

    y = df[target_column]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    val_ratio = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        stratify=y_train_val,
        random_state=random_state
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple:
    """
    Standard scaling.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val)

    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )
import pandas as pd

EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw dataset and validate required columns.
    """

    df = pd.read_csv(filepath)

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate dataset summary statistics.
    """

    numeric_cols = df.select_dtypes(include="number").columns

    numeric_stats = {}

    for col in numeric_cols:
        numeric_stats[col] = {
            "mean": round(df[col].mean(), 4),
            "std": round(df[col].std(), 4),
            "min": round(df[col].min(), 4),
            "max": round(df[col].max(), 4)
        }

    missing_values = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if df[col].isna().sum() > 0
    }

    class_distribution = (
        df["Machine failure"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "missing_values": missing_values,
        "class_distribution": class_distribution,
        "numeric_stats": numeric_stats
    }
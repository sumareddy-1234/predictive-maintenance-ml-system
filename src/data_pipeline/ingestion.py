from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

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


def download_dataset():
    """
    Downloads AI4I dataset and saves it to data/raw/ai4i2020.csv
    """

    dataset = fetch_ucirepo(id=601)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    df = pd.concat([X, y], axis=1)

    # Add missing columns if UCI API version excludes them
    if "UDI" not in df.columns:
        df.insert(0, "UDI", range(1, len(df) + 1))

    if "Product ID" not in df.columns:
        df.insert(1, "Product ID", [f"P{i}" for i in range(1, len(df) + 1)])

    # Rename columns to match assignment exactly
    rename_map = {
        "Air temperature": "Air temperature [K]",
        "Process temperature": "Process temperature [K]",
        "Rotational speed": "Rotational speed [rpm]",
        "Torque": "Torque [Nm]",
        "Tool wear": "Tool wear [min]"
    }

    df = df.rename(columns=rename_map)

    # Reorder columns
    df = df[EXPECTED_COLUMNS]

    Path("data/raw").mkdir(parents=True, exist_ok=True)

    save_path = "data/raw/ai4i2020.csv"
    df.to_csv(save_path, index=False)

    print("Dataset saved successfully!")
    print("Shape:", df.shape)

    return df


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset and validate schema.
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
    Generate summary statistics required by assignment.
    """

    numeric_stats = {}

    numeric_columns = df.select_dtypes(include="number").columns

    for col in numeric_columns:
        numeric_stats[col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
        }

    missing_values = (
        df.isnull()
        .sum()
    )

    missing_values = {
        col: int(count)
        for col, count in missing_values.items()
        if count > 0
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


if __name__ == "__main__":
    download_dataset()
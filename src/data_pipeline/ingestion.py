from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path


def load_dataset():
    dataset = fetch_ucirepo(id=601)

    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    Path("data/raw").mkdir(parents=True, exist_ok=True)

    df.to_csv("data/raw/ai4i2020.csv", index=False)

    print("Dataset saved successfully!")
    print("Shape:", df.shape)

    return df


if __name__ == "__main__":
    load_dataset()
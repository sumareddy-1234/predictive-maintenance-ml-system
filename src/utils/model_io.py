import joblib
from pathlib import Path


def save_model(
    model,
    filepath: str
):

    Path(filepath).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        filepath
    )


def load_model(
    filepath: str
):

    return joblib.load(
        filepath
    )


def save_scaler(
    scaler,
    filepath: str
):

    Path(filepath).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        scaler,
        filepath
    )


def load_scaler(
    filepath: str
):

    return joblib.load(
        filepath
    )


def save_feature_list(
    features,
    filepath: str
):

    Path(filepath).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        features,
        filepath
    )


def load_feature_list(
    filepath: str
):

    return joblib.load(
        filepath
    )
import pandas as pd

from src.utils.model_io import (
    load_model,
    load_scaler,
    load_feature_list
)


def load_prediction_artifacts():

    model = load_model(
        "artifacts/random_forest.pkl"
    )

    scaler = load_scaler(
        "artifacts/scaler.pkl"
    )

    features = load_feature_list(
        "artifacts/features.pkl"
    )

    return model, scaler, features


def prepare_input_data(
    input_dict: dict,
    feature_list: list
):

    df = pd.DataFrame(
        [input_dict]
    )

    df = df[feature_list]

    return df


def predict_failure(
    input_dict: dict
):

    model, scaler, feature_list = (
        load_prediction_artifacts()
    )

    input_df = prepare_input_data(
        input_dict,
        feature_list
    )

    scaled_input = scaler.transform(
        input_df
    )

    prediction = model.predict(
        scaled_input
    )[0]

    probability = model.predict_proba(
        scaled_input
    )[0][1]

    return {
        "prediction": int(prediction),
        "failure_probability": round(
            float(probability),
            4
        )
    }
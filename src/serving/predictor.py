import numpy as np
import pandas as pd
import os


class PredictiveMaintenancePredictor:

    def __init__(self, model: object, scaler: object, feature_names: list, threshold: float = 0.5):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.threshold = threshold

    def preprocess_input(self, raw_input: dict) -> np.ndarray:
        """
        Validate and scale a single raw input dictionary.
        """
        missing = [f for f in self.feature_names if f not in raw_input]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Create DataFrame with columns in correct order
        df = pd.DataFrame([raw_input])
        df = df[self.feature_names]

        # Apply scaling
        scaled = self.scaler.transform(df)
        return scaled

    def predict(self, raw_input: dict) -> dict:
        """
        Predict failure risk for a single raw input dictionary.
        """
        X = self.preprocess_input(raw_input)
        
        # Get probability for positive class (failure)
        proba = float(self.model.predict_proba(X)[0][1])
        pred = bool(proba >= self.threshold)

        if proba < 0.3:
            risk = "LOW"
        elif proba < 0.6:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        return {
            "failure_predicted": pred,
            "failure_probability": round(proba, 4),
            "risk_level": risk,
            "threshold_used": self.threshold
        }

    def predict_batch(self, raw_inputs: list) -> list:
        """
        Predict failure risk for a list of raw input dictionaries in a vectorized way.
        """
        if not raw_inputs:
            return []

        # Validate that all inputs have all features
        for raw_input in raw_inputs:
            missing = [f for f in self.feature_names if f not in raw_input]
            if missing:
                raise ValueError(f"Missing features: {missing}")

        # Convert to DataFrame and align features
        df = pd.DataFrame(raw_inputs)
        df = df[self.feature_names]

        # Apply scaling
        X = self.scaler.transform(df)
        
        # Predict probabilities
        probs = self.model.predict_proba(X)[:, 1]

        results = []
        for p in probs:
            p_val = float(p)
            pred = bool(p_val >= self.threshold)

            if p_val < 0.3:
                risk = "LOW"
            elif p_val < 0.6:
                risk = "MEDIUM"
            else:
                risk = "HIGH"

            results.append({
                "failure_predicted": pred,
                "failure_probability": round(p_val, 4),
                "risk_level": risk,
                "threshold_used": self.threshold
            })

        return results


def predict_failure(raw_input: dict) -> dict:
    """
    Helper function for legacy test compatibility.
    Loads default random forest model artifacts and predicts.
    """
    from src.utils.model_io import load_model, load_scaler, load_feature_list
    
    # Try multiple common relative directory targets to find artifacts/
    paths = [
        "artifacts/random_forest.pkl",
        "../artifacts/random_forest.pkl",
        "../../artifacts/random_forest.pkl"
    ]
    model_path = None
    for p in paths:
        if os.path.exists(p):
            model_path = p
            break
            
    if model_path is None:
        # Construct fallback relative path from this file
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "artifacts", "random_forest.pkl")
        scaler_path = os.path.join(base_dir, "artifacts", "scaler.pkl")
        features_path = os.path.join(base_dir, "artifacts", "features.pkl")
    else:
        scaler_path = model_path.replace("random_forest.pkl", "scaler.pkl")
        features_path = model_path.replace("random_forest.pkl", "features.pkl")

    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    features = load_feature_list(features_path)

    predictor = PredictiveMaintenancePredictor(model, scaler, features)
    return predictor.predict(raw_input)
import os
import pickle
from typing import Any, List


# -----------------------------
# 1. Generic Save Function
# -----------------------------
def save_object(obj: Any, filepath: str):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


# -----------------------------
# 2. Generic Load Function
# -----------------------------
def load_object(filepath: str):

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "rb") as f:
        return pickle.load(f)


# -----------------------------
# 3. Model Save / Load
# -----------------------------
def save_model(model, filepath: str):

    save_object(model, filepath)


def load_model(filepath: str):

    return load_object(filepath)


# -----------------------------
# 4. Scaler Save / Load
# -----------------------------
def save_scaler(scaler, filepath: str):

    save_object(scaler, filepath)


def load_scaler(filepath: str):

    return load_object(filepath)


# -----------------------------
# 5. Feature List Save / Load
# -----------------------------
def save_feature_list(features: List[str], filepath: str):

    if not isinstance(features, list):
        raise ValueError("features must be a list of column names")

    save_object(features, filepath)


def load_feature_list(filepath: str):

    return load_object(filepath)


# -----------------------------
# 6. Artifact Manager (Optional Helper)
# -----------------------------
class ArtifactManager:

    def __init__(self, base_path: str = "artifacts"):

        self.base_path = base_path

        os.makedirs(base_path, exist_ok=True)

    def model_path(self, filename: str):
        return os.path.join(self.base_path, filename)

    def save(self, obj, filename: str):

        path = self.model_path(filename)
        save_object(obj, path)
        return path

    def load(self, filename: str):

        path = self.model_path(filename)
        return load_object(path)
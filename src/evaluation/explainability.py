import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def compute_permutation_importance(
    model: object,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance for features on validation set.
    """
    result = permutation_importance(
        model, X_val, y_val, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    
    df = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": np.round(result.importances_mean, 6),
        "importance_std": np.round(result.importances_std, 6)
    })
    
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def compute_shap_summary(
    model: object,
    X_sample: pd.DataFrame,
    model_type: str = "tree"
) -> dict:
    """
    Compute SHAP explainability summary for tree or linear models.
    """
    # Safety: sample max 500 rows
    if len(X_sample) > 500:
        X_sample = X_sample.sample(500, random_state=42)

    feature_names = list(X_sample.columns)

    # TREE MODELS
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # binary classification handling
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # extract expected_value safely
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            if len(explainer.expected_value) > 1:
                expected_value = float(explainer.expected_value[1])
            else:
                expected_value = float(explainer.expected_value[0])
        else:
            expected_value = float(explainer.expected_value)

    # LINEAR MODELS
    else:
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            expected_value = float(explainer.expected_value[0])
        else:
            expected_value = float(explainer.expected_value)


    shap_values = np.array(shap_values)

    # ensure correct shape handling
    if shap_values.ndim == 3:  # sometimes SHAP returns (n, features, classes)
        shap_values = shap_values[:, :, 1]

    # Mean absolute SHAP per feature rounded to 6 decimal places
    mean_abs_shap = {
        feature_names[i]: round(float(np.mean(np.abs(shap_values[:, i]))), 6)
        for i in range(len(feature_names))
    }

    return {
        "shap_values": shap_values,
        "expected_value": expected_value,
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs_shap
    }


def get_top_failure_drivers(
    shap_summary: dict,
    n_top: int = 5
) -> list:
    """
    Extract the top features driven failure from SHAP summary.
    """
    mean_abs_shap = shap_summary["mean_abs_shap"]

    sorted_features = sorted(
        mean_abs_shap.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [name for name, _ in sorted_features[:n_top]]
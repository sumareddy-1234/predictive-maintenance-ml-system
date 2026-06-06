import numpy as np
import pandas as pd


def compute_psi(
    reference_data: np.ndarray,
    current_data: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Population Stability Index (PSI) between reference and current distributions.
    PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))
    """
    reference_data = np.array(reference_data)
    current_data = np.array(current_data)

    # Use equal-frequency binning based on the reference data distribution
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference_data, percentiles)

    # Handle duplicate edges (e.g. constant features)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    ref_counts = np.histogram(reference_data, bins=bin_edges)[0]
    curr_counts = np.histogram(current_data, bins=bin_edges)[0]

    # Convert to percentages
    ref_perc = ref_counts / len(reference_data) if len(reference_data) > 0 else np.zeros_like(ref_counts)
    curr_perc = curr_counts / len(current_data) if len(current_data) > 0 else np.zeros_like(curr_counts)

    # Add small epsilon of 1e-4 to all percentages before computing log
    epsilon = 1e-4
    ref_perc = ref_perc + epsilon
    curr_perc = curr_perc + epsilon

    psi = np.sum((curr_perc - ref_perc) * np.log(curr_perc / ref_perc))

    return round(float(psi), 4)


def compute_feature_drift_report(
    X_reference: pd.DataFrame,
    X_current: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute feature-level drift report using PSI.
    """
    results = []

    for col in X_reference.columns:
        psi_value = compute_psi(
            X_reference[col].values,
            X_current[col].values
        )

        if psi_value < 0.1:
            status = "stable"
        elif psi_value < 0.25:
            status = "moderate_drift"
        else:
            status = "significant_drift"

        results.append({
            "feature": col,
            "psi": psi_value,
            "drift_status": status
        })

    df = pd.DataFrame(results)
    df = df.sort_values("psi", ascending=False).reset_index(drop=True)
    return df


def detect_prediction_drift(
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray
) -> dict:
    """
    Detect prediction drift using PSI and KL Divergence on predicted probabilities.
    """
    reference_predictions = np.array(reference_predictions)
    current_predictions = np.array(current_predictions)

    # Compute PSI on the predictions
    psi_value = compute_psi(reference_predictions, current_predictions)

    # Compute KL divergence with epsilon smoothing of 1e-4
    epsilon = 1e-4
    
    # Bin prediction probabilities using 10 equal-width bins on [0, 1] range
    bins = np.linspace(0, 1, 11)
    
    ref_hist, _ = np.histogram(reference_predictions, bins=bins)
    curr_hist, _ = np.histogram(current_predictions, bins=bins)

    ref_prob = ref_hist / len(reference_predictions) if len(reference_predictions) > 0 else np.zeros_like(ref_hist)
    curr_prob = curr_hist / len(current_predictions) if len(current_predictions) > 0 else np.zeros_like(curr_hist)

    # Add epsilon smoothing
    ref_prob += epsilon
    curr_prob += epsilon

    kl_div = np.sum(ref_prob * np.log(ref_prob / curr_prob))

    severity = "none"
    if psi_value >= 0.25:
        severity = "significant"
    elif psi_value >= 0.1:
        severity = "moderate"

    return {
        "psi": psi_value,
        "kl_divergence": round(float(kl_div), 4),
        "drift_detected": psi_value >= 0.1,
        "severity": severity
    }
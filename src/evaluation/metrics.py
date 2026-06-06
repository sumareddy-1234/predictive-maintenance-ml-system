import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix, hamming_loss
)


def calculate_metrics(y_true, y_pred, y_proba):
    return compute_classification_report(y_true, y_pred, y_proba)


def print_metrics(results: dict):
    print("\n===== METRICS =====")
    for k, v in results.items():
        print(f"{k}: {v}")


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> dict:
    """
    Compute standard classification metrics.
    Note: precision and recall are macro-averaged.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "average_precision": round(float(average_precision_score(y_true, y_proba)), 4),
        "matthews_corrcoef": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }


def compute_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list = None
) -> pd.DataFrame:
    """
    Analyze precision, recall, F1, and predicted positives across a range of thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pred_positives = int(np.sum(y_pred))

        rows.append({
            "threshold": round(float(t), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "predicted_positives": pred_positives
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("threshold", ascending=True).reset_index(drop=True)
    return df


def compute_cost_sensitive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: dict
) -> dict:
    """
    Compute total business cost and cost per sample based on outcomes.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = (
        tp * cost_matrix.get("TP", 0) +
        tn * cost_matrix.get("TN", 0) +
        fp * cost_matrix.get("FP", 0) +
        fn * cost_matrix.get("FN", 0)
    )

    cost_per_sample = total_cost / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_cost": float(total_cost),
        "cost_per_sample": round(float(cost_per_sample), 4)
    }


def compute_multilabel_metrics(
    y_true_multilabel: np.ndarray,
    y_pred_multilabel: np.ndarray
) -> dict:
    """
    Compute metrics for the multi-label failure mode classification task.
    """
    labels = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    per_label_f1 = {}
    
    for i, label in enumerate(labels):
        f1 = f1_score(y_true_multilabel[:, i], y_pred_multilabel[:, i], zero_division=0)
        per_label_f1[label] = round(float(f1), 4)

    h_loss = hamming_loss(y_true_multilabel, y_pred_multilabel)
    subset_acc = accuracy_score(y_true_multilabel, y_pred_multilabel)
    macro_f1 = f1_score(y_true_multilabel, y_pred_multilabel, average="macro", zero_division=0)

    return {
        "hamming_loss": round(float(h_loss), 4),
        "subset_accuracy": round(float(subset_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_label_f1": per_label_f1
    }
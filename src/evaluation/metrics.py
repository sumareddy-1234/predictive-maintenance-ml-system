import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def calculate_metrics(
    y_true,
    y_pred,
    y_proba
):

    return {
        "accuracy": round(
            accuracy_score(y_true, y_pred),
            4
        ),

        "precision": round(
            precision_score(y_true, y_pred),
            4
        ),

        "recall": round(
            recall_score(y_true, y_pred),
            4
        ),

        "f1_score": round(
            f1_score(y_true, y_pred),
            4
        ),

        "roc_auc": round(
            roc_auc_score(y_true, y_proba),
            4
        )
    }


def get_confusion_matrix(
    y_true,
    y_pred
):

    return confusion_matrix(
        y_true,
        y_pred
    )


def print_metrics(results):

    print("\n===== Evaluation Metrics =====")

    for key, value in results.items():

        print(
            f"{key}: {value}"
        )
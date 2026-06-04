import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def train_logistic_regression(
    X_train,
    y_train,
    class_weight="balanced",
    random_state=42
):

    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    return model


def train_decision_tree(
    X_train,
    y_train,
    max_depth=5,
    random_state=42
):

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state
    )

    model.fit(X_train, y_train)

    return model


def get_baseline_predictions(
    model,
    X,
    threshold=0.5
):

    y_proba = model.predict_proba(X)[:, 1]

    y_pred = (
        y_proba >= threshold
    ).astype(int)

    return {
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def evaluate_model(
    model,
    X,
    y
):

    predictions = get_baseline_predictions(
        model,
        X
    )

    y_pred = predictions["y_pred"]
    y_proba = predictions["y_proba"]

    return {
        "accuracy": round(
            accuracy_score(y, y_pred),
            4
        ),
        "precision": round(
            precision_score(y, y_pred),
            4
        ),
        "recall": round(
            recall_score(y, y_pred),
            4
        ),
        "f1_score": round(
            f1_score(y, y_pred),
            4
        ),
        "roc_auc": round(
            roc_auc_score(y, y_proba),
            4
        )
    }
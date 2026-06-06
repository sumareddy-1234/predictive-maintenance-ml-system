import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# -------------------------
# 1. Logistic Regression
# -------------------------
def train_logistic_regression(
    X_train,
    y_train,
    class_weight="balanced",
    random_state=42
):

    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_state,
        solver="liblinear"
    )

    model.fit(X_train, y_train)

    return model


# -------------------------
# 2. Decision Tree
# -------------------------
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


# -------------------------
# 3. Predictions (STRICT FORMAT)
# -------------------------
def get_baseline_predictions(
    model,
    X,
    threshold=0.5
):

    y_proba = model.predict_proba(X)[:, 1]

    y_pred = (y_proba >= threshold).astype(int)

    return {
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    from src.evaluation.metrics import compute_classification_report
    return compute_classification_report(y, y_pred, y_proba)
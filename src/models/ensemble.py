import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# REQUIRED BY YOUR PIPELINE (FIXES YOUR ERROR)
def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth=None,
    random_state: int = 42
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_voting_classifier(X_train, y_train, random_state: int = 42):
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state)

    model = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
        voting="soft"
    )

    model.fit(X_train, y_train)
    return model


def train_stacking_classifier(X_train, y_train, random_state: int = 42):
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state)

    meta = LogisticRegression(max_iter=1000, random_state=random_state)

    model = StackingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
        final_estimator=meta,
        cv=5,
        passthrough=False
    )

    model.fit(X_train, y_train)
    return model


def evaluate_ensemble_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    from src.evaluation.metrics import compute_classification_report
    return compute_classification_report(y, y_pred, y_proba)



def compare_ensemble_models(models: dict, X_val: np.ndarray, y_val: np.ndarray):
    rows = []

    for name, model in models.items():
        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]

        rows.append({
            "model_name": name,
            "accuracy": round(float(accuracy_score(y_val, preds)), 4),
            "precision": round(float(precision_score(y_val, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_val, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_val, preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_val, proba)), 4)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    return df
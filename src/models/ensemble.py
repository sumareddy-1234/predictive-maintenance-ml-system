from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def train_voting_classifier(
    X_train,
    y_train
):

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    dt = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("dt", dt),
            ("rf", rf)
        ],
        voting="soft"
    )

    model.fit(X_train, y_train)

    return model


def evaluate_ensemble_model(
    model,
    X,
    y
):

    y_pred = model.predict(X)

    y_proba = model.predict_proba(X)[:, 1]

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
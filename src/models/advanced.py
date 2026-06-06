import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier


def train_random_forest(X_train, y_train, n_estimators=200, max_depth=None, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def tune_hyperparameters(model_name, X_train, y_train, cv=5, random_state=42):

    if model_name == "random_forest":
        model = RandomForestClassifier(class_weight="balanced", random_state=random_state)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }

    elif model_name == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=random_state)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6],
            "subsample": [0.7, 0.8, 1.0]
        }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=20,
        scoring="f1",
        cv=cv,
        random_state=random_state
    )

    if model_name == "gradient_boosting":
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        search.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        search.fit(X_train, y_train)

    return {
        "best_params": search.best_params_,
        "best_score": round(search.best_score_, 4),
        "best_estimator": search.best_estimator_
    }


def tune_random_forest(X_train, y_train):
    res = tune_hyperparameters("random_forest", X_train, y_train)
    return res["best_estimator"], res["best_params"], res["best_score"]



def train_multilabel_classifier(X_train, y_multilabel_train, random_state=42):
    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state
        )
    )
    model.fit(X_train, y_multilabel_train)
    return model


def predict_failure_modes(multilabel_model, X):
    preds = np.array(multilabel_model.predict(X))
    probs = np.array([
        est.predict_proba(X)[:, 1] for est in multilabel_model.estimators_
    ]).T

    return {
        "predictions": preds,
        "failure_mode_names": ["TWF", "HDF", "PWF", "OSF", "RNF"],
        "probabilities": probs
    }
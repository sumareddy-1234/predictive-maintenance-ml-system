from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_random_forest(
    X_train,
    y_train,
    n_estimators=200,
    max_depth=10,
    random_state=42
):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def tune_random_forest(
    X_train,
    y_train
):

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_
    )
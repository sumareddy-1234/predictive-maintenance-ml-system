import unittest
import numpy as np
import pandas as pd
from src.models.baseline import (
    train_logistic_regression,
    train_decision_tree,
    get_baseline_predictions
)
from src.models.advanced import (
    train_random_forest,
    train_gradient_boosting,
    tune_hyperparameters,
    train_multilabel_classifier,
    predict_failure_modes
)
from src.models.ensemble import (
    train_voting_classifier,
    train_stacking_classifier,
    compare_ensemble_models
)


class TestModels(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.normal(0, 1, (100, 5))
        self.y_train = np.random.choice([0, 1], 100, p=[0.9, 0.1])
        
        self.X_val = np.random.normal(0, 1, (20, 5))
        self.y_val = np.random.choice([0, 1], 20, p=[0.9, 0.1])

    def test_train_logistic_regression(self):
        model = train_logistic_regression(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        preds = get_baseline_predictions(model, self.X_val)
        self.assertEqual(len(preds["y_pred"]), 20)
        self.assertEqual(len(preds["y_proba"]), 20)

    def test_train_decision_tree(self):
        model = train_decision_tree(self.X_train, self.y_train)
        self.assertIsNotNone(model)

    def test_train_random_forest(self):
        model = train_random_forest(self.X_train, self.y_train, n_estimators=10)
        self.assertIsNotNone(model)

    def test_train_gradient_boosting(self):
        model = train_gradient_boosting(self.X_train, self.y_train, n_estimators=10)
        self.assertIsNotNone(model)

    def test_tune_hyperparameters(self):
        # cv=2 and n_iter=2 to keep testing fast
        from sklearn.model_selection import RandomizedSearchCV
        
        # We override standard CV to test the logic quickly
        res = tune_hyperparameters("random_forest", self.X_train, self.y_train, cv=2, random_state=42)
        self.assertIn("best_params", res)
        self.assertIn("best_score", res)
        self.assertIn("best_estimator", res)

    def test_train_voting_classifier(self):
        voting = train_voting_classifier(self.X_train, self.y_train)
        self.assertIsNotNone(voting)

    def test_train_stacking_classifier(self):
        stacking = train_stacking_classifier(self.X_train, self.y_train)
        self.assertIsNotNone(stacking)

    def test_compare_ensemble_models(self):
        voting = train_voting_classifier(self.X_train, self.y_train)
        comparison = compare_ensemble_models({"voting": voting}, self.X_val, self.y_val)
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(list(comparison.columns), ["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"])
        self.assertEqual(comparison.iloc[0]["model_name"], "voting")

    def test_multilabel_classification(self):
        y_multi_train = np.random.choice([0, 1], (100, 5))
        multilabel_model = train_multilabel_classifier(self.X_train, y_multi_train)
        res = predict_failure_modes(multilabel_model, self.X_val)
        self.assertEqual(res["predictions"].shape, (20, 5))
        self.assertEqual(res["probabilities"].shape, (20, 5))
        self.assertEqual(res["failure_mode_names"], ["TWF", "HDF", "PWF", "OSF", "RNF"])


if __name__ == "__main__":
    unittest.main()

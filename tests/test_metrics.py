import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.evaluation.metrics import (
    compute_classification_report,
    compute_threshold_analysis,
    compute_cost_sensitive_metrics,
    compute_multilabel_metrics
)
from src.evaluation.explainability import (
    compute_permutation_importance,
    compute_shap_summary,
    get_top_failure_drivers
)


class TestMetricsAndExplainability(unittest.TestCase):

    def test_compute_classification_report(self):
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])
        
        report = compute_classification_report(y_true, y_pred, y_proba)
        self.assertIn("accuracy", report)
        self.assertIn("precision", report)
        self.assertIn("recall", report)
        self.assertIn("matthews_corrcoef", report)
        self.assertEqual(report["confusion_matrix"], [[3, 1], [0, 2]])

    def test_compute_threshold_analysis(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])
        df = compute_threshold_analysis(y_true, y_proba, thresholds=[0.3, 0.5])
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["threshold"]), [0.3, 0.5])
        self.assertEqual(list(df["predicted_positives"]), [3, 2])

    def test_compute_cost_sensitive_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1]) # TN, FP, FN, TP
        cost_matrix = {"TP": -500, "TN": 0, "FP": 200, "FN": 5000}
        
        res = compute_cost_sensitive_metrics(y_true, y_pred, cost_matrix)
        # tp=1 (-500), tn=1 (0), fp=1 (200), fn=1 (5000)
        # total_cost = -500 + 0 + 200 + 5000 = 4700
        # cost_per_sample = 4700 / 4 = 1175.0
        self.assertEqual(res["total_cost"], 4700.0)
        self.assertEqual(res["cost_per_sample"], 1175.0)

    def test_compute_multilabel_metrics(self):
        y_true = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        y_pred = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0]])
        res = compute_multilabel_metrics(y_true, y_pred)

        self.assertIn("hamming_loss", res)
        self.assertIn("subset_accuracy", res)
        self.assertIn("macro_f1", res)
        self.assertIn("per_label_f1", res)

    def test_explainability_permutation(self):
        X = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0], "feat2": [4.0, 3.0, 2.0, 1.0]})
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression()
        model.fit(X, y)
        
        perm = compute_permutation_importance(model, X, y, n_repeats=2, random_state=42)
        self.assertEqual(len(perm), 2)
        self.assertIn("feature", perm.columns)
        self.assertIn("importance_mean", perm.columns)

    def test_explainability_shap(self):
        X = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0], "feat2": [4.0, 3.0, 2.0, 1.0]})
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression()
        model.fit(X, y)
        
        shap_res = compute_shap_summary(model, X, model_type="linear")
        self.assertEqual(shap_res["shap_values"].shape, (4, 2))
        self.assertIn("mean_abs_shap", shap_res)
        
        top = get_top_failure_drivers(shap_res, n_top=1)
        self.assertEqual(len(top), 1)


if __name__ == "__main__":
    unittest.main()

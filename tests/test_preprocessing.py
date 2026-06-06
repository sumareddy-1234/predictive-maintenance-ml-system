import unittest
import pandas as pd
import numpy as np
from src.data_pipeline.preprocessing import (
    encode_product_type,
    remove_identifier_columns,
    handle_class_imbalance_info,
    split_data,
    scale_features
)


class TestPreprocessing(unittest.TestCase):

    def test_encode_product_type(self):
        df = pd.DataFrame({"Type": ["L", "M", "H"]})
        df_encoded = encode_product_type(df)
        self.assertEqual(list(df_encoded["Type"]), [0, 1, 2])

    def test_remove_identifier_columns(self):
        df = pd.DataFrame({"UDI": [1], "Product ID": ["L47333"], "Type": ["L"]})
        df_cleaned = remove_identifier_columns(df)
        self.assertNotIn("UDI", df_cleaned.columns)
        self.assertNotIn("Product ID", df_cleaned.columns)
        self.assertIn("Type", df_cleaned.columns)

    def test_handle_class_imbalance_info(self):
        df = pd.DataFrame({"Machine failure": [0, 0, 0, 0, 1]})
        res = handle_class_imbalance_info(df)
        self.assertEqual(res["majority_class_count"], 4)
        self.assertEqual(res["minority_class_count"], 1)
        self.assertEqual(res["imbalance_ratio"], 4.0)
        self.assertIn(res["recommended_strategy"], ["SMOTE", "class_weight", "oversampling"])

    def test_split_data(self):
        df = pd.DataFrame({
            "feature1": range(100),
            "Machine failure": [0] * 90 + [1] * 10
        })
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df, "Machine failure", test_size=0.2, val_size=0.1, random_state=42
        )
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), 100)
        self.assertEqual(len(y_train) + len(y_val) + len(y_test), 100)
        self.assertTrue(65 <= len(X_train) <= 75)
        self.assertTrue(8 <= len(X_val) <= 12)
        self.assertTrue(18 <= len(X_test) <= 22)

        self.assertEqual(list(y_train).count(1), 7)
        self.assertEqual(list(y_val).count(1), 1)
        self.assertEqual(list(y_test).count(1), 2)

    def test_scale_features(self):
        X_train = pd.DataFrame({"feat": [1.0, 2.0, 3.0]})
        X_val = pd.DataFrame({"feat": [2.0]})
        X_test = pd.DataFrame({"feat": [3.0]})
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
        self.assertEqual(X_train_scaled.shape, (3, 1))
        self.assertEqual(X_val_scaled.shape, (1, 1))
        self.assertEqual(X_test_scaled.shape, (1, 1))
        self.assertAlmostEqual(X_train_scaled[0, 0], -1.22474487)
        self.assertAlmostEqual(X_train_scaled[1, 0], 0.0)


if __name__ == "__main__":
    unittest.main()

import unittest
import pandas as pd
import numpy as np
from src.data_pipeline.feature_store import (
    compute_power_feature,
    compute_temperature_delta,
    compute_tool_wear_interaction,
    compute_strain_index,
    build_feature_matrix,
    select_features_by_importance
)


class TestFeatureStore(unittest.TestCase):

    def test_compute_power_feature(self):
        # Power = Torque * (2 * pi * RPM / 60)
        # For Torque = 40 Nm, RPM = 1500 rpm:
        # Power = 40 * (2 * pi * 1500 / 60) = 40 * (50 * pi) = 2000 * pi = 6283.1853
        df = pd.DataFrame({
            "Torque [Nm]": [40.0],
            "Rotational speed [rpm]": [1500]
        })
        df_feat = compute_power_feature(df)
        self.assertIn("power_watts", df_feat.columns)
        self.assertAlmostEqual(df_feat["power_watts"].iloc[0], 6283.1853, places=3)

    def test_compute_temperature_delta(self):
        df = pd.DataFrame({
            "Process temperature [K]": [310.5],
            "Air temperature [K]": [300.2]
        })
        df_feat = compute_temperature_delta(df)
        self.assertIn("temp_delta_K", df_feat.columns)
        self.assertAlmostEqual(df_feat["temp_delta_K"].iloc[0], 10.3, places=3)

    def test_compute_tool_wear_interaction(self):
        df = pd.DataFrame({
            "Tool wear [min]": [10.0],
            "Torque [Nm]": [40.0],
            "Rotational speed [rpm]": [1500.0]
        })
        df_feat = compute_tool_wear_interaction(df)
        self.assertIn("tool_wear_torque_interaction", df_feat.columns)
        self.assertIn("tool_wear_rpm_interaction", df_feat.columns)
        self.assertEqual(df_feat["tool_wear_torque_interaction"].iloc[0], 400.0)
        self.assertEqual(df_feat["tool_wear_rpm_interaction"].iloc[0], 15000.0)

    def test_compute_strain_index(self):
        # strain_index = (Torque * Tool wear) / (RPM + 1)
        # (40 * 10) / (1499 + 1) = 400 / 1500 = 0.2667
        df = pd.DataFrame({
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [10.0],
            "Rotational speed [rpm]": [1499]
        })
        df_feat = compute_strain_index(df)
        self.assertIn("strain_index", df_feat.columns)
        self.assertAlmostEqual(df_feat["strain_index"].iloc[0], 0.2667, places=4)

    def test_build_feature_matrix(self):
        df = pd.DataFrame({
            "Torque [Nm]": [40.0],
            "Rotational speed [rpm]": [1500],
            "Process temperature [K]": [310.0],
            "Air temperature [K]": [300.0],
            "Tool wear [min]": [10.0]
        })
        df_feat = build_feature_matrix(df)
        self.assertIn("power_watts", df_feat.columns)
        self.assertIn("temp_delta_K", df_feat.columns)
        self.assertIn("tool_wear_torque_interaction", df_feat.columns)
        self.assertIn("tool_wear_rpm_interaction", df_feat.columns)
        self.assertIn("strain_index", df_feat.columns)

    def test_select_features_by_importance(self):
        np.random.seed(42)
        X = pd.DataFrame({
            "important_feat": np.random.normal(0, 1, 100) + np.array([0, 2] * 50),
            "unimportant_feat": np.random.normal(0, 1, 100)
        })
        y = pd.Series([0, 1] * 50)
        selected = select_features_by_importance(X, y, n_features=1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0], "important_feat")


if __name__ == "__main__":
    unittest.main()

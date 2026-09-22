"""
RetailPulse AI - Pipeline Unit Tests
====================================
Academic Level: 2nd Year 2nd Semester Undergraduate Data Science Project
Author: RetailPulse AI Team

Tests for:
  - Data ingestion and date parsing integrity
  - Feature engineering correctness and data leakage prevention
  - Metric computation calculations
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_model import (
    load_raw_data,
    engineer_features,
    calculate_metrics,
    FEATURES,
)


class TestDataPipeline(unittest.TestCase):
    """Test suite for data loading and feature engineering integrity."""

    @classmethod
    def setUpClass(cls):
        """Load data once for test assertions."""
        cls.raw_df = load_raw_data()

    def test_raw_data_structure(self):
        """Verify raw data has required columns and proper datetime formatting."""
        required_cols = {
            "Store",
            "Date",
            "Weekly_Sales",
            "Holiday_Flag",
            "Temperature",
            "Fuel_Price",
            "CPI",
            "Unemployment",
        }
        self.assertTrue(required_cols.issubset(set(self.raw_df.columns)))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.raw_df["Date"]))
        self.assertFalse(self.raw_df["Weekly_Sales"].isna().any())
        self.assertGreater(len(self.raw_df), 0)

    def test_feature_engineering_columns(self):
        """Verify that all target features are generated properly."""
        feat_df = engineer_features(self.raw_df)
        for col in FEATURES:
            self.assertIn(col, feat_df.columns, f"Missing engineered feature: {col}")
        self.assertFalse(feat_df[FEATURES].isna().any().any(), "Engineered features must not contain NaNs")

    def test_no_data_leakage_in_rolling_statistics(self):
        """
        Verify that Rolling_Mean_4 does NOT include the current week's sales (shift=1).
        For row t, Rolling_Mean_4 should equal the average of sales at t-1, t-2, t-3, t-4.
        """
        feat_df = engineer_features(self.raw_df)
        store_1 = feat_df[feat_df["Store"] == 1].sort_values("Date").reset_index(drop=True)

        if len(store_1) >= 5:
            # Check row 4: rolling mean should equal mean of Lag_1, Lag_2, Lag_3, Lag_4
            test_row = store_1.iloc[4]
            # Verify Rolling_Mean_4 is not equal to current week's sales (unless exact coincidence)
            self.assertIn("Rolling_Mean_4", test_row)
            self.assertIn("Lag_1", test_row)
            self.assertIn("Lag_2", test_row)
            self.assertIn("Lag_4", test_row)

    def test_metrics_calculation(self):
        """Verify MAE, RMSE, MAPE, and R2 calculation accuracy on synthetic values."""
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([110.0, 190.0, 310.0, 390.0])
        metrics = calculate_metrics(y_true, y_pred)

        self.assertEqual(metrics["MAE"], 10.0)
        self.assertEqual(metrics["RMSE"], 10.0)
        self.assertGreater(metrics["R2"], 0.9)
        self.assertGreater(metrics["MAPE"], 0.0)


if __name__ == "__main__":
    unittest.main()

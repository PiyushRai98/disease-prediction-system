"""
Tests for the data loading module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    load_diabetes,
    load_dataset,
    load_all_datasets,
)


class TestDataLoader:
    """Test cases for data loading functions."""

    def test_load_diabetes(self):
        """Test diabetes dataset loading."""
        df = load_diabetes()
        assert isinstance(df, pd.DataFrame)
        assert "target" in df.columns
        assert len(df) > 0
        assert df["target"].isin([0, 1]).all()

    def test_load_diabetes_features(self):
        """Test diabetes dataset has expected features."""
        df = load_diabetes()
        expected_features = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]
        for feature in expected_features:
            assert feature in df.columns, f"Missing feature: {feature}"

    def test_load_dataset_invalid_name(self):
        """Test that invalid dataset name raises ValueError."""
        with pytest.raises(ValueError):
            load_dataset("invalid_dataset")

    def test_load_dataset_diabetes(self):
        """Test loading diabetes through the generic loader."""
        df = load_dataset("diabetes")
        assert isinstance(df, pd.DataFrame)
        assert "target" in df.columns

    def test_dataset_no_null_target(self):
        """Test that target column has no null values."""
        df = load_diabetes()
        assert df["target"].isnull().sum() == 0

    def test_dataset_binary_target(self):
        """Test target is binary (0 or 1)."""
        df = load_diabetes()
        unique_values = set(df["target"].unique())
        assert unique_values.issubset({0, 1})

    def test_dataset_shape(self):
        """Test dataset has reasonable shape."""
        df = load_diabetes()
        assert df.shape[0] > 100  # At least 100 samples
        assert df.shape[1] > 2  # At least 2 columns (features + target)


class TestLoadAllDatasets:
    """Test cases for loading all datasets."""

    def test_returns_dict(self):
        """Test that load_all_datasets returns a dictionary."""
        # Only test with diabetes since it doesn't require internet
        datasets = {"diabetes": load_dataset("diabetes")}
        assert isinstance(datasets, dict)
        assert "diabetes" in datasets

    def test_dict_values_are_dataframes(self):
        """Test that all values are DataFrames."""
        datasets = {"diabetes": load_dataset("diabetes")}
        for name, df in datasets.items():
            assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"

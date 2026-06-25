"""
Tests for the preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    PreprocessingPipeline,
    detect_missing_values,
    detect_outliers_iqr,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n) * 10 + 50,
            "feature3": np.random.choice(["A", "B", "C"], n),
            "target": np.random.choice([0, 1], n),
        }
    )


@pytest.fixture
def sample_with_missing():
    """Create a DataFrame with missing values."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "target": np.random.choice([0, 1], n),
        }
    )
    # Introduce missing values
    df.loc[0:5, "feature1"] = np.nan
    df.loc[10:15, "feature2"] = np.nan
    return df


class TestPreprocessingPipeline:
    """Test cases for the preprocessing pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initializes with correct defaults."""
        pipeline = PreprocessingPipeline()
        assert pipeline.scale_features is True
        assert pipeline.handle_outliers is True
        assert pipeline.outlier_method == "iqr"
        assert pipeline.impute_strategy == "median"

    def test_fit_transform_returns_correct_types(self, sample_dataframe):
        """Test that fit_transform returns numpy arrays."""
        pipeline = PreprocessingPipeline()
        X_train, X_test, y_train, y_test = pipeline.fit_transform(
            sample_dataframe
        )

        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

    def test_fit_transform_shape(self, sample_dataframe):
        """Test output shapes are correct."""
        pipeline = PreprocessingPipeline()
        X_train, X_test, y_train, y_test = pipeline.fit_transform(
            sample_dataframe
        )

        total = len(X_train) + len(X_test)
        assert total <= len(sample_dataframe)
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

    def test_fit_transform_no_missing(self, sample_with_missing):
        """Test that output has no missing values after preprocessing."""
        pipeline = PreprocessingPipeline()
        X_train, X_test, y_train, y_test = pipeline.fit_transform(
            sample_with_missing
        )

        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

    def test_pipeline_report(self, sample_dataframe):
        """Test that preprocessing report is generated."""
        pipeline = PreprocessingPipeline()
        pipeline.fit_transform(sample_dataframe)
        report = pipeline.get_report()

        assert "original_shape" in report
        assert "final_shape" in report
        assert "train_shape" in report
        assert "test_shape" in report

    def test_no_scaling(self, sample_dataframe):
        """Test pipeline without scaling."""
        pipeline = PreprocessingPipeline(scale_features=False)
        X_train, X_test, y_train, y_test = pipeline.fit_transform(
            sample_dataframe
        )
        assert pipeline.scaler is None

    def test_stratified_split(self, sample_dataframe):
        """Test that train/test split maintains class proportions."""
        pipeline = PreprocessingPipeline()
        X_train, X_test, y_train, y_test = pipeline.fit_transform(
            sample_dataframe
        )

        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        # Ratios should be similar (within 10%)
        assert abs(train_ratio - test_ratio) < 0.1


class TestDetectMissingValues:
    """Test cases for missing value detection."""

    def test_no_missing(self):
        """Test with no missing values."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        report = detect_missing_values(df)
        assert report.empty

    def test_with_missing(self):
        """Test detection of missing values."""
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan]})
        report = detect_missing_values(df)
        assert len(report) == 2
        assert "Missing Count" in report.columns
        assert "Missing Percent" in report.columns


class TestDetectOutliers:
    """Test cases for outlier detection."""

    def test_no_outliers(self):
        """Test with no outliers."""
        df = pd.DataFrame({"a": np.ones(100)})
        report = detect_outliers_iqr(df)
        assert report["Outlier Count"].sum() == 0

    def test_with_outliers(self):
        """Test detection of outliers."""
        np.random.seed(42)
        data = np.random.randn(100)
        data[0] = 100  # Clear outlier
        df = pd.DataFrame({"a": data})
        report = detect_outliers_iqr(df)
        assert report["Outlier Count"].iloc[0] > 0

    def test_report_columns(self):
        """Test report has correct columns."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        report = detect_outliers_iqr(df)
        expected_cols = [
            "Column",
            "Lower Bound",
            "Upper Bound",
            "Outlier Count",
            "Outlier Percent",
        ]
        for col in expected_cols:
            assert col in report.columns

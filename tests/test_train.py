"""
Tests for the model training module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import get_model_instance, train_model


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.choice([0, 1], 100)
    return X_train, y_train


class TestGetModelInstance:
    """Test cases for model instance creation."""

    def test_logistic_regression(self):
        """Test Logistic Regression instantiation."""
        model = get_model_instance("Logistic Regression")
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_svm(self):
        """Test SVM instantiation."""
        model = get_model_instance("SVM")
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_random_forest(self):
        """Test Random Forest instantiation."""
        model = get_model_instance("Random Forest")
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_xgboost(self):
        """Test XGBoost instantiation."""
        model = get_model_instance("XGBoost")
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_invalid_model_name(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError):
            get_model_instance("InvalidModel")


class TestTrainModel:
    """Test cases for model training."""

    def test_train_logistic_regression(self, sample_data):
        """Test training Logistic Regression without tuning."""
        X_train, y_train = sample_data
        model, info = train_model(
            "Logistic Regression", X_train, y_train, tune_hyperparameters=False
        )
        assert model is not None
        assert "cv_mean" in info
        assert "cv_std" in info
        assert 0 <= info["cv_mean"] <= 1

    def test_train_random_forest(self, sample_data):
        """Test training Random Forest without tuning."""
        X_train, y_train = sample_data
        model, info = train_model(
            "Random Forest", X_train, y_train, tune_hyperparameters=False
        )
        assert model is not None
        assert hasattr(model, "feature_importances_")

    def test_model_can_predict(self, sample_data):
        """Test trained model can make predictions."""
        X_train, y_train = sample_data
        model, _ = train_model(
            "Logistic Regression", X_train, y_train, tune_hyperparameters=False
        )

        X_test = np.random.randn(10, 5)
        predictions = model.predict(X_test)
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_model_predict_proba(self, sample_data):
        """Test trained model can output probabilities."""
        X_train, y_train = sample_data
        model, _ = train_model(
            "Random Forest", X_train, y_train, tune_hyperparameters=False
        )

        X_test = np.random.randn(10, 5)
        probas = model.predict_proba(X_test)
        assert probas.shape == (10, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_training_info_structure(self, sample_data):
        """Test training info contains expected keys."""
        X_train, y_train = sample_data
        _, info = train_model(
            "Logistic Regression", X_train, y_train, tune_hyperparameters=False
        )

        assert "model_name" in info
        assert "cv_scores" in info
        assert "cv_mean" in info
        assert "cv_std" in info
        assert info["model_name"] == "Logistic Regression"

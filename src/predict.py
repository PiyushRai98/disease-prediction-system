"""
Prediction Module for Disease Prediction System.
Provides prediction interface for single samples and batch predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from src.config import MODELS_DIR, DATASETS
from src.utils import (
    logger,
    load_model,
    get_risk_category,
    get_risk_color,
)
from src.preprocessing import PreprocessingPipeline


class DiseasePredictor:
    """
    Production-ready prediction interface for all disease models.
    Handles loading models, preprocessing inputs, and generating predictions.
    """

    def __init__(self, dataset_name: str, model_name: str = "random_forest"):
        """
        Initialize the predictor.

        Args:
            dataset_name: Name of the disease dataset ('heart', 'diabetes', 'cancer').
            model_name: Name of the model file (without extension).
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model = None
        self.pipeline: Optional[PreprocessingPipeline] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model from disk."""
        model_path = MODELS_DIR / self.dataset_name / f"{self.model_name}.pkl"
        try:
            self.model = load_model(model_path)
            logger.info(
                f"Loaded model: {self.model_name} for {self.dataset_name}"
            )
        except FileNotFoundError:
            logger.error(f"Model not found: {model_path}")
            raise

    def predict_single(
        self, features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Make a prediction for a single patient.

        Args:
            features: Dictionary of feature name to value.

        Returns:
            Dictionary with prediction, probability, and risk assessment.
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Convert to DataFrame for consistent processing
        input_df = pd.DataFrame([features])

        # Get prediction and probability
        prediction = int(self.model.predict(input_df)[0])
        probability = float(self.model.predict_proba(input_df)[0][1])

        # Determine risk category
        risk_category = get_risk_category(probability)
        risk_color = get_risk_color(risk_category)

        # Get disease-specific labels
        dataset_config = DATASETS[self.dataset_name]
        diagnosis = (
            dataset_config["positive_label"]
            if prediction == 1
            else dataset_config["negative_label"]
        )

        result = {
            "prediction": prediction,
            "diagnosis": diagnosis,
            "probability": round(probability, 4),
            "confidence": round(max(probability, 1 - probability) * 100, 2),
            "risk_category": risk_category,
            "risk_color": risk_color,
            "risk_percentage": round(probability * 100, 2),
            "features_used": features,
        }

        logger.info(
            f"Prediction for {self.dataset_name}: "
            f"{diagnosis} ({probability:.4f})"
        )

        return result

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for a batch of patients.

        Args:
            df: DataFrame with patient features.

        Returns:
            DataFrame with predictions and risk assessments.
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)[:, 1]

        dataset_config = DATASETS[self.dataset_name]

        results_df = df.copy()
        results_df["Prediction"] = predictions
        results_df["Probability"] = probabilities.round(4)
        results_df["Risk Percentage"] = (probabilities * 100).round(2)
        results_df["Risk Category"] = [
            get_risk_category(p) for p in probabilities
        ]
        results_df["Diagnosis"] = [
            dataset_config["positive_label"]
            if p == 1
            else dataset_config["negative_label"]
            for p in predictions
        ]

        logger.info(
            f"Batch prediction complete: {len(df)} samples, "
            f"{predictions.sum()} positive"
        )

        return results_df

    def get_feature_importance_for_prediction(
        self, features: Dict[str, float]
    ) -> Optional[Dict[str, float]]:
        """
        Get feature contributions for a specific prediction.

        Args:
            features: Dictionary of feature values.

        Returns:
            Dictionary mapping feature names to their contribution scores.
        """
        if self.model is None:
            return None

        try:
            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_
                feature_names = list(features.keys())
                return {
                    name: round(float(imp), 4)
                    for name, imp in zip(feature_names, importance)
                }
            elif hasattr(self.model, "coef_"):
                importance = np.abs(self.model.coef_[0])
                feature_names = list(features.keys())
                return {
                    name: round(float(imp), 4)
                    for name, imp in zip(feature_names, importance)
                }
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")

        return None


def get_available_models(dataset_name: str) -> List[str]:
    """
    Get list of available trained models for a dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        List of available model names.
    """
    model_dir = MODELS_DIR / dataset_name
    if not model_dir.exists():
        return []

    models = [f.stem for f in model_dir.glob("*.pkl")]
    return models


def predict_from_csv(
    csv_path: str,
    dataset_name: str,
    model_name: str = "random_forest",
) -> pd.DataFrame:
    """
    Make predictions from a CSV file.

    Args:
        csv_path: Path to input CSV file.
        dataset_name: Name of the disease dataset.
        model_name: Name of the model to use.

    Returns:
        DataFrame with predictions.
    """
    df = pd.read_csv(csv_path)
    predictor = DiseasePredictor(dataset_name, model_name)
    return predictor.predict_batch(df)

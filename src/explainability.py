"""
Explainability Module for Disease Prediction System.
Implements SHAP-based model interpretability for all trained models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.config import FIGURES_DIR, MODELS_DIR
from src.utils import logger, timer


@timer
def compute_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str,
) -> pd.DataFrame:
    """
    Compute feature importance for a trained model.

    Args:
        model: Trained model object.
        feature_names: List of feature names.
        model_name: Name of the model.

    Returns:
        DataFrame with feature importance scores.
    """
    importance = None

    if hasattr(model, "feature_importances_"):
        # Tree-based models (Random Forest, XGBoost)
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Linear models (Logistic Regression, SVM with linear kernel)
        importance = np.abs(model.coef_[0])
    else:
        logger.warning(
            f"Cannot extract feature importance from {model_name}"
        )
        return pd.DataFrame()

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    ).sort_values("Importance", ascending=False)

    return importance_df


@timer
def compute_shap_values(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    model_name: str,
    max_samples: int = 100,
) -> Optional[shap.Explanation]:
    """
    Compute SHAP values for model predictions.

    Args:
        model: Trained model object.
        X_train: Training data (for background).
        X_test: Test data to explain.
        feature_names: List of feature names.
        model_name: Name of the model.
        max_samples: Maximum number of samples to explain.

    Returns:
        SHAP Explanation object or None if computation fails.
    """
    try:
        # Limit samples for computational efficiency
        n_samples = min(max_samples, X_test.shape[0])
        X_explain = X_test[:n_samples]
        n_background = min(100, X_train.shape[0])
        X_background = X_train[:n_background]

        # Choose appropriate explainer
        if model_name in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(
                pd.DataFrame(X_explain, columns=feature_names)
            )
        else:
            # Use KernelExplainer for non-tree models
            explainer = shap.KernelExplainer(
                model.predict_proba,
                pd.DataFrame(X_background, columns=feature_names),
            )
            shap_values_raw = explainer.shap_values(
                pd.DataFrame(X_explain, columns=feature_names),
                nsamples=50,
            )
            # For binary classification, take positive class
            if isinstance(shap_values_raw, list):
                shap_values_raw = shap_values_raw[1]

            shap_values = shap.Explanation(
                values=shap_values_raw,
                base_values=explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value,
                data=X_explain,
                feature_names=feature_names,
            )

        logger.info(f"SHAP values computed for {model_name}")
        return shap_values

    except Exception as e:
        logger.error(f"Error computing SHAP values for {model_name}: {e}")
        return None


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot feature importance bar chart.

    Args:
        importance_df: DataFrame with feature importance.
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        top_n: Number of top features to show.
        save_path: Path to save the figure.
    """
    if importance_df.empty:
        return

    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(len(top_features)),
        top_features["Importance"],
        color="steelblue",
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["Feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance - {model_name}\n({dataset_name})")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    if save_path is None:
        save_path = (
            FIGURES_DIR
            / f"feature_importance_{dataset_name}_{model_name.lower().replace(' ', '_')}.png"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_summary(
    shap_values: shap.Explanation,
    model_name: str,
    dataset_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate SHAP summary plot.

    Args:
        shap_values: SHAP Explanation object.
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        save_path: Path to save the figure.
    """
    if shap_values is None:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, show=False)
    plt.title(f"SHAP Summary - {model_name} ({dataset_name})")
    plt.tight_layout()

    if save_path is None:
        save_path = (
            FIGURES_DIR
            / f"shap_summary_{dataset_name}_{model_name.lower().replace(' ', '_')}.png"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_force(
    shap_values: shap.Explanation,
    sample_index: int,
    model_name: str,
    dataset_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate SHAP force plot for a single prediction.

    Args:
        shap_values: SHAP Explanation object.
        sample_index: Index of the sample to explain.
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        save_path: Path to save the figure.
    """
    if shap_values is None:
        return

    if save_path is None:
        save_path = (
            FIGURES_DIR
            / f"shap_force_{dataset_name}_{model_name.lower().replace(' ', '_')}.html"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        force_plot = shap.force_plot(
            shap_values[sample_index],
            matplotlib=False,
        )
        shap.save_html(str(save_path), force_plot)
        logger.info(f"SHAP force plot saved to {save_path}")
    except Exception as e:
        logger.warning(f"Could not generate force plot: {e}")


@timer
def generate_explainability_report(
    models: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
) -> Dict[str, Any]:
    """
    Generate complete explainability report for all models on a dataset.

    Args:
        models: Dictionary mapping model names to trained model objects.
        X_train: Training features.
        X_test: Test features.
        feature_names: List of feature names.
        dataset_name: Name of the dataset.

    Returns:
        Dictionary with explainability results.
    """
    report = {}

    for model_name, model_data in models.items():
        if "error" in model_data:
            continue

        model = model_data["model"]
        logger.info(f"Generating explainability for {model_name} ({dataset_name})")

        # Feature importance
        importance_df = compute_feature_importance(
            model, feature_names, model_name
        )
        if not importance_df.empty:
            plot_feature_importance(importance_df, model_name, dataset_name)

        # SHAP values
        shap_values = compute_shap_values(
            model, X_train, X_test, feature_names, model_name
        )

        if shap_values is not None:
            plot_shap_summary(shap_values, model_name, dataset_name)
            plot_shap_force(shap_values, 0, model_name, dataset_name)

        report[model_name] = {
            "feature_importance": importance_df.to_dict()
            if not importance_df.empty
            else None,
            "shap_computed": shap_values is not None,
        }

    return report

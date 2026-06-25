"""
Model Evaluation Module for Disease Prediction System.
Generates comprehensive evaluation metrics, plots, and comparison tables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
)

from src.config import FIGURES_DIR, MODEL_RESULTS_DIR, METRICS
from src.utils import logger, timer, save_results


@timer
def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        model_name: Name of the model.

    Returns:
        Dictionary with all evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
    }

    logger.info(
        f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
        f"F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}"
    )

    return metrics


@timer
def evaluate_all_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_name: str,
    save_results_flag: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all trained models and generate comparison.

    Args:
        models: Dictionary mapping model names to trained model objects.
        X_test: Test features.
        y_test: Test labels.
        dataset_name: Name of the dataset.
        save_results_flag: Whether to save results to disk.

    Returns:
        Dictionary of evaluation results for all models.
    """
    all_results = {}

    for model_name, model_data in models.items():
        if "error" in model_data:
            continue

        model = model_data["model"]
        results = evaluate_model(model, X_test, y_test, model_name)
        all_results[model_name] = results

    if save_results_flag:
        # Save comparison table
        comparison_df = create_comparison_table(all_results)
        comparison_path = MODEL_RESULTS_DIR / f"{dataset_name}_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Comparison table saved to {comparison_path}")

    return all_results


def create_comparison_table(
    results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create a comparison table of all models.

    Args:
        results: Dictionary of evaluation results.

    Returns:
        DataFrame with model comparison metrics.
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(metrics["accuracy"], 4),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1 Score": round(metrics["f1"], 4),
                "ROC-AUC": round(metrics["roc_auc"], 4),
            }
        )

    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    return df


def get_best_model(results: Dict[str, Dict[str, Any]], metric: str = "roc_auc") -> str:
    """
    Identify the best model based on a specific metric.

    Args:
        results: Dictionary of evaluation results.
        metric: Metric to use for comparison.

    Returns:
        Name of the best model.
    """
    best_model = max(results.items(), key=lambda x: x[1][metric])
    return best_model[0]


# ============================================================
# Visualization Functions
# ============================================================
def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    dataset_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        y_test: True labels.
        y_pred: Predicted labels.
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        save_path: Path to save the figure.
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}\n({dataset_name})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    if save_path is None:
        save_path = (
            FIGURES_DIR
            / f"cm_{dataset_name}_{model_name.lower().replace(' ', '_')}.png"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    dataset_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot ROC curves for all models on a single figure.

    Args:
        results: Dictionary of evaluation results.
        y_test: True labels.
        dataset_name: Name of the dataset.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, metrics in results.items():
        y_prob = np.array(metrics["y_prob"])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = metrics["roc_auc"]

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC = {roc_auc_val:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves - {dataset_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"roc_curves_{dataset_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curves(
    results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    dataset_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot Precision-Recall curves for all models.

    Args:
        results: Dictionary of evaluation results.
        y_test: True labels.
        dataset_name: Name of the dataset.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, metrics in results.items():
        y_prob = np.array(metrics["y_prob"])
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)

        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{model_name} (AUC = {pr_auc:.4f})",
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curves - {dataset_name}", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"pr_curves_{dataset_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    dataset_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot model comparison bar chart.

    Args:
        comparison_df: DataFrame with model comparison metrics.
        dataset_name: Name of the dataset.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    x = np.arange(len(comparison_df))
    width = 0.15

    for i, metric in enumerate(metrics_to_plot):
        ax.bar(
            x + i * width,
            comparison_df[metric],
            width,
            label=metric,
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Model Comparison - {dataset_name}", fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(comparison_df["Model"], rotation=15)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"model_comparison_{dataset_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_plots(
    results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    dataset_name: str,
) -> None:
    """
    Generate all evaluation plots for a dataset.

    Args:
        results: Dictionary of evaluation results.
        y_test: True test labels.
        dataset_name: Name of the dataset.
    """
    logger.info(f"Generating plots for {dataset_name}...")

    # ROC curves
    plot_roc_curves(results, y_test, dataset_name)

    # PR curves
    plot_precision_recall_curves(results, y_test, dataset_name)

    # Confusion matrices for each model
    for model_name, metrics in results.items():
        y_pred = np.array(metrics["y_pred"])
        plot_confusion_matrix(y_pred, y_test, model_name, dataset_name)

    # Comparison bar chart
    comparison_df = create_comparison_table(results)
    plot_model_comparison(comparison_df, dataset_name)

    logger.info(f"All plots saved to {FIGURES_DIR}")

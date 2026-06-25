"""
Model Training Module for Disease Prediction System.
Handles training, cross-validation, and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
)

from src.config import (
    MODELS_CONFIG,
    RANDOM_STATE,
    CV_FOLDS,
    MODELS_DIR,
)
from src.utils import logger, timer, save_model


def get_model_instance(model_name: str) -> Any:
    """
    Get an initialized model instance by name.

    Args:
        model_name: Name of the model.

    Returns:
        Initialized model object.

    Raises:
        ValueError: If model name is not recognized.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            **MODELS_CONFIG["Logistic Regression"]["params"]
        ),
        "SVM": SVC(**MODELS_CONFIG["SVM"]["params"]),
        "Random Forest": RandomForestClassifier(
            **MODELS_CONFIG["Random Forest"]["params"]
        ),
        "XGBoost": XGBClassifier(**MODELS_CONFIG["XGBoost"]["params"]),
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {list(models.keys())}"
        )

    return models[model_name]


@timer
def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune_hyperparameters: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a single model with optional hyperparameter tuning.

    Args:
        model_name: Name of the model to train.
        X_train: Training features.
        y_train: Training labels.
        tune_hyperparameters: Whether to perform GridSearchCV.

    Returns:
        Tuple of (trained_model, training_info).
    """
    logger.info(f"Training {model_name}...")

    model = get_model_instance(model_name)
    training_info: Dict[str, Any] = {"model_name": model_name}

    if tune_hyperparameters:
        # Perform GridSearchCV
        grid_params = MODELS_CONFIG[model_name]["grid_params"]
        cv = StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_params,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

        training_info["best_params"] = grid_search.best_params_
        training_info["best_cv_score"] = grid_search.best_score_
        training_info["tuned"] = True

        logger.info(
            f"{model_name} - Best CV Score: {grid_search.best_score_:.4f}"
        )
        logger.info(f"{model_name} - Best Params: {grid_search.best_params_}")

    else:
        # Train without tuning
        model.fit(X_train, y_train)
        training_info["tuned"] = False

    # Cross-validation scores
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    training_info["cv_scores"] = cv_scores.tolist()
    training_info["cv_mean"] = cv_scores.mean()
    training_info["cv_std"] = cv_scores.std()

    logger.info(
        f"{model_name} - CV Accuracy: {cv_scores.mean():.4f} "
        f"(+/- {cv_scores.std():.4f})"
    )

    return model, training_info


@timer
def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dataset_name: str,
    tune_hyperparameters: bool = True,
    save_models: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Train all models for a specific dataset.

    Args:
        X_train: Training features.
        y_train: Training labels.
        dataset_name: Name of the dataset (for saving models).
        tune_hyperparameters: Whether to perform hyperparameter tuning.
        save_models: Whether to save trained models to disk.

    Returns:
        Dictionary mapping model names to their results.
    """
    results = {}

    for model_name in MODELS_CONFIG.keys():
        try:
            model, training_info = train_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                tune_hyperparameters=tune_hyperparameters,
            )

            results[model_name] = {
                "model": model,
                "training_info": training_info,
            }

            # Save model
            if save_models:
                model_path = (
                    MODELS_DIR
                    / dataset_name
                    / f"{model_name.lower().replace(' ', '_')}.pkl"
                )
                save_model(model, model_path)

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    return results


def get_training_summary(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Generate a summary table of training results.

    Args:
        results: Dictionary of training results from train_all_models.

    Returns:
        DataFrame with training summary.
    """
    summary_rows = []

    for model_name, result in results.items():
        if "error" in result:
            continue

        info = result["training_info"]
        row = {
            "Model": model_name,
            "CV Mean Accuracy": round(info["cv_mean"], 4),
            "CV Std": round(info["cv_std"], 4),
            "Tuned": info["tuned"],
        }

        if "best_params" in info:
            row["Best Params"] = str(info["best_params"])

        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values(
        "CV Mean Accuracy", ascending=False
    )

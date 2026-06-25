"""
Utility functions for the Disease Prediction System.
Provides logging, timing, and helper functions used across modules.
"""

import logging
import time
import functools
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import joblib
import pandas as pd
import numpy as np


# ============================================================
# Logging Configuration
# ============================================================
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a configured logger instance.

    Args:
        name: Name for the logger.
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger("disease_prediction")


# ============================================================
# Decorators
# ============================================================
def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed:.2f}s")
        return result

    return wrapper


# ============================================================
# Model I/O
# ============================================================
def save_model(model: Any, filepath: Path) -> None:
    """
    Save a trained model to disk.

    Args:
        model: Trained model object.
        filepath: Path to save the model.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: Path) -> Any:
    """
    Load a trained model from disk.

    Args:
        filepath: Path to the saved model.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found at {filepath}")
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


# ============================================================
# Data I/O
# ============================================================
def save_dataframe(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save DataFrame to CSV.

    Args:
        df: DataFrame to save.
        filepath: Output file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"DataFrame saved to {filepath} (shape: {df.shape})")


def load_dataframe(filepath: Path) -> pd.DataFrame:
    """
    Load DataFrame from CSV.

    Args:
        filepath: Path to CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"DataFrame loaded from {filepath} (shape: {df.shape})")
    return df


# ============================================================
# Results I/O
# ============================================================
def save_results(results: Dict[str, Any], filepath: Path) -> None:
    """
    Save results dictionary to JSON.

    Args:
        results: Results dictionary.
        filepath: Output file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python native types
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(results, default=convert))
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: Path) -> Dict[str, Any]:
    """
    Load results from JSON file.

    Args:
        filepath: Path to JSON file.

    Returns:
        Results dictionary.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Results not found at {filepath}")
    with open(filepath, "r") as f:
        results = json.load(f)
    return results


# ============================================================
# Risk Assessment
# ============================================================
def get_risk_category(probability: float) -> str:
    """
    Determine risk category based on prediction probability.

    Args:
        probability: Prediction probability (0-1).

    Returns:
        Risk category string.
    """
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"


def get_risk_color(category: str) -> str:
    """
    Get color associated with risk category.

    Args:
        category: Risk category string.

    Returns:
        Color hex code.
    """
    colors = {
        "Low Risk": "#28a745",
        "Medium Risk": "#ffc107",
        "High Risk": "#dc3545",
    }
    return colors.get(category, "#6c757d")


# ============================================================
# Validation
# ============================================================
def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list] = None,
    min_rows: int = 10,
) -> bool:
    """
    Validate a DataFrame meets minimum requirements.

    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
        min_rows: Minimum number of rows required.

    Returns:
        True if valid, raises ValueError otherwise.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    if len(df) < min_rows:
        raise ValueError(
            f"DataFrame has {len(df)} rows, minimum {min_rows} required"
        )

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    return True

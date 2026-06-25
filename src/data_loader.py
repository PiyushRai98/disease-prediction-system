"""
Data Loading Module for Disease Prediction System.
Handles fetching datasets from UCI ML Repository and local files.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

from src.config import DATASETS, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils import logger, save_dataframe, timer


@timer
def load_heart_disease() -> pd.DataFrame:
    """
    Load Heart Disease dataset from UCI ML Repository.

    The target variable is converted to binary:
        0 = No Disease (original value 0)
        1 = Disease (original values 1, 2, 3, 4)

    Returns:
        DataFrame with heart disease data.
    """
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=DATASETS["heart"]["uci_id"])
        X = dataset.data.features
        y = dataset.data.targets

        df = pd.concat([X, y], axis=1)

        # Convert target to binary classification
        target_col = DATASETS["heart"]["target_col"]
        df["target"] = (df[target_col] > 0).astype(int)
        df.drop(columns=[target_col], inplace=True)

        logger.info(f"Heart Disease dataset loaded: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading Heart Disease dataset: {e}")
        # Fallback: try loading from processed directory
        fallback_path = PROCESSED_DATA_DIR / "heart_disease.csv"
        if fallback_path.exists():
            logger.info("Loading from fallback processed file")
            return pd.read_csv(fallback_path)
        raise


@timer
def load_diabetes() -> pd.DataFrame:
    """
    Load Diabetes dataset from local CSV file.

    Target variable: Outcome (0 = Non-Diabetic, 1 = Diabetic)

    Returns:
        DataFrame with diabetes data.
    """
    file_path = Path(DATASETS["diabetes"]["file_path"])

    if not file_path.exists():
        raise FileNotFoundError(
            f"Diabetes dataset not found at {file_path}. "
            "Please place diabetes.csv in data/raw/ directory."
        )

    df = pd.read_csv(file_path)

    # Rename target column for consistency
    if "Outcome" in df.columns:
        df.rename(columns={"Outcome": "target"}, inplace=True)

    logger.info(f"Diabetes dataset loaded: {df.shape}")
    return df


@timer
def load_breast_cancer() -> pd.DataFrame:
    """
    Load Breast Cancer Wisconsin Diagnostic dataset from UCI ML Repository.

    Target variable encoding:
        B (Benign) = 0
        M (Malignant) = 1

    Returns:
        DataFrame with breast cancer data.
    """
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=DATASETS["cancer"]["uci_id"])
        X = dataset.data.features
        y = dataset.data.targets

        df = pd.concat([X, y], axis=1)

        # Encode target: B=0, M=1
        target_col = DATASETS["cancer"]["target_col"]
        df["target"] = df[target_col].map({"B": 0, "M": 1})
        df.drop(columns=[target_col], inplace=True)

        # Drop ID column if present
        if "ID" in df.columns:
            df.drop(columns=["ID"], inplace=True)

        logger.info(f"Breast Cancer dataset loaded: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading Breast Cancer dataset: {e}")
        fallback_path = PROCESSED_DATA_DIR / "breast_cancer.csv"
        if fallback_path.exists():
            logger.info("Loading from fallback processed file")
            return pd.read_csv(fallback_path)
        raise


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a dataset by name.

    Args:
        name: Dataset identifier ('heart', 'diabetes', 'cancer').

    Returns:
        DataFrame with the requested dataset.

    Raises:
        ValueError: If dataset name is not recognized.
    """
    loaders = {
        "heart": load_heart_disease,
        "diabetes": load_diabetes,
        "cancer": load_breast_cancer,
    }

    if name not in loaders:
        raise ValueError(
            f"Unknown dataset: '{name}'. Choose from: {list(loaders.keys())}"
        )

    return loaders[name]()


def load_all_datasets() -> dict:
    """
    Load all datasets and return as dictionary.

    Returns:
        Dictionary mapping dataset names to DataFrames.
    """
    datasets = {}
    for name in ["heart", "diabetes", "cancer"]:
        try:
            datasets[name] = load_dataset(name)
            logger.info(f"Successfully loaded: {name}")
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

    return datasets


def save_processed_datasets(datasets: dict) -> None:
    """
    Save processed datasets to the processed data directory.

    Args:
        datasets: Dictionary mapping dataset names to DataFrames.
    """
    for name, df in datasets.items():
        filepath = PROCESSED_DATA_DIR / f"{name}.csv"
        save_dataframe(df, filepath)


def get_feature_names(dataset_name: str) -> list:
    """
    Get feature names for a specific dataset (excluding target).

    Args:
        dataset_name: Name of the dataset.

    Returns:
        List of feature column names.
    """
    df = load_dataset(dataset_name)
    return [col for col in df.columns if col != "target"]


def get_dataset_info(dataset_name: str) -> dict:
    """
    Get metadata information about a dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        Dictionary with dataset metadata.
    """
    config = DATASETS[dataset_name]
    df = load_dataset(dataset_name)

    return {
        "name": config["name"],
        "description": config["description"],
        "n_samples": len(df),
        "n_features": len(df.columns) - 1,
        "target_distribution": df["target"].value_counts().to_dict(),
        "feature_names": [col for col in df.columns if col != "target"],
        "positive_label": config["positive_label"],
        "negative_label": config["negative_label"],
    }

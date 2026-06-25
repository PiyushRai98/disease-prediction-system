"""
Configuration module for the Disease Prediction System.
Centralizes all project paths, model parameters, and settings.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_RESULTS_DIR = REPORTS_DIR / "model_results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ============================================================
# Dataset Configuration
# ============================================================
DATASETS = {
    "heart": {
        "uci_id": 45,
        "name": "Heart Disease",
        "target_col": "num",
        "description": "Cleveland Heart Disease Dataset",
        "positive_label": "Disease",
        "negative_label": "No Disease",
    },
    "diabetes": {
        "file_path": str(RAW_DATA_DIR / "diabetes.csv"),
        "name": "Diabetes",
        "target_col": "Outcome",
        "description": "Pima Indians Diabetes Dataset",
        "positive_label": "Diabetic",
        "negative_label": "Non-Diabetic",
    },
    "cancer": {
        "uci_id": 17,
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "target_col": "Diagnosis",
        "description": "Breast Cancer Wisconsin Diagnostic Dataset",
        "positive_label": "Malignant",
        "negative_label": "Benign",
    },
}

# ============================================================
# Model Configuration
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    "Logistic Regression": {
        "class": "sklearn.linear_model.LogisticRegression",
        "params": {
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
        },
        "grid_params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
        },
    },
    "SVM": {
        "class": "sklearn.svm.SVC",
        "params": {
            "random_state": RANDOM_STATE,
            "probability": True,
        },
        "grid_params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    },
    "Random Forest": {
        "class": "sklearn.ensemble.RandomForestClassifier",
        "params": {
            "random_state": RANDOM_STATE,
        },
        "grid_params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
    },
    "XGBoost": {
        "class": "xgboost.XGBClassifier",
        "params": {
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        },
        "grid_params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
        },
    },
}

# ============================================================
# Evaluation Metrics
# ============================================================
METRICS: List[str] = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

# ============================================================
# Risk Categories
# ============================================================
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 1.0,
}

# ============================================================
# Streamlit Configuration
# ============================================================
STREAMLIT_CONFIG = {
    "page_title": "Disease Prediction System",
    "page_icon": "🏥",
    "layout": "wide",
}

# ============================================================
# Create directories if they don't exist
# ============================================================
def create_directories() -> None:
    """Create all required project directories."""
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR / "heart",
        MODELS_DIR / "diabetes",
        MODELS_DIR / "cancer",
        FIGURES_DIR,
        MODEL_RESULTS_DIR,
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


create_directories()

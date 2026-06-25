"""
Preprocessing Module for Disease Prediction System.
Implements a complete, reusable preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from src.config import RANDOM_STATE, TEST_SIZE
from src.utils import logger, timer


class PreprocessingPipeline:
    """
    Reusable preprocessing pipeline for medical datasets.

    Handles:
        - Missing value detection and imputation
        - Duplicate removal
        - Outlier detection and handling
        - Feature scaling
        - Label encoding
        - Train/test splitting with stratification
    """

    def __init__(
        self,
        scale_features: bool = True,
        handle_outliers: bool = True,
        outlier_method: str = "iqr",
        impute_strategy: str = "median",
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            scale_features: Whether to apply feature scaling.
            handle_outliers: Whether to detect and handle outliers.
            outlier_method: Method for outlier detection ('iqr' or 'zscore').
            impute_strategy: Strategy for imputing missing values.
        """
        self.scale_features = scale_features
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.impute_strategy = impute_strategy
        self.scaler: Optional[StandardScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: list = []
        self.preprocessing_report: Dict[str, Any] = {}

    @timer
    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "target"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the pipeline and transform the data.

        Args:
            df: Input DataFrame with features and target.
            target_col: Name of the target column.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        logger.info(f"Starting preprocessing pipeline (shape: {df.shape})")
        self.preprocessing_report = {"original_shape": df.shape}

        # Step 1: Separate features and target
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
        self.feature_names = X.columns.tolist()

        # Step 2: Handle missing values
        X = self._handle_missing_values(X)

        # Step 3: Remove duplicates
        X, y = self._remove_duplicates(X, y)

        # Step 4: Encode categorical features
        X = self._encode_categorical(X)

        # Step 5: Handle outliers
        if self.handle_outliers:
            X = self._handle_outliers(X)

        # Step 6: Feature scaling
        if self.scale_features:
            X = self._scale_features(X)

        # Step 7: Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        self.preprocessing_report["final_shape"] = X.shape
        self.preprocessing_report["train_shape"] = X_train.shape
        self.preprocessing_report["test_shape"] = X_test.shape

        logger.info(
            f"Preprocessing complete. "
            f"Train: {X_train.shape}, Test: {X_test.shape}"
        )

        # Convert to numpy arrays for consistent output
        X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        X_test_arr = X_test.values if isinstance(X_test, pd.DataFrame) else np.array(X_test)

        return X_train_arr, X_test_arr, y_train.values, y_test.values

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted pipeline.

        Args:
            X: Input DataFrame with features.

        Returns:
            Transformed feature array.
        """
        X = X.copy()

        # Impute missing values
        if self.imputer is not None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = self.imputer.transform(X[numeric_cols])

        # Encode categorical features
        for col, encoder in self.label_encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col])

        # Scale features
        if self.scale_features and self.scaler is not None:
            X = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index,
            )

        return X.values

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and impute missing values.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with imputed values.
        """
        missing_before = X.isnull().sum().sum()
        missing_cols = X.columns[X.isnull().any()].tolist()

        self.preprocessing_report["missing_values_before"] = missing_before
        self.preprocessing_report["columns_with_missing"] = missing_cols

        if missing_before > 0:
            logger.info(
                f"Found {missing_before} missing values in {len(missing_cols)} columns"
            )

            # Replace zeros with NaN for specific medical features
            # (In medical data, 0 often means missing for certain features)
            zero_not_valid = [
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
            ]
            for col in zero_not_valid:
                if col in X.columns:
                    X[col] = X[col].replace(0, np.nan)

            # Impute numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.imputer = SimpleImputer(strategy=self.impute_strategy)
                X[numeric_cols] = self.imputer.fit_transform(X[numeric_cols])

            # Impute categorical columns with mode
            cat_cols = X.select_dtypes(include=["object"]).columns
            if len(cat_cols) > 0:
                cat_imputer = SimpleImputer(strategy="most_frequent")
                X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

        else:
            # Still check for zero-as-missing in diabetes dataset
            zero_not_valid = [
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
            ]
            cols_to_impute = [col for col in zero_not_valid if col in X.columns]
            if cols_to_impute:
                for col in cols_to_impute:
                    X[col] = X[col].replace(0, np.nan)

                numeric_cols = X.select_dtypes(include=[np.number]).columns
                self.imputer = SimpleImputer(strategy=self.impute_strategy)
                X[numeric_cols] = self.imputer.fit_transform(X[numeric_cols])

        missing_after = X.isnull().sum().sum()
        self.preprocessing_report["missing_values_after"] = missing_after
        logger.info(f"Missing values after imputation: {missing_after}")

        return X

    def _remove_duplicates(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove duplicate rows.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            Tuple of deduplicated (X, y).
        """
        combined = X.copy()
        combined["_target"] = y.values

        n_before = len(combined)
        combined = combined.drop_duplicates()
        n_after = len(combined)
        n_removed = n_before - n_after

        self.preprocessing_report["duplicates_removed"] = n_removed

        if n_removed > 0:
            logger.info(f"Removed {n_removed} duplicate rows")

        y = combined["_target"]
        X = combined.drop(columns=["_target"])

        return X, y

    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with encoded categorical features.
        """
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

        self.preprocessing_report["categorical_columns"] = cat_cols

        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"Encoded column '{col}' ({len(le.classes_)} classes)")

        return X

    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and cap outliers using IQR method.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with capped outliers.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        outliers_found = 0

        if self.outlier_method == "iqr":
            for col in numeric_cols:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (X[col] < lower_bound) | (X[col] > upper_bound)
                n_outliers = outlier_mask.sum()
                outliers_found += n_outliers

                # Cap outliers instead of removing
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)

        elif self.outlier_method == "zscore":
            from scipy import stats

            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(X[col].dropna()))
                outlier_mask = z_scores > 3
                n_outliers = outlier_mask.sum()
                outliers_found += n_outliers

                median_val = X[col].median()
                X.loc[X.index[outlier_mask], col] = median_val

        self.preprocessing_report["outliers_capped"] = outliers_found
        logger.info(
            f"Outliers handled: {outliers_found} values capped "
            f"(method: {self.outlier_method})"
        )

        return X

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply StandardScaler to all numeric features.

        Args:
            X: Input DataFrame.

        Returns:
            Scaled DataFrame.
        """
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )
        logger.info("Feature scaling applied (StandardScaler)")
        return X_scaled

    def get_report(self) -> Dict[str, Any]:
        """
        Get the preprocessing report.

        Returns:
            Dictionary with preprocessing statistics.
        """
        return self.preprocessing_report


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a detailed missing value report.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with missing value statistics per column.
    """
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    dtypes = df.dtypes

    report = pd.DataFrame(
        {
            "Missing Count": missing,
            "Missing Percent": percent.round(2),
            "Data Type": dtypes,
        }
    )
    report = report[report["Missing Count"] > 0].sort_values(
        "Missing Percent", ascending=False
    )

    return report


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using IQR method for all numeric columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with outlier counts per column.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_report = []

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_report.append(
            {
                "Column": col,
                "Lower Bound": round(lower, 2),
                "Upper Bound": round(upper, 2),
                "Outlier Count": n_outliers,
                "Outlier Percent": round(n_outliers / len(df) * 100, 2),
            }
        )

    return pd.DataFrame(outlier_report).sort_values(
        "Outlier Count", ascending=False
    )

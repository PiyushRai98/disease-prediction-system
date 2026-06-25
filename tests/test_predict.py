"""
Tests for the prediction module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import get_available_models
from src.utils import get_risk_category, get_risk_color


class TestRiskAssessment:
    """Test cases for risk assessment utilities."""

    def test_low_risk(self):
        """Test low risk categorization."""
        assert get_risk_category(0.1) == "Low Risk"
        assert get_risk_category(0.0) == "Low Risk"
        assert get_risk_category(0.29) == "Low Risk"

    def test_medium_risk(self):
        """Test medium risk categorization."""
        assert get_risk_category(0.3) == "Medium Risk"
        assert get_risk_category(0.5) == "Medium Risk"
        assert get_risk_category(0.59) == "Medium Risk"

    def test_high_risk(self):
        """Test high risk categorization."""
        assert get_risk_category(0.6) == "High Risk"
        assert get_risk_category(0.8) == "High Risk"
        assert get_risk_category(1.0) == "High Risk"

    def test_risk_colors(self):
        """Test risk color assignment."""
        assert get_risk_color("Low Risk") == "#28a745"
        assert get_risk_color("Medium Risk") == "#ffc107"
        assert get_risk_color("High Risk") == "#dc3545"

    def test_unknown_risk_color(self):
        """Test unknown risk category returns default color."""
        assert get_risk_color("Unknown") == "#6c757d"


class TestGetAvailableModels:
    """Test cases for model discovery."""

    def test_nonexistent_dataset(self):
        """Test with non-existent dataset returns empty list."""
        models = get_available_models("nonexistent")
        assert models == []

    def test_returns_list(self):
        """Test that function returns a list."""
        models = get_available_models("heart")
        assert isinstance(models, list)

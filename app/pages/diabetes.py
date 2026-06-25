"""
Diabetes Prediction Page.
Provides input form and prediction interface for diabetes risk.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import DiseasePredictor, get_available_models
from src.utils import get_risk_category, get_risk_color


def show_diabetes_page():
    """Display the diabetes prediction page."""
    st.header("🩸 Diabetes Prediction")
    st.markdown(
        "Assess diabetes risk based on diagnostic measurements."
    )
    st.markdown("---")

    # Model selection
    available_models = get_available_models("diabetes")
    if not available_models:
        st.error(
            "⚠️ No trained models found for Diabetes. "
            "Please run the training pipeline first."
        )
        st.code("python -m src.train", language="bash")
        return

    model_choice = st.selectbox(
        "Select Model",
        available_models,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    st.markdown("---")

    # Input form
    st.subheader("📋 Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0, max_value=20, value=1,
        )
        glucose = st.number_input(
            "Plasma Glucose Concentration (mg/dl)",
            min_value=0, max_value=250, value=120,
            help="2-hour plasma glucose concentration in oral glucose tolerance test",
        )
        blood_pressure = st.number_input(
            "Diastolic Blood Pressure (mm Hg)",
            min_value=0, max_value=150, value=70,
        )
        skin_thickness = st.number_input(
            "Triceps Skin Fold Thickness (mm)",
            min_value=0, max_value=100, value=20,
        )

    with col2:
        insulin = st.number_input(
            "2-Hour Serum Insulin (mu U/ml)",
            min_value=0, max_value=900, value=80,
        )
        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=0.0, max_value=70.0, value=25.0, step=0.1,
            help="Weight in kg / (Height in m)²",
        )
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0, max_value=2.5, value=0.5, step=0.01,
            help="Genetic predisposition score",
        )
        age = st.number_input(
            "Age (years)",
            min_value=1, max_value=120, value=30,
        )

    st.markdown("---")

    # Prediction
    if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
        features = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }

        try:
            predictor = DiseasePredictor("diabetes", model_choice)
            result = predictor.predict_single(features)

            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Diagnosis", result["diagnosis"])
            with col2:
                st.metric("Confidence", f"{result['confidence']}%")
            with col3:
                st.metric("Risk Score", f"{result['risk_percentage']}%")

            # Risk indicator
            risk_class = result["risk_category"].lower().replace(" ", "-")
            st.markdown(
                f'<div class="risk-{risk_class.split("-")[0]}">'
                f'<h3 style="text-align:center;">{result["risk_category"]}</h3>'
                f'<p style="text-align:center;">Probability: {result["probability"]:.4f}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )

            # Feature importance
            importance = predictor.get_feature_importance_for_prediction(features)
            if importance:
                st.subheader("📈 Top Contributing Factors")
                imp_df = pd.DataFrame(
                    list(importance.items()),
                    columns=["Feature", "Importance"],
                ).sort_values("Importance", ascending=False).head(5)
                st.bar_chart(imp_df.set_index("Feature"))

            # Health tips based on inputs
            st.subheader("💡 Health Insights")
            if glucose > 140:
                st.warning("⚠️ Glucose levels are elevated. Consider consulting a doctor.")
            if bmi > 30:
                st.warning("⚠️ BMI indicates obesity. Regular exercise recommended.")
            if blood_pressure > 90:
                st.warning("⚠️ Blood pressure is high. Monitor regularly.")

        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Batch prediction
    st.markdown("---")
    st.subheader("📁 Batch Prediction")
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=["csv"],
        key="diabetes_batch",
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} records")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("Run Batch Prediction", key="diabetes_batch_btn"):
                predictor = DiseasePredictor("diabetes", model_choice)
                results_df = predictor.predict_batch(batch_df)
                st.dataframe(results_df, use_container_width=True)

                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "diabetes_predictions.csv",
                    "text/csv",
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

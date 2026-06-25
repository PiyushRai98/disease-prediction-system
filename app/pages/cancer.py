"""
Breast Cancer Prediction Page.
Provides input form and prediction interface for breast cancer diagnosis.
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


def show_cancer_page():
    """Display the breast cancer prediction page."""
    st.header("🔬 Breast Cancer Prediction")
    st.markdown(
        "Classify breast tumors as benign or malignant based on cell nucleus features."
    )
    st.markdown("---")

    # Model selection
    available_models = get_available_models("cancer")
    if not available_models:
        st.error(
            "⚠️ No trained models found for Breast Cancer. "
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

    # Input form with tabs for feature groups
    st.subheader("📋 Cell Nucleus Features")
    st.info(
        "Enter computed features from digitized images of fine needle aspirate (FNA) "
        "of breast mass. Features describe characteristics of cell nuclei."
    )

    tab1, tab2, tab3 = st.tabs(["Mean Values", "SE Values", "Worst Values"])

    with tab1:
        st.markdown("**Mean values of cell nucleus features:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            radius_mean = st.number_input("Radius (mean)", value=14.0, step=0.1, key="rm")
            texture_mean = st.number_input("Texture (mean)", value=19.0, step=0.1, key="tm")
            perimeter_mean = st.number_input("Perimeter (mean)", value=92.0, step=0.1, key="pm")
            area_mean = st.number_input("Area (mean)", value=650.0, step=1.0, key="am")

        with col2:
            smoothness_mean = st.number_input("Smoothness (mean)", value=0.1, step=0.001, format="%.4f", key="sm")
            compactness_mean = st.number_input("Compactness (mean)", value=0.1, step=0.001, format="%.4f", key="cm")
            concavity_mean = st.number_input("Concavity (mean)", value=0.09, step=0.001, format="%.4f", key="conm")
            concave_points_mean = st.number_input("Concave Points (mean)", value=0.05, step=0.001, format="%.4f", key="cpm")

        with col3:
            symmetry_mean = st.number_input("Symmetry (mean)", value=0.18, step=0.001, format="%.4f", key="sym")
            fractal_dimension_mean = st.number_input("Fractal Dimension (mean)", value=0.06, step=0.001, format="%.4f", key="fdm")

    with tab2:
        st.markdown("**Standard Error of cell nucleus features:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            radius_se = st.number_input("Radius (SE)", value=0.4, step=0.01, key="rse")
            texture_se = st.number_input("Texture (SE)", value=1.2, step=0.01, key="tse")
            perimeter_se = st.number_input("Perimeter (SE)", value=3.0, step=0.1, key="pse")
            area_se = st.number_input("Area (SE)", value=40.0, step=1.0, key="ase")

        with col2:
            smoothness_se = st.number_input("Smoothness (SE)", value=0.007, step=0.001, format="%.4f", key="sse")
            compactness_se = st.number_input("Compactness (SE)", value=0.025, step=0.001, format="%.4f", key="cse")
            concavity_se = st.number_input("Concavity (SE)", value=0.03, step=0.001, format="%.4f", key="conse")
            concave_points_se = st.number_input("Concave Points (SE)", value=0.01, step=0.001, format="%.4f", key="cpse")

        with col3:
            symmetry_se = st.number_input("Symmetry (SE)", value=0.02, step=0.001, format="%.4f", key="syse")
            fractal_dimension_se = st.number_input("Fractal Dimension (SE)", value=0.003, step=0.001, format="%.4f", key="fdse")

    with tab3:
        st.markdown("**Worst (largest) values of cell nucleus features:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            radius_worst = st.number_input("Radius (worst)", value=16.0, step=0.1, key="rw")
            texture_worst = st.number_input("Texture (worst)", value=25.0, step=0.1, key="tw")
            perimeter_worst = st.number_input("Perimeter (worst)", value=107.0, step=0.1, key="pw")
            area_worst = st.number_input("Area (worst)", value=880.0, step=1.0, key="aw")

        with col2:
            smoothness_worst = st.number_input("Smoothness (worst)", value=0.13, step=0.001, format="%.4f", key="sw")
            compactness_worst = st.number_input("Compactness (worst)", value=0.25, step=0.001, format="%.4f", key="cw")
            concavity_worst = st.number_input("Concavity (worst)", value=0.27, step=0.001, format="%.4f", key="conw")
            concave_points_worst = st.number_input("Concave Points (worst)", value=0.11, step=0.001, format="%.4f", key="cpw")

        with col3:
            symmetry_worst = st.number_input("Symmetry (worst)", value=0.29, step=0.001, format="%.4f", key="syw")
            fractal_dimension_worst = st.number_input("Fractal Dimension (worst)", value=0.08, step=0.001, format="%.4f", key="fdw")

    st.markdown("---")

    # Prediction
    if st.button("🔍 Predict Cancer Diagnosis", type="primary", use_container_width=True):
        features = {
            "radius1": radius_mean,
            "texture1": texture_mean,
            "perimeter1": perimeter_mean,
            "area1": area_mean,
            "smoothness1": smoothness_mean,
            "compactness1": compactness_mean,
            "concavity1": concavity_mean,
            "concave_points1": concave_points_mean,
            "symmetry1": symmetry_mean,
            "fractal_dimension1": fractal_dimension_mean,
            "radius2": radius_se,
            "texture2": texture_se,
            "perimeter2": perimeter_se,
            "area2": area_se,
            "smoothness2": smoothness_se,
            "compactness2": compactness_se,
            "concavity2": concavity_se,
            "concave_points2": concave_points_se,
            "symmetry2": symmetry_se,
            "fractal_dimension2": fractal_dimension_se,
            "radius3": radius_worst,
            "texture3": texture_worst,
            "perimeter3": perimeter_worst,
            "area3": area_worst,
            "smoothness3": smoothness_worst,
            "compactness3": compactness_worst,
            "concavity3": concavity_worst,
            "concave_points3": concave_points_worst,
            "symmetry3": symmetry_worst,
            "fractal_dimension3": fractal_dimension_worst,
        }

        try:
            predictor = DiseasePredictor("cancer", model_choice)
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
                st.metric("Malignancy Probability", f"{result['risk_percentage']}%")

            # Risk indicator
            risk_class = result["risk_category"].lower().replace(" ", "-")
            st.markdown(
                f'<div class="risk-{risk_class.split("-")[0]}">'
                f'<h3 style="text-align:center;">{result["risk_category"]}</h3>'
                f'<p style="text-align:center;">'
                f"Malignancy Probability: {result['probability']:.4f}"
                f"</p></div>",
                unsafe_allow_html=True,
            )

            # Feature importance
            importance = predictor.get_feature_importance_for_prediction(features)
            if importance:
                st.subheader("📈 Top Contributing Features")
                imp_df = pd.DataFrame(
                    list(importance.items()),
                    columns=["Feature", "Importance"],
                ).sort_values("Importance", ascending=False).head(10)
                st.bar_chart(imp_df.set_index("Feature"))

        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Batch prediction
    st.markdown("---")
    st.subheader("📁 Batch Prediction")
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=["csv"],
        key="cancer_batch",
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} records")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("Run Batch Prediction", key="cancer_batch_btn"):
                predictor = DiseasePredictor("cancer", model_choice)
                results_df = predictor.predict_batch(batch_df)
                st.dataframe(results_df, use_container_width=True)

                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "cancer_predictions.csv",
                    "text/csv",
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

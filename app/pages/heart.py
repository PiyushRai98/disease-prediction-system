"""
Heart Disease Prediction Page.
Provides input form and prediction interface for cardiovascular disease.
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


def show_heart_page():
    """Display the heart disease prediction page."""
    st.header("❤️ Heart Disease Prediction")
    st.markdown(
        "Predict cardiovascular disease risk based on clinical parameters."
    )
    st.markdown("---")

    # Model selection
    available_models = get_available_models("heart")
    if not available_models:
        st.error(
            "⚠️ No trained models found for Heart Disease. "
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
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox(
            "Chest Pain Type",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical Angina",
                2: "Atypical Angina",
                3: "Non-anginal Pain",
                4: "Asymptomatic",
            }[x],
        )
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80, max_value=250, value=130,
        )

    with col2:
        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=100, max_value=600, value=245,
        )
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )
        restecg = st.selectbox(
            "Resting ECG Results",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy",
            }[x],
        )
        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=60, max_value=220, value=150,
        )

    with col3:
        exang = st.selectbox(
            "Exercise Induced Angina",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )
        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0, max_value=7.0, value=1.0, step=0.1,
        )
        slope = st.selectbox(
            "Slope of Peak Exercise ST",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Upsloping",
                2: "Flat",
                3: "Downsloping",
            }[x],
        )
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia",
            [3, 6, 7],
            format_func=lambda x: {
                3: "Normal",
                6: "Fixed Defect",
                7: "Reversible Defect",
            }[x],
        )

    st.markdown("---")

    # Prediction button
    if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
        features = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }

        try:
            predictor = DiseasePredictor("heart", model_choice)
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

            # Risk category with color
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

        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Batch prediction
    st.markdown("---")
    st.subheader("📁 Batch Prediction")
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=["csv"],
        key="heart_batch",
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} records")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("Run Batch Prediction", key="heart_batch_btn"):
                predictor = DiseasePredictor("heart", model_choice)
                results_df = predictor.predict_batch(batch_df)
                st.dataframe(results_df, use_container_width=True)

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "heart_predictions.csv",
                    "text/csv",
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

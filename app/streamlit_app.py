"""
Disease Prediction System - Streamlit Application
Main entry point for the multi-disease prediction web application.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for medical theme
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1B4F72;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5D6D7E;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 1rem;
        border-radius: 10px;
    }
    .risk-medium {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
    }
    .risk-high {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        padding: 1rem;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #1B4F72;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2E86C1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main application entry point."""

    # Sidebar navigation
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/000000/heart-health.png",
            width=80,
        )
        st.title("Navigation")

        selected = option_menu(
            menu_title=None,
            options=[
                "Home",
                "Heart Disease",
                "Diabetes",
                "Breast Cancer",
                "Model Comparison",
                "Explainability",
            ],
            icons=[
                "house",
                "heart-pulse",
                "droplet",
                "virus",
                "bar-chart",
                "lightbulb",
            ],
            default_index=0,
            styles={
                "container": {"padding": "5px"},
                "icon": {"color": "#1B4F72", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "2px",
                },
                "nav-link-selected": {"background-color": "#1B4F72"},
            },
        )

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "AI-powered multi-disease prediction system using "
            "machine learning models."
        )
        st.markdown("**Version:** 1.0.0")
        st.markdown("**Author:** Piyush Kumar Rai")

    # Route to appropriate page
    if selected == "Home":
        show_home_page()
    elif selected == "Heart Disease":
        from app.pages.heart import show_heart_page
        show_heart_page()
    elif selected == "Diabetes":
        from app.pages.diabetes import show_diabetes_page
        show_diabetes_page()
    elif selected == "Breast Cancer":
        from app.pages.cancer import show_cancer_page
        show_cancer_page()
    elif selected == "Model Comparison":
        show_comparison_page()
    elif selected == "Explainability":
        show_explainability_page()


def show_home_page():
    """Display the home page."""
    st.markdown(
        '<p class="main-header">🏥 Disease Prediction System</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">'
        "AI-Powered Multi-Disease Prediction Using Machine Learning"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Overview cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #e74c3c, #c0392b); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center;">
                <h2>❤️ Heart Disease</h2>
                <p>Predict cardiovascular disease risk using 13 clinical features</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #3498db, #2980b9); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center;">
                <h2>🩸 Diabetes</h2>
                <p>Assess diabetes risk based on diagnostic measurements</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #9b59b6, #8e44ad); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center;">
                <h2>🔬 Breast Cancer</h2>
                <p>Classify tumors as benign or malignant using cell features</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # How it works
    st.header("How It Works")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 1️⃣ Input Data")
        st.markdown("Enter patient medical parameters through the form")

    with col2:
        st.markdown("### 2️⃣ Processing")
        st.markdown("Data is preprocessed and normalized")

    with col3:
        st.markdown("### 3️⃣ Prediction")
        st.markdown("ML models analyze the data")

    with col4:
        st.markdown("### 4️⃣ Results")
        st.markdown("Get risk assessment with explanations")

    st.markdown("---")

    # Key Features
    st.header("Key Features")
    features = [
        "✅ Multiple disease prediction models",
        "✅ Explainable AI with SHAP analysis",
        "✅ Risk scoring and categorization",
        "✅ Model comparison dashboard",
        "✅ Batch prediction from CSV",
        "✅ Downloadable prediction reports",
        "✅ Interactive visualizations",
        "✅ Production-ready ML pipeline",
    ]
    col1, col2 = st.columns(2)
    for i, feature in enumerate(features):
        if i < 4:
            col1.markdown(feature)
        else:
            col2.markdown(feature)

    # Models used
    st.markdown("---")
    st.header("Models Implemented")
    model_col1, model_col2, model_col3, model_col4 = st.columns(4)

    with model_col1:
        st.metric("Logistic Regression", "Linear Model")
    with model_col2:
        st.metric("SVM", "Kernel-based")
    with model_col3:
        st.metric("Random Forest", "Ensemble")
    with model_col4:
        st.metric("XGBoost", "Gradient Boosting")


def show_comparison_page():
    """Display model comparison dashboard."""
    st.header("📊 Model Comparison Dashboard")
    st.markdown("Compare performance of all models across datasets.")

    import pandas as pd
    from src.config import MODEL_RESULTS_DIR

    datasets = ["heart", "diabetes", "cancer"]
    dataset_labels = {
        "heart": "Heart Disease",
        "diabetes": "Diabetes",
        "cancer": "Breast Cancer",
    }

    for dataset in datasets:
        comparison_path = MODEL_RESULTS_DIR / f"{dataset}_comparison.csv"
        if comparison_path.exists():
            st.subheader(f"🔹 {dataset_labels[dataset]}")
            df = pd.read_csv(comparison_path)
            st.dataframe(df, use_container_width=True)

            # Highlight best model
            best_idx = df["ROC-AUC"].idxmax()
            best_model = df.loc[best_idx, "Model"]
            best_auc = df.loc[best_idx, "ROC-AUC"]
            st.success(
                f"🏆 Best Model: **{best_model}** (ROC-AUC: {best_auc:.4f})"
            )
            st.markdown("---")
        else:
            st.warning(
                f"No results found for {dataset_labels[dataset]}. "
                "Please train models first."
            )


def show_explainability_page():
    """Display explainability dashboard."""
    st.header("🔍 Model Explainability Dashboard")
    st.markdown(
        "Understand how models make predictions using SHAP analysis."
    )

    from src.config import FIGURES_DIR

    dataset_choice = st.selectbox(
        "Select Dataset",
        ["heart", "diabetes", "cancer"],
        format_func=lambda x: {
            "heart": "Heart Disease",
            "diabetes": "Diabetes",
            "cancer": "Breast Cancer",
        }[x],
    )

    # Show SHAP summary plots
    st.subheader("SHAP Summary Plots")
    models = ["random_forest", "xgboost", "logistic_regression", "svm"]
    model_labels = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "logistic_regression": "Logistic Regression",
        "svm": "SVM",
    }

    for model in models:
        shap_path = FIGURES_DIR / f"shap_summary_{dataset_choice}_{model}.png"
        if shap_path.exists():
            st.image(
                str(shap_path),
                caption=f"SHAP Summary - {model_labels[model]}",
                use_column_width=True,
            )

    # Show feature importance
    st.subheader("Feature Importance")
    for model in models:
        fi_path = (
            FIGURES_DIR
            / f"feature_importance_{dataset_choice}_{model}.png"
        )
        if fi_path.exists():
            st.image(
                str(fi_path),
                caption=f"Feature Importance - {model_labels[model]}",
                use_column_width=True,
            )


if __name__ == "__main__":
    main()

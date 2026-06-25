# 🏥 Disease Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.2-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

**AI-Powered Multi-Disease Prediction Using Machine Learning**

[Live Demo](#deployment) · [Documentation](#documentation) · [Report](reports/project_report.md)

</div>

---

## 📋 Overview

A production-ready **Multi-Disease Prediction System** that predicts three major diseases using machine learning:

| Disease | Dataset | Samples | Features | Best Model AUC |
|---------|---------|---------|----------|----------------|
| ❤️ Heart Disease | UCI Cleveland | 303 | 13 | ~0.90 |
| 🩸 Diabetes | Pima Indians | 768 | 8 | ~0.82 |
| 🔬 Breast Cancer | Wisconsin Diagnostic | 569 | 30 | ~0.99 |

The system compares **Logistic Regression**, **SVM**, **Random Forest**, and **XGBoost**, providing explainable predictions with SHAP analysis.

---

## 🏗️ Architecture

```
disease-prediction-system/
├── 📁 data/                    # Raw and processed datasets
│   ├── raw/diabetes.csv
│   └── processed/
├── 📁 notebooks/               # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_comparison.ipynb
├── 📁 models/                  # Trained model artifacts
│   ├── heart/
│   ├── diabetes/
│   └── cancer/
├── 📁 reports/                 # Generated reports and figures
│   ├── figures/
│   ├── model_results/
│   └── project_report.md
├── 📁 src/                     # Core source modules
│   ├── config.py               # Centralized configuration
│   ├── data_loader.py          # Dataset loading utilities
│   ├── preprocessing.py        # Preprocessing pipeline
│   ├── train.py                # Model training with GridSearchCV
│   ├── evaluate.py             # Evaluation metrics & plots
│   ├── explainability.py       # SHAP analysis
│   ├── predict.py              # Prediction interface
│   └── utils.py                # Utilities & helpers
├── 📁 app/                     # Streamlit web application
│   ├── streamlit_app.py        # Main app entry point
│   └── pages/                  # Individual disease pages
├── 📁 tests/                   # Unit tests
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/disease-prediction-system.git
cd disease-prediction-system

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Train Models

```bash
# Run the training notebook or use the pipeline
python -c "
from src.data_loader import load_dataset
from src.preprocessing import PreprocessingPipeline
from src.train import train_all_models
from src.evaluate import evaluate_all_models, generate_all_plots

for dataset_name in ['heart', 'diabetes', 'cancer']:
    df = load_dataset(dataset_name)
    pipeline = PreprocessingPipeline()
    X_train, X_test, y_train, y_test = pipeline.fit_transform(df)
    models = train_all_models(X_train, y_train, dataset_name)
    results = evaluate_all_models(models, X_test, y_test, dataset_name)
    generate_all_plots(results, y_test, dataset_name)
    print(f'{dataset_name} complete!')
"
```

### Launch Application

```bash
streamlit run app/streamlit_app.py
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 🧠 Models & Methodology

### Preprocessing Pipeline
- ✅ Missing value imputation (median strategy)
- ✅ Zero-as-missing replacement (medical features)
- ✅ Duplicate removal
- ✅ Outlier capping (IQR method)
- ✅ Label encoding for categorical features
- ✅ StandardScaler normalization
- ✅ Stratified train/test split (80/20)

### Classification Models
| Model | Type | Strengths |
|-------|------|-----------|
| Logistic Regression | Linear | Interpretable, fast, good baseline |
| SVM | Kernel-based | Effective in high dimensions |
| Random Forest | Ensemble | Handles non-linearity, robust |
| XGBoost | Gradient Boosting | State-of-the-art, handles imbalance |

### Hyperparameter Tuning
- GridSearchCV with Stratified 5-Fold Cross-Validation
- Optimized parameters per model per dataset
- Best model selection based on ROC-AUC

---

## 📊 Evaluation Metrics

For each model on each dataset:
- Accuracy, Precision, Recall, F1 Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curves & Precision-Recall Curves
- Cross-validation scores with standard deviation

---

## 🔍 Explainability (XAI)

- **Feature Importance:** Extracted from tree-based models and linear coefficients
- **SHAP Analysis:** 
  - Summary plots showing global feature impact
  - Force plots for individual prediction explanations
  - Feature interaction effects

---

## 🖥️ Streamlit Application

The web application provides:
- 🏠 **Home** - System overview and capabilities
- ❤️ **Heart Disease** - Prediction with 13 clinical inputs
- 🩸 **Diabetes** - Risk assessment with 8 diagnostic measurements
- 🔬 **Breast Cancer** - Classification with 30 cell features
- 📊 **Model Comparison** - Performance leaderboard
- 🔍 **Explainability** - SHAP visualizations

### Features:
- Individual patient prediction
- Batch CSV prediction
- Risk scoring (Low/Medium/High)
- Confidence percentage
- Feature contribution analysis
- Downloadable prediction reports

---

## 📈 Results Summary

### Best Models by Disease:
- **Heart Disease:** Random Forest (ROC-AUC ~0.90)
- **Diabetes:** Logistic Regression (ROC-AUC ~0.82)
- **Breast Cancer:** All models achieve ~0.99 ROC-AUC

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🚢 Deployment

### Streamlit Community Cloud
1. Push to GitHub
2. Connect repository at [share.streamlit.io](https://share.streamlit.io)
3. Set main file: `app/streamlit_app.py`

### Hugging Face Spaces
1. Create a new Space (Streamlit SDK)
2. Push code to the Space repository
3. Application deploys automatically

### Render
1. Create a new Web Service
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run app/streamlit_app.py --server.port $PORT`

---

## 📚 Documentation

- [Project Report](reports/project_report.md) - Detailed methodology and findings
- [EDA Notebook](notebooks/01_eda.ipynb) - Exploratory data analysis
- [Training Notebook](notebooks/02_model_training.ipynb) - Model training pipeline
- [Comparison Notebook](notebooks/03_model_comparison.ipynb) - Model comparison & SHAP

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.10+ |
| ML | Scikit-Learn, XGBoost |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Explainability | SHAP |
| Web App | Streamlit |
| Testing | Pytest |

---

## 👤 Author

**Piyush Kumar Rai**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for datasets
- SHAP library for explainability tools
- Streamlit for the web framework
- Scikit-Learn and XGBoost communities

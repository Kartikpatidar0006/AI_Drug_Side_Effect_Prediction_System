# 💊 AI Drug Side Effect Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**A production-ready AI-powered healthcare web application for predicting drug treatment outcomes using a pre-trained Random Forest Classifier.**

</div>

---

## 🎯 Project Overview

The **AI Drug Side Effect Prediction System** is a professional-grade Streamlit application that predicts whether a patient will experience a **positive outcome** (Recovered / Recovering) or an **adverse outcome** (Hospitalized / Fatal) based on:

- 🏥 Clinical data (drug name, dosage, side effect, severity)
- 👤 Demographics (age, gender, country)
- 🧬 Lifestyle factors (smoking status, alcohol use)
- ⚕️ Medical history (chronic conditions)

The application leverages a **Random Forest Classifier** trained on 100,000 real-world-inspired drug reports, achieving **71.2% accuracy** and **97.96% precision**.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏠 **Dashboard** | Hero section, key metrics, interactive EDA charts, dataset overview |
| 🔬 **Prediction** | Patient input form, real-time AI prediction, confidence gauge, probability breakdown |
| 📊 **Model Info** | Accuracy metrics, confusion matrix, feature importance, full parameter table |
| ℹ️ **About** | ML pipeline walkthrough, EDA findings, technology stack, developer info |
| 🎨 **Custom UI** | Dark glassmorphism theme, animated cards, Plotly charts, responsive layout |
| 🔒 **Error Handling** | Friendly error messages for missing inputs, model errors, and dataset issues |

---

## 🗂️ Project Structure

```
Drug-side-effect-model/
│
├── app.py                          # Main Streamlit entry point
├── model_loader.py                 # Model & scaler loading with caching
├── predictor.py                    # Feature engineering & prediction logic
├── utils.py                        # Shared utilities, charts, CSS helpers
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── Drug_sideeffect_model.pkl       # Pre-trained Random Forest model
├── scaler.pkl                      # Fitted StandardScaler
├── drug_side_effects_100k_dataset.csv  # Training dataset (100K rows)
│
├── assets/
│   └── style.css                   # Custom CSS (dark theme + animations)
│
└── pages/
    ├── Dashboard.py                # Home / dashboard page
    ├── Prediction.py               # Prediction form & results page
    ├── Model_Info.py               # Model metrics & analysis page
    └── About.py                    # Project overview & developer info
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11 or higher
- pip package manager

### 1. Clone / Download the Project

```bash
git clone <repository-url>
cd Drug-side-effect-model
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

### 3. Install Dependencies

> ⚠️ **Important:** The model was trained with `scikit-learn==1.6.1`. Installing this exact version prevents version mismatch warnings.

```bash
pip install -r requirements.txt
```

### 4. Verify Required Files

Ensure the following files are present in the project root:

```
✅ Drug_sideeffect_model.pkl
✅ scaler.pkl
✅ drug_side_effects_100k_dataset.csv
```

---

## ▶️ How to Run

```bash
streamlit run app.py
```

The application will open at **http://localhost:8501** in your browser.

---

## 📋 Requirements

```
streamlit>=1.32.0
scikit-learn==1.6.1
pandas>=2.0.0
numpy>=1.26.0
joblib>=1.3.0
plotly>=5.18.0
matplotlib>=3.8.0
```

---

## 🤖 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Classifier |
| n_estimators | 100 |
| max_depth | 15 |
| class_weight | balanced |
| criterion | gini |
| max_features | sqrt |
| random_state | 42 |
| **Accuracy** | **71.22%** |
| **Precision** | **97.96%** |
| **Recall** | **68.82%** |
| **F1 Score** | **80.85%** |

### Target Classes
- **Class 0** — Adverse Outcome (Hospitalized / Fatal)
- **Class 1** — Positive Outcome (Recovered / Recovering)

### Feature Engineering
- One-Hot Encoding with `drop_first=True` on all categorical features
- StandardScaler applied to all 46 encoded features
- Reference categories: Female, Australia, Amlodipine, Abdominal Pain / Blurred Vision, Mild, None / Asthma, No smoker, None / Frequent alcohol

---

## 📸 Screenshots

> *Add screenshots of the application here after running it locally.*

| Dashboard | Prediction | Model Info |
|-----------|------------|------------|
| ![Dashboard](assets/dashboard.png) | ![Prediction](assets/prediction.png) | ![Model Info](assets/model_info.png) |

---

## 🔮 Future Improvements

- [ ] XGBoost / LightGBM ensemble for improved accuracy
- [ ] SHAP explainability plots per prediction
- [ ] FHIR-compliant API for EHR integration
- [ ] Mobile-responsive Progressive Web App (PWA)
- [ ] Role-based authentication & audit logging
- [ ] Multi-language interface support
- [ ] Batch prediction via CSV upload
- [ ] Real-time model monitoring dashboard

---

## ⚠️ Medical Disclaimer

> This application is intended for **research and educational purposes only**.
> It does **not** constitute medical advice and should **not** be used to make
> clinical decisions. Always consult a qualified healthcare professional before
> making any treatment decisions.

---

## 📄 License

MIT License — Free to use for educational and portfolio purposes.

---

<div align="center">
Built with ❤️ using Python, Streamlit & Scikit-Learn
</div>

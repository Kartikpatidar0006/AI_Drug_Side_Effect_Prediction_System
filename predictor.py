"""
predictor.py
------------
Handles feature engineering and prediction logic.
Encodes user inputs into the exact feature vector format
that matches the trained Random Forest model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import warnings


# ── Feature metadata ──────────────────────────────────────────────────────────

# Exact feature order expected by the scaler / model
FEATURE_COLUMNS = [
    'age', 'dosage_mg',
    'gender_Male',
    'country_Canada', 'country_Germany', 'country_India',
    'country_Pakistan', 'country_UK', 'country_USA',
    'drug_name_Amoxicillin', 'drug_name_Atorvastatin', 'drug_name_Ibuprofen',
    'drug_name_Insulin', 'drug_name_Lisinopril', 'drug_name_Metformin',
    'drug_name_Omeprazole', 'drug_name_Paracetamol', 'drug_name_Sertraline',
    'side_effect_Anxiety', 'side_effect_Constipation', 'side_effect_Diarrhea',
    'side_effect_Dizziness', 'side_effect_Dry Cough', 'side_effect_Dry Mouth',
    'side_effect_Fatigue', 'side_effect_Headache', 'side_effect_Heartburn',
    'side_effect_Hypoglycemia', 'side_effect_Insomnia',
    'side_effect_Liver Toxicity', 'side_effect_Muscle Pain',
    'side_effect_Nausea', 'side_effect_Palpitations', 'side_effect_Rash',
    'side_effect_Stomach Pain', 'side_effect_Sweating', 'side_effect_Swelling',
    'side_effect_Weight Gain',
    'severity_Moderate', 'severity_Severe',
    'chronic_condition_Diabetes', 'chronic_condition_Heart Disease',
    'chronic_condition_Hypertension', 'chronic_condition_Kidney Disease',
    'smoker_Yes',
    'alcohol_use_Occasional',
]

# Valid categorical values
GENDERS = ["Female", "Male"]
COUNTRIES = ["Australia", "Canada", "Germany", "India", "Pakistan", "UK", "USA"]
DRUG_NAMES = [
    "Amlodipine", "Amoxicillin", "Atorvastatin", "Ibuprofen", "Insulin",
    "Lisinopril", "Metformin", "Omeprazole", "Paracetamol", "Sertraline",
]
SIDE_EFFECTS = [
    "Abdominal Pain", "Anxiety", "Blurred Vision", "Constipation", "Diarrhea",
    "Dizziness", "Dry Cough", "Dry Mouth", "Fatigue", "Headache", "Heartburn",
    "Hypoglycemia", "Insomnia", "Liver Toxicity", "Muscle Pain", "Nausea",
    "Palpitations", "Rash", "Stomach Pain", "Sweating", "Swelling",
    "Weight Gain",
]
SEVERITIES = ["Mild", "Moderate", "Severe"]
CHRONIC_CONDITIONS = [
    "None", "Asthma", "Diabetes", "Heart Disease", "Hypertension",
    "Kidney Disease",
]
SMOKER_OPTIONS = ["No", "Yes"]
ALCOHOL_OPTIONS = ["None", "Occasional", "Frequent"]
DOSAGE_OPTIONS = [5, 10, 20, 25, 50, 100, 250, 500]

# Class labels
CLASS_LABELS = {0: "Adverse Outcome", 1: "Positive Outcome"}
CLASS_DESCRIPTIONS = {
    0: "The model predicts an **adverse outcome** (Hospitalized / Fatal). "
       "The patient's condition may worsen or require hospitalization.",
    1: "The model predicts a **positive outcome** (Recovered / Recovering). "
       "The patient is likely to recover successfully from treatment.",
}


def build_feature_vector(inputs: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert raw user inputs into a one-hot-encoded DataFrame
    that exactly matches the feature vector used during training.

    Parameters
    ----------
    inputs : dict
        Dictionary with keys: age, dosage_mg, gender, country, drug_name,
        side_effect, severity, chronic_condition, smoker, alcohol_use.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all 46 features in the correct order.
    """
    row: Dict[str, int | float] = {col: 0 for col in FEATURE_COLUMNS}

    # Numeric features
    row['age'] = float(inputs['age'])
    row['dosage_mg'] = float(inputs['dosage_mg'])

    # Binary / one-hot features
    if inputs['gender'] == 'Male':
        row['gender_Male'] = 1

    # Country (drop_first → reference = Australia)
    country_key = f"country_{inputs['country']}"
    if country_key in row:
        row[country_key] = 1

    # Drug name (drop_first → reference = Amlodipine)
    drug_key = f"drug_name_{inputs['drug_name']}"
    if drug_key in row:
        row[drug_key] = 1

    # Side effect (drop_first → reference = Abdominal Pain / Blurred Vision)
    se_key = f"side_effect_{inputs['side_effect']}"
    if se_key in row:
        row[se_key] = 1

    # Severity (drop_first → reference = Mild)
    if inputs['severity'] == 'Moderate':
        row['severity_Moderate'] = 1
    elif inputs['severity'] == 'Severe':
        row['severity_Severe'] = 1

    # Chronic condition (drop_first → reference = None / Asthma)
    cc_key = f"chronic_condition_{inputs['chronic_condition']}"
    if cc_key in row:
        row[cc_key] = 1

    # Smoker
    if inputs['smoker'] == 'Yes':
        row['smoker_Yes'] = 1

    # Alcohol use (drop_first → reference = None / Frequent)
    if inputs['alcohol_use'] == 'Occasional':
        row['alcohol_use_Occasional'] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict(
    inputs: Dict[str, Any],
    model: Any,
    scaler: Any,
) -> Tuple[int, float, np.ndarray]:
    """
    Run the full prediction pipeline.

    Parameters
    ----------
    inputs : dict
        Raw user inputs.
    model : sklearn estimator
        Loaded RandomForestClassifier.
    scaler : sklearn transformer
        Loaded StandardScaler.

    Returns
    -------
    Tuple[int, float, np.ndarray]
        (predicted_class, confidence_score, probability_array)
    """
    feature_df = build_feature_vector(inputs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_scaled = scaler.transform(feature_df)
        predicted_class = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]

    confidence = float(probabilities[predicted_class])
    return predicted_class, confidence, probabilities


def get_risk_level(predicted_class: int, confidence: float) -> Tuple[str, str]:
    """
    Derive a human-readable risk level from the prediction.

    Returns
    -------
    Tuple[str, str]
        (risk_label, risk_color_hex)
    """
    if predicted_class == 0:
        if confidence >= 0.80:
            return "🔴 Critical Risk", "#FF4B4B"
        else:
            return "🟠 High Risk", "#FF8C00"
    else:
        if confidence >= 0.80:
            return "🟢 Low Risk", "#00C48C"
        else:
            return "🟡 Moderate Risk", "#FFD700"

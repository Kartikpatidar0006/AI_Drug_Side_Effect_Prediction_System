"""
model_loader.py
---------------
Handles loading of the pre-trained Random Forest model and StandardScaler.
Uses caching to avoid reloading on every Streamlit rerun.
"""

import joblib
import streamlit as st
import os
from typing import Tuple, Any


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "Drug_sideeffect_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    """
    Load the pre-trained Random Forest Classifier from disk.
    
    Returns:
        Loaded sklearn RandomForestClassifier model.
    
    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: For any other loading error.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Please ensure it exists in the project root."
        )
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")


@st.cache_resource(show_spinner=False)
def load_scaler() -> Any:
    """
    Load the pre-fitted StandardScaler from disk.
    
    Returns:
        Loaded sklearn StandardScaler object.
    
    Raises:
        FileNotFoundError: If the scaler file does not exist.
        Exception: For any other loading error.
    """
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler file '{SCALER_PATH}' not found. "
            "Please ensure it exists in the project root."
        )
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = joblib.load(SCALER_PATH)
        return scaler
    except Exception as e:
        raise Exception(f"Failed to load scaler: {str(e)}")


def load_model_and_scaler() -> Tuple[Any, Any]:
    """
    Convenience function to load both model and scaler.
    
    Returns:
        Tuple of (model, scaler).
    """
    model = load_model()
    scaler = load_scaler()
    return model, scaler

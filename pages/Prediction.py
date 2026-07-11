"""
pages/Prediction.py
-------------------
Prediction page.
Accepts patient clinical and lifestyle inputs, encodes them
exactly as during training, applies the scaler, and predicts
treatment outcome using the Random Forest model.
"""

import streamlit as st
import plotly.graph_objects as go
from predictor import (
    predict,
    get_risk_level,
    build_feature_vector,
    CLASS_LABELS,
    CLASS_DESCRIPTIONS,
    GENDERS,
    COUNTRIES,
    DRUG_NAMES,
    SIDE_EFFECTS,
    SEVERITIES,
    CHRONIC_CONDITIONS,
    SMOKER_OPTIONS,
    ALCOHOL_OPTIONS,
    DOSAGE_OPTIONS,
)
from utils import make_probability_bar, section_header, COLORS


# ── Recommendation text based on risk level ────────────────────────────────────

_RECOMMENDATIONS = {
    "🔴 Critical Risk": (
        "Immediate medical review is strongly advised. Consider hospitalisation, "
        "medication adjustment, or escalation of care. Do not delay clinical evaluation."
    ),
    "🟠 High Risk": (
        "Close monitoring is recommended. Consult a physician promptly. "
        "Review dosage and assess suitability of the current medication regimen."
    ),
    "🟡 Moderate Risk": (
        "Follow-up appointment recommended within 1–2 weeks. "
        "Monitor for worsening symptoms and maintain adherence to the prescribed regimen."
    ),
    "🟢 Low Risk": (
        "Patient is likely recovering well. Continue current treatment as prescribed. "
        "Routine check-up at the next scheduled appointment."
    ),
}


def _render_form() -> dict:
    """
    Render the patient input form and return raw input values.
    Returns a dict that predictor.build_feature_vector() can consume.
    """

    # ── Section 1: Patient Information ────────────────────────────────────────
    st.markdown("""
    <div class="form-section-card">
        <div class="form-section-title">👤 Patient Information</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider(
            "Patient Age (years)", min_value=18, max_value=90, value=45,
            help="Age of the patient in years.",
        )
    with c2:
        gender = st.selectbox(
            "Gender", options=GENDERS,
            help="Biological sex of the patient.",
        )
    with c3:
        country = st.selectbox(
            "Country", options=COUNTRIES,
            help="Country where the treatment was administered.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Medication Information ─────────────────────────────────────
    st.markdown("""
    <div class="form-section-card">
        <div class="form-section-title">💊 Medication Information</div>
    </div>
    """, unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        drug_name = st.selectbox(
            "Drug Name", options=DRUG_NAMES,
            help="Name of the drug being evaluated.",
        )
    with c5:
        dosage_mg = st.selectbox(
            "Dosage (mg)", options=DOSAGE_OPTIONS,
            help="Prescribed dosage in milligrams.",
        )
    with c6:
        side_effect = st.selectbox(
            "Reported Side Effect", options=SIDE_EFFECTS,
            help="The primary side effect reported by the patient.",
        )

    c7, c8 = st.columns(2)
    with c7:
        severity = st.selectbox(
            "Severity Level", options=SEVERITIES,
            help="How severe is the reported side effect?",
        )
    with c8:
        chronic_condition = st.selectbox(
            "Chronic Condition", options=CHRONIC_CONDITIONS,
            help="Select 'None' if the patient has no chronic condition.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 3: Lifestyle Information ──────────────────────────────────────
    st.markdown("""
    <div class="form-section-card">
        <div class="form-section-title">🏃 Lifestyle Information</div>
    </div>
    """, unsafe_allow_html=True)

    c9, c10 = st.columns(2)
    with c9:
        smoker = st.selectbox(
            "Smoker?", options=SMOKER_OPTIONS,
            help="Does the patient smoke?",
        )
    with c10:
        alcohol_use = st.selectbox(
            "Alcohol Use", options=ALCOHOL_OPTIONS,
            help="Frequency of alcohol consumption.",
        )

    return {
        "age":               age,
        "dosage_mg":         dosage_mg,
        "gender":            gender,
        "country":           country,
        "drug_name":         drug_name,
        "side_effect":       side_effect,
        "severity":          severity,
        "chronic_condition": chronic_condition,
        "smoker":            smoker,
        "alcohol_use":       alcohol_use,
    }


def _render_result(inputs: dict, model, scaler) -> None:
    """Run prediction and render the premium result section."""
    with st.spinner("🔍 Analysing patient data…"):
        try:
            predicted_class, confidence, probabilities = predict(inputs, model, scaler)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            return

    risk_label, risk_color = get_risk_level(predicted_class, confidence)
    label       = CLASS_LABELS[predicted_class]
    description = CLASS_DESCRIPTIONS[predicted_class]
    recommendation = _RECOMMENDATIONS.get(risk_label, "Consult a qualified healthcare professional.")

    # ── Result Card ────────────────────────────────────────────────────────────
    st.markdown("---")
    section_header("🎯", "Prediction Result", "AI-generated treatment outcome analysis")

    card_class   = "success" if predicted_class == 1 else "danger"
    outcome_icon = "✅" if predicted_class == 1 else "🚨"

    conf_bg    = "rgba(16,185,129,0.12)"  if predicted_class == 1 else "rgba(239,68,68,0.12)"
    conf_color = COLORS["success"]         if predicted_class == 1 else COLORS["danger"]
    conf_border = "rgba(16,185,129,0.4)"  if predicted_class == 1 else "rgba(239,68,68,0.4)"

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-headline">{outcome_icon} {label}</div>
        <p class="result-sub">{description}</p>
        <br>
        <span class="confidence-badge"
              style="background:{conf_bg};color:{conf_color};border:1px solid {conf_border};">
            {risk_label}
        </span>
        &nbsp;&nbsp;
        <span class="confidence-badge"
              style="background:rgba(37,99,235,0.1);color:{COLORS['primary']};border:1px solid rgba(37,99,235,0.3);">
            🎯 Confidence: {confidence*100:.1f}%
        </span>
        <div class="recommendation-card" style="margin-top:1.2rem;
            background:{'rgba(16,185,129,0.06)' if predicted_class==1 else 'rgba(239,68,68,0.06)'};
            border:1px solid {conf_border};">
            <strong style="color:{conf_color};">📋 Clinical Recommendation</strong>
            <p>{recommendation}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability breakdown ──────────────────────────────────────────────────
    col_prob, col_details = st.columns([3, 2])

    with col_prob:
        st.markdown("**📊 Probability Distribution**")
        st.plotly_chart(
            make_probability_bar(float(probabilities[0]), float(probabilities[1])),
            use_container_width=True,
        )

        # Confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 36, "color": COLORS["text"]}},
            title={"text": "Model Confidence", "font": {"size": 14, "color": COLORS["muted"]}},
            delta={"reference": 70, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
                "bar":  {"color": risk_color, "thickness": 0.25},
                "bgcolor": COLORS["bg_card2"],
                "bordercolor": COLORS["border"],
                "steps": [
                    {"range": [0,  50], "color": "#F1F5F9"},
                    {"range": [50, 75], "color": "#EFF6FF"},
                    {"range": [75, 100], "color": "#DCFCE7"},
                ],
                "threshold": {
                    "line": {"color": risk_color, "width": 3},
                    "thickness": 0.75, "value": confidence * 100,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=COLORS["text"],
            height=240,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_details:
        # Prediction summary
        st.markdown("**📋 Prediction Summary**")
        rows = [
            ("Prediction",              label),
            ("Probability — Positive",  f"{probabilities[1]*100:.2f}%"),
            ("Probability — Adverse",   f"{probabilities[0]*100:.2f}%"),
            ("Confidence",              f"{confidence*100:.2f}%"),
            ("Risk Level",              risk_label),
        ]
        for row_title, row_val in rows:
            st.markdown(f"""
            <div class="summary-row">
                <span class="summary-label">{row_title}</span>
                <span class="summary-value">{row_val}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Input summary
        st.markdown("**🧬 Input Summary**")
        input_display = {
            "Age":               inputs["age"],
            "Gender":            inputs["gender"],
            "Country":           inputs["country"],
            "Drug":              inputs["drug_name"],
            "Dosage":            f"{inputs['dosage_mg']} mg",
            "Side Effect":       inputs["side_effect"],
            "Severity":          inputs["severity"],
            "Chronic Condition": inputs["chronic_condition"],
            "Smoker":            inputs["smoker"],
            "Alcohol Use":       inputs["alcohol_use"],
        }
        for k, v in input_display.items():
            st.markdown(f"""
            <div class="summary-row">
                <span class="summary-label">{k}</span>
                <span style="color:{COLORS['primary']};font-weight:500;font-size:0.82rem;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Feature Vector (expander) ──────────────────────────────────────────────
    with st.expander("🔢 View Encoded Feature Vector"):
        feature_df = build_feature_vector(inputs)
        st.dataframe(feature_df.T.rename(columns={0: "Value"}), use_container_width=True)

    # ── Medical Disclaimer ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This AI prediction is intended for
        research and educational purposes only. It does <em>not</em> constitute
        medical advice and should not be used to make clinical decisions.
        Always consult a qualified healthcare professional before making any
        treatment decisions. Prediction accuracy is limited by the training
        data and model constraints.
    </div>
    """, unsafe_allow_html=True)


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    """Render the Prediction page."""

    model  = st.session_state.get("model")
    scaler = st.session_state.get("scaler")
    err    = st.session_state.get("model_error")

    # Page header
    section_header("🔬", "Treatment Outcome Predictor",
                   "Enter patient details below to receive an AI-generated prediction")

    # Model availability check
    if err:
        st.error(f"❌ Model could not be loaded: {err}")
        st.info("💡 Ensure `Drug_sideeffect_model.pkl` and `scaler.pkl` exist in the project root.")
        return

    if model is None or scaler is None:
        st.warning("⏳ Model is still loading. Please wait and refresh.")
        return

    # Input form
    with st.form(key="prediction_form", clear_on_submit=False):
        inputs = _render_form()

        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns([2, 5])
        with col_btn1:
            submitted = st.form_submit_button(
                "🚀 Generate Prediction",
                use_container_width=True,
            )
        with col_btn2:
            st.markdown(
                f'<p style="color:{COLORS["muted"]};font-size:0.82rem;padding-top:0.7rem;">'
                '&nbsp; Click to run the Random Forest model on the entered patient data.</p>',
                unsafe_allow_html=True,
            )

    # Result section (rendered outside the form)
    if submitted:
        _render_result(inputs, model, scaler)
    else:
        st.markdown("""
        <div class="custom-alert alert-info">
            💡 Fill in all the patient details above and click
            <strong>Generate Prediction</strong> to see the AI outcome analysis.
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <span>AI Drug Side Effect Prediction System</span> · v1.0 ·
        Random Forest Classifier | Scikit-Learn
    </div>
    """, unsafe_allow_html=True)

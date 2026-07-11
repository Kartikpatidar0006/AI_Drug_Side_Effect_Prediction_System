"""
pages/About.py
--------------
About page.
Explains the project overview, ML pipeline, EDA, feature engineering,
technology stack, and developer information.
"""

import streamlit as st
from utils import COLORS, TESTING_ACCURACY, section_header


def render() -> None:
    """Render the About page."""

    # ── Hero ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-container fade-in">
        <div class="hero-badge">📖 About This Project</div>
        <h1 class="hero-title">AI Drug Side Effect<br>Prediction System</h1>
        <p class="hero-desc">
            A production-ready AI application that leverages machine learning to support
            healthcare professionals with evidence-based treatment outcome prediction.
            This project showcases a complete end-to-end ML pipeline — from raw data
            to deployed model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Project Overview ───────────────────────────────────────────────────────
    section_header("🏥", "Project Overview",
                   "The problem, solution, and business impact")

    o1, o2 = st.columns(2)
    with o1:
        st.markdown(f"""
        <div class="info-card">
            <h4>🎯 Objective</h4>
            <p>
            Develop an intelligent clinical decision support tool that predicts
            whether a patient will experience a <strong>positive outcome</strong>
            (Recovered / Recovering) or an <strong>adverse outcome</strong>
            (Hospitalized / Fatal) based on drug, demographic, and lifestyle inputs.
            </p>
        </div>

        <div class="info-card">
            <h4>📊 Dataset</h4>
            <p>
            The model was trained on a synthetic but realistic dataset of
            <strong>100,000 drug side-effect reports</strong> covering 10 widely-used
            medications across 7 countries, with 16 original features spanning clinical,
            demographic, and lifestyle domains.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with o2:
        st.markdown(f"""
        <div class="info-card">
            <h4>⚕️ Healthcare Impact</h4>
            <p>
            Early identification of patients at risk of adverse outcomes allows
            healthcare teams to intervene proactively — adjusting dosages,
            switching medications, or escalating care pathways before
            complications arise.
            </p>
        </div>

        <div class="info-card">
            <h4>🔒 Responsible AI</h4>
            <p>
            This tool is designed for educational and research purposes only.
            All predictions come with confidence scores, probability breakdowns,
            a clinical recommendation, and a prominent medical disclaimer —
            reinforcing that it supplements, not replaces, clinical judgment.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── ML Pipeline ────────────────────────────────────────────────────────────
    section_header("⚙️", "Machine Learning Pipeline",
                   "Step-by-step walkthrough of the model development process")

    pipeline_steps = [
        ("Data Collection",
         "Collected 100,000 synthetic patient records with 16 features including patient "
         "demographics, drug information, side effect type, severity, chronic conditions, "
         "and lifestyle factors."),
        ("Exploratory Data Analysis (EDA)",
         "Analysed distributions, identified class imbalance (Recovered 75% vs Hospitalized/Fatal 25%), "
         "explored correlations between features and outcomes, and visualised drug-by-outcome patterns."),
        ("Feature Engineering",
         "Created a binary target variable: 1 (Recovered/Recovering), 0 (Hospitalized/Fatal). "
         "Dropped non-predictive columns (patient_id, dates, hospitalized flag, recovery_days). "
         "Applied one-hot encoding with drop_first=True to all categorical variables."),
        ("Feature Scaling",
         "Applied StandardScaler to all 46 encoded features. Fitting was done only on training data "
         "to prevent data leakage. The fitted scaler was serialised as scaler.pkl."),
        ("Model Training",
         "Trained a Random Forest Classifier (100 trees, max_depth=15, class_weight='balanced') "
         "using Scikit-Learn. The balanced class weight compensates for the outcome imbalance."),
        ("Model Evaluation",
         f"Achieved {TESTING_ACCURACY}% testing accuracy with high precision, recall, and F1-Score. "
         "High precision ensures false-positive predictions (predicting Positive when Adverse) "
         "are minimised."),
        ("Model Serialisation",
         "Saved the trained model as Drug_sideeffect_model.pkl and the scaler as scaler.pkl "
         "using Joblib for efficient loading in production."),
        ("Deployment",
         "Built a full-stack Streamlit web application with custom CSS, interactive forms, "
         "Plotly visualisations, and modular page routing."),
    ]

    for i, (title, desc) in enumerate(pipeline_steps, 1):
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="step-number">{i}</div>
            <div class="step-content">
                <h5>{title}</h5>
                <p>{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── EDA Highlights ─────────────────────────────────────────────────────────
    section_header("🔍", "EDA Highlights",
                   "Key findings from the exploratory data analysis")

    e1, e2, e3 = st.columns(3)
    eda_items = [
        (e1, "📈 Severity is Dominant",
         "Side effect severity accounts for over 74% of total feature importance, "
         "making it the single most predictive feature in the dataset."),
        (e2, "👴 Age Matters",
         "Patients over 65 show 3× higher adverse outcome rates compared to patients "
         "under 30, confirming age as a critical risk factor."),
        (e3, "🚬 Lifestyle Risk",
         "Smokers have a higher chance of adverse outcomes. "
         "Combined with frequent alcohol use, the overall risk increases further."),
    ]
    for col, title, desc in eda_items:
        with col:
            st.markdown(f"""
            <div class="info-card">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Prediction Flow ────────────────────────────────────────────────────────
    section_header("🔄", "Prediction Flow",
                   "How the app transforms raw inputs into a prediction")

    flow_steps = [
        ("User Input",
         "The clinician enters patient data via the web form: age, gender, country, drug, "
         "dosage, side effect, severity, chronic condition, smoking status, and alcohol use."),
        ("One-Hot Encoding",
         "Categorical fields are transformed into binary indicator columns using the same "
         "drop_first=True scheme as training. The exact 46-column feature vector is constructed "
         "in the correct order defined by scaler.feature_names_in_."),
        ("Feature Scaling",
         "The 46-column vector is passed through the loaded StandardScaler (scaler.pkl), "
         "which applies the mean and standard deviation fitted during training."),
        ("Random Forest Prediction",
         "The scaled vector is fed to the Random Forest Classifier (Drug_sideeffect_model.pkl). "
         "Each of the 100 decision trees votes, and the majority vote determines the class."),
        ("Output Generation",
         "The app returns the predicted class (0/1), class probabilities, confidence score, "
         "risk level badge, clinical recommendation, and a human-readable description of the outcome."),
    ]

    for i, (step_title, step_desc) in enumerate(flow_steps, 1):
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="step-number">{i}</div>
            <div class="step-content">
                <h5>{step_title}</h5>
                <p>{step_desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Technology Stack ───────────────────────────────────────────────────────
    section_header("🛠️", "Technology Stack",
                   "Libraries and frameworks powering this application")

    tech_cols = st.columns(2)
    tech_groups = [
        ("🐍 Backend & ML", [
            ("Python 3.11+",  "Core programming language"),
            ("Scikit-Learn",  "Random Forest model & preprocessing"),
            ("Pandas",        "Data manipulation & feature engineering"),
            ("NumPy",         "Numerical operations & array processing"),
            ("Joblib",        "Model serialisation / deserialisation"),
        ]),
        ("🎨 Frontend & Visualisation", [
            ("Streamlit",     "Web application framework"),
            ("Plotly",        "Interactive charts & gauges"),
            ("Custom CSS3",   "Professional light-mode design system"),
            ("Google Fonts",  "Inter & Outfit typography"),
        ]),
    ]

    for col, (group_title, techs) in zip(tech_cols, tech_groups):
        with col:
            st.markdown(f"""
            <div class="info-card">
                <h4>{group_title}</h4>
            </div>
            """, unsafe_allow_html=True)
            for tech_name, tech_desc in techs:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.75rem;
                            padding:0.5rem 0;border-bottom:1px solid {COLORS['border']};">
                    <span class="tech-badge">{tech_name}</span>
                    <span style="color:{COLORS['muted']};font-size:0.82rem;">{tech_desc}</span>
                </div>
                """, unsafe_allow_html=True)

    # ── Future Improvements ────────────────────────────────────────────────────
    section_header("🚀", "Future Improvements",
                   "Planned enhancements for the next version")

    f1, f2 = st.columns(2)
    future_items = [
        ("🧠 Advanced Models",
         "XGBoost, LightGBM, and Neural Network ensemble models for improved accuracy and recall."),
        ("📱 Mobile App",
         "React Native / Flutter companion app for on-the-go clinical decision support."),
        ("🔗 EHR Integration",
         "FHIR-compliant API to integrate with Electronic Health Record systems like Epic and Cerner."),
        ("📊 SHAP Explainability",
         "Per-prediction SHAP waterfall plots to explain exactly why the model made each decision."),
        ("🌐 Multi-language Support",
         "Interface localisation for global healthcare deployments."),
        ("🔐 Authentication & Audit Logs",
         "Role-based access control and full audit trail for clinical-grade compliance."),
    ]

    for i, (title, desc) in enumerate(future_items):
        col = f1 if i % 2 == 0 else f2
        with col:
            st.markdown(f"""
            <div class="info-card">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Developer Info ─────────────────────────────────────────────────────────
    section_header("👨‍💻", "Developer Information", "About the creator of this project")

    st.markdown(f"""
    <div class="info-card" style="padding:2rem;">
        <div style="display:flex;align-items:center;gap:1.5rem;margin-bottom:1.2rem;">
            <div style="width:72px;height:72px;border-radius:50%;
                        background:linear-gradient(135deg,{COLORS['primary']},{COLORS['secondary']});
                        display:flex;align-items:center;justify-content:center;
                        font-size:2rem;flex-shrink:0;
                        box-shadow:0 4px 16px rgba(37,99,235,0.3);">👨‍⚕️</div>
            <div>
                <h3 style="margin:0;font-family:'Outfit',sans-serif;font-size:1.3rem;
                           color:{COLORS['text']};">AI &amp; Machine Learning Engineer</h3>
                <p style="margin:0.2rem 0 0;color:{COLORS['muted']};font-size:0.88rem;">
                    Full Stack Developer · Healthcare AI Specialist
                </p>
            </div>
        </div>
        <p style="color:{COLORS['muted']};font-size:0.9rem;line-height:1.75;">
            This project was developed as a portfolio-grade demonstration of end-to-end
            machine learning engineering — from raw data processing and model training,
            to production-ready deployment with a professional-grade web interface.
            It showcases expertise in healthcare AI, feature engineering, model evaluation,
            and modern UI/UX design principles.
        </p>
        <div style="margin-top:1rem;">
            <span class="tech-badge">Python</span>
            <span class="tech-badge">Machine Learning</span>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">Healthcare AI</span>
            <span class="tech-badge">Data Science</span>
            <span class="tech-badge">Feature Engineering</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Version & License ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;gap:1.5rem;margin-top:1.2rem;">
        <div class="info-card" style="flex:1;text-align:center;">
            <p style="color:{COLORS['muted']};font-size:0.78rem;margin:0 0 0.3rem;">Version</p>
            <h3 style="margin:0;color:{COLORS['primary']};font-family:'Outfit',sans-serif;">v1.0.0</h3>
        </div>
        <div class="info-card" style="flex:1;text-align:center;">
            <p style="color:{COLORS['muted']};font-size:0.78rem;margin:0 0 0.3rem;">License</p>
            <h3 style="margin:0;color:{COLORS['primary']};font-family:'Outfit',sans-serif;">MIT</h3>
        </div>
        <div class="info-card" style="flex:1;text-align:center;">
            <p style="color:{COLORS['muted']};font-size:0.78rem;margin:0 0 0.3rem;">Model</p>
            <h3 style="margin:0;color:{COLORS['primary']};font-family:'Outfit',sans-serif;">Random Forest</h3>
        </div>
        <div class="info-card" style="flex:1;text-align:center;">
            <p style="color:{COLORS['muted']};font-size:0.78rem;margin:0 0 0.3rem;">Status</p>
            <h3 style="margin:0;color:{COLORS['success']};font-family:'Outfit',sans-serif;">🟢 Live</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        <span>AI Drug Side Effect Prediction System</span> · v1.0 ·
        Open-source for educational purposes
    </div>
    """, unsafe_allow_html=True)

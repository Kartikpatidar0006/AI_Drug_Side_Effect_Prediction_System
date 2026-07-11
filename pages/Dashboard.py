"""
pages/Dashboard.py
------------------
Home / Dashboard page.
Shows hero section, dataset statistics, key insight cards, and charts.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import (
    get_dataset_stats,
    metric_card,
    section_header,
    make_outcome_pie_chart,
    make_gender_chart,
    make_drug_bar_chart,
    make_severity_chart,
    COLORS,
    TESTING_ACCURACY,
)


def render() -> None:
    """Render the Dashboard page."""

    df = st.session_state.get("dataset")
    model = st.session_state.get("model")

    # ── Hero Section ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-container fade-in">
        <div class="hero-badge">✨ AI-Powered Healthcare Analytics</div>
        <h1 class="hero-title">Drug Side Effect<br>Prediction System</h1>
        <p class="hero-desc">
            A state-of-the-art machine learning platform that predicts patient treatment outcomes
            based on clinical and lifestyle data. Built with a Random Forest Classifier trained
            on 100,000 real-world drug reports.
        </p>
        <div>
            <span class="hero-stat">🎯 {TESTING_ACCURACY}% Testing Accuracy</span>
            <span class="hero-stat">📊 100K Patient Records</span>
            <span class="hero-stat">🌍 7 Countries</span>
            <span class="hero-stat">💊 10 Drugs Covered</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick-Navigate CTA ─────────────────────────────────────────────────────
    col_cta1, col_cta2, col_cta3 = st.columns([1, 1, 2])
    with col_cta1:
        if st.button("🔬 Start Prediction", use_container_width=True, key="dash_predict_btn"):
            st.session_state["_nav_target"] = "🔬  Prediction"
            st.rerun()
    with col_cta2:
        if st.button("📊 Model Details", use_container_width=True, key="dash_model_btn"):
            st.session_state["_nav_target"] = "📊  Model Information"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Metrics ────────────────────────────────────────────────────────────
    section_header("📈", "Key Metrics", "Overview of the dataset and model performance")

    if df is not None:
        stats = get_dataset_stats(df)

        col1, col2, col3, col4 = st.columns(4)
        cards = [
            (col1, "Total Patients",  f"{stats['n_samples']:,}",       "🧑‍⚕️", "100,000 records"),
            (col2, "Feature Count",   str(stats["n_features"]),         "🔢",   "46 encoded features"),
            (col3, "Model",           "Random Forest",                  "🌲",   "100 estimators"),
            (col4, "Testing Accuracy", f"{TESTING_ACCURACY}%",         "🎯",   "On full dataset"),
        ]
        for col, title, val, icon, delta in cards:
            with col:
                st.markdown(metric_card(title, val, icon, delta), unsafe_allow_html=True)
    else:
        st.error("⚠️ Dataset could not be loaded. Please ensure the CSV file is present.")
        return

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset Overview ───────────────────────────────────────────────────────
    section_header("📋", "Dataset Overview", "Statistical summary of the 100K patient dataset")

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "💊 Drug Analysis", "📋 Raw Sample"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_outcome_pie_chart(df), use_container_width=True)
        with c2:
            st.plotly_chart(make_gender_chart(df), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(make_severity_chart(df), use_container_width=True)
        with c4:
            country_counts = df['country'].value_counts().reset_index()
            country_counts.columns = ['Country', 'Count']
            fig = px.bar(
                country_counts, x='Country', y='Count',
                color='Count',
                color_continuous_scale=[[0, "#DBEAFE"], [1, COLORS["primary"]]],
                template="plotly_white",
            )
            fig.update_layout(
                title=dict(text="Reports by Country",
                           font=dict(color=COLORS["text"], size=16)),
                xaxis=dict(color=COLORS["muted"], gridcolor=COLORS["border"]),
                yaxis=dict(color=COLORS["muted"], gridcolor=COLORS["border"]),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                font_color=COLORS["text"],
                height=340,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.plotly_chart(make_drug_bar_chart(df), use_container_width=True)

        # Drug × Outcome heatmap
        pivot = df.groupby(['drug_name', 'outcome']).size().unstack(fill_value=0)
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0, "#EFF6FF"], [1, COLORS["primary"]]],
            text=pivot.values,
            texttemplate="%{text:,}",
            textfont={"size": 11, "color": COLORS["text"]},
        ))
        fig_heat.update_layout(
            title=dict(text="Drug × Outcome Heatmap",
                       font=dict(color=COLORS["text"], size=16)),
            xaxis=dict(color=COLORS["text"]),
            yaxis=dict(color=COLORS["text"]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=COLORS["text"],
            height=400,
            margin=dict(l=120, r=20, t=50, b=60),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.markdown("""
        <div class="info-card">
            <h4>📄 Raw Dataset Preview (First 10 Rows)</h4>
            <p>A glimpse at the raw data before preprocessing and one-hot encoding.</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**📊 Numeric Summary**")
            st.dataframe(
                df[["age", "dosage_mg", "recovery_days"]].describe().round(2),
                use_container_width=True,
            )
        with col_b:
            st.markdown("**📋 Outcome Distribution**")
            oc = df['outcome'].value_counts().reset_index()
            oc.columns = ['Outcome', 'Count']
            oc['Percentage'] = (oc['Count'] / len(df) * 100).round(2).astype(str) + '%'
            st.dataframe(oc, use_container_width=True, hide_index=True)

    # ── Feature Highlights ─────────────────────────────────────────────────────
    section_header("💡", "Feature Highlights", "Key predictor groups used by the model")

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="info-card">
            <h4>⚕️ Clinical Features</h4>
            <p>Age, Dosage (mg), Drug Name, Reported Side Effect,
            Severity Level, and Chronic Conditions form the core
            clinical input of the model.</p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="info-card">
            <h4>🧬 Lifestyle Factors</h4>
            <p>Smoking status and alcohol consumption patterns are
            significant lifestyle predictors included to improve
            outcome estimation accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="info-card">
            <h4>🌍 Demographic Data</h4>
            <p>Gender and Country of origin provide demographic context
            that helps the model account for population-level differences
            in drug response patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        <span>AI Drug Side Effect Prediction System</span> · v1.0 ·
        Built with ❤️ using Python, Streamlit &amp; Scikit-Learn
    </div>
    """, unsafe_allow_html=True)

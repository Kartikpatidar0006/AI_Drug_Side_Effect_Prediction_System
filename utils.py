"""
utils.py
--------
Shared utility functions used across multiple pages.
Includes data loading, metric computation, chart generation,
and shared UI helpers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional
import warnings
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "drug_side_effects_100k_dataset.csv")

# ── Single source of truth for tested model accuracy ──────────────────────────
TESTING_ACCURACY: float = 78.2

# ── Color palette (Professional Healthcare Light Theme) ───────────────────────
COLORS = {
    "primary":   "#2563EB",
    "secondary": "#0EA5E9",
    "accent":    "#06B6D4",
    "success":   "#10B981",
    "danger":    "#EF4444",
    "warning":   "#F59E0B",
    "bg_main":   "#F8FAFC",
    "bg_card":   "#FFFFFF",
    "bg_card2":  "#F1F5F9",
    "text":      "#1E293B",
    "muted":     "#64748B",
    "border":    "#E2E8F0",
}

PLOTLY_TEMPLATE = "plotly_white"

# Plotly color scales that work well on the light theme
_CHART_COLORS = [
    COLORS["primary"], COLORS["secondary"], COLORS["success"],
    COLORS["warning"], COLORS["danger"], COLORS["accent"],
]


# ── Dataset ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataset() -> Optional[pd.DataFrame]:
    """Load the raw dataset, returning None on failure."""
    try:
        df = pd.read_csv(DATASET_PATH)
        return df
    except FileNotFoundError:
        st.error(f"Dataset file '{DATASET_PATH}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def get_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Return high-level statistics about the dataset."""
    return {
        "n_samples":   len(df),
        "n_features":  len(df.columns) - 1,   # exclude patient_id
        "n_drugs":     df["drug_name"].nunique(),
        "n_countries": df["country"].nunique(),
        "outcome_dist": df["outcome"].value_counts().to_dict(),
        "avg_age":     round(df["age"].mean(), 1),
        "avg_dosage":  round(df["dosage_mg"].mean(), 1),
    }


# ── Model metrics ─────────────────────────────────────────────────────────────

def compute_model_metrics(model: Any, scaler: Any, df: pd.DataFrame) -> Dict[str, float]:
    """
    Re-compute evaluation metrics using the full dataset.

    Returns a dict with accuracy, precision, recall, f1, and confusion matrix.
    Does NOT retrain — only calls model.predict() on existing data.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix,
    )

    # Replicate training pre-processing exactly
    df = df.copy()
    df['target'] = df['outcome'].apply(
        lambda x: 1 if x in ['Recovered', 'Recovering'] else 0
    )
    drop_cols = [
        'patient_id', 'report_date', 'treatment_start_date',
        'outcome', 'hospitalized', 'recovery_days',
    ]
    df_feat = df.drop(columns=drop_cols)
    cat_cols = [
        'gender', 'country', 'drug_name', 'side_effect',
        'severity', 'chronic_condition', 'smoker', 'alcohol_use',
    ]
    df_encoded = pd.get_dummies(df_feat, columns=cat_cols, drop_first=True)
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    X_aligned = X.reindex(columns=scaler.feature_names_in_, fill_value=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_scaled = scaler.transform(X_aligned)
        y_pred = model.predict(X_scaled)

    cm = confusion_matrix(y, y_pred)
    return {
        "accuracy":  round(accuracy_score(y, y_pred) * 100, 2),
        "precision": round(precision_score(y, y_pred) * 100, 2),
        "recall":    round(recall_score(y, y_pred) * 100, 2),
        "f1":        round(f1_score(y, y_pred) * 100, 2),
        "confusion_matrix": cm,
    }


# ── Shared UI helper ──────────────────────────────────────────────────────────

def section_header(icon: str, title: str, subtitle: str = "") -> None:
    """Render a consistent section header with icon, title and optional subtitle."""
    sub_html = f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon">{icon}</div>
        <div>
            <h2 class="section-title">{title}</h2>
            {sub_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Metric card HTML ──────────────────────────────────────────────────────────

def metric_card(title: str, value: str, icon: str = "📊", delta: str = "") -> str:
    """Return HTML string for a styled metric card."""
    delta_html = f'<p class="metric-delta">{delta}</p>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-content">
            <p class="metric-title">{title}</p>
            <h2 class="metric-value">{value}</h2>
            {delta_html}
        </div>
    </div>
    """


# ── Charts ─────────────────────────────────────────────────────────────────────

def make_gauge_chart(value: float, title: str, color: str) -> go.Figure:
    """Create a Plotly gauge chart for a percentage metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 28, "color": COLORS["text"]}},
        title={"text": title, "font": {"size": 14, "color": COLORS["muted"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": COLORS["bg_card2"],
            "bordercolor": COLORS["border"],
            "steps": [
                {"range": [0,  60], "color": "#F1F5F9"},
                {"range": [60, 80], "color": "#E0F2FE"},
                {"range": [80, 100], "color": "#DCFCE7"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=200,
        font_color=COLORS["text"],
    )
    return fig


def make_confusion_matrix_chart(cm: np.ndarray) -> go.Figure:
    """Create an annotated heatmap for the confusion matrix."""
    labels = ["Adverse (0)", "Positive (1)"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0, "#EFF6FF"], [1, COLORS["primary"]]],
        showscale=False,
        text=cm,
        texttemplate="%{text:,}",
        textfont={"size": 18, "color": COLORS["text"]},
    ))
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(color=COLORS["text"], size=16)),
        xaxis=dict(title="Predicted", color=COLORS["muted"]),
        yaxis=dict(title="Actual", color=COLORS["muted"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text"],
        height=320,
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig


def make_feature_importance_chart(model: Any, feature_names: list) -> go.Figure:
    """Horizontal bar chart for top-20 feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(20).sort_values()

    colors = [
        COLORS["accent"] if i >= len(top) - 3 else COLORS["primary"]
        for i in range(len(top))
    ]

    fig = go.Figure(go.Bar(
        x=top.values,
        y=top.index,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in top.values],
        textposition='outside',
        textfont=dict(color=COLORS["muted"], size=10),
    ))
    fig.update_layout(
        title=dict(text="Top 20 Feature Importances", font=dict(color=COLORS["text"], size=16)),
        xaxis=dict(title="Importance Score", color=COLORS["muted"], gridcolor=COLORS["border"]),
        yaxis=dict(color=COLORS["text"], tickfont=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text"],
        height=600,
        margin=dict(l=180, r=80, t=50, b=40),
        template=PLOTLY_TEMPLATE,
    )
    return fig


def make_outcome_pie_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart of outcome distribution."""
    counts = df['outcome'].value_counts()
    palette = [COLORS["success"], COLORS["primary"], COLORS["warning"], COLORS["danger"]]
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.55,
        marker_colors=palette,
        textinfo='label+percent',
        textfont=dict(color=COLORS["text"], size=13),
    ))
    fig.update_layout(
        title=dict(text="Outcome Distribution", font=dict(color=COLORS["text"], size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text"],
        legend=dict(font=dict(color=COLORS["muted"])),
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_gender_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart of gender distribution."""
    counts = df['gender'].value_counts()
    palette = [COLORS["primary"], COLORS["accent"]]
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.55,
        marker_colors=palette,
        textinfo='label+percent',
        textfont=dict(color=COLORS["text"], size=13),
    ))
    fig.update_layout(
        title=dict(text="Gender Distribution", font=dict(color=COLORS["text"], size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text"],
        legend=dict(font=dict(color=COLORS["muted"])),
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_severity_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of severity vs outcome."""
    grouped = df.groupby(['severity', 'outcome']).size().reset_index(name='count')
    fig = px.bar(
        grouped, x='severity', y='count', color='outcome',
        barmode='group',
        color_discrete_sequence=_CHART_COLORS,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        title=dict(text="Severity vs Outcome", font=dict(color=COLORS["text"], size=16)),
        xaxis=dict(title="Severity", color=COLORS["muted"]),
        yaxis=dict(title="Count", color=COLORS["muted"], gridcolor=COLORS["border"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text"],
        legend=dict(font=dict(color=COLORS["muted"])),
        height=340,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def make_drug_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of drug usage frequency."""
    counts = df['drug_name'].value_counts().reset_index()
    counts.columns = ['Drug', 'Count']
    fig = px.bar(
        counts, x='Count', y='Drug', orientation='h',
        color='Count',
        color_continuous_scale=[[0, "#DBEAFE"], [1, COLORS["primary"]]],
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        title=dict(text="Drug Usage Frequency", font=dict(color=COLORS["text"], size=16)),
        xaxis=dict(title="Number of Reports", color=COLORS["muted"], gridcolor=COLORS["border"]),
        yaxis=dict(color=COLORS["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        font_color=COLORS["text"],
        height=360,
        margin=dict(l=120, r=20, t=50, b=40),
    )
    return fig


def make_probability_bar(prob_adverse: float, prob_positive: float) -> go.Figure:
    """Horizontal probability bar chart for prediction output."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Adverse Outcome",
        x=[prob_adverse * 100],
        y=["Probability"],
        orientation='h',
        marker_color=COLORS["danger"],
        text=[f"{prob_adverse*100:.1f}%"],
        textposition='inside',
    ))
    fig.add_trace(go.Bar(
        name="Positive Outcome",
        x=[prob_positive * 100],
        y=["Probability"],
        orientation='h',
        marker_color=COLORS["success"],
        text=[f"{prob_positive*100:.1f}%"],
        textposition='inside',
    ))
    fig.update_layout(
        barmode='stack',
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text"],
        legend=dict(orientation='h', y=-0.3, font=dict(color=COLORS["muted"])),
        xaxis=dict(range=[0, 100], title="Probability (%)", color=COLORS["muted"],
                   gridcolor=COLORS["border"]),
        yaxis=dict(showticklabels=False),
        height=120,
        margin=dict(l=10, r=10, t=10, b=40),
        template=PLOTLY_TEMPLATE,
    )
    return fig


# ── CSS helper ────────────────────────────────────────────────────────────────

def load_css(path: str = "assets/style.css") -> None:
    css_path = os.path.join(BASE_DIR, path)

    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

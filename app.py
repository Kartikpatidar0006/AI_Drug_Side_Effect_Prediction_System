"""
app.py
------
Entry point for the AI Drug Side Effect Prediction System.
Configures the Streamlit app, applies global styling, renders the
sidebar navigation, and routes to the selected page module.
"""

import streamlit as st
from utils import load_css, load_dataset, get_dataset_stats, COLORS, TESTING_ACCURACY
from model_loader import load_model_and_scaler

# ── Page configuration (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="AI Drug Side Effect Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "AI Drug Side Effect Prediction System v1.0",
    },
)

# ── Inject global CSS ──────────────────────────────────────────────────────────
load_css("assets/style.css")

# ── Pre-load resources into session state (cached) ────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_resources():
    try:
        model, scaler = load_model_and_scaler()
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

# Load silently — pages will access via session_state
if "model" not in st.session_state:
    model, scaler, err = _get_resources()
    st.session_state["model"]        = model
    st.session_state["scaler"]       = scaler
    st.session_state["model_error"]  = err

if "dataset" not in st.session_state:
    st.session_state["dataset"] = load_dataset()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo & branding
    st.markdown("""
    <div class="sidebar-logo-container">
        <div class="sidebar-logo">💊</div>
        <div class="sidebar-title">AI Drug Side Effect<br>Prediction System</div>
        <div class="sidebar-subtitle">Powered by Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown('<p class="nav-section-label">Navigation</p>', unsafe_allow_html=True)

    NAV_OPTIONS = {
        "🏠  Dashboard":         "Dashboard",
        "🔬  Prediction":        "Prediction",
        "📊  Model Information": "Model Info",
        "ℹ️  About":             "About",
    }

    # Apply any programmatic navigation requested by CTA buttons.
    # Must run BEFORE st.radio() is instantiated — the only moment
    # Streamlit permits writing to a widget's own session state key.
    if "_nav_target" in st.session_state:
        st.session_state["navigation"] = st.session_state.pop("_nav_target")

    selected_label = st.radio(
        label="",
        options=list(NAV_OPTIONS.keys()),
        label_visibility="collapsed",
        key="navigation",
    )
    selected_page = NAV_OPTIONS[selected_label]

    # Dataset quick stats
    df = st.session_state.get("dataset")
    if df is not None:
        stats = get_dataset_stats(df)
        st.markdown("---")
        st.markdown('<p class="nav-section-label">Quick Stats</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Samples", f"{stats['n_samples']:,}", label_visibility="visible")
        with col2:
            st.metric("Features", stats["n_features"], label_visibility="visible")

    # Model status indicator
    st.markdown("---")
    if st.session_state.get("model") is not None:
        st.markdown(f"""
        <div class="sidebar-status">
            <span class="sidebar-status-dot"></span>
            <span class="sidebar-status-text">Model Loaded</span>
        </div>
        <div class="sidebar-accuracy-pill">
            🎯 Testing Accuracy: {TESTING_ACCURACY}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:0.5rem;font-size:0.82rem;color:#EF4444;">'
            '<span style="width:8px;height:8px;border-radius:50%;background:#EF4444;'
            'display:inline-block;"></span>'
            ' Model Offline</div>',
            unsafe_allow_html=True,
        )


# ── Route to page ──────────────────────────────────────────────────────────────
if selected_page == "Dashboard":
    from pages.Dashboard import render
    render()
elif selected_page == "Prediction":
    from pages.Prediction import render
    render()
elif selected_page == "Model Info":
    from pages.Model_Info import render
    render()
elif selected_page == "About":
    from pages.About import render
    render()

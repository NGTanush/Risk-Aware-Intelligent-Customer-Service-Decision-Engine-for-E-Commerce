import os
import sys
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure local package import
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.decision_pipeline import CustomerServiceDecisionEngine

# ============================================================
# PAGE CONFIGURATION & METADATA
# ============================================================

st.set_page_config(
    page_title="Risk-Aware AI Decision Operations Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MINIMAL LUXURY ENTERPRISE AI CSS STYLING
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary: #060B14;
        --bg-secondary: #0E1628;
        --bg-panel: rgba(18, 25, 40, 0.85);
        --bg-card: rgba(23, 34, 56, 0.6);
        --border-color: #22304A;
        --border-hover: #00D9FF;
        --accent-cyan: #00D9FF;
        --accent-violet: #7C3AED;
        --color-safe: #10B981;
        --color-warn: #F59E0B;
        --color-danger: #EF4444;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-heading: 'Space Grotesk', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    /* Reset & Base */
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: var(--font-main);
    }

    .stAppViewContainer {
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(0, 217, 255, 0.03) 0%, transparent 40%),
            radial-gradient(circle at 85% 85%, rgba(124, 58, 237, 0.03) 0%, transparent 40%),
            linear-gradient(rgba(34, 48, 74, 0.15) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34, 48, 74, 0.15) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
    }

    /* Hide default Streamlit headers and footers */
    header[data-testid="stHeader"] { background: transparent !important; }
    footer { visibility: hidden; }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

    /* Sidebar Control Panel */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    
    .sidebar-brand-card {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.08) 0%, rgba(124, 58, 237, 0.08) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .sidebar-title {
        font-family: var(--font-heading);
        font-weight: 700;
        font-size: 1.25rem;
        letter-spacing: 0.5px;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: var(--color-safe);
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 12px var(--color-safe);
        animation: pulse-ring 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    @keyframes pulse-ring {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .telemetry-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px dashed rgba(34, 48, 74, 0.5);
        font-size: 0.85rem;
    }
    .telemetry-label { color: var(--text-secondary); }
    .telemetry-value { font-family: var(--font-mono); font-weight: 600; color: var(--accent-cyan); }

    /* Top Sticky Header */
    .top-header {
        position: sticky;
        top: 0;
        z-index: 99;
        background: rgba(6, 11, 20, 0.9);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 18px 28px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }

    .top-header-title {
        font-family: var(--font-heading);
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        background: linear-gradient(90deg, #FFFFFF 0%, var(--accent-cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .top-header-subtitle {
        color: var(--text-secondary);
        font-size: 0.82rem;
        letter-spacing: 0.5px;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-secondary);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: var(--text-secondary);
        font-family: var(--font-heading);
        font-weight: 500;
        font-size: 0.88rem;
        border: none !important;
        transition: all 0.2s ease;
        padding: 0 16px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.15) 0%, rgba(124, 58, 237, 0.15) 100%) !important;
        color: var(--accent-cyan) !important;
        border: 1px solid var(--accent-cyan) !important;
        font-weight: 700;
    }

    /* Glass Panels & Cards */
    .glass-panel {
        background: var(--bg-panel);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: border-color 0.3s ease;
    }

    .glass-panel:hover {
        border-color: rgba(0, 217, 255, 0.4);
    }

    /* Hero Query Container */
    .hero-title-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 217, 255, 0.1);
        border: 1px solid rgba(0, 217, 255, 0.3);
        color: var(--accent-cyan);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-family: var(--font-mono);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }

    /* Text Area Override */
    .stTextArea textarea {
        background-color: rgba(14, 22, 40, 0.8) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: var(--font-main) !important;
        font-size: 1rem !important;
        padding: 14px 18px !important;
        transition: all 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.25) !important;
    }

    /* Primary Glowing Button */
    div.stButton > button {
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-violet) 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        font-family: var(--font-heading) !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 1px !important;
        padding: 14px 28px !important;
        border-radius: 10px !important;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.35) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        cursor: pointer;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.6) !important;
        filter: brightness(1.1);
    }

    /* Pipeline Visualization Nodes */
    .pipeline-wrapper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(14, 22, 40, 0.9);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 20px;
        margin: 20px 0;
        overflow-x: auto;
    }

    .pipeline-step {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
        min-width: 110px;
        flex: 1;
        margin: 0 4px;
        transition: all 0.3s ease;
    }

    .pipeline-step.active {
        border-color: var(--accent-cyan);
        background: linear-gradient(180deg, rgba(0, 217, 255, 0.1) 0%, rgba(14, 22, 40, 0.9) 100%);
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.25);
    }

    .pipeline-step-num {
        font-size: 0.68rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .pipeline-step-title {
        font-family: var(--font-heading);
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 2px;
    }

    .pipeline-step-val {
        font-family: var(--font-mono);
        font-size: 0.85rem;
        font-weight: 700;
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .pipeline-arrow {
        color: var(--border-color);
        font-size: 1.1rem;
        font-weight: bold;
    }

    /* Metric Glass Cards */
    .metric-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: var(--accent-cyan);
        box-shadow: 0 8px 24px rgba(0, 217, 255, 0.15);
    }

    .metric-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .metric-value {
        font-family: var(--font-heading);
        font-size: 1.35rem;
        font-weight: 700;
        margin: 8px 0 4px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .metric-sub {
        font-size: 0.72rem;
        color: var(--text-muted);
    }

    /* Horizontal Risk Track */
    .risk-meter-box {
        background: var(--bg-panel);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 20px;
        margin-top: 20px;
    }

    .risk-track-bg {
        height: 12px;
        border-radius: 6px;
        background: linear-gradient(90deg, #10B981 0%, #F59E0B 50%, #EF4444 100%);
        position: relative;
        margin: 12px 0 6px 0;
    }

    .risk-pin {
        position: absolute;
        top: -4px;
        width: 20px;
        height: 20px;
        background: #FFFFFF;
        border: 3px solid var(--bg-primary);
        border-radius: 50%;
        transform: translateX(-50%);
        box-shadow: 0 0 12px rgba(255, 255, 255, 0.9);
        transition: left 0.5s ease;
    }

    /* Verdict Display Cards */
    .verdict-card {
        border-radius: 16px;
        padding: 28px;
        margin: 24px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    }

    .verdict-card.automate {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(14, 22, 40, 0.95) 100%);
        border: 2px solid var(--color-safe);
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.2);
    }

    .verdict-card.clarify {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(14, 22, 40, 0.95) 100%);
        border: 2px solid var(--color-warn);
        box-shadow: 0 0 35px rgba(245, 158, 11, 0.2);
    }

    .verdict-card.escalate {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(14, 22, 40, 0.95) 100%);
        border: 2px solid var(--color-danger);
        box-shadow: 0 0 35px rgba(239, 68, 68, 0.2);
    }

    .verdict-header {
        font-family: var(--font-heading);
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Response Container */
    .response-container {
        background: rgba(10, 16, 28, 0.9);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 22px;
        margin-top: 16px;
        font-family: var(--font-main);
        font-size: 0.98rem;
        line-height: 1.6;
        color: var(--text-primary);
        position: relative;
    }

    .response-badge {
        display: inline-block;
        background: rgba(0, 217, 255, 0.15);
        border: 1px solid var(--accent-cyan);
        color: var(--accent-cyan);
        font-family: var(--font-mono);
        font-size: 0.72rem;
        padding: 2px 10px;
        border-radius: 12px;
        margin-top: 10px;
        margin-right: 6px;
    }

    /* Footer */
    .app-footer {
        border-top: 1px solid var(--border-color);
        padding-top: 20px;
        margin-top: 50px;
        text-align: center;
        color: var(--text-muted);
        font-family: var(--font-mono);
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS & CACHED RESOURCES
# ============================================================

def format_intent_label(intent_raw: str) -> str:
    if not intent_raw:
        return "Unknown Intent"
    return intent_raw.replace("_", " ").title()

@st.cache_resource
def load_engine():
    return CustomerServiceDecisionEngine()

engine = load_engine()

if "session_logs" not in st.session_state:
    st.session_state.session_logs = []


# ============================================================
# SIDEBAR: ENGINE CONTROL PANEL
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-card">
        <div class="sidebar-title">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00D9FF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path>
            </svg>
            <span>Risk-Aware AI</span>
        </div>
        <div style="margin-top: 8px; font-size: 0.8rem; color: #94A3B8; display: flex; align-items: center; gap: 8px;">
            <span class="pulse-dot"></span>
            <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600;">SYSTEM ONLINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 0.95rem; color: #F8FAFC; margin-bottom: 8px;'>SELECTIVE AUTOMATION POLICY</div>", unsafe_allow_html=True)
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.85,
        step=0.05,
        help="Minimum calibrated probability required to permit automatic answer generation."
    )

    st.markdown(f"""
    <div style="background: rgba(0,217,255,0.05); border: 1px solid rgba(0,217,255,0.2); border-radius: 8px; padding: 10px; margin-bottom: 20px; font-size: 0.8rem; text-align: center;">
        Active Operating Cutoff: <b style="color: #00D9FF; font-family: 'JetBrains Mono', monospace;">{confidence_threshold:.0%}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 0.95rem; color: #F8FAFC; margin-bottom: 12px;'>AI MODEL TELEMETRY</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(14,22,40,0.6); border: 1px solid #22304A; border-radius: 10px; padding: 12px;">
        <div class="telemetry-item">
            <span class="telemetry-label">Classifier Model</span>
            <span class="telemetry-value">Linear SVM</span>
        </div>
        <div class="telemetry-item">
            <span class="telemetry-label">Calibration</span>
            <span class="telemetry-value">Platt Scaling</span>
        </div>
        <div class="telemetry-item">
            <span class="telemetry-label">Macro F1-Score</span>
            <span class="telemetry-value" style="color: #00D9FF;">99.04%</span>
        </div>
        <div class="telemetry-item">
            <span class="telemetry-label">Brier Score</span>
            <span class="telemetry-value" style="color: #00D9FF;">0.0072</span>
        </div>
        <div class="telemetry-item" style="border-bottom: none;">
            <span class="telemetry-label">OOD Detector</span>
            <span class="telemetry-value" style="color: #10B981;">ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.caption("Risk-Aware Customer Service Decision Center v2.0")


# ============================================================
# TOP FIXED COMMAND HEADER
# ============================================================

current_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

st.markdown(f"""
<div class="top-header">
    <div>
        <div class="top-header-title">AI CUSTOMER SERVICE DECISION CENTER</div>
        <div class="top-header-subtitle">
            <span>Confidence Calibration</span> • <span>Risk Assessment</span> • <span>Human-in-the-Loop Safeguards</span>
        </div>
    </div>
    <div style="text-align: right;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #F8FAFC; font-weight: 600; display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
            <span class="pulse-dot"></span> SYSTEM ONLINE
        </div>
        <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px; font-family: 'JetBrains Mono', monospace;">
            Threshold: <span style="color: #00D9FF;">{confidence_threshold:.0%}</span> | Model: <span style="color: #7C3AED;">Calibrated LinearSVC</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DASHBOARD NAVIGATION TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 DECISION CENTER",
    "📊 MODEL PERFORMANCE",
    "📈 CONFIDENCE LAB",
    "🛡️ RISK MONITOR",
    "🧪 ROBUSTNESS LAB",
    "🏗️ SYSTEM ARCHITECTURE"
])


# ============================================================
# TAB 1: DECISION CENTER (CORE PIPELINE)
# ============================================================

with tab1:
    st.markdown("""
    <div class="hero-title-badge">AI OPERATIONS CENTER</div>
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">
        Real-Time Customer Request Risk Analysis
    </div>
    <div style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 24px;">
        Evaluates query intent, calibrated uncertainty, customer sentiment, and domain safety before routing.
    </div>
    """, unsafe_allow_html=True)

    # Input Container
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        user_query = st.text_area(
            "Incoming Customer Request",
            value="Where can I track my package?",
            placeholder="Type a customer query... e.g. 'I want to request a refund for my order'",
            height=80,
            label_visibility="collapsed"
        )
    with col_btn:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("⚡ ANALYZE REQUEST", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Run Backend Execution
    query_text = user_query if user_query.strip() else "Where can I track my package?"
    result = engine.process(query_text, custom_threshold=confidence_threshold)
    
    if analyze_btn and isinstance(st.session_state.session_logs, list):
        st.session_state.session_logs.append({
            "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "Query": result["query"],
            "Intent": format_intent_label(result["predicted_intent"]),
            "Confidence": f"{result['confidence_percentage']}%",
            "Sentiment": result["sentiment"],
            "Risk Level": result["risk_level"],
            "Domain": result["domain_status"],
            "Decision": result["decision"]
        })

    # Pipeline Node Graph Visualization
    dec = result["decision"]
    dec_color = "#10B981" if dec == "AUTOMATE" else ("#F59E0B" if dec == "CLARIFY" else "#EF4444")
    intent_display = format_intent_label(result["predicted_intent"])

    st.markdown(f"""
    <div class="pipeline-wrapper">
        <div class="pipeline-step active">
            <div class="pipeline-step-num">Stage 1</div>
            <div class="pipeline-step-title">Customer Query</div>
            <div class="pipeline-step-val" style="color: #00D9FF;">INCOMING</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step active">
            <div class="pipeline-step-num">Stage 2</div>
            <div class="pipeline-step-title">Intent Classifier</div>
            <div class="pipeline-step-val" style="color: #7C3AED;">{intent_display}</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step active">
            <div class="pipeline-step-num">Stage 3</div>
            <div class="pipeline-step-title">Platt Calibration</div>
            <div class="pipeline-step-val" style="color: #00D9FF;">{result['confidence_percentage']}%</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step active">
            <div class="pipeline-step-num">Stage 4</div>
            <div class="pipeline-step-title">Sentiment Analysis</div>
            <div class="pipeline-step-val" style="color: {'#EF4444' if result['sentiment']=='NEGATIVE' else '#00D9FF'};">{result['sentiment']}</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step active">
            <div class="pipeline-step-num">Stage 5</div>
            <div class="pipeline-step-title">OOD Detector</div>
            <div class="pipeline-step-val" style="color: {'#10B981' if result['domain_status']=='IN_DOMAIN' else '#EF4444'};">{result['domain_status']}</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step active">
            <div class="pipeline-step-num">Stage 6</div>
            <div class="pipeline-step-title">Risk Engine</div>
            <div class="pipeline-step-val" style="color: {dec_color};">{result['risk_level']} ({result['risk_score']:.2f})</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step active" style="border-color: {dec_color}; background: linear-gradient(180deg, rgba(0,0,0,0.3) 0%, {dec_color}22 100%);">
            <div class="pipeline-step-num">Stage 7</div>
            <div class="pipeline-step-title">Decision Engine</div>
            <div class="pipeline-step-val" style="color: {dec_color}; font-weight: 800;">{dec}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric Cards Row (5 Columns)
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Intent</div>
            <div class="metric-value" style="color: #7C3AED; font-size: 1.1rem;">{intent_display}</div>
            <div class="metric-sub">Classifier Category</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        conf_color = "#10B981" if result['confidence'] >= confidence_threshold else "#F59E0B"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Calibrated Confidence</div>
            <div class="metric-value" style="color: {conf_color};">{result['confidence_percentage']}%</div>
            <div class="metric-sub">Platt Scaled Probability</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        sent_color = "#EF4444" if result['sentiment'] == "NEGATIVE" else "#00D9FF"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Customer Sentiment</div>
            <div class="metric-value" style="color: {sent_color};">{result['sentiment']}</div>
            <div class="metric-sub">Score: {result['sentiment_score']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        r_color = "#EF4444" if result['risk_level'] == "HIGH" else ("#F59E0B" if result['risk_level'] == "MEDIUM" else "#10B981")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Composite Risk Score</div>
            <div class="metric-value" style="color: {r_color};">{result['risk_level']}</div>
            <div class="metric-sub">Index: {result['risk_score']:.2f} / 1.00</div>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        dom_color = "#10B981" if result['domain_status'] == "IN_DOMAIN" else "#EF4444"
        dom_label = "IN-DOMAIN" if result['domain_status'] == "IN_DOMAIN" else "OUT-OF-DOMAIN"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Domain Relevance</div>
            <div class="metric-value" style="color: {dom_color}; font-size: 1.05rem;">{dom_label}</div>
            <div class="metric-sub">Similarity: {result['similarity_score']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    # Risk Meter & Circular Confidence Gauge Row
    col_gauge, col_risk = st.columns([1, 2])

    with col_gauge:
        conf_val = float(result['confidence_percentage'])
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = conf_val,
            number = {'suffix': "%", 'font': {'color': '#F8FAFC', 'family': 'Space Grotesk', 'size': 32}},
            title = {'text': "CONFIDENCE GAUGE", 'font': {'color': '#94A3B8', 'size': 12, 'family': 'Space Grotesk'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#22304A"},
                'bar': {'color': "#00D9FF"},
                'bgcolor': "#0E1628",
                'bordercolor': "#22304A",
                'steps': [
                    {'range': [0, confidence_threshold*100], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [confidence_threshold*100, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "#7C3AED", 'width': 4},
                    'thickness': 0.8,
                    'value': confidence_threshold*100
                }
            }
        ))
        fig_gauge.update_layout(
            height=180,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#F8FAFC"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    with col_risk:
        pin_pos = min(98.0, max(2.0, result['risk_score'] * 100.0))
        st.markdown(f"""
        <div class="risk-meter-box">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; font-weight: 700; color: #94A3B8;">COMPOSITE RISK SPECTRUM</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: {dec_color}; font-weight: 700;">{result['risk_level']} RISK ({result['risk_score']:.2f})</span>
            </div>
            <div class="risk-track-bg">
                <div class="risk-pin" style="left: {pin_pos}%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748B; font-family: 'JetBrains Mono', monospace;">
                <span>LOW (0.00 - 0.34)</span>
                <span>MEDIUM (0.35 - 0.64)</span>
                <span>HIGH (0.65 - 1.00)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Verdict Card & Response Panel
    if dec == "AUTOMATE":
        st.markdown(f"""
        <div class="verdict-card automate">
            <div class="verdict-header" style="color: #10B981;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                <span>AUTOMATION APPROVED</span>
            </div>
            <div style="margin-top: 10px; color: #F8FAFC; font-size: 0.98rem;">
                <b>Decision Reason:</b> {result['reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 8px;'>🤖 GENERATED AI RESPONSE</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="response-container">
            {result['response']}
            <div style="margin-top: 14px;">
                <span class="response-badge">Method: {result['generation_method']}</span>
                <span class="response-badge" style="border-color: #10B981; color: #10B981;">Confidence: {result['confidence_percentage']}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif dec == "CLARIFY":
        st.markdown(f"""
        <div class="verdict-card clarify">
            <div class="verdict-header" style="color: #F59E0B;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                <span>CLARIFICATION REQUIRED</span>
            </div>
            <div style="margin-top: 10px; color: #F8FAFC; font-size: 0.98rem;">
                <b>Decision Reason:</b> {result['reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 8px;'>❓ SUGGESTED CLARIFICATION PROMPT</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="response-container" style="border-color: #F59E0B;">
            {result['response']}
        </div>
        """, unsafe_allow_html=True)

    else: # HUMAN_ESCALATION
        st.markdown(f"""
        <div class="verdict-card escalate">
            <div class="verdict-header" style="color: #EF4444;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2.5"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                <span>MANDATORY HUMAN ESCALATION</span>
            </div>
            <div style="margin-top: 10px; color: #F8FAFC; font-size: 0.98rem;">
                <b>Decision Reason:</b> {result['reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 8px;'>🚨 SUPPORT ROUTING PROTOCOL</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="response-container" style="border-color: #EF4444;">
            {result['response']}
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 2: MODEL PERFORMANCE (RESEARCH LAB)
# ============================================================

with tab2:
    st.markdown("""
    <div class="hero-title-badge">ML RESEARCH LABORATORY</div>
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">
        Classifier Benchmark & Platt Calibration Telemetry
    </div>
    <div style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 24px;">
        Empirical metrics across 46 customer service intent classes on 44,884 samples.
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Overview Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value" style="color: #10B981;">99.09%</div>
            <div class="metric-sub">LinearSVC Classifier</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Macro F1-Score</div>
            <div class="metric-value" style="color: #00D9FF;">99.04%</div>
            <div class="metric-sub">46 Intent Classes</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Brier Score</div>
            <div class="metric-value" style="color: #7C3AED;">0.0072</div>
            <div class="metric-sub">Platt Sigmoidal Error</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ECE Index</div>
            <div class="metric-value" style="color: #10B981;">0.0034</div>
            <div class="metric-sub">Expected Calibration Error</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    c_m1, c_m2 = st.columns(2)

    with c_m1:
        st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 12px;'>🏆 Baseline Model Benchmark Comparison</div>", unsafe_allow_html=True)
        if os.path.exists("evaluation/model_comparison.csv"):
            comp_df = pd.read_csv("evaluation/model_comparison.csv")
            st.dataframe(comp_df, use_container_width=True)

    with c_m2:
        st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 12px;'>🎯 Calibrated Model Performance Summary</div>", unsafe_allow_html=True)
        if os.path.exists("evaluation/calibration_results.csv"):
            calib_df = pd.read_csv("evaluation/calibration_results.csv")
            st.dataframe(calib_df, use_container_width=True)

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 12px;'>🔥 Confusion Matrix Heatmap</div>", unsafe_allow_html=True)
        if os.path.exists("evaluation/calibrated_svm_confusion_matrix.png"):
            st.image("evaluation/calibrated_svm_confusion_matrix.png", caption="Calibrated Linear SVM Confusion Matrix (46 Classes)", use_container_width=True)

    with col_cm2:
        st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 12px;'>📈 Platt Probability Calibration Curve</div>", unsafe_allow_html=True)
        if os.path.exists("evaluation/calibration_curve.png"):
            st.image("evaluation/calibration_curve.png", caption="Platt Sigmoidal Probability Calibration Curve", use_container_width=True)


# ============================================================
# TAB 3: CONFIDENCE LAB (SELECTIVE AUTOMATION)
# ============================================================

with tab3:
    st.markdown("""
    <div class="hero-title-badge">AI RELIABILITY LABORATORY</div>
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">
        Selective Automation Trade-off Analysis
    </div>
    <div style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 24px;">
        Evaluate automatic response coverage vs selective accuracy across confidence thresholds.
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists("evaluation/confidence_threshold_results.csv"):
        thresh_df = pd.read_csv("evaluation/confidence_threshold_results.csv")
        
        col_t1, col_t2 = st.columns([1, 2])

        with col_t1:
            st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.05rem; color: #F8FAFC; margin-bottom: 10px;'>Threshold Evaluation Data</div>", unsafe_allow_html=True)
            st.dataframe(thresh_df, use_container_width=True, height=320)

        with col_t2:
            st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.05rem; color: #F8FAFC; margin-bottom: 10px;'>Coverage vs Selective Accuracy Curve</div>", unsafe_allow_html=True)
            
            fig_thresh = go.Figure()
            cov_vals = thresh_df["Coverage"] * 100 if thresh_df["Coverage"].max() <= 1.0 else thresh_df["Coverage"]
            acc_vals = thresh_df["Selective Accuracy"] * 100 if thresh_df["Selective Accuracy"].max() <= 1.0 else thresh_df["Selective Accuracy"]
            
            fig_thresh.add_trace(go.Scatter(
                x=thresh_df["Threshold"], y=cov_vals,
                mode='lines+markers', name='Coverage (%)', line=dict(color='#00D9FF', width=3)
            ))
            fig_thresh.add_trace(go.Scatter(
                x=thresh_df["Threshold"], y=acc_vals,
                mode='lines+markers', name='Selective Accuracy (%)', line=dict(color='#10B981', width=3)
            ))
            fig_thresh.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(14,22,40,0.6)',
                font=dict(color='#94A3B8', family='Space Grotesk'),
                xaxis=dict(title='Confidence Threshold', gridcolor='#22304A'),
                yaxis=dict(title='Percentage (%)', gridcolor='#22304A'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=320,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_thresh, use_container_width=True)

    st.markdown("""
    <div style="background: rgba(0,217,255,0.06); border: 1px solid rgba(0,217,255,0.25); border-radius: 12px; padding: 16px; margin-top: 20px;">
        <b style="color: #00D9FF;">Selected Production Operating Point: 85% Cutoff</b><br/>
        Achieves <b>97.58% Automatic Coverage</b> with <b>99.77% Selective Accuracy</b> while maintaining a low <b>2.42% Escalation Rate</b>.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 4: RISK MONITOR (SECURITY OPERATIONS CENTER)
# ============================================================

with tab4:
    st.markdown("""
    <div class="hero-title-badge">SECURITY OPERATIONS CENTER</div>
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">
        Live Risk Telemetry & Session Audit Logs
    </div>
    <div style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 24px;">
        Real-time monitoring of evaluated request risk distribution and automated decision routing.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.session_logs:
        logs_df = pd.DataFrame(st.session_state.session_logs)
        
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.05rem; color: #F8FAFC; margin-bottom: 10px;'>Risk Level Breakdown</div>", unsafe_allow_html=True)
            risk_counts = logs_df["Risk Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            fig_risk = px.pie(
                risk_counts, values="Count", names="Risk Level",
                color="Risk Level",
                color_discrete_map={"LOW": "#10B981", "MEDIUM": "#F59E0B", "HIGH": "#EF4444"},
                hole=0.5
            )
            fig_risk.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC', family='Space Grotesk'), height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_risk, use_container_width=True)

        with r2:
            st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.05rem; color: #F8FAFC; margin-bottom: 10px;'>Decision Routing Distribution</div>", unsafe_allow_html=True)
            dec_counts = logs_df["Decision"].value_counts().reset_index()
            dec_counts.columns = ["Decision", "Count"]
            fig_dec = px.pie(
                dec_counts, values="Count", names="Decision",
                color="Decision",
                color_discrete_map={"AUTOMATE": "#10B981", "CLARIFY": "#F59E0B", "HUMAN_ESCALATION": "#EF4444"},
                hole=0.5
            )
            fig_dec.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC', family='Space Grotesk'), height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_dec, use_container_width=True)

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 10px;'>📜 Live Session Audit Stream</div>", unsafe_allow_html=True)
        st.dataframe(logs_df, use_container_width=True)
    else:
        st.info("🛡️ No queries evaluated in this session yet. Run requests in DECISION CENTER or ROBUSTNESS LAB to stream live security telemetry!")


# ============================================================
# TAB 5: ROBUSTNESS LAB (TEST SUITE)
# ============================================================

with tab5:
    st.markdown("""
    <div class="hero-title-badge">AI TESTING LABORATORY</div>
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">
        Robustness & Stress Testing Suite
    </div>
    <div style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 24px;">
        Test model stability against paraphrasing, noisy typos, angry sentiment, and out-of-domain queries.
    </div>
    """, unsafe_allow_html=True)

    sample_tests = {
        "1. Standard In-Domain Query (Low Risk)": "Where can I track my package?",
        "2. Rephrased Query": "Could you please tell me where my parcel is right now?",
        "3. Typo / Noisy Query": "wheres my ordr pkg track status?",
        "4. Angry Customer (Low Risk Intent)": "WHERE IS MY PACKAGE?! I HAVE BEEN WAITING FOREVER AND THIS IS RIDICULOUS!",
        "5. Ambiguous Query": "I need help with my order.",
        "6. High-Risk Refund Request": "I want to request a refund for my order. I am very disappointed!",
        "7. Account Security / Password (High Risk)": "I need to recover my password immediately.",
        "8. Out-Of-Domain (Sports)": "Who will win tomorrow's cricket match?"
    }

    selected_test_name = st.selectbox("Select Benchmark Scenario:", list(sample_tests.keys()))
    selected_query_text = sample_tests[selected_test_name]

    st.markdown(f"""
    <div style="background: rgba(14,22,40,0.8); border: 1px solid #22304A; border-radius: 10px; padding: 14px; margin: 14px 0;">
        <b>Target Test Query:</b> <code style="color: #00D9FF;">{selected_query_text}</code>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔬 EXECUTE ROBUSTNESS DIAGNOSTIC"):
        test_res = engine.process(selected_query_text, custom_threshold=confidence_threshold)
        
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("<div style='font-family: Space Grotesk, sans-serif; font-weight: 700; font-size: 1.05rem; color: #F8FAFC; margin-bottom: 10px;'>Evaluated Diagnostic Metrics</div>", unsafe_allow_html=True)
            st.json({
                "Predicted Intent": format_intent_label(test_res["predicted_intent"]),
                "Calibrated Confidence": f"{test_res['confidence_percentage']}%",
                "Sentiment": test_res["sentiment"],
                "Domain Status": test_res["domain_status"],
                "Risk Level": test_res["risk_level"],
                "Risk Score": test_res["risk_score"]
            })
        with r_col2:
            dec_t = test_res['decision']
            dec_color_t = "#10B981" if dec_t == "AUTOMATE" else ("#F59E0B" if dec_t == "CLARIFY" else "#EF4444")
            st.markdown(f"""
            <div style="background: rgba(14,22,40,0.9); border: 2px solid {dec_color_t}; border-radius: 14px; padding: 20px;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 800; color: {dec_color_t};">VERDICT: {dec_t}</div>
                <div style="margin-top: 10px; font-size: 0.92rem; color: #F8FAFC;"><b>Reason:</b> {test_res['reason']}</div>
                <div style="margin-top: 10px; font-size: 0.9rem; color: #94A3B8;"><b>Response:</b> {test_res['response']}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# TAB 6: SYSTEM ARCHITECTURE
# ============================================================

with tab6:
    st.markdown("""
    <div class="hero-title-badge">TECHNICAL ARCHITECTURE</div>
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">
        End-to-End System Pipeline Infographic
    </div>
    <div style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 24px;">
        Modular architecture combining NLP feature extraction, calibrated classification, sentiment analysis, and risk scoring.
    </div>
    """, unsafe_allow_html=True)

    arch_c1, arch_c2, arch_c3, arch_c4 = st.columns(4)

    with arch_c1:
        st.markdown("""
        <div class="glass-panel" style="text-align: center; min-height: 280px;">
            <div style="font-size: 1.8rem;">🔍</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.1rem; color: #00D9FF; margin: 10px 0;">1. UNDERSTAND</div>
            <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
                • TF-IDF N-Gram Vectorization (20,000 features)<br/>
                • Tokenization & Preprocessing<br/>
                • Cosine Similarity Out-of-Domain Detection
            </div>
        </div>
        """, unsafe_allow_html=True)

    with arch_c2:
        st.markdown("""
        <div class="glass-panel" style="text-align: center; min-height: 280px;">
            <div style="font-size: 1.8rem;">⚖️</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.1rem; color: #7C3AED; margin: 10px 0;">2. ASSESS</div>
            <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
                • Calibrated Linear SVM Classifier<br/>
                • Platt Sigmoidal Probability Scaling<br/>
                • VADER Customer Sentiment & Intensity Scoring
            </div>
        </div>
        """, unsafe_allow_html=True)

    with arch_c3:
        st.markdown("""
        <div class="glass-panel" style="text-align: center; min-height: 280px;">
            <div style="font-size: 1.8rem;">🛡️</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.1rem; color: #F59E0B; margin: 10px 0;">3. DECIDE</div>
            <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
                • Composite Risk Matrix Evaluation<br/>
                • Selective Automation Policy Enforcement<br/>
                • 3-Way Routing (Automate / Clarify / Escalate)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with arch_c4:
        st.markdown("""
        <div class="glass-panel" style="text-align: center; min-height: 280px;">
            <div style="font-size: 1.8rem;">⚡</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.1rem; color: #10B981; margin: 10px 0;">4. RESPOND</div>
            <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
                • OpenAI LLM Response Generation<br/>
                • Cosine Similarity Dataset Fallback<br/>
                • Mandatory Human Support Handover
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# APP FOOTER
# ============================================================

st.markdown("""
<div class="app-footer">
    Selective Automation • Platt Calibration • Risk Engine • Human-in-the-Loop<br/>
    <b style="color: #00D9FF;">ENTERPRISE AI CUSTOMER SERVICE DECISION ENGINE</b>
</div>
""", unsafe_allow_html=True)

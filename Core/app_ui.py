import os
import sys
import streamlit as st
import datetime

# Ensure parent directory is in path for imports
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from Core.engine.data import get_market_data, add_indicators
from gui.chart_engine import create_trading_chart, display_terminal, display_stats, display_live_log
from gui.ml_gui import display_ml_training_page
from gui.dataset_gui import display_dataset_page
from gui.news_sidebar import display_news_page
from Core.engine.signal import execute_trade_decision

# App Configuration
st.set_page_config(page_title="AI Quantum Trader Pro v6.2", layout="wide")

# --- INITIALIZE SESSION STATE ---
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = "SI=F"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "silver_ema_v1"
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("👨‍🚀 Control Center")
    page = st.radio("Navigation", ["📈 Live Chart", "🧠 ML Training", "📂 Datasets"])
    st.divider()

    # Model Selection (Dynamically from /models folder)
    st.subheader("🤖 AI Model")
    if os.path.exists("models"):
        modelle = [f.replace(".pkl", "") for f in os.listdir("models") if f.endswith(".pkl")]
        if modelle:
            st.session_state.selected_model = st.selectbox("Active Model:", modelle)
        else:
            st.warning("No Model (.pkl) found.")
    else:
        st.error("'models' folder missing.")

# --- PAGE 1: LIVE CHART ---
if page == "📈 Live Chart":
    st.subheader(f"Trading Terminal: {st.session_state.current_symbol}")
    col_main, col_side = st.columns([2.8, 1.5])

    with col_main:
        # Load data
        df_raw = get_market_data(st.session_state.current_symbol)
        if df_raw is not None and not df_raw.empty:
            df = add_indicators(df_raw)
            # Display chart
            create_trading_chart(df)
            # Log below chart
            display_live_log()
        else:
            st.error("Connecting to Yahoo Finance or invalid symbol...")

    with col_side:
        st.markdown(f"### ⚡ Signal Analysis")
        if 'df' in locals() and df is not None:
            # Calculate AI Signal
            signal, conf = execute_trade_decision(df, st.session_state.selected_model)
            st.metric("AI SIGNAL", signal, f"Conf: {conf}")

            # Trade Log Logic (Entry on signal change)
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            if not st.session_state.trade_log or st.session_state.trade_log[0]['Signal'] != signal:
                if signal in ["BUY", "SELL"]:
                    st.session_state.trade_log.insert(0, {
                        "Time": ts, "Symbol": st.session_state.current_symbol,
                        "Signal": signal, "Conf": conf, "Price": round(df['Close'].iloc[-1], 2)
                    })

            # Stats & Terminal
            display_stats(df)
            new_sym = display_terminal(st.session_state.current_symbol)
            if new_sym and new_sym != st.session_state.current_symbol:
                st.session_state.current_symbol = new_sym
                st.rerun()
        
        # --- NEWS SIDEBAR IN RIGHT COLUMN ---
        st.divider()
        with st.expander("📰 Asset News", expanded=False):
            display_news_page(st.session_state.current_symbol)

# --- PAGE 2: ML TRAINING ---
elif page == "🧠 ML Training":
    display_ml_training_page()

# --- PAGE 3: DATASETS ---
elif page == "📂 Datasets":
    display_dataset_page()
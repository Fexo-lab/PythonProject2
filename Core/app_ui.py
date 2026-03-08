import sys
import os
import streamlit as st
import datetime

# Pfad-Fix für lokale Module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .engine.data import get_market_data, add_indicators
from gui.chart_engine import create_trading_chart, display_terminal, display_stats, display_live_log
from gui.ml_gui import display_ml_training_page
from gui.dataset_gui import display_dataset_page
from .engine.signal import execute_trade_decision

# App Konfiguration
st.set_page_config(page_title="AI Quantum Trader Pro v6.2", layout="wide")

# --- INITIALISIERUNG SESSION STATE ---
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

    # Modell-Auswahl (Dynamisch aus dem Ordner /models)
    st.subheader("🤖 KI Modell")
    if os.path.exists("models"):
        modelle = [f.replace(".pkl", "") for f in os.listdir("models") if f.endswith(".pkl")]
        if modelle:
            st.session_state.selected_model = st.selectbox("Aktives Modell:", modelle)
        else:
            st.warning("Kein Modell (.pkl) gefunden.")
    else:
        st.error("Ordner 'models' fehlt.")

# --- SEITE 1: LIVE CHART ---
if page == "📈 Live Chart":
    st.subheader(f"Trading Terminal: {st.session_state.current_symbol}")
    col_main, col_side = st.columns([3.5, 1.2])

    with col_main:
        # Daten laden
        df_raw = get_market_data(st.session_state.current_symbol)
        if df_raw is not None and not df_raw.empty:
            df = add_indicators(df_raw)
            # Chart anzeigen
            create_trading_chart(df)
            # Logbuch unter dem Chart
            display_live_log()
        else:
            st.error("Verbindung zu Yahoo Finance wird aufgebaut oder Symbol ungültig...")

    with col_side:
        st.markdown(f"### ⚡ Signal Analysis")
        if 'df' in locals() and df is not None:
            # KI Signal berechnen
            signal, conf = execute_trade_decision(df, st.session_state.selected_model)
            st.metric("KI-VORGABE", signal, f"Conf: {conf}")

            # Logbuch-Logik (Eintrag bei Signal-Wechsel)
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            if not st.session_state.trade_log or st.session_state.trade_log[0]['Signal'] != signal:
                if signal in ["BUY", "SELL"]:
                    st.session_state.trade_log.insert(0, {
                        "Zeit": ts, "Symbol": st.session_state.current_symbol,
                        "Signal": signal, "Conf": conf, "Preis": round(df['Close'].iloc[-1], 2)
                    })

            # Statistiken & Terminal
            display_stats(df)
            new_sym = display_terminal(st.session_state.current_symbol)
            if new_sym and new_sym != st.session_state.current_symbol:
                st.session_state.current_symbol = new_sym
                st.rerun()

# --- SEITE 2: ML TRAINING ---
elif page == "🧠 ML Training":
    display_ml_training_page()

# --- SEITE 3: DATASETS (Hier war der Fehler!) ---
elif page == "📂 Datasets":
    display_dataset_page()
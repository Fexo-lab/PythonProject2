import streamlit as st
import pandas as pd
import os
import yfinance as yf
import requests

DATASET_DIR = "datasets"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)


def search_ticker(query):
    """Sucht nach Symbolen über die Yahoo Finance API."""
    if not query or len(query) < 2: return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        return [f"{q.get('symbol')} | {q.get('shortname', 'Asset')}" for q in data.get('quotes', [])]
    except:
        return []


def display_dataset_page():
    st.subheader("📂 Dataset Management & Import")

    # --- TEIL 1: IMPORT NEUER DATEN ---
    with st.expander("📥 Neuen Datensatz herunterladen", expanded=True):
        search_query = st.text_input("Asset suchen (z.B. Silver, Gold, BTC)", placeholder="Name eingeben...")

        selected_ticker = None
        if search_query:
            suggestions = search_ticker(search_query)
            if suggestions:
                choice = st.selectbox("Gefundene Symbole:", suggestions)
                selected_ticker = choice.split(" | ")[0]

        col1, col2 = st.columns(2)
        period = col1.selectbox("Zeitraum", ["1y", "2y", "5y", "max"])
        interval = col2.selectbox("Intervall", ["1h", "1d", "15m"])

        if st.button("🚀 Daten jetzt importieren"):
            if not selected_ticker:
                st.error("Bitte wähle zuerst ein Asset aus!")
            else:
                with st.status(f"Lade {selected_ticker} herunter...") as status:
                    data = yf.download(selected_ticker, period=period, interval=interval)
                    if not data.empty:
                        # MultiIndex Fix für 2026er yfinance Version
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)

                        filename = f"{selected_ticker.replace('=', '')}_{interval}.csv"
                        path = os.path.join(DATASET_DIR, filename)
                        data.to_csv(path)
                        status.update(label=f"Gespeichert als {filename}!", state="complete")
                        st.success(f"Datei erfolgreich in {DATASET_DIR} abgelegt.")
                    else:
                        st.error("Keine Daten gefunden.")

    st.divider()

    # --- TEIL 2: VORHANDENE DATEIEN VERWALTEN ---
    st.markdown("### 📊 Vorhandene Datensätze")
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]

    if files:
        for file in files:
            col_a, col_b = st.columns([3, 1])
            col_a.code(file)
            if col_b.button("Löschen", key=file):
                os.remove(os.path.join(DATASET_DIR, file))
                st.rerun()
    else:
        st.info("Noch keine CSV-Dateien im Ordner 'datasets' vorhanden.")
import streamlit as st
import pandas as pd
import os
import yfinance as yf
import requests
from Core.config import DATASETS_DIR

DATASET_DIR = DATASETS_DIR
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)


def search_ticker(query):
    """Search for symbols via the Yahoo Finance API."""
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

    # --- SECTION 1: IMPORT NEW DATA ---
    with st.expander("📥 Download new dataset", expanded=True):
        search_query = st.text_input("Search Asset (e.g. Silver, Gold, BTC)", placeholder="Enter name...")

        selected_ticker = None
        if search_query:
            suggestions = search_ticker(search_query)
            if suggestions:
                choice = st.selectbox("Found Symbols:", suggestions)
                selected_ticker = choice.split(" | ")[0]

        col1, col2 = st.columns(2)
        period = col1.selectbox("Time Period", ["1y", "2y", "5y", "max"])
        interval = col2.selectbox("Interval", ["1h", "1d", "15m"])

        if st.button("🚀 Import Data Now"):
            if not selected_ticker:
                st.error("Please select an asset first!")
            else:
                with st.status(f"Downloading {selected_ticker}...") as status:
                    data = yf.download(selected_ticker, period=period, interval=interval)
                    if not data.empty:
                        # MultiIndex fix for yfinance
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)

                        filename = f"{selected_ticker.replace('=', '')}_{interval}.csv"
                        path = os.path.join(DATASET_DIR, filename)
                        data.to_csv(path)
                        status.update(label=f"Saved as {filename}!", state="complete")
                        st.success(f"File successfully saved to {DATASET_DIR}.")
                    else:
                        st.error("No data found.")

    st.divider()

    # --- SECTION 2: MANAGE EXISTING FILES ---
    st.markdown("### 📊 Available Datasets")
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]

    if files:
        for file in files:
            col_a, col_b = st.columns([3, 1])
            col_a.code(file)
            if col_b.button("Delete", key=file):
                os.remove(os.path.join(DATASET_DIR, file))
                st.rerun()
    else:
        st.info("No CSV files in 'datasets' folder yet.")
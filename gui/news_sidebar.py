import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from Core.engine.data import get_asset_news, load_live_assets
from Core.config import NEWS_STORAGE_DIR

# Pfade basierend auf Projektstruktur
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = NEWS_STORAGE_DIR

if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)


def analyze_sentiment(title):
    """Bewertet den Titel auf bullische oder bärische Keywords."""
    bullish = ['rise', 'high', 'jump', 'bull', 'gain', 'growth', 'positive', 'up', 'surge', 'rally', 'breakout']
    bearish = ['fall', 'low', 'drop', 'bear', 'loss', 'decline', 'negative', 'down', 'crash', 'slump', 'risk']
    score = 0
    t = title.lower()
    for w in bullish:
        if w in t: score += 1
    for w in bearish:
        if w in t: score -= 1
    return score


def get_sentiment_trend_data(search_term, days=7):
    """Erstellt einen Pandas DataFrame mit dem Sentiment-Verlauf."""
    safe_file_name = "".join([c for c in search_term if c.isalnum()])
    file_path = os.path.join(HISTORY_DIR, f"{safe_file_name}_history.json")

    if not os.path.exists(file_path):
        return pd.DataFrame()

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except:
            return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Zeitstempel umwandeln und sortieren
    df['date_added'] = pd.to_datetime(df['date_added'])
    df = df.sort_values('date_added')

    # Filter auf Zeitraum
    cutoff = datetime.now() - timedelta(days=days)
    df = df[df['date_added'] > cutoff]

    # Gleitender Durchschnitt für den Trend-Index
    if not df.empty:
        df['sentiment_index'] = df['score'].rolling(window=3, min_periods=1).mean()

    return df


def update_and_get_history(search_term, fresh_news):
    """Speichert News mit korrekter Lokalzeit."""
    safe_file_name = "".join([c for c in search_term if c.isalnum()])
    file_path = os.path.join(HISTORY_DIR, f"{safe_file_name}_history.json")

    asset_history = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                asset_history = json.load(f)
            except:
                asset_history = []

    existing_titles = {item['title'] for item in asset_history}
    added_count = 0

    for n in fresh_news:
        title = n.get('title') or n.get('text')

        if title and title not in existing_titles:
            # Zeitzonen-Fix: UTC zu Lokal
            raw_ts = n.get('providerPublishTime')
            if isinstance(raw_ts, (int, float)):
                dt_utc = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
                dt_local = dt_utc.astimezone()
                date_str = dt_local.strftime("%Y-%m-%d %H:%M")
            else:
                date_str = "0000-00-00 00:00"

            entry = {
                'title': title,
                'score': analyze_sentiment(title),
                'publisher': n.get('publisher') or "Yahoo Finance",
                'link': n.get('link'),
                'date_added': date_str
            }
            asset_history.append(entry)
            existing_titles.add(title)
            added_count += 1

    if added_count > 0:
        asset_history.sort(key=lambda x: x.get('date_added', ''), reverse=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asset_history, f, indent=4, ensure_ascii=False)

    return asset_history, added_count, file_path


def display_news_page(selected_symbol):
    """
    Optimierte News-Sentinel UI.
    Der globale Sync läuft bereits in app_ui.py.
    Hier konzentrieren wir uns rein auf die Visualisierung.
    """
    st.title("📰 News Maximizer & Sentinel")

    assets_data = load_live_assets()
    if not assets_data:
        st.warning("No assets found.")
        return

    # Asset-Auswahl
    display_names = list(assets_data.keys())
    clean_ticker = str(selected_symbol).split('|')[0].strip()
    current_name = next((n for n, t in assets_data.items() if t == clean_ticker), display_names[0])
    selected_display_name = st.selectbox("Focus:", options=display_names, index=display_names.index(current_name))

    search_term = selected_display_name.split(' ')[0].strip()

    try:
        # Lokale News-Daten laden
        # Wir rufen get_asset_news trotzdem auf, um sicherzugehen, dass das Fokus-Asset absolut aktuell ist
        fresh_api_news = get_asset_news(search_term)
        all_news, _, _ = update_and_get_history(search_term, fresh_api_news)

        # --- SENTIMENT TREND CHART ---
        st.subheader(f"📈 Sentiment Verlauf: {search_term}")
        trend_df = get_sentiment_trend_data(search_term)

        if not trend_df.empty:
            # Modernisierte Layout-Syntax für 2026 [cite: 2026-03-07]
            st.area_chart(data=trend_df, x='date_added', y='sentiment_index', width='stretch')
        else:
            st.info("Collecting data for trend analysis...")

        # News Liste mit Farbindikatoren
        st.write("---")
        for item in all_news[:100]:
            color = "#00ff00" if item['score'] > 0 else "#ff4b4b" if item['score'] < 0 else "#777777"
            # Bullet-Point Design für bessere Lesbarkeit
            st.markdown(f"<span style='color:{color}'>●</span> **{item['title']}**", unsafe_allow_html=True)
            st.caption(f"{item['publisher']} | {item['date_added']}")

    except Exception as e:
        st.error(f"Fehler: {e}")
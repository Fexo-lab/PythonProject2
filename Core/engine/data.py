import yfinance as yf
import pandas as pd
import os
import json
from Core.ml.preprocessor import get_feature_matrix

LIVE_CONFIG = "live_assets.json"

def load_live_assets():
    """Load the list of tradeable assets."""
    if os.path.exists(LIVE_CONFIG):
        try:
            with open(LIVE_CONFIG, "r") as f:
                return json.load(f)
        except:
            return {"Gold": "GC=F", "Silver": "SI=F"}
    return {"Gold": "GC=F", "Silver": "SI=F"}

def save_live_assets(assets):
    """Saves new assets to configuration (was missing earlier)."""
    with open(LIVE_CONFIG, "w") as f:
        json.dump(assets, f)

def get_market_data(ticker_or_path, period="1mo", interval="1h"):
    # ... (Rest der Funktion bleibt wie vorhin gegeben)
    if not ticker_or_path: return None
    is_csv = str(ticker_or_path).lower().endswith(".csv")
    try:
        if is_csv:
            df = pd.read_csv(ticker_or_path, index_col=0, engine='python')
            df.index = pd.to_datetime(df.index, errors='coerce')
            return df.sort_index()
        else:
            df = yf.download(ticker_or_path, period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return None


def add_indicators(df):
    if df is None or len(df) < 20: return df

    # The central matrix calculates everything (volatility, returns, etc.)
    from Core.ml.preprocessor import get_feature_matrix
    df_ki, _ = get_feature_matrix(df)

    # IMPORTANT: We must return calculated columns in the original DataFrame
    # This ensures the data flow is preserved correctly
    for col in ['volatility', 'rsi', 'returns']:
        if col in df_ki.columns:
            df[col] = df_ki[col]

    # Absolute EMAs for the chart
    for period in [20, 50, 100]:
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

    return df


def get_asset_news(search_term):
    """Fetch news for a given asset using yfinance."""
    try:
        # Create a ticker object with the asset name or symbol
        assets_map = load_live_assets()
        ticker_symbol = assets_map.get(search_term, search_term)
        
        # Fetch news using yfinance
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        
        if news:
            return news
        else:
            return []
    except Exception as e:
        print(f"News Fetch Error for {search_term}: {e}")
        return []
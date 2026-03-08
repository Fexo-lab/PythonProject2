import numpy as np
import pandas as pd
from ..config import FEATURE_COLS


def prepare_live_features(df, weights):
    temp_df = df.copy()
    # Identische Logik wie im Preprocessor für alle 9 Features
    temp_df['returns'] = temp_df['Close'].pct_change()
    temp_df['volatility'] = temp_df['returns'].rolling(window=10).std()
    temp_df['range'] = (temp_df['High'] - temp_df['Low']) / (temp_df['Close'] + 1e-9)
    tr = pd.concat([(temp_df['High'] - temp_df['Low']), (temp_df['High'] - temp_df['Close'].shift()).abs(),
                    (temp_df['Low'] - temp_df['Close'].shift()).abs()], axis=1).max(axis=1)
    temp_df['atr'] = tr.rolling(window=14).mean() / (temp_df['Close'] + 1e-9)

    for period in [20, 50, 100]:
        ema = temp_df['Close'].ewm(span=period, adjust=False).mean()
        temp_df[f'ema_{period}_dist'] = (temp_df['Close'] - ema) / (ema + 1e-9) * 100

    delta = temp_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    temp_df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-5))))
    temp_df['volume_force'] = temp_df['Volume'] / (
        temp_df['Volume'].rolling(window=20).mean().replace(0, 1)) if 'Volume' in temp_df.columns else 1.0

    # Normalisierung (identisch zum Preprocessor)
    df_feat = temp_df[FEATURE_COLS].copy()
    temp_df[FEATURE_COLS] = (df_feat - df_feat.mean()) / (df_feat.std() + 1e-9)

    # Alle 9 Spalten nutzen
    features = temp_df[FEATURE_COLS].fillna(0)
    return features.values * weights

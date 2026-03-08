import pandas as pd
import numpy as np
import os

from Core.config import FEATURE_COLS, DATASETS_DIR


def get_feature_matrix(df):
    """
    Berechnet alle technischen Indikatoren und normalisiert sie,
    damit die DNA-Gewichtung mathematisch korrekt greifen kann.
    """
    df = df.copy()

    # 1. Basis-Indikatoren
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=10).std()
    df['range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)

    # ATR Berechnung
    tr = pd.concat([
        (df['High'] - df['Low']),
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean() / (df['Close'] + 1e-9)

    # EMA Abstände
    for period in [20, 50, 100]:
        ema = df['Close'].ewm(span=period, adjust=False).mean()
        df[f'ema_{period}_dist'] = (df['Close'] - ema) / (ema + 1e-9) * 100

    # RSI Berechnung
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-5))))

    # Volumen Kraft
    if 'Volume' in df.columns:
        df['volume_force'] = df['Volume'] / (df['Volume'].rolling(window=20).mean().replace(0, 1))
    else:
        df['volume_force'] = 1.0

    # 2. NORMALISIERUNG (Der Ursprungs-Fix)
    # Bringt alle Features in den Bereich von ca. -3.0 bis +3.0
    # Ohne dies würde der RSI (100) die Returns (0.001) einfach "erdrücken"
    df_feat = df[FEATURE_COLS].copy()
    df[FEATURE_COLS] = (df_feat - df_feat.mean()) / (df_feat.std() + 1e-9)

    return df.dropna(), FEATURE_COLS


def create_simulated_training_set(csv_name, max_sl_pct):
    """
    Erstellt ein Trainings-Set mit Labels basierend auf einem dynamischen Trailing Stop.
    """
    file_path = os.path.join(DATASETS_DIR, csv_name)
    if not os.path.exists(file_path):
        return None, None

    df_raw = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # Wichtig: get_feature_matrix liefert die normalisierten Daten
    df, _ = get_feature_matrix(df_raw)

    labels, pip_results = [], []
    prices = df['Close'].values

    # Simulation des Trailing Stops für Labeling
    for i in range(len(prices)):
        label, pip_diff = 0, 0
        trade_found = False
        entry_p = prices[i]

        # BUY Simulation
        curr_sl = entry_p * (1 - max_sl_pct)
        max_seen = entry_p
        for j in range(1, 150):
            if i + j >= len(prices): break
            p = prices[i + j]
            if p > max_seen:
                max_seen = p
                curr_sl = max_seen * (1 - (max_sl_pct * 0.5))
            if p <= curr_sl:
                pip_diff = (curr_sl - entry_p) / entry_p * 1000
                # Label basierend auf ob Gewinn oder Verlust (NICHT auf Profit-Schwelle!)
                label = 1 if pip_diff > 0 else 0
                trade_found = True
                break

        # SELL Simulation (falls BUY nicht erfolgreich war)
        if not trade_found:
            curr_sl = entry_p * (1 + max_sl_pct)
            min_seen = entry_p
            for j in range(1, 150):
                if i + j >= len(prices): break
                p = prices[i + j]
                if p < min_seen:
                    min_seen = p
                    curr_sl = min_seen * (1 + (max_sl_pct * 0.5))
                if p >= curr_sl:
                    pip_diff = (entry_p - curr_sl) / entry_p * 1000
                    # Label basierend auf ob Gewinn oder Verlust (NICHT auf Profit-Schwelle!)
                    label = 2 if pip_diff > 0 else 0
                    trade_found = True
                    break

        # If no trade within 150 bars, mark as 0
        # (This indicates poor conditions or lack of opportunity)
        if not trade_found:
            label = 0
            pip_diff = 0

        labels.append(label)
        pip_results.append(pip_diff)

    df['target_label'] = labels
    df['pip_result'] = pip_results

    # Entferne die ersten 100 Zeilen (wegen Indikatoren-Warmup)
    full_df = df.iloc[100:].copy().reset_index(drop=True)

    # 80/20 Train-Test Split
    split_idx = int(len(full_df) * 0.8)
    return full_df.iloc[:split_idx].copy(), full_df.iloc[split_idx:].copy()
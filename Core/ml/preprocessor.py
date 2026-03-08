import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Core.config import FEATURE_COLS, DATASETS_DIR


def get_feature_matrix(df):
    """
    Calculates all technical indicators and normalizes them
    so that DNA weighting works mathematically correctly.
    """
    df = df.copy()

    # 1. Base indicators
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=10).std()
    df['range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)

    # ATR Calculation
    tr = pd.concat([
        (df['High'] - df['Low']),
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean() / (df['Close'] + 1e-9)

    # EMA Distances
    for period in [20, 50, 100]:
        ema = df['Close'].ewm(span=period, adjust=False).mean()
        df[f'ema_{period}_dist'] = (df['Close'] - ema) / (ema + 1e-9) * 100

    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-5))))

    # Volume Force
    if 'Volume' in df.columns:
        df['volume_force'] = df['Volume'] / (df['Volume'].rolling(window=20).mean().replace(0, 1))
    else:
        df['volume_force'] = 1.0

    # 2. NORMALIZATION (The Normalization Fix)
    # Brings all features into the range of approximately -3.0 to +3.0
    # Without this, RSI (100) would simply "crush" Returns (0.001)
    df_feat = df[FEATURE_COLS].copy()
    df[FEATURE_COLS] = (df_feat - df_feat.mean()) / (df_feat.std() + 1e-9)

    return df.dropna(), FEATURE_COLS


def create_simulated_training_set(csv_name, max_sl_pct):
    """
    Creates a training set with labels based on a dynamic trailing stop.
    """
    file_path = os.path.join(DATASETS_DIR, csv_name)
    if not os.path.exists(file_path):
        return None, None

    df_raw = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # Important: get_feature_matrix provides normalized data
    df, _ = get_feature_matrix(df_raw)

    labels, pip_results = [], []
    prices = df['Close'].values

    # Trailing Stop Simulation for Labeling
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

        # SELL Simulation (if BUY was not successful)
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
                    # Label based on profit or loss (NOT on profit threshold!)
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

    # Remove the first 100 rows (due to indicator warmup)
    full_df = df.iloc[100:].copy().reset_index(drop=True)

    # STRATIFIED TRAIN-TEST SPLIT (ensures balanced class distribution)
    # This prevents one set from having disproportionately more BUY/SELL/HOLD
    train_df, test_df = train_test_split(
        full_df, 
        test_size=0.2,
        random_state=42,
        stratify=full_df['target_label']  # Balance class distribution
    )
    
    return train_df.copy().reset_index(drop=True), test_df.copy().reset_index(drop=True)


def analyze_feature_importance(X_train, X_test, y_train, y_test):
    """
    Analyzes which features contribute most to predictions.
    Returns important features to remove noise and reduce overfitting.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Train a minimal model just for feature analysis
    model = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Get feature importance
    importances = np.abs(model.feature_importances_)
    feature_importance_dict = dict(zip(X_train.columns, importances))
    
    # Rank features
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    return feature_importance_dict, sorted_features


def get_top_features(feature_importance_dict, top_n=6):
    """
    Returns top N features by importance.
    Only use the best features to reduce noise and overfitting.
    """
    sorted_dict = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_dict[:top_n]]
    return top_features
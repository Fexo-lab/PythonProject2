import pandas as pd
import numpy as np
import os

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


def create_simulated_training_set(csv_name, max_sl_pct, punishment_pips=0, use_atr_sl=True, atr_multiplier=1.5):
    """
    Creates a training set with labels based on independent BUY/SELL logic.
    IMPORTANTLY: Uses Walk-Forward validation (time-respecting) for training data.
    
    IMPROVED: Uses ATR-based Stop Loss (volatility-adaptive) instead of fixed %
    
    Args:
        csv_name: Dataset file name
        max_sl_pct: Fallback Stop-Loss percentage (if use_atr_sl=False)
        punishment_pips: Entry cost (spread/slippage) to subtract from each trade result
        use_atr_sl: Use ATR-based stop loss (True) or fixed % (False)
        atr_multiplier: ATR multiplier for stop loss distance (1.5 = 1.5*ATR)
    """
    file_path = os.path.join(DATASETS_DIR, csv_name)
    if not os.path.exists(file_path):
        return None, None

    df_raw = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # Important: get_feature_matrix provides normalized data
    df, _ = get_feature_matrix(df_raw)

    # Calculate RAW (non-normalized) ATR for stop loss calculations
    # The ATR in get_feature_matrix is normalized (divided by Close) for feature engineering
    # But for stop loss, we need the actual ATR in price units
    if use_atr_sl:
        tr_raw = pd.concat([
            (df_raw['High'] - df_raw['Low']),
            (df_raw['High'] - df_raw['Close'].shift()).abs(),
            (df_raw['Low'] - df_raw['Close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr_raw = tr_raw.rolling(window=14).mean()  # RAW ATR in points, not normalized
        df['atr_raw'] = atr_raw.values  # Store for later use
    else:
        df['atr_raw'] = None
    
    labels, pip_results = [], []
    prices = df['Close'].values
    atr_raw_values = df['atr_raw'].values if use_atr_sl else None

    # Independent BUY/SELL Labeling (Not "SELL only if BUY fails")
    for i in range(len(prices)):
        entry_p = prices[i]
        best_label = 0  # Default: HOLD
        best_pip_diff = 0
        
        # Get RAW ATR value at entry point (in price units, not normalized)
        atr_val = atr_raw_values[i] if use_atr_sl and i < len(atr_raw_values) and atr_raw_values[i] > 0 else None

        # ===== BUY SCENARIO (Label 1) - ATR-based =====
        if use_atr_sl and atr_val is not None:
            # ATR Stop Loss: Entry - (ATR * Multiplier) in price units
            sl_distance = atr_val * atr_multiplier
            curr_sl_buy = entry_p - sl_distance
            # Trailing stop: if new high, reduce stop loss by half the new distance
            max_seen = entry_p
        else:
            # Fallback: percentage-based
            curr_sl_buy = entry_p * (1 - max_sl_pct)
            max_seen = entry_p
        
        buy_pips = 0
        for j in range(1, 150):
            if i + j >= len(prices): break
            p = prices[i + j]
            if p > max_seen:
                max_seen = p
                if use_atr_sl and atr_val is not None:
                    # Trailing stop: reduce stop distance to half ATR
                    new_atr_dist = atr_val * (atr_multiplier * 0.5)
                    curr_sl_buy = max(curr_sl_buy, max_seen - new_atr_dist)
                else:
                    curr_sl_buy = max(curr_sl_buy, max_seen * (1 - (max_sl_pct * 0.5)))
            
            if p <= curr_sl_buy:
                buy_pips = (curr_sl_buy - entry_p) / entry_p * 1000 - punishment_pips
                if buy_pips > 0:  # Only if profitable after entry cost
                    best_label = 1
                    best_pip_diff = buy_pips
                break

        # ===== SELL SCENARIO (Label 2) - ATR-based - INDEPENDENT of BUY =====
        if use_atr_sl and atr_val is not None:
            # ATR Stop Loss: Entry + (ATR * Multiplier)
            sl_distance = atr_val * atr_multiplier
            curr_sl_sell = entry_p + sl_distance
            # Trailing stop for short
            min_seen = entry_p
        else:
            # Fallback: percentage-based
            curr_sl_sell = entry_p * (1 + max_sl_pct)
            min_seen = entry_p
        
        sell_pips = 0
        for j in range(1, 150):
            if i + j >= len(prices): break
            p = prices[i + j]
            if p < min_seen:
                min_seen = p
                if use_atr_sl and atr_val is not None:
                    # Trailing stop: reduce stop distance to half ATR
                    new_atr_dist = atr_val * (atr_multiplier * 0.5)
                    curr_sl_sell = min(curr_sl_sell, min_seen + new_atr_dist)
                else:
                    curr_sl_sell = min(curr_sl_sell, min_seen * (1 + (max_sl_pct * 0.5)))
            
            if p >= curr_sl_sell:
                sell_pips = (entry_p - curr_sl_sell) / entry_p * 1000 - punishment_pips
                if sell_pips > 0:  # Only if profitable after entry cost
                    # Prefer SELL over BUY if it's more profitable
                    if sell_pips > best_pip_diff:
                        best_label = 2
                        best_pip_diff = sell_pips
                break

        labels.append(best_label)
        pip_results.append(best_pip_diff)

    df['target_label'] = labels
    df['pip_result'] = pip_results

    # Remove the first 100 rows (due to indicator warmup)
    full_df = df.iloc[100:].copy().reset_index(drop=True)

    # WALK-FORWARD VALIDATION (Time-respecting, not shuffled random)
    # 80% training, 20% testing (sequential in time - correct for time-series)
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = full_df.iloc[split_idx:].copy().reset_index(drop=True)
    
    return train_df, test_df


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
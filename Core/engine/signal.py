import os, joblib, json, numpy as np
from ..ml.preprocessor import get_feature_matrix, FEATURE_COLS


def execute_trade_decision(df, model_name):
    if df is None or len(df) < 100: return "WAITING", 0.0
    model_path, dna_path = f"models/{model_name}.pkl", f"models/{model_name}_dna.json"
    if not os.path.exists(model_path): return "NO_MODEL", 0.0

    try:
        df_processed, _ = get_feature_matrix(df)
        with open(dna_path, 'r') as f:
            dna = json.load(f)
            weights = np.array(dna.get('weights', [1.0]*9))
            biases = np.array(dna.get('biases', [0.0]*9))

        # Nutzt jetzt Weights UND Biases für Feature-Skalierung
        X_live = (df_processed[FEATURE_COLS].tail(1).values * weights) + biases
        model = joblib.load(model_path)

        prediction = model.predict(X_live)[0]
        probs = model.predict_proba(X_live)[0]

        if prediction == 1: return "BUY", round(probs[1], 3)
        if prediction == 2: return "SELL", round(probs[2], 3)
        return "NEUTRAL", round(probs[0], 3)
    except Exception as e:
        return f"ERROR: {str(e)}", 0.0
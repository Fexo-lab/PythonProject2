# Core/config.py

# Zentraler Anker für alle Parameter - Jetzt auf 9 Features erweitert
FEATURE_COLS = [
    'returns',
    'volatility',
    'range',
    'atr',               # NEU: Markt-Volatilität (Gaps & Kraft)
    'ema_20_dist',
    'ema_50_dist',
    'ema_100_dist',
    'rsi',
    'volume_force'       # NEU: Volumen vs. Durchschnitt
]

# GA Parameter (Genetischer Algorithmus)
POPULATION_SIZE = 15
MUTATION_RATE = 0.7
DEFIBRILLATOR_LEVEL = 0.1
STAGNATION_THRESHOLD = 20 # Zyklen ohne Rekord vor Mutations-Erhöhung

# Modell & Daten Pfade
MODELS_DIR = "models"
DATASETS_DIR = "datasets"
NEWS_STORAGE_DIR = "news_storage"

# Streamlit UI Konfiguration
STREAMLIT_PORT = 8501
STREAMLIT_HOST = "localhost"
UI_WIDTH = 1280
UI_HEIGHT = 900
UI_TITLE = "AI Quantum Trader Pro v1.0"

# Timeouts & Delays
STREAMLIT_STARTUP_DELAY = 4  # Sekunden
API_TIMEOUT = 5  # Sekunden

# Model Quality Thresholds
MIN_TRADES_FOR_SIGNIFICANCE = 50  # Weniger als 50 Trades = nicht aussagekräftig
OVERFIT_THRESHOLD = 0.3  # Wenn (train_fit - test_fit) / abs(test_fit) > 0.3 = overfitting
REALISTIC_WR_MIN = 30.0  # Win Rate sollte nicht unter 30% sein (unrealistisch)
REALISTIC_WR_MAX = 70.0  # Win Rate sollte nicht über 70% sein (zu optimistisch)
REALISTIC_PF_MIN = 0.8  # Profit Factor unter 0.8 = nicht sinnvoll
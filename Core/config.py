# Core/config.py

# Central anchor for all parameters - Now extended to 9 features
FEATURE_COLS = [
    'returns',
    'volatility',
    'range',
    'atr',               # NEW: Market Volatility (Gaps & Strength)
    'ema_20_dist',
    'ema_50_dist',
    'ema_100_dist',
    'rsi',
    'volume_force'       # NEW: Volume vs. Average
]

# GA Parameters (Genetic Algorithm)
POPULATION_SIZE = 15
MUTATION_RATE = 0.7
DEFIBRILLATOR_LEVEL = 0.1
STAGNATION_THRESHOLD = 40  # Cycles without record before increasing mutations (was 20 - too aggressive for long runs)

# Training Safeguards (for 10h+ training sessions)
MAX_FITNESS_PATIENCE = 100  # Stop early if no improvement after N cycles
MIN_TRADES_FOR_MODEL_SAVE = 30  # Only save model if it generates 30+ trades
MAX_TRAINING_CYCLES = 10000  # Failsafe to prevent infinite loops

# Model & Data Paths
MODELS_DIR = "models"
DATASETS_DIR = "datasets"
NEWS_STORAGE_DIR = "news_storage"

# Streamlit UI Configuration
STREAMLIT_PORT = 8501
STREAMLIT_HOST = "localhost"
UI_WIDTH = 1280
UI_HEIGHT = 900
UI_TITLE = "AI Quantum Trader Pro v1.0"

# Timeouts & Delays
STREAMLIT_STARTUP_DELAY = 4  # Seconds
API_TIMEOUT = 5  # Seconds

# Model Quality Thresholds
MIN_TRADES_FOR_SIGNIFICANCE = 50  # Less than 50 trades = not statistically valid
OVERFIT_THRESHOLD = 0.25  # Relaxed: Trading data is inherently noisy; 25% difference is reasonable
REALISTIC_WR_MIN = 30.0  # Win Rate should not be below 30% (unrealistic)
REALISTIC_WR_MAX = 70.0  # Win Rate should not be above 70% (too optimistic)
REALISTIC_PF_MIN = 0.8  # Profit Factor below 0.8 = not meaningful
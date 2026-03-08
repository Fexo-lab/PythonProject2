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

# Trading Parameter
EMA_WINDOW_MAX = 100     # Minimum Datenpunkte für saubere EMAs
STAGNATION_THRESHOLD = 20 # Zyklen ohne Rekord vor Mutations-Erhöhung
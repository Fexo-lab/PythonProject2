"""
Quantum Trading Bot - Core Module
==================================

Главные Komponenten:
- ml: Machine Learning & Evolution
- engine: Trading Engine & Marktdaten
- config: Zentrale Konfiguration
"""

from . import ml
from . import engine
from .config import FEATURE_COLS

__version__ = "1.0.0"
__author__ = "Fexo-lab"

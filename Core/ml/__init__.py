"""
ML Module - Genetischer Algorithmus & Datenverarbeitung
========================================================

Enthält:
- core: Evolutionärer Algorithmus für Feature-Gewichtungen
- preprocessor: Datenaufbereitung und Label-Generierung
- brain: Live-Feature-Vorbereitung für Predictions
"""

from .core import EvolutionCore
from .preprocessor import get_feature_matrix, create_simulated_training_set
from .brain import prepare_live_features

__all__ = ['EvolutionCore', 'get_feature_matrix', 'create_simulated_training_set', 'prepare_live_features']

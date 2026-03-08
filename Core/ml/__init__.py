"""
ML Module - Genetic Algorithm & Data Processing
========================================================

Contains:
- core: Evolutionary Algorithm for Feature Weights
- preprocessor: Data Preparation and Label Generation
- brain: Live Feature Preparation for Predictions
"""

# Lazy imports to avoid circular dependencies
__all__ = ['EvolutionCore', 'get_feature_matrix', 'create_simulated_training_set', 'prepare_live_features']

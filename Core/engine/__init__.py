"""
Engine Module - Trading & Daten-Engine
======================================

Enthält:
- data: Marktdaten-Abruf und Indikatoren-Berechnung
- signal: Trade-Signal-Generierung
"""

# Lazy imports to avoid circular dependencies
__all__ = ['get_market_data', 'add_indicators', 'load_live_assets', 'save_live_assets', 
           'get_asset_news', 'execute_trade_decision']

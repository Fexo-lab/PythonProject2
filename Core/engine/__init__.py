"""
Engine Module - Trading & Data Engine
======================================

Contains:
- data: Market Data Fetching and Indicator Calculation
- signal: Trade Signal Generation
"""

# Lazy imports to avoid circular dependencies
__all__ = ['get_market_data', 'add_indicators', 'load_live_assets', 'save_live_assets', 
           'get_asset_news', 'execute_trade_decision']

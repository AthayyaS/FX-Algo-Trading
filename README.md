FX-Folder Repository Overview
What is this repo?
This is a GBP/USD Algorithmic Trading System that backtests three different macro-technical trading strategies using historical market data and macroeconomic indicators.

Key Features
3 Trading Strategies:
1. Kalman Filter + Rate Differential - Ultra-simplified approach
2. Hybrid Macro-Technical with ATR - Comprehensive strategy with multiple indicators
3. Rate Differential Only - Simplest approach using interest rate spreads

Data Sources:
1. Market data: Yahoo Finance (GBP/USD prices)
2. Macro data: FRED API (interest rates, inflation, GDP)
3. Analysis Tools: Win rates, Sharpe ratios, drawdowns, yearly performance, regime shift detection

Results
All outputs are saved to the Results folder:
- s1_train_backtest.png, s1_test_backtest.png (plots)
- s1_train_backtest.xlsx, s1_test_backtest.xlsx (detailed metrics)
- Similar files for strategies 2 and 3

Directory Structure
main.py - Core backtesting engine
strategy_kalman_ratediff.py - Strategy 1
strategy_hybrid_atr.py - Strategy 2
strategy_rate_only.py - Strategy 3
requirements.txt - Dependencies
Results - Auto-generated backtest outputs
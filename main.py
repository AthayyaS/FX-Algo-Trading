"""
GBP/USD Algorithmic Trading System
Hybrid Macro-Technical Strategies with Multiple Models

This system backtests 3 different trading strategies:
1. Kalman + Rate_Diff (Ultra-Simplified)
2. Hybrid Macro-Technical with ATR (Comprehensive)
3. Rate Differential Only (Simplest)

Uses data from:
- yfinance for market prices (GBP/USD)
- FRED API for macro fundamentals (rates, inflation, GDP)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from fredapi import Fred
from scipy import signal
from sklearn.model_selection import train_test_split
from datetime import datetime

# Import strategy modules
from strategy_kalman_ratediff import generate_trading_signals as kalman_ratediff_signals
from strategy_hybrid_atr import generate_trading_signals as hybrid_signals
from strategy_rate_only import generate_trading_signals as rate_only_signals

# ============================================================
# SECTION 1: DATA FETCHING & PREPARATION
# ============================================================

def fetch_market_data(symbol="GBPUSD=X", start="2010-01-01", end="2025-01-01"):
    """Fetch FX market data from yfinance."""
    data = yf.download(symbol, start=start, end=end)
    
    # Remove MultiIndex columns (flatten)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)
    
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Rename columns
    data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    
    return data


def fetch_macro_data():
    """Fetch and merge UK/US macro data from FRED with publication lags"""
    os.environ["FRED_API_KEY"] = "ceae23455b8523f81afc08b15171a2e9"
    fred = Fred()
    
    # Interest Rates (no lag - same-day publication)
    uk_rate = fred.get_series("IRSTCI01GBM156N")
    us_rate = fred.get_series("FEDFUNDS")
    
    # Inflation (2-week lag)
    uk_cpi = fred.get_series("GBRCPIALLMINMEI").shift(14, freq='D')
    us_cpi = fred.get_series("USACPALTT01CTGYM").shift(14, freq='D')
    
    # GDP (1-month lag)
    uk_gdp = fred.get_series("UKNGDP").shift(30, freq='D')
    us_gdp = fred.get_series("GDPC1").shift(30, freq='D')
    
    # Merge into one DataFrame
    macro = (
        pd.DataFrame({"UK_rate": uk_rate, "US_rate": us_rate})
        .join(pd.DataFrame({"UK_CPI": uk_cpi, "US_CPI": us_cpi}), how="outer")
        .join(pd.DataFrame({"UK_GDP": uk_gdp, "US_GDP": us_gdp}), how="outer")
    ).ffill()
    
    # Forward fill GDP (quarterly → monthly)
    macro = macro.ffill()

    # Reset index to make date a column
    macro = macro.reset_index()
    macro.rename(columns={"index": "Date"}, inplace=True)

    return macro


def calculate_kalman_filter(df, process_variance=0.001, measurement_variance=0.1):
    """
    Calculate Kalman Filter to estimate fair value of GBP/USD.
     
    Args:
        df: DataFrame with Close prices
        process_variance: How much the price can deviate (tuning parameter)
        measurement_variance: Measurement noise (tuning parameter)
    
    Returns:
        DataFrame with Kalman Filter estimates
    """
    close_prices = df["Close"].values
    n = len(close_prices)
    
    # Initialize
    x = np.zeros(n)  # State (estimated fair value)
    p = np.zeros(n)  # Estimation error
    k = np.zeros(n)  # Kalman gain
    
    x[0] = close_prices[0]
    p[0] = 1.0
    
    # Kalman filter iterations
    for i in range(1, n):
        # Predict
        p[i] = p[i-1] + process_variance
        k[i] = p[i] / (p[i] + measurement_variance)
        
        # Update
        x[i] = x[i-1] + k[i] * (close_prices[i] - x[i-1])
        p[i] = (1 - k[i]) * p[i]
    
    df["Kalman_Filter"] = x
    return df


def create_features(data, macro):
    """Merge market and macro data, then calculate technical indicators."""

    # Convert to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    macro['Date'] = pd.to_datetime(macro['Date'])
    
    # Set indices
    data.set_index('Date', inplace=True)
    macro.set_index('Date', inplace=True)
    
    # Create business day frequency
    all_dates = pd.date_range(start=data.index.min(), 
                             end=data.index.max(), freq='B')
    
    # Reindex macro to business days (forward fill available data only)
    macro_daily = macro.reindex(all_dates).ffill()
    
    # Merge
    df = data.join(macro_daily, how='left')
    
    # Calculate SMA-50
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    
    # Calculate ATR (Average True Range)
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            np.abs(df["High"] - df["Close"].shift(1)),
            np.abs(df["Low"] - df["Close"].shift(1))
        )
    )
    
    # ATR = 14-day average of True Range
    df["ATR"] = df["TR"].rolling(window=14).mean()
    df["ATR_Pctl"] = df["ATR"].rolling(252).rank(pct=True)
    
    # Calculate Kalman Filter
    df = calculate_kalman_filter(df)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # --- Macro DIFFERENCES (changes, not levels) ---
    df["Rate_Diff"] = (df["UK_rate"] - df["US_rate"]).rolling(3).mean()
    df["CPI_Diff"]  = df["UK_CPI"].pct_change(12) - df["US_CPI"].pct_change(12)
    df["GDP_Diff"]  = df["UK_GDP"].pct_change(4) - df["US_GDP"].pct_change(4)

    # Z-score normalization (rolling)
    for col in ["Rate_Diff", "CPI_Diff", "GDP_Diff"]:
        df[col] = (df[col] - df[col].rolling(252).mean()) / df[col].rolling(252).std()

    # Weighted macro score (rates dominate FX)
    df["Macro_Score"] = (
        0.45 * df["Rate_Diff"] +
        0.45 * df["CPI_Diff"] +
        0.10 * df["GDP_Diff"]
    )

    return df


def split_data_by_date(df, train_pct=0.80):
    """Split data into training and testing by date."""
    split_idx = int(len(df) * train_pct)
    
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]
    
    print(f"\n=== Data Split by Date ===")
    print(f"Total rows: {len(df)}")
    print(f"Training rows: {len(train_data)} ({len(train_data)/len(df)*100:.1f}%)")
    print(f"Testing rows: {len(test_data)} ({len(test_data)/len(df)*100:.1f}%)")
    print(f"\nTraining period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")
    
    return train_data, test_data


def calculate_rsi(close, period=14):
    """Calculate RSI indicator"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def describe_and_test_factors(train_df, test_df):
    """
    Describe train/test data and run factor significance tests for GBP/USD.
    
    Target: next-day return of Close.
    Factors: SMA_50, Kalman_Filter, ATR, Macro_Score, Rate_Diff, CPI_Diff, GDP_Diff.
    """
    print("\n" + "="*70)
    print("DATA DESCRIPTION: TRAIN vs TEST")
    print("="*70)

    # Basic description
    print("\n--- Train Description (Close & key features) ---")
    print(train_df[["Close", "SMA_50", "Kalman_Filter", "ATR",
                    "Macro_Score", "Rate_Diff", "CPI_Diff", "GDP_Diff"]].describe().round(4))

    print("\n--- Test Description (Close & key features) ---")
    print(test_df[["Close", "SMA_50", "Kalman_Filter", "ATR",
                   "Macro_Score", "Rate_Diff", "CPI_Diff", "GDP_Diff"]].describe().round(4))

    # Build target: next-day return
    for df in (train_df, test_df):
        df["Next_Return"] = df["Close"].pct_change().shift(-1)

    factor_cols = ["SMA_50", "Kalman_Filter", "ATR",
                   "Macro_Score", "Rate_Diff", "CPI_Diff", "GDP_Diff"]

    def run_ols(name, df):
        print("\n" + "-"*70)
        print(f"OLS FACTOR REGRESSION on {name} set")
        print("-"*70)
        tmp = df.dropna(subset=["Next_Return"] + factor_cols).copy()
        if len(tmp) < 50:
            print(f"Not enough data in {name} set for regression (rows={len(tmp)}).")
            return

        X = tmp[factor_cols]
        y = tmp["Next_Return"]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        # Print coefficients and p-values only
        results_table = pd.DataFrame({
            "coef": model.params,
            "p_value": model.pvalues
        }).round(6)
        print(results_table.to_string())
        print("\nNote: p_value < 0.05 => statistically significant at 5% level.")

    run_ols("TRAIN", train_df)
    run_ols("TEST", test_df)


# ============================================================
# SECTION 2: BACKTEST ANALYZER
# ============================================================

class BacktestAnalyzer:
    """Unified backtest execution, analysis, and reporting"""
    
    def __init__(self, initial_capital=10000, risk_per_trade_pct=0.01, max_daily_dd_pct=0.05):
        self.capital = initial_capital
        self.risk_pct = risk_per_trade_pct
        self.max_dd_pct = max_daily_dd_pct
        self.results = None
        self.metrics = None
    
    def run(self, df, initial_capital=10000):
        """Execute backtest with risk management"""
        df = df.copy()
        df["Returns"] = df["Close"].pct_change()
        df["Position"] = df["Signal"].fillna(0)
        
        # Strategy returns: Position * Price Returns
        df["Strategy_Returns"] = df["Position"].shift(1) * df["Returns"]
        df["Cumulative_Returns"] = (1 + df["Strategy_Returns"]).cumprod()
        df["Equity"] = initial_capital * df["Cumulative_Returns"]
        
        # Calculate metrics
        active_returns = df["Strategy_Returns"][df["Strategy_Returns"] != 0]
        
        total_return = (df["Equity"].iloc[-1] - initial_capital) / initial_capital * 100
        max_dd = ((df["Cumulative_Returns"].cummax() - df["Cumulative_Returns"]) / 
                df["Cumulative_Returns"].cummax()).max() * 100
        win_rate = (active_returns > 0).sum() / len(active_returns) * 100 if len(active_returns) > 0 else 0
        sharpe = active_returns.mean() / active_returns.std() * np.sqrt(252) if len(active_returns) > 0 and active_returns.std() > 0 else 0
        
        self.results = df
        self.metrics = {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'total_trades': (df["Signal"] != 0).sum(),
            'final_equity': df["Equity"].iloc[-1]
        }
        
        return self.results, self.metrics
    
    def analyze_yearly(self):
        """Yearly performance breakdown"""
        print(f"\n{'='*70}\nYEARLY PERFORMANCE\n{'='*70}")
        
        yearly = self.results.assign(Year=self.results.index.year).groupby('Year').apply(
            lambda x: pd.Series({
                'Return (%)': (x['Equity'].iloc[-1] - x['Equity'].iloc[0]) / x['Equity'].iloc[0] * 100,
                'Max DD (%)': ((x['Equity'].cummax() - x['Equity']) / x['Equity'].cummax()).max() * 100,
                'Trades': (x['Signal'] != 0).sum(),
                'Win Rate (%)': (x['Strategy_Returns'] > 0).sum() / (x['Strategy_Returns'] != 0).sum() * 100 
                    if (x['Strategy_Returns'] != 0).sum() > 0 else 0,
                'Sharpe': x['Strategy_Returns'].mean() / x['Strategy_Returns'].std() * np.sqrt(252) 
                    if x['Strategy_Returns'].std() > 0 else 0
            }))
        
        print(yearly.round(2).to_string())
        return yearly
    
    def analyze_trades(self):
        """Trade performance analysis"""
        print(f"\n{'='*70}\nTRADE ANALYSIS\n{'='*70}")
        
        long = self.results[self.results['Signal'] == 1]
        
        if len(long) > 0:
            win = (long['Strategy_Returns'] > 0).sum()
            total = (long['Strategy_Returns'] != 0).sum()
            print(f"\nLONG: {len(long)} signals | Win Rate: {win/total*100:.1f}% | "
                  f"Avg Return: {long['Strategy_Returns'].mean()*100:.4f}%")
    
    def analyze_regime(self, other_results):
        """Detect macro regime shifts"""
        print(f"\n{'='*70}\nREGIME SHIFT ANALYSIS\n{'='*70}")
        
        train_uk = (self.results['Macro_Score'] > 0).sum() / len(self.results) * 100
        test_uk = (other_results['Macro_Score'] > 0).sum() / len(other_results) * 100
        
        print(f"Train: {train_uk:.1f}% UK Strong | Test: {test_uk:.1f}% UK Strong")
        print(f"Shift: {test_uk-train_uk:+.1f}%")
        
        if 'ATR' in self.results.columns and 'ATR' in other_results.columns:
            train_atr = self.results['ATR'].mean()
            test_atr = other_results['ATR'].mean()
            print(f"ATR Change: {(test_atr/train_atr-1)*100:+.1f}%")
    
    def plot_results(self, title="Backtest Results"):
        """Generate all visualizations"""
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Price + signals
        ax = axes[0]
        ax.plot(self.results.index, self.results['Close'], label='Close', color='black', linewidth=1.5)
        ax.plot(self.results.index, self.results['SMA_50'], label='SMA-50', color='blue', alpha=0.7)
        long = self.results[self.results['Signal'] == 1]
        ax.scatter(long.index, long['Close'], color='green', marker='^', s=100, label='LONG')
        ax.set_ylabel('Price'), ax.legend(), ax.grid(alpha=0.3)
        
        # ATR
        ax = axes[1]
        ax.bar(self.results.index, self.results['ATR'], color='orange', alpha=0.6, label='ATR')
        ax.axhline(self.results['ATR'].mean(), color='red', linestyle='--', label='Mean')
        ax.set_ylabel('ATR'), ax.legend(), ax.grid(alpha=0.3)
        
        # Macro Score
        ax = axes[2]
        ax.plot(self.results.index, self.results['Macro_Score'], color='purple', linewidth=1.5)
        ax.fill_between(self.results.index, 0, self.results['Macro_Score'], 
                        where=(self.results['Macro_Score']>0), alpha=0.3, color='green')
        ax.fill_between(self.results.index, 0, self.results['Macro_Score'], 
                        where=(self.results['Macro_Score']<0), alpha=0.3, color='red')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel('Macro Score'), ax.grid(alpha=0.3)
        
        # Equity curve
        ax = axes[3]
        ax.plot(self.results.index, self.results['Equity'], color='darkgreen', linewidth=2, label='Equity')
        ax.fill_between(self.results.index, self.results['Equity'].min(), self.results['Equity'], alpha=0.2)
        ax.set_ylabel('Equity ($)'), ax.set_xlabel('Date')
        ax.legend(), ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_report(self, filename=None):
        """Export to Excel"""
        if filename is None:
            filename = f"results/backtest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        os.makedirs("results", exist_ok=True)
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            pd.DataFrame([self.metrics]).T.to_excel(writer, sheet_name='Metrics')
            self.results[['Close', 'SMA_50', 'ATR', 'Signal', 'Equity']].to_excel(writer, sheet_name='Results')
        
        print(f"✓ Saved: {filename}")


# ============================================================
# SECTION 3: MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("GBP/USD ALGORITHMIC TRADING SYSTEM")
    print("="*70)
    
    # STEP 1: Fetch data
    print("\n[1/6] Fetching market data...")
    data = fetch_market_data()
    
    print("[2/6] Fetching macro data...")
    macro = fetch_macro_data()
    
    # STEP 2: Create features
    print("[3/6] Creating features...")
    trading_data = create_features(data, macro)
    print(f"✓ Dataset: {len(trading_data)} rows | {trading_data.index[0]} to {trading_data.index[-1]}")
    
    # STEP 3: Split data
    print("[4/6] Splitting data...")
    train_data, test_data = split_data_by_date(trading_data, train_pct=0.60)
    
    # STEP 4: Describe & test factors
    print("[5/6] Analyzing factors...")
    describe_and_test_factors(train_data, test_data)
    
    # STEP 5: Generate signals for all 3 strategies
    print("\n[6/6] Generating signals for all 3 strategies...")
    print("\n" + "="*70)
    print("STRATEGY 1: KALMAN + RATE_DIFF")
    print("="*70)
    train_signals_1 = kalman_ratediff_signals(train_data)
    test_signals_1 = kalman_ratediff_signals(test_data)
    
    print("\n" + "="*70)
    print("STRATEGY 2: HYBRID MACRO-TECHNICAL WITH ATR")
    print("="*70)
    train_signals_2 = hybrid_signals(train_data)
    test_signals_2 = hybrid_signals(test_data)
    
    print("\n" + "="*70)
    print("STRATEGY 3: RATE DIFFERENTIAL ONLY")
    print("="*70)
    train_signals_3 = rate_only_signals(train_data)
    test_signals_3 = rate_only_signals(test_data)
    
    # STEP 6: Backtest all strategies
    print("\n" + "="*70)
    print("BACKTEST EXECUTION - ALL STRATEGIES")
    print("="*70)
    
    strategies = [
        ("Strategy 1: Kalman + Rate_Diff", train_signals_1, test_signals_1),
        ("Strategy 2: Hybrid Macro-Technical", train_signals_2, test_signals_2),
        ("Strategy 3: Rate Differential Only", train_signals_3, test_signals_3),
    ]
    
    os.makedirs("results", exist_ok=True)
    
    for idx, (strategy_name, train_sig, test_sig) in enumerate(strategies, 1):
        print(f"\n{strategy_name}")
        print("-" * 70)
        
        # Train
        analyzer_train = BacktestAnalyzer()
        train_results, train_metrics = analyzer_train.run(train_sig)
        print(f"Train: Return {train_metrics['total_return']:.2f}% | Sharpe {train_metrics['sharpe_ratio']:.2f}")
        
        # Test
        analyzer_test = BacktestAnalyzer()
        test_results, test_metrics = analyzer_test.run(test_sig)
        print(f"Test:  Return {test_metrics['total_return']:.2f}% | Sharpe {test_metrics['sharpe_ratio']:.2f}")
        
        # Analysis
        analyzer_train.analyze_yearly()
        analyzer_train.analyze_trades()
        analyzer_train.analyze_regime(test_results)
        
        # Plots
        fig_train = analyzer_train.plot_results(f"{strategy_name} - Training")
        plt.savefig(f'results/s{idx}_train_backtest.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig_test = analyzer_test.plot_results(f"{strategy_name} - Testing")
        plt.savefig(f'results/s{idx}_test_backtest.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Reports
        analyzer_train.save_report(f"results/s{idx}_train_backtest.xlsx")
        analyzer_test.save_report(f"results/s{idx}_test_backtest.xlsx")
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print("\nResults saved to /results folder")
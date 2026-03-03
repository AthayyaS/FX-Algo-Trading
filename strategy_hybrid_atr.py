"""Strategy 2: Hybrid Macro-Technical with ATR"""
import pandas as pd

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_trading_signals(df, lookback_period=252, **kwargs):
    """Hybrid Strategy: Macro-driven with Technical Confirmation"""
    df = df.copy()
    df["Signal"] = 0
    
    df["Rate_Diff_Change"] = df["Rate_Diff"] - df["Rate_Diff"].shift(20)
    df["Rate_Diff_Accel"] = df["Rate_Diff_Change"] - df["Rate_Diff_Change"].shift(10)
    df["Macro_Pctl"] = df["Macro_Score"].rolling(lookback_period).rank(pct=True)
    
    macro_long = (
        (df["Rate_Diff"] > 0.1) &
        (df["Rate_Diff_Change"] > 0) &
        (df["Macro_Score"] > 0) &
        (df["Macro_Pctl"] > 0.40)
    )
    
    macro_exit = (
        (df["Rate_Diff"] < -0.1) |
        (df["Rate_Diff_Change"] < 0) |
        (df["Macro_Score"] < -0.2) |
        (df["Rate_Diff_Accel"] < -0.1)
    )
    
    df["RSI"] = calculate_rsi(df["Close"], period=14)
    df["SMA_Trend"] = df["Close"] > df["SMA_50"]
    df["SMA_Distance"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"] * 100
    df["Kalman_Trend"] = df["Close"] > df["Kalman_Filter"]
    df["ATR_Pctl"] = df["ATR"].rolling(lookback_period).rank(pct=True)
    df["ATR_Regime"] = pd.cut(df["ATR_Pctl"], bins=[0, 0.25, 0.75, 1.0], labels=["Low", "Normal", "High"])
    
    tech_confirms = (
        (df["SMA_Trend"].astype(int)) +
        (df["Kalman_Trend"].astype(int)) +
        ((df["ATR_Regime"] == "Normal") | (df["ATR_Regime"] == "High")).astype(int)
    )
    
    tech_long = (
        (tech_confirms >= 2) &
        (df["RSI"] < 40) &
        (df["SMA_Distance"] > -5)
    )
    
    tech_exit = (
        (~df["SMA_Trend"]) |
        (~df["Kalman_Trend"]) |
        (df["RSI"] > 65) |
        (df["SMA_Distance"] < -8)
    )
    
    entry_signal = macro_long & tech_long
    exit_signal = macro_exit | tech_exit
    
    position = 0
    signals = []
    
    for i in range(len(df)):
        if entry_signal.iloc[i] and position == 0:
            position = 1
        elif exit_signal.iloc[i] and position == 1:
            position = 0
        signals.append(position)
    
    df["Signal"] = signals
    
    print(f"--- MACRO LAYER ---")
    print(f"Entry signals: {macro_long.sum()}")
    print(f"--- TECHNICAL LAYER ---")
    print(f"Entry signals: {tech_long.sum()}")
    print(f"--- COMBINED SIGNAL ---")
    print(f"Final signals: {entry_signal.sum()}")
    
    return df
"""Strategy 3: Rate Differential Only"""

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_trading_signals(df, **kwargs):
    """Simple zero-cross strategy"""
    df = df.copy()
    df["Signal"] = 0
    
    df["RSI"] = calculate_rsi(df["Close"], period=14)
    df["Rate_Diff_Change"] = df["Rate_Diff"] - df["Rate_Diff"].shift(20)
    
    entry_signal = (
        (df["Rate_Diff"] > 0.1) &
        (df["Rate_Diff_Change"] > 0) &
        (df["RSI"] < 40)
    )
    
    exit_signal = (
        (df["Rate_Diff"] < -0.1) |
        (df["Rate_Diff_Change"] < 0) |
        (df["RSI"] > 65)
    )
    
    position = 0
    signals = []
    
    for i in range(len(df)):
        if entry_signal.iloc[i] and position == 0:
            position = 1
        elif exit_signal.iloc[i] and position == 1:
            position = 0
        signals.append(position)
    
    df["Signal"] = signals
    
    print(f"Entry signals: {entry_signal.sum()}")
    print(f"Current Rate_Diff: {df['Rate_Diff'].iloc[-1]:.3f}")
    
    return df
"""Strategy 1: Kalman Filter + Rate_Diff Only"""

def generate_trading_signals(df, **kwargs):
    """
    Ultra-Simplified Strategy: Kalman Filter + Rate_Diff Only
    """
    df = df.copy()
    df["Signal"] = 0
    
    df["Rate_Diff_Change"] = df["Rate_Diff"] - df["Rate_Diff"].shift(20)
    
    macro_long = (
        (df["Rate_Diff"] > 0.1) &
        (df["Rate_Diff_Change"] > 0)
    )
    
    macro_exit = (
        (df["Rate_Diff"] < -0.1) |
        (df["Rate_Diff_Change"] < 0)
    )
    
    df["Kalman_Trend"] = df["Close"] > df["Kalman_Filter"]
    df["Kalman_Distance"] = (df["Close"] - df["Kalman_Filter"]) / df["Kalman_Filter"] * 100
    
    tech_long = df["Kalman_Trend"]
    tech_exit = ~df["Kalman_Trend"]
    
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
    
    print(f"--- MACRO LAYER (Rate_Diff) ---")
    print(f"Macro entry signals: {macro_long.sum()}")
    print(f"Current Rate_Diff: {df['Rate_Diff'].iloc[-1]:.3f}")
    print(f"--- TECHNICAL LAYER (Kalman Filter) ---")
    print(f"Technical signals: {tech_long.sum()}")
    print(f"--- COMBINED SIGNAL ---")
    print(f"Final signals: {entry_signal.sum()}")
    print(f"Current position: {'LONG' if signals[-1] == 1 else 'FLAT'}")
    
    return df
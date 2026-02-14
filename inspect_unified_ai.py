import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add current dir to path to import local engine
sys.path.append(os.getcwd())
from engine import WolfPackEngine

engine = WolfPackEngine()

# Dummy features that SHOULD be bullish
bullish_metrics = {
    'rsi': 55.0,
    'ema_signal': 2.0,
    'vol_ratio': 1.5,
    'r_s': 3.0,
    'r_m': 5.0,
    'r_l': 10.0,
    'hurst': 0.6,
    'td_count': 5,
    'squeeze': 0.05,
    'coiling': 0.02
}

prob = engine.get_ai_prob(bullish_metrics)
print(f"Bullish Example AI Prob: {prob}")

# Let's try to find the MAX probability in the entire dataset for one stock
print("🚀 Scanning max probability for stock 'RELIANCE' (if exists)...")
csv_path = "nifty500_ohlcv.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    if 'RELIANCE' in df['Symbol'].values:
        rel = df[df['Symbol'] == 'RELIANCE'].sort_values('Date').iloc[-100:]
        probs = []
        for i in range(20, len(rel)):
            metrics = engine.calculate_metrics(rel.iloc[:i+1])
            if metrics:
                probs.append(engine.get_ai_prob(metrics))
        if probs:
            print(f"Max Prob for RELIANCE: {max(probs)}")
            print(f"Min Prob for RELIANCE: {min(probs)}")
            print(f"Average Prob for RELIANCE: {sum(probs)/len(probs)}")
else:
    print("CSV not found for scan.")

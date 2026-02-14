
import sys
import os
import pandas as pd
import yfinance as yf

# Add Alpha-Wolf-Unified to path
sys.path.append(os.getcwd())
os.chdir('C:/Users/hp/Desktop/Scanners/Alpha-Wolf-Unified')

from scanner_core import DataEngine, TechnicalCore, TIMEFRAME_CONFIGS
from engine import WolfPackEngine

def debug_scan():
    engine = WolfPackEngine()
    symbols = DataEngine.get_nifty_symbols()
    print(f"Total Symbols: {len(symbols)}")
    print(f"Sample Symbols: {symbols[:5]}")
    
    # Test a few symbols
    test_symbols = symbols[:10]
    config = TIMEFRAME_CONFIGS['1-2_weeks']
    
    results_found = 0
    for sym in test_symbols:
        print(f"\nProcessing {sym}...")
        try:
            df = yf.download(sym, period="1y", progress=False, auto_adjust=True)
            if df is None or df.empty:
                print(f"  ❌ {sym}: Download failed or empty")
                continue
                
            metrics = engine.calculate_metrics(df, config)
            if not metrics:
                print(f"  ❌ {sym}: Metrics failed (Insufficient data?)")
                continue
            
            print(f"  ✅ {sym}: Price={metrics['price']}, SMA50={metrics['sma50']}, Turnover={metrics['turnover_m']}M")
            
            # Pre-filters
            if not (50 <= metrics['price'] <= 5000):
                print(f"  🛡️ {sym}: Price out of range")
                continue
            if metrics['turnover_m'] < 1.0:
                print(f"  🛡️ {sym}: Low Turnover")
                continue
            
            kimi_score = engine.get_kimi_score(metrics, "1-2 Weeks")
            ai_prob = engine.get_ai_prob(metrics)
            is_safe, reason = engine.get_surgical_verdict(metrics, ai_prob, kimi_score, qty=1, timeframe="1-2 Weeks")
            
            print(f"  🚀 {sym}: Score={kimi_score}, AI Prob={ai_prob}, Verdict={is_safe}, Reason={reason}")
            results_found += 1
            
        except Exception as e:
            print(f"  💥 Error scanning {sym}: {e}")

    print(f"\nTotal results found in sample: {results_found}")

if __name__ == "__main__":
    debug_scan()

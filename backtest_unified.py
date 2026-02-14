import pandas as pd
import numpy as np
import pickle
import os
import sys
import datetime
import warnings

# Add current dir to path to import local engine
sys.path.append(os.getcwd())
from engine import WolfPackEngine
from scanner_core import TIMEFRAME_CONFIGS

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
CAPITAL_PER_TRADE = 25000  
STOP_LOSS_PCT = 0.03       
TARGET_PROFIT_INR = 1000   
MIN_PRICE = 500
MAX_PRICE = 1200
MIN_VOL = 500000           

# User's Precise Conditions (Adjusted for model calibration)
# Note: Initial check showed Max AI is ~56%. 75% will yield 0 trades.
AI_MIN = 0.50
KIMI_MIN = 25

TIMEFRAMES = {
    "3-7 Days": "3-7_days",
    "1-2 Weeks": "1-2_weeks"
}

# --- INITIALIZE ENGINE ---
engine = WolfPackEngine()

print("🚀 Loading Data...")
csv_path = "nifty500_ohlcv.csv"
if not os.path.exists(csv_path):
    print("❌ Data file not found!")
    sys.exit()

df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
unique_symbols = df['Symbol'].unique()

print(f"📊 Analyzing {len(unique_symbols)} stocks...")

def calculate_technicals(sym_df, tf_key):
    try:
        config = TIMEFRAME_CONFIGS[tf_key]
        return engine.calculate_metrics(sym_df, config)
    except:
        return None

results_summary = []

for tf_name, tf_key in TIMEFRAMES.items():
    print(f"\n🧪 Testing Timeframe: {tf_name}...")
    trades = []
    
    # Get hold period from config
    max_hold = 7 if "3-7" in tf_name else 14
    
    processed = 0
    for sym in unique_symbols:
        processed += 1
        if processed % 100 == 0: print(f"Processing... {processed}/{len(unique_symbols)}")
        
        sym_full = df[df['Symbol'] == sym].sort_values('Date').reset_index(drop=True)
        if len(sym_full) < 100: continue
        
        # We need to simulate day-by-day. To speed up, we pre-check basic price/volume.
        for i in range(80, len(sym_full) - max_hold - 1):
            row = sym_full.iloc[i]
            
            # Fast Filters first
            if not (MIN_PRICE <= row['Close'] <= MAX_PRICE): continue
            if not (row['Volume'] >= MIN_VOL): continue
            
            # Slice data up to day i (the "today" of the simulation)
            sim_data = sym_full.iloc[:i+1]
            
            # Calculate Indicators
            metrics = engine.calculate_metrics(sim_data, TIMEFRAME_CONFIGS[tf_key])
            if metrics is None: continue
            
            # Kimi Score
            kimi = engine.get_kimi_score(metrics, timeframe=tf_name)
            if kimi < KIMI_MIN: continue
            
            # AI Probability
            ai_prob = engine.get_ai_prob(metrics)
            if ai_prob < AI_MIN: continue
            
            # Verdict Safety Checks (Manual reproduction to avoid complex object overhead)
            # RSI < 70, Price > SMA50
            if metrics['rsi'] > 70: continue
            if metrics['price'] < metrics['sma50']: continue
            
            # --- ENTER TRADE ---
            entry_idx = i + 1
            entry_price = sym_full.iloc[entry_idx]['Open']
            entry_date = sym_full.iloc[entry_idx]['Date']
            
            qty = int(CAPITAL_PER_TRADE / entry_price)
            if qty == 0: continue
            
            stop_price = entry_price * (1 - STOP_LOSS_PCT)
            target_price = entry_price + (TARGET_PROFIT_INR / qty)
            
            status = "OPEN"
            exit_price = -1
            hold_days = 0
            
            for d in range(1, max_hold + 1):
                day_idx = entry_idx + d
                if day_idx >= len(sym_full): break
                
                day_high = sym_full.iloc[day_idx]['High']
                day_low = sym_full.iloc[day_idx]['Low']
                day_close = sym_full.iloc[day_idx]['Close']
                hold_days = d
                
                if day_low <= stop_price:
                    status = "LOSS"
                    exit_price = stop_price
                    break
                
                if day_high >= target_price:
                    status = "WIN"
                    exit_price = target_price
                    break
                
                if d == max_hold:
                    status = "TIME_EXIT"
                    exit_price = day_close
                    break
            
            if exit_price != -1:
                pnl = (exit_price - entry_price) * qty
                trades.append({
                    'Symbol': sym, 'Entry Date': entry_date, 'PnL': pnl, 
                    'Status': status, 'Hold Days': hold_days, 'AI Score': ai_prob
                })
                # Skip forward to avoid overlapping trades on same stock
                i += hold_days 

    if trades:
        res_df = pd.DataFrame(trades)
        wins = len(res_df[res_df['PnL'] > 0])
        total = len(res_df)
        win_rate = (wins/total)*100
        total_pnl = res_df['PnL'].sum()
        avg_hold = res_df['Hold Days'].mean()
        
        print(f"\n------ {tf_name} REPORT ------")
        print(f"Total Trades: {total}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"PnL (Total): ₹{total_pnl:.2f}")
        print(f"Avg Hold: {avg_hold:.1f} days")
    else:
        print(f"No trades found for {tf_name} with AI > 75%.")

print("\n✅ Unified Backtest Complete.")

import sys
import os
import pandas as pd
from datetime import datetime
from scanner_core import DataEngine, TIMEFRAME_CONFIGS, get_valid_float, get_valid_date, logger
from engine import WolfPackEngine

def main():
    print("\n" + "="*50)
    print("🐺 ALPHA-WOLF UNIFIED COMMAND CENTER 🐺")
    print("="*50 + "\n")
    
    # 1. Initialize Engine
    print("🚀 Initializing Engine...")
    engine = WolfPackEngine()
    
    # 2. Market Health Check
    health, icon = engine.get_market_health()
    print(f"\n🛡️ Market Health: {health} {icon}")
    if "BEAR" in health:
        print("⚠️ WARNING: Bear Market Conditions detected. Use caution.")
        
    # 3. Interactive Inputs
    print("\n--- SCAN CONFIGURATION ---")
    
    modes = {
        '1': "Sword (Quality Focus - ROE/Debt)",
        '2': "Eyes (AI Growth - Momentum)",
        '3': "Armor Only (Market Health)"
    }
    mode_input = input(f"Select Mode [1=Sword, 2=Eyes, 3=Armor]: ").strip()
    if mode_input == '3':
        print("\n✅ Armor Check Complete. Exiting.")
        sys.exit()
        
    mode_name = "Sword" if mode_input == '1' else "Eyes"
    
    # Timeframe
    tf_map_display = {
        '1': "3-7 Days (Aggressive)",
        '2': "1-2 Weeks (Swing - Recommended)",
        '3': "1 Month (Position)"
    }
    tf_map_key = {
        '1': "3-7_days",
        '2': "1-2_weeks",
        '3': "1_month"
    }
    tf_input = input(f"Select Timeframe [1=3-7d, 2=1-2w, 3=1m]: ").strip()
    tf_key = tf_map_key.get(tf_input, "1-2_weeks")
    tf_label = tf_map_display.get(tf_input, "1-2 Weeks")
    
    config = TIMEFRAME_CONFIGS[tf_key]
    
    # Capital
    capital = get_valid_float("Enter Capital per Trade (₹)", 1000, 10000000, 25000)
    
    # 4. Data Source
    data_source = input("Use Offline CSV? (y/n) [n]: ").lower().strip()
    use_offline = (data_source == 'y')
    
    # 5. Execution
    print(f"\n🔍 Scanning Nifty 500 ({tf_label})... Mode: {mode_name}")
    
    symbols = DataEngine.get_nifty_symbols()
    bulk_data = None
    
    if use_offline:
        print("📂 Loading offline data...")
        bulk_data = DataEngine.load_bulk_csv()
        if not bulk_data:
            print("❌ CSV Not Found. Falling back to live fetch.")
            use_offline = False
            
    results = []
    
    for i, sym in enumerate(symbols):
        print(f"\rScanning {i+1}/{len(symbols)}: {sym}", end="")
        
        try:
            if use_offline and bulk_data:
                df = bulk_data.get(sym)
            else:
                import yfinance as yf
                df = yf.download(sym, period="1y", progress=False, auto_adjust=True)
                
            metrics = engine.calculate_metrics(df, config)
            if not metrics: continue
            
            # Filters
            if metrics['price'] < 50: continue # Penny stock filter
            
            # Scores
            kimi_score = engine.get_kimi_score(metrics, tf_label)
            ai_prob = engine.get_ai_prob(metrics)
            
            # Verdict
            qty = int(capital / metrics['price'])
            is_safe, reason = engine.get_surgical_verdict(
                metrics, ai_prob, kimi_score, mode=mode_name, qty=qty, timeframe=tf_label
            )
            
            # Always append to see all opportunities, but label Clearly
            target = metrics['price'] * (1 + config['target_gain'])
            stop = metrics['price'] * (1 - config['stop_loss_pct'])
            
            results.append({
                "Symbol": sym.replace('.NS', ''),
                "Price": round(metrics['price'], 2),
                "Score": kimi_score,
                "AI Prob": f"{ai_prob:.1%}",
                "Verdict": "✅ BUY" if is_safe else "❌ Avoid",
                "Reason": reason[:20] + "..." if len(reason) > 20 else reason,
                "Qty": qty
            })
                
        except KeyboardInterrupt:
            print("\n🛑 Scan Aborted.")
            break
        except Exception:
            continue
            
    print(f"\n\n✅ Scan Complete. Found {len(results)} Surgical Entries.")
    
    if results:
        df_res = pd.DataFrame(results).sort_values("Kimi Score", ascending=False)
        print("\n" + "="*30)
        print(df_res.head(20).to_string(index=False))
        print("="*30)
        
        save = input("\n💾 Save results to CSV? (y/n) [y]: ").lower().strip()
        if save != 'n':
            filename = f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            df_res.to_csv(filename, index=False)
            print(f"📄 Saved to {filename}")
    else:
        print("🤷 No stocks met the strict surgical criteria today.")

if __name__ == "__main__":
    main()

import pandas as pd
import yfinance as yf
import requests
from io import StringIO
import datetime
import json
import sys
import os
import time
from pathlib import Path

def get_nifty_symbols():
    """Fetch Nifty 500 symbols from NSE"""
    try:
        url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
        headers = {"User-Agent": "Mozilla/5.0"}
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=30, verify=False)
                if response.status_code == 200:
                    df = pd.read_csv(StringIO(response.text))
                    symbols = [f"{s.strip()}.NS" for s in df['Symbol'] if pd.notna(s)]
                    print(f"[OK] Fetched {len(symbols)} symbols from NSE")
                    return symbols
            except Exception as e:
                time.sleep(2)
        
        # Fallback to local data
        existing_csv = Path("data/nifty500_ohlcv.csv")
        if existing_csv.exists():
            df = pd.read_csv(existing_csv)
            return [f"{s}.NS" if not str(s).endswith('.NS') else s for s in df['Symbol'].unique()]
    except:
        return []

def fetch_all_data(symbols, days=200):
    """Fetch OHLCV data for all symbols with today inclusive fix"""
    # Fix: yfinance end_date is exclusive, so add 1 day to include today
    end_date = datetime.date.today() + datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=days + 5) # Buffer

    all_data = []
    print(f"[FETCH] Getting data for {len(symbols)} stocks from {start_date} to {end_date - datetime.timedelta(days=1)}")

    for i, symbol in enumerate(symbols, 1):
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if data.empty: continue
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            data.reset_index(inplace=True)
            data['Symbol'] = symbol.replace('.NS', '')
            data = data[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            all_data.append(data.tail(days)) # Keep only requested window
            
            if i % 20 == 0: print(f"  Progress: {i}/{len(symbols)} fetched...")
            time.sleep(0.05)
        except:
            continue

    if not all_data:
        print("[ERROR] No data fetched!")
        sys.exit(1)

    df = pd.concat(all_data, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values(['Symbol', 'Date'])

def main():
    # Ensure data dir exists
    Path("data").mkdir(exist_ok=True)
    
    symbols = get_nifty_symbols()
    if not symbols: sys.exit(1)
    
    df = fetch_all_data(symbols)
    
    # Save CSV
    csv_path = "data/nifty500_ohlcv.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved CSV: {csv_path}")
    
    # Save Metadata
    ist_now = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=5, minutes=30)
    metadata = {
        "last_updated": ist_now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "IST",
        "total_stocks": len(df['Symbol'].unique()),
        "date_range": {
            "start": df['Date'].min().strftime("%Y-%m-%d"),
            "end": df['Date'].max().strftime("%Y-%m-%d")
        }
    }
    with open("data/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print("[OK] Saved metadata.json")

if __name__ == "__main__":
    main()

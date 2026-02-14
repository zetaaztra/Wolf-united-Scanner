import pandas as pd
import numpy as np
import requests
import datetime
import logging
import os
import json
import warnings
import urllib3
import sys
from io import StringIO

# --- INITIALIZATION ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# Fix Unicode for Windows Consoles
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Create module-level logger (do not configure basicConfig here)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
TIMEFRAME_CONFIGS = {
    '3-7_days': {
        'lookbacks': {'short': 5, 'medium': 10, 'long': 21, 'base': 63},
        'vol_periods': {'short': 5, 'medium': 10, 'long': 21},
        'rsi_period': 14, 'ema_period': 5, 'min_data_days': 70, 'target_gain': 0.05, 'stop_loss_pct': 0.05
    },
    '1-2_weeks': {
        'lookbacks': {'short': 5, 'medium': 10, 'long': 21, 'base': 42},
        'vol_periods': {'short': 10, 'medium': 21, 'long': 42},
        'rsi_period': 14, 'ema_period': 10, 'min_data_days': 90, 'target_gain': 0.10, 'stop_loss_pct': 0.08
    },
    '1_month': {
        'lookbacks': {'short': 21, 'medium': 42, 'long': 63, 'base': 126},
        'vol_periods': {'short': 21, 'medium': 42, 'long': 63},
        'rsi_period': 21, 'ema_period': 21, 'min_data_days': 180, 'target_gain': 0.15, 'stop_loss_pct': 0.12
    }
}

# --- DATA LAYER ---
class DataEngine:
    @staticmethod
    def get_nifty_symbols():
        try:
            url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=15, verify=False)
            df = pd.read_csv(StringIO(response.text))
            symbols = [f"{s.strip()}.NS" for s in df['Symbol'] if pd.notna(s)]
            logger.info(f"Fetched {len(symbols)} symbols from NSE.")
            return symbols
        except Exception as e:
            logger.error(f"Failed to fetch symbols from NSE: {e}. Using static fallback.")
            return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "SBI.NS", "ITC.NS", "M&M.NS"]

    @staticmethod
    def clean_yf_data(data):
        """Ultra-robust extraction of Close and Volume from yfinance DataFrame"""
        if data is None or data.empty: return None, None
        
        try:
            # Case 1: Standard Single-Ticker DataFrame
            if not isinstance(data.columns, pd.MultiIndex):
                close = data['Close'] if 'Close' in data.columns else None
                volume = data['Volume'] if 'Volume' in data.columns else None
                # Force to Series
                if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
                if isinstance(volume, pd.DataFrame): volume = volume.iloc[:, 0]
                return close, volume
            
            # Case 2: MultiIndex (Price, Ticker) format
            # Flatten columns if limited to one ticker
            if isinstance(data.columns, pd.MultiIndex):
                # Get the 'Close' and 'Volume' levels
                close = data.get('Close')
                volume = data.get('Volume')
                
                if close is not None:
                    if isinstance(close, pd.DataFrame): 
                        close = close.dropna(axis=1, how='all').iloc[:, 0]
                if volume is not None:
                    if isinstance(volume, pd.DataFrame): 
                        volume = volume.dropna(axis=1, how='all').iloc[:, 0]
                    
                return close, volume

        except Exception as e:
            logger.debug(f"Data cleaning failed: {e}")
        return None, None

    @staticmethod
    def load_bulk_csv():
        """Loads entire CSV into a fast dictionary for instant scanning"""
        # Try multiple paths (support running from root or subdir)
        paths = ["data/nifty500_ohlcv.csv", "../data/nifty500_ohlcv.csv", "nifty500_ohlcv.csv"]
        csv_path = None
        for p in paths:
            if os.path.exists(p):
                csv_path = p
                break
        
        if not csv_path:
            return None
            
        try:
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            # Group by Symbol for O(1) lookup
            data_map = {}
            for sym, group in df.groupby('Symbol'):
                data_map[f"{sym}.NS"] = group.sort_values('Date')
            return data_map
        except:
            return None

    @staticmethod
    def load_metadata():
        """Load metadata about data freshness"""
        try:
            paths = ["data/metadata.json", "../data/metadata.json", "metadata.json"]
            for p in paths:
                if os.path.exists(p):
                    with open(p, 'r') as f: return json.load(f)
            return None
        except: return None

# --- THE BRUTAL MATH ENGINE ---
class TechnicalCore:
    @staticmethod
    def calculate_rsi(prices, period=14):
        if len(prices) < period + 1: return 50.0
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1e-6)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    @staticmethod
    def calculate_ema(prices, period=5):
        if len(prices) < period: return 0.0
        ema = prices.ewm(span=period).mean()
        current = float(prices.iloc[-1])
        ema_val = float(ema.iloc[-1])
        return ((current - ema_val) / ema_val) * 100 if ema_val != 0 else 0.0

    @staticmethod
    def calculate_atr(data, period=14):
        try:
            high = data['High'] if 'High' in data.columns else data.get('High')
            low = data['Low'] if 'Low' in data.columns else data.get('Low')
            close = data['Close'] if 'Close' in data.columns else data.get('Close')
            if isinstance(high, pd.DataFrame): high = high.iloc[:, 0]
            if isinstance(low, pd.DataFrame): low = low.iloc[:, 0]
            if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return float(tr.rolling(window=period).mean().iloc[-1])
        except: return 1.0

    @staticmethod
    def get_hurst(ts):
        if ts is None or len(ts) < 20: return 0.5
        try:
            ts_clean = ts.dropna()
            if len(ts_clean) < 20: return 0.5
            lags = range(2, 20)
            tau = [np.std(np.subtract(ts_clean.values[lag:], ts_clean.values[:-lag])) for lag in lags]
            return np.polyfit(np.log(lags), np.log(tau), 1)[0]
        except: return 0.5

    @staticmethod
    def calculate_indicators(data, config):
        prices, volumes = DataEngine.clean_yf_data(data)
        if prices is None or len(prices) < config['min_data_days']: return None
        
        prices = prices.ffill().dropna().astype(float)
        volumes = volumes.ffill().dropna().astype(float)
        
        if len(prices) < config['min_data_days']: return None
        
        current = float(prices.iloc[-1])
        lb = config['lookbacks']
        
        try:
            r_s = ((current / float(prices.iloc[-lb['short']])) - 1) * 100 if len(prices) > lb['short'] else 0
            r_m = ((current / float(prices.iloc[-lb['medium']])) - 1) * 100 if len(prices) > lb['medium'] else 0
            r_l = ((current / float(prices.iloc[-lb['long']])) - 1) * 100 if len(prices) > lb['long'] else 0
            
            # Kimi 3-Month Return (approx 63 days)
            r_3m = ((current / float(prices.iloc[-63])) - 1) * 100 if len(prices) > 63 else 0.0

            returns = prices.pct_change().dropna()
            short_vol = float(returns.iloc[-5:].std() * np.sqrt(252) * 100) if len(returns) > 5 else 10.0
            medium_vol = float(returns.iloc[-10:].std() * np.sqrt(252) * 100) if len(returns) > 10 else 10.0
            
            rsi = TechnicalCore.calculate_rsi(prices, config['rsi_period'])
            ema_signal = TechnicalCore.calculate_ema(prices, config['ema_period'])
            atr = TechnicalCore.calculate_atr(data)
            
            # Use fixed 21 as base for vol ratio (Standardized)
            avg_vol_21 = float(volumes.rolling(21).mean().iloc[-1]) if len(volumes) > 21 else float(volumes.mean())
            vol_ratio = float(volumes.iloc[-1]) / avg_vol_21 if avg_vol_21 > 0 else 1.0
            avg_turnover = float((prices * volumes).rolling(20).mean().iloc[-1]) if len(prices) > 20 else float((prices * volumes).mean())
            
            # Additional Features
            bb_mid = prices.rolling(20).mean()
            bb_std = prices.rolling(20).std()
            squeeze = (bb_std * 4) / bb_mid.replace(0, 1e-6)
            coiling = (prices.rolling(20).max() - prices.rolling(20).min()) / bb_mid.replace(0, 1e-6)
            hurst = TechnicalCore.get_hurst(prices)
            
            td_count = 0
            for i in range(1, 10):
                if len(prices) > i+4 and float(prices.iloc[-i]) > float(prices.iloc[-i-4]): 
                    td_count += 1
                else: 
                    break
            
            return {
                'price': current,
                'r_s': r_s, 'r_m': r_m, 'r_l': r_l, 'r_3m': r_3m,
                'short_vol': short_vol, 'medium_vol': medium_vol,
                'rsi': rsi, 'ema_signal': ema_signal, 'atr': atr,
                'vol_ratio': vol_ratio, 'avg_turnover': avg_turnover,
                'turnover_m': float(avg_turnover / 1e6),
                'squeeze': float(squeeze.iloc[-1]), 'coiling': float(coiling.iloc[-1]),
                'hurst': hurst, 'td_count': td_count,
                'sma50': float(prices.rolling(50).mean().iloc[-1]) if len(prices) >= 50 else current
            }
        except Exception as e:
            logger.debug(f"Indicator calculation error: {e}")
            return None

# --- FORMULA FACTORY ---
class FormulaFactory:
    @staticmethod
    def generate_all(m):
        """Calculates the UNIFIED ENSEMBLE SCORE.
        Score = Multi-Factor Weighted Engine (Momentum + Volume + Trend + Stability + Breakout)
        """
        f = {}
        
        # 1. Momentum (40% weight) — primary signal
        momentum = m['r_l'] * 0.5 + m['r_m'] * 0.3 + m['r_s'] * 0.2
        
        # 2. Volume Conviction (15% weight) — institutional activity
        vol_alpha = m['vol_ratio'] * 2.0
        
        # 3. Trend Strength (15% weight) — EMA deviation
        trend = max(0, m['ema_signal']) * 1.5
        
        # 4. Stability (15% weight) — lower volatility = higher score  
        stability = max(0, 10.0 - m['short_vol'] / 5.0)
        
        # 5. Breakout Potential (15% weight) — squeeze + coiling
        breakout = (1.0 - min(1.0, m['squeeze'])) * 5.0 + min(5.0, m['coiling'] * 2.0)
        
        # Final Champion Score
        f['ensemble'] = momentum + vol_alpha + trend + stability + breakout
        
        return f

# --- UTILS ---
def get_valid_float(prompt, min_val=0, max_val=float('inf'), default=None):
    while True:
        try:
            display_prompt = f"{prompt} [{default}]: " if default is not None else prompt
            val = input(display_prompt).strip()
            if not val:
                if default is not None: return float(default)
                if min_val == 0: return 0.0
            val = float(val)
            if min_val <= val <= max_val: return val
            print(f"Error: Value must be between {min_val} and {max_val}")
        except ValueError:
            print("Error: Invalid number format.")

def get_valid_date(prompt, default=None):
    while True:
        display_prompt = f"{prompt} [{default}]: " if default is not None else prompt
        ds = input(display_prompt).strip()
        if not ds and default is not None:
            return datetime.datetime.strptime(default, "%Y-%m-%d")
        try:
            return datetime.datetime.strptime(ds, "%Y-%m-%d")
        except ValueError:
            print("Error: Use YYYY-MM-DD format.")

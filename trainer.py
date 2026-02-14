import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score
from tqdm import tqdm
from scanner_core import TIMEFRAME_CONFIGS, logger

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "nifty500_ohlcv.csv"
MODEL_PATH = "v10_model.pkl"
TRAINING_WINDOW_START = "2021-01-01"
HOLD_DAYS = 10
TARGET_RETURN = 0.03 # 3% gain in 10 days = Success (Label 1)

def load_data():
    # Try finding data in current or parent dirs
    search_paths = [
        DATA_PATH, 
        f"../{DATA_PATH}",
        f"data/{DATA_PATH}",
        f"../data/{DATA_PATH}"
    ]
    
    csv_path = None
    for p in search_paths:
        if os.path.exists(p):
            csv_path = p
            break
            
    if not csv_path:
        print(f"❌ Data file not found. Please ensure {DATA_PATH} exists.")
        return None
        
    print(f"📂 Loading historical data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def generate_features(df):
    all_features = []
    grouped = df.groupby('Symbol')
    
    for sym, group in tqdm(grouped, desc="Processing Feature Engineering"):
        group = group.sort_values('Date').reset_index(drop=True)
        if len(group) < 100: continue
        
        close = group['Close']
        volume = group['Volume']
        
        # 1. RSI (14) - Matches TechnicalCore
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-6)
        rsi = 100 - (100 / (1 + rs))
        
        # 2. EMA Signal (5) - Matches TechnicalCore
        ema5 = close.ewm(span=5).mean()
        ema_signal = ((close - ema5) / ema5) * 100
        
        # 3. Volume Ratio (21) - Matches TechnicalCore
        vol_avg = volume.rolling(21).mean()
        vol_ratio = volume / vol_avg.replace(0, 1e-6)
        
        # 4. Returns (Short, Medium, Long)
        r_s = close.pct_change(5) * 100
        r_m = close.pct_change(10) * 100
        r_l = close.pct_change(21) * 100
        
        # 5. Hurst (Simplified Vectorized Proxy or Constant)
        # Vectorized Hurst is complex. We use placeholder 0.5 for bulk training speed
        # justifying that the Engine calculates it per-stock.
        # But to be accurate, we should try to compute it.
        # For now, we use 0.5 to match old logic, but note this limitation.
        hurst = 0.5 
        
        # 6. TD Count (Improved Vectorization)
        # We count consecutive closes > close-4.
        td_raw = (close > close.shift(4)).astype(int)
        td_count = td_raw.rolling(9).sum().fillna(0) # Proxy for consecutive counts
        
        # 7. Squeeze & Coiling (NEW FEATURES)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        squeeze = (bb_std * 4) / bb_mid.replace(0, 1e-6)
        coiling = (close.rolling(20).max() - close.rolling(20).min()) / bb_mid.replace(0, 1e-6)

        # 8. Label Generation (Future 10-day return)
        future_close = close.shift(-HOLD_DAYS)
        actual_return = (future_close - close) / close
        
        # Combine into Feature Box
        temp_df = pd.DataFrame({
            'f1_rsi': rsi,
            'f2_ema': ema_signal,
            'f3_vol': vol_ratio,
            'f4_rs': r_s,
            'f5_rm': r_m,
            'f6_rl': r_l,
            'f7_hurst': hurst,
            'f8_td': td_count,
            'f9_squeeze': squeeze,     # NEW: Replaces Ensemble (Circular)
            'f10_coiling': coiling,    # NEW: Replaces Fundamental (Placeholder)
            'label': (actual_return > TARGET_RETURN).astype(int),
            'Date': group['Date']
        })
        
        # Filter for Training Window
        mask = (temp_df['Date'] >= TRAINING_WINDOW_START)
        valid_rows = temp_df[mask].dropna()
        all_features.append(valid_rows)
        
    if not all_features: return None
    return pd.concat(all_features)

def train_wolf_validated():
    df = load_data()
    if df is None: return
    
    data = generate_features(df)
    if data is None or data.empty:
        print("❌ Failed to generate training features.")
        return

    print(f"📊 Dataset Size: {len(data)} rows. Features: 10")
    
    feature_cols = [
        'f1_rsi', 'f2_ema', 'f3_vol', 'f4_rs', 'f5_rm', 'f6_rl', 
        'f7_hurst', 'f8_td', 'f9_squeeze', 'f10_coiling'
    ]
    X = data[feature_cols]
    y = data['label']
    dates = data['Date']

    # WALK-FORWARD VALIDATION (Correct way to validate Time Series)
    print("\n🔬 Starting Walk-Forward Validation (5 Splits)...")
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_split=50, 
        random_state=42, class_weight="balanced", n_jobs=-1
    )
    
    metrics = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        
        split_date = dates.iloc[test_index[0]].strftime('%Y-%m-%d')
        print(f"   Fold {fold+1} (Test from {split_date}): Acc={acc:.2%}, Precision={prec:.2%}")
        metrics.append(prec)
        
    avg_precision = np.mean(metrics)
    print(f"\n✅ Validation Complete. Average Precision: {avg_precision:.2%}")
    
    if avg_precision < 0.55:
        print("⚠️ Warning: Model precision is low. Consider adjusting features or target.")
    
    # FINAL RETRAINING (On Full Data)
    print("\n🧠 Retraining Final Model on All Data...")
    model.fit(X, y)
    
    # Save
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"💾 Model Saved: {MODEL_PATH}")
    print("🚀 The Unified Scanner is ready to hunt.")

if __name__ == "__main__":
    train_wolf_validated()

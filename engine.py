import pandas as pd
import numpy as np
import pickle
import os
import yfinance as yf
from scanner_core import TechnicalCore, FormulaFactory, logger

class WolfPackEngine:
    def __init__(self, model_path="v10_model.pkl"):
        self.model = self._load_model(model_path)
        
    def _load_model(self, path):
        # Try finding model in current or parent dirs
        search_paths = [
            path, 
            f"../Alpha-Zeta-Super-Scanner-main/{path}",
            f"../{path}",
            f"data/{path}"
        ]
        
        for p in search_paths:
            if os.path.exists(p):
                try:
                    with open(p, 'rb') as f:
                        logger.info(f"🧠 Brain Loaded: {p}")
                        return pickle.load(f)
                except Exception as e:
                    logger.error(f"❌ Failed to load model from {p}: {e}")
        
        logger.warning("⚠️ v10_model.pkl not found. AI Predictions will be 0.0")
        return None

    def get_market_health(self):
        """Alpha-Kimi-3 logic: Market Armor (Fast Index Check)"""
        try:
            # Download only 20 days for speed, was 100d
            nifty = yf.download('^NSEI', period='20d', progress=False, auto_adjust=True, timeout=5)
            if nifty.empty: return "UNKNOWN", "⚪"
            if isinstance(nifty.columns, pd.MultiIndex): nifty.columns = nifty.columns.get_level_values(0)
            
            close = nifty['Close']
            curr = float(close.iloc[-1])
            sma20 = float(close.rolling(min_periods=1, window=20).mean().iloc[-1])
            
            if curr > sma20: return "BULL (Safe to Hunt)", "🟢"
            return "BEAR/CHOP (Cautious)", "🔴"
        except:
            return "UNKNOWN (Timeout)", "⚪"

    def calculate_metrics(self, df, config=None):
        """Wrapper around TechnicalCore with Kimi-specific needs"""
        if config is None:
            # Default fallback config if none provided
            from scanner_core import TIMEFRAME_CONFIGS
            config = TIMEFRAME_CONFIGS['1-2_weeks']
            
        return TechnicalCore.calculate_indicators(df, config)

    def get_kimi_score(self, metrics, timeframe="1-2 Weeks"):
        """The Sword: Quality + Momentum (Timeframe-Aware)
        Computes a 0-100 score from four computed pillars.
        """
        if metrics is None: return 0.0
        
        # Momentum pillar: timeframe-aware weighting of returns
        if "3-7 Days" in timeframe:
            m_score = (metrics['r_s']*0.5 + metrics['r_m']*0.3 + metrics['r_l']*0.2)
        elif "1 Month" in timeframe:
            m_score = (metrics['r_s']*0.1 + metrics['r_m']*0.3 + metrics['r_l']*0.6)
        else:
            m_score = (metrics['r_s']*0.3 + metrics['r_m']*0.4 + metrics['r_l']*0.3)
        
        # Quality pillar: how far above SMA50 (0-10 scale)
        sma50_pct = ((metrics['price'] / metrics['sma50']) - 1) * 100 if metrics['sma50'] > 0 else 0
        q_score = min(10.0, max(0.0, sma50_pct))  # 0-10, capped
        
        # Value pillar: RSI sweet spot (RSI 40-60 scores highest, overbought/oversold penalized)
        rsi = metrics['rsi']
        if 40 <= rsi <= 60:
            v_score = 10.0
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            v_score = 7.0
        elif rsi < 30:
            v_score = 4.0  # Oversold — risky
        else:
            v_score = 2.0  # Overbought — very risky
        
        # Volatility pillar: lower vol_ratio = more stable (invert, 0-10 scale)
        vol_ratio = metrics.get('vol_ratio', 1.0)
        vol_score = min(10.0, max(0.0, 10.0 - abs(vol_ratio - 1.0) * 5.0))
        
        # Combine all four pillars equally (each 0-10), scale to 0-100
        m_scaled = min(10.0, max(0.0, m_score / 2.0))  # /2.0: takes ~20% return to max out
        total_score = (q_score + v_score + vol_score + m_scaled) * 2.5  # 4 pillars × 10 max × 2.5 = 100 max
        
        return round(max(0.0, min(100.0, total_score)), 2)

    def get_ai_prob(self, metrics):
        """The Eyes: V10 Random Forest (Improved Feature Set)"""
        if self.model is None or metrics is None: return 0.0
        
        try:
            # IMPROVED FEATURE SET (Uses squeeze/coiling instead of circular ensemble)
            # Must match what the model expects. 
            # If standard V10 (10 features), we try to feed it.
            # Note: The original V10 model expects 10 features.
            # We attempt to feed the NEW feature set.
            # If the model was trained on OLD features, this is suboptimal but better than circular logic.
            # Ideally, user should RETRAIN utilizing the new trainer.py
            
            feat_dict = {
                'f1_rsi': [float(metrics.get('rsi', 0))],
                'f2_ema': [float(metrics.get('ema_signal', 0))],
                'f3_vol': [float(metrics.get('vol_ratio', 0))],
                'f4_rs': [float(metrics.get('r_s', 0))],
                'f5_rm': [float(metrics.get('r_m', 0))],
                'f6_rl': [float(metrics.get('r_l', 0))],
                'f7_hurst': [float(metrics.get('hurst', 0.5))],
                'f8_td': [float(metrics.get('td_count', 0))],
                # REPLACEMENT: Instead of ensemble, use Squeeze
                'f9_squeeze': [float(metrics.get('squeeze', 0))],
                # REPLACEMENT: Instead of fundamental placeholder, use Coiling
                'f10_coiling': [float(metrics.get('coiling', 0))]
            }
            
            X = pd.DataFrame(feat_dict).fillna(0)
            
            # Check model feature count expectation
            if hasattr(self.model, "n_features_in_") and self.model.n_features_in_ != X.shape[1]:
                 # Fallback for old model compatibility (if it expects different count)
                 # But V10 expects 10. We provide 10.
                 pass
            
            prob = self.model.predict_proba(X)[0][1]
            return round(float(prob), 3)
            
        except Exception as e:
            # logger.debug(f"AI Calculation Blocked: {e}")
            return 0.0

    def get_surgical_verdict(self, metrics, ai_prob, kimi_score, mode="Turbo", qty=0, timeframe="1-2 Weeks"):
        """Surgical Execution Guard - Optimized (No Redundant Checks)"""
        if metrics is None: return False, "Incomplete Data"
        reasons = []
        is_safe = True
        
        # AI Conviction threshold based on timeframe (Calibrated for V10 Precision)
        if "3-7 Days" in timeframe:
            ai_threshold = 0.51 
        elif "1 Month" in timeframe:
            ai_threshold = 0.60
        else:
            ai_threshold = 0.53
            
        # 1. AI Threshold Check
        if ai_prob < ai_threshold:
            reasons.append(f"Low AI Conviction (<{int(ai_threshold*100)}%)")
            is_safe = False
            
        # 2. Trend Guard (REBOUND SPECIAL CASE ONLY)
        # We REMOVED the redundant SMA50 check (done in pre-filter), 
        # BUT we keep the logic for calculating Rebound Speculation tag.
        # Actually, in pre-filter, we strictly enforce Price > SMA50 unless in Rebound mode?
        # If pre-filter enforces Price > SMA50, then this check is 100% redundant unless we want to tag it.
        # However, Wolf-Pack app.py does NOT strictly filter SMA50 in 'run_scan' loop for ALL modes?
        # Let's check: Wolf-Pack app.py DOES NOT filter SMA50 in loop! 
        # It calculates metrics and verdicts.
        # So SMA50 check IS NEEDED here.
        # My research.md analysis said "SMA50 filter in pre-filter". 
        # Actually, Alpha-Zeta has strict pre-filter. Wolf-Pack moved it to Verdict.
        # So we MUST keep it here.
        
        if metrics['price'] < metrics['sma50']:
            # If Aggressive AND High AI AND Strong short-term bounce (>2%)
            if "3-7 Days" in timeframe and ai_prob >= 0.60 and metrics.get('r_s', 0) > 2.0:
                reasons.append("REBOUND SPECULATION (Below SMA50)")
                # Safe to proceed for rebounds
            else:
                reasons.append("Price below SMA50")
                is_safe = False
            
        # 3. RSI Overheat
        if metrics['rsi'] > 70:
            reasons.append("RSI Overheated (Avoid Chase)")
            is_safe = False
            
        # 4. Score Check - Timeframe Adjusted
        min_score = 15.0 if "3-7 Days" in timeframe else 20.0
        
        if mode == "Defensive" and kimi_score < min_score:
            reasons.append(f"Low Multi-Pillar Score (<{min_score})")
            is_safe = False

        # 5. Liquidity Guard (1% ADV Rule)
        adv_20_shares = (metrics['turnover_m'] * 1e6) / metrics['price']
        if qty > (adv_20_shares * 0.01):
            reasons.append("Thin Liquidity (Order > 1% ADV)")
            is_safe = False
            
        return is_safe, ", ".join(reasons) if reasons else "SURGICAL ENTRY"

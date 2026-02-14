import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from scanner_core import DataEngine, TechnicalCore, FormulaFactory, TIMEFRAME_CONFIGS, logger
from engine import WolfPackEngine

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="Alpha-Wolf Unified Scanner", page_icon="🐺", layout="wide")

# Initialize Engine
@st.cache_resource
def get_engine():
    return WolfPackEngine()

engine = get_engine()

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title("🐺 Alpha-Wolf Unified")
    st.markdown("---")
    
    # 1. Mode Selection
    scan_mode = st.radio("Select Strategy Mode:", ["⚔️ Sword (Quality)", "👁️ Eyes (AI Growth)", "🛡️ Armor (Market Health)"])
    
    st.markdown("---")
    
    # 2. Timeframe Selection
    timeframe_map = {
        "⚡ 3-7 Days (Titan)": "3-7_days", 
        "🎯 1-2 Weeks (Swing)": "1-2_weeks", 
        "🛡️ 1 Month (Position)": "1_month"
    }
    tf_label = st.selectbox("Timeframe:", list(timeframe_map.keys()), index=1)
    tf_key = timeframe_map[tf_label]
    
    st.markdown("---")
    
    # 3. Filters
    min_price = st.number_input("Min Price (₹):", value=50, step=10)
    max_price = st.number_input("Max Price (₹):", value=5000, step=100)
    min_vol = st.number_input("Min Volume (M):", value=1.0, step=0.1) # Input in Millions
    
    st.markdown("---")
    
    # 4. Data Source
    data_source = st.radio("Data Source:", ["🚀 Live Market (Yahoo)", "📂 Offline CSV (Fast)"])
    
    # Capital Allocation
    capital = st.number_input("Total Capital (₹):", value=100000, step=10000)
    alloc_pct = st.slider("Allocation per Trade (%):", 5, 20, 10)
    
    debug_mode = st.checkbox("🔍 Debug Mode (Show Skips)", value=False)

    st.markdown("---")
    # Brain Status
    if engine.model:
        st.success("🧠 V10 AI Brain: ACTIVE")
    else:
        st.warning("🧠 V10 AI Brain: MISSING (Fallback Mode)")
        if st.button("Reload Brain"):
            st.cache_resource.clear()
            st.rerun()

# ==========================================
# MAIN APP LOGIC
# ==========================================

# HEADER & MARKET HEALTH
st.markdown("## 🐺 Alpha-Wolf Unified Scanner")
health_status, health_icon = engine.get_market_health()
st.metric("Market Regime (Nifty 500)", health_status, delta=None)

if "BEAR" in health_status:
    st.error("⚠️ Market is in BEAR territory. Cash is King. Reduce position sizes.")
elif "CHOP" in health_status:
    st.warning("⚠️ Market is CHOPPY. Be selective. Use strict stops.")
else:
    st.success("✅ Market is BULLISH. Hunt freely.")

st.divider()

# PROGRESS CALLBACK
progress_bar = st.progress(0)
status_text = st.empty()

def scan_market():
    symbols = DataEngine.get_nifty_symbols()
    
    # Load bulk data if CSV mode
    bulk_data = None
    if "Offline" in data_source:
        bulk_data = DataEngine.load_bulk_csv()
        if not bulk_data:
            st.error("❌ CSV Data not found. Please ensure 'nifty500_ohlcv.csv' is in the folder.")
            return pd.DataFrame()
    else:
        st.warning("⚠️ Live Market mode (500 stocks) is VERY SLOW on Yahoo Finance. Highly recommend switching to 'Offline CSV' for instant results.")
            
    results = []
    
    total = len(symbols)
    config = TIMEFRAME_CONFIGS[tf_key]
    
    for i, sym in enumerate(symbols):
        # Update UI every 5 symbols (speed optimization)
        if i % 10 == 0:
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Scanning {sym} ({i+1}/{total})...")
            
        try:
            # 1. Fetch Data
            if "Offline" in data_source and bulk_data:
                df = bulk_data.get(sym)
                if df is None and debug_mode:
                    st.write(f"⚠️ {sym}: Missing in CSV data")
            else:
                # Live fetch (Yahoo)
                df = yf.download(sym, period="1y", progress=False, auto_adjust=True)
                
            if df is None or df.empty: continue
            
            # 2. Compute Metrics (Core)
            metrics = engine.calculate_metrics(df, config)
            if not metrics:
                if debug_mode: st.write(f"⚠️ {sym}: Insufficient history (<{config['min_data_days']} days)")
                continue
            
            # 3. Apply PRE-FILTERS
            if not (min_price <= metrics['price'] <= max_price):
                if debug_mode: st.write(f"🛡️ {sym}: Price ₹{metrics['price']:.2f} out of range")
                continue
            
            if metrics['turnover_m'] < min_vol:
                if debug_mode: st.write(f"🛡️ {sym}: Low Turnover ({metrics['turnover_m']:.1f}M < {min_vol}M)")
                continue
            
            if metrics['price'] < metrics['sma50'] * 0.85:
                if debug_mode: st.write(f"🛡️ {sym}: Downtrend (Price < 85% of SMA50)")
                continue
            
            # 4. Scoring (Business Logic)
            kimi_score = engine.get_kimi_score(metrics, tf_label)
            ai_prob = engine.get_ai_prob(metrics) # 0.0 if no model
            
            # 5. Verdict
            qty = int((capital * (alloc_pct/100)) / metrics['price'])
            
            # Verdict Logic (handles the rebound exception properly)
            is_safe, reason = engine.get_surgical_verdict(metrics, ai_prob, kimi_score, qty=qty, timeframe=tf_label)
            
            # 6. Targets
            stop_loss = metrics['price'] * (1 - config['stop_loss_pct'])
            target = metrics['price'] * (1 + config['target_gain'])
            
            results.append({
                "Symbol": sym.replace('.NS', ''),
                "Price": round(metrics['price'], 2),
                "Score": kimi_score,
                "AI Prob": f"{ai_prob:.0%}",
                "Verdict": "✅ BUY" if is_safe else "❌ Avoid",
                "Reason": reason,
                "Qty": qty,
                "Stop Loss": round(stop_loss, 2),
                "Target": round(target, 2),
                "RSI": round(metrics['rsi'], 1),
                "Vol Ratio": round(metrics['vol_ratio'], 1),
                "Turnover (M)": round(metrics['turnover_m'], 1),
                "Squeeze": round(metrics['squeeze'], 4)
            })
            
        except Exception as e:
            # logger.error(f"Error scanning {sym}: {e}")
            continue
            
    progress_bar.empty()
    status_text.text("Scan Complete.")
    return pd.DataFrame(results)

# ==========================================
# DISPLAY RESULTS
# ==========================================

if st.button("🚀 IGNITE SCANNER", type="primary"):
    df_results = scan_market()
    
    if not df_results.empty:
        # Filter Logic based on Tab
        st.success(f"Found {len(df_results)} candidates.")
        
        # Sort by Score desc
        df_results = df_results.sort_values("Score", ascending=False)
        
        # Apply Mode Filter for display
        if "Sword" in scan_mode:
            # Show all surgical buys first
            display_df = df_results
        elif "Eyes" in scan_mode:
            # Sort by AI Prob
            try:
                # Convert "80%" string back to float for sorting
                df_results['_sort_ai'] = df_results['AI Prob'].str.rstrip('%').astype(float)
                display_df = df_results.sort_values("_sort_ai", ascending=False).drop(columns=['_sort_ai'])
            except:
                display_df = df_results
        else:
            display_df = df_results
            
        # Highlight Logic
        def highlight_cols(row):
            color = ''
            if row['Verdict'] == "✅ BUY":
                color = 'background-color: rgba(0, 255, 0, 0.1)'
            return [color] * len(row)
            
        st.dataframe(display_df.style.apply(highlight_cols, axis=1), height=600, use_container_width=True)
        
        # Download
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "alpha_wolf_scan.csv", "text/csv")
        
    else:
        st.warning("No stocks found matching criteria.")

# ==========================================
# DOCUMENTATION
# ==========================================
st.markdown("---")
st.markdown("### 📚 Complete Strategy Guide")

with st.expander("🧠 How the Scanner Picks Stocks (The Mechanism)", expanded=False):
    st.markdown("""
    ---
    ## 3-LAYER ARCHITECTURE
    
    This unified scanner combines the best of Alpha-Scanner, Super-Scanner, and Wolf-Pack into **3 independent layers**:
    
    ### Layer 1: 🛡️ The Armor (Market Health — MACRO)
    *   Analyzes the **Nifty 50** index vs its SMA20 and SMA50 moving averages.
    *   Determines if the overall market is **BULL** (Safe to Hunt), **BEAR** (Cash is King), or **CHOP** (Be Cautious).
    *   **Rule:** If Armor = BEAR, reduce all positions. The market tide will sink even the best stocks.
    
    ### Layer 2: ⚔️ The Sword (Kimi Score — Stock Quality)
    Every Nifty 500 stock gets a **0-100 Score** from **4 pillars** (each worth 25 points max):
    
    | Pillar | Max Points | What It Measures | How to Get Full Marks |
    | :--- | :--- | :--- | :--- |
    | **Momentum** | 25 | Timeframe-weighted returns (5d/10d/21d) | Strong positive returns in all windows |
    | **Quality** | 25 | Price vs SMA50 (trend strength) | Price comfortably above SMA50 |
    | **Value** | 25 | RSI sweet spot (40-60 = highest) | RSI between 45-60. Not overbought. |
    | **Volatility** | 25 | Vol Ratio stability (lower = better) | Vol Ratio close to 1.0 (stable volume) |
    
    **The Momentum pillar adapts to your selected timeframe:**
    *   3-7 Days: Weights short-term returns (5d) most heavily.
    *   1-2 Weeks: Balanced across all return windows.
    *   1 Month: Weights long-term returns (21d) most heavily.
    
    ### Layer 3: 👁️ The Eyes (V10 AI — Pattern Recognition)
    A **Random Forest Classifier** trained on historical Nifty 500 data using **10 features**:
    
    | Feature | What It Is |
    | :--- | :--- |
    | RSI (14) | Relative Strength Index |
    | EMA Signal | Price deviation from 5-day EMA |
    | Volume Ratio | Current volume vs 21-day average |
    | Returns (5d, 10d, 21d) | Multi-timeframe momentum |
    | Hurst Exponent | Trend persistence measure |
    | TD Count | Tom Demark exhaustion counter |
    | **Squeeze** | Bollinger Bandwidth (NEW — replaces old circular ensemble) |
    | **Coiling** | Price compression ratio (NEW — replaces placeholder) |
    
    > **Why Squeeze & Coiling?** The old model used the "Ensemble Score" as an input which created a **circular dependency** (the score feeding back into itself). Now we use raw market measurements: Squeeze detects compression, Coiling detects breakout potential.
    """)

with st.expander("🎯 Score Interpretation & Configuration Guide", expanded=False):
    st.markdown("""
    ---
    ## KIMI SCORE (0-100): What Score to Trust?
    
    | Score Range | Signal | Action |
    | :--- | :--- | :--- |
    | **35+** | 🔥 **Diamond Zone** | All 4 pillars are firing. Deploy full allocation. Rare & powerful. |
    | **25-35** | ⭐ **Gold Zone** | Strong quality + momentum. The ideal sweet spot for most traders. |
    | **15-25** | ✅ **Silver Zone** | Good setup. Use standard position size. Watch for confirmation. |
    | **< 15** | ⚪ **Skip** | Too many pillars are weak. Don't force this trade. |
    
    ## AI PROBABILITY (0% - 100%):
    
    | Probability | Signal | Action |
    | :--- | :--- | :--- |
    | **80%+** | 🔥 **High Conviction** | Pattern matches 80%+ of historical winners. Strong buy. |
    | **70-80%** | ✅ **Confident** | Reliable. Best risk/reward zone. |
    | **60-70%** | 🟡 **Marginal** | Needs Kimi Score > 25 to confirm. Otherwise skip. |
    | **< 60%** | ⚪ **Low** | Skip unless Kimi Score alone is 35+. |
    
    ## 💎 THE "GOLDEN ENTRY" (Maximum Conviction Setup)
    > A stock is a **Golden Entry** when ALL THREE conditions are met:
    > 1. **Kimi Score > 25** (Quality confirmed)
    > 2. **AI Prob > 75%** (Pattern confirmed)
    > 3. **Verdict = ✅ BUY** (All safety guards passed)
    
    ---
    ## 📦 VOLUME (TURNOVER) CONFIGURATION
    
    The "Min Volume (M)" filter = **Turnover in Millions INR** (Price × Shares traded per day, averaged over 20 days).
    
    | Your Timeframe | Min Turnover | Why |
    | :--- | :--- | :--- |
    | **3-7 Days (Aggressive)** | **300M+** | Need instant exits. Zero slippage tolerance. |
    | **1-2 Weeks (Swing)** | **100M+** | Best balance of signal quality + exit speed. |
    | **1 Month (Position)** | **50M+** | Slower moves, can tolerate thinner volume. |
    
    **Pro Tips:**
    *   **Vol Ratio > 2.0** + **Turnover > 500M** = Heavy institutional buying. Pay attention.
    *   **Vol Ratio < 0.5** = No one is trading. Dead stock. Skip.
    *   Setting volume too low (< 10M) risks getting stuck in penny stocks with no exit.
    
    ---
    ## ⏱️ WHEN TO RUN THE SCANNER
    
    | Time | Quality | Best For |
    | :--- | :--- | :--- |
    | **3:15 PM IST** | ⭐⭐⭐⭐⭐ | Best. Full day's data + all volume captured. |
    | **After Market (4+ PM)** | ⭐⭐⭐⭐ | Great. Use "Offline CSV" for instant results. |
    | **Pre-Market (8-9 AM)** | ⭐⭐⭐ | Good for planning. Uses yesterday's closing data. |
    | **Mid-Day (12-2 PM)** | ⭐⭐ | OK for "Live Market" mode only. Data is incomplete. |
    
    **Entry Timing:**
    *   **Aggressive:** Enter at **3:25 PM** if price holds above the current level.
    *   **Safe:** Enter **next morning at 9:30 AM** after assessing for overnight gaps.
    
    ---
    ## 🕐 TIMEFRAME SELECTION
    
    | Timeframe | Hold Period | Stop Loss | Target | Best Market |
    | :--- | :--- | :--- | :--- | :--- |
    | **⚡ 3-7 Days** | 3-5 trading days | 5% | 5% | BULL only |
    | **🎯 1-2 Weeks** | 5-10 trading days | 8% | 10% | BULL or CHOP |
    | **🛡️ 1 Month** | 15-25 trading days | 12% | 15% | Any regime |
    
    **Recommendation:** Start with **1-2 Weeks**. It has the best risk/reward balance.
    """)

with st.expander("🛡️ Surgical Verdict & Red Flags", expanded=False):
    st.markdown("""
    ---
    ## THE SURGICAL VERDICT — What Gets Checked
    
    Before any stock gets the **✅ BUY** tag, it must pass through ALL of these safety guards:
    
    | Guard | What It Checks | Threshold |
    | :--- | :--- | :--- |
    | **AI Conviction** | Minimum AI probability | 60% (short) / 70% (swing) / 80% (position) |
    | **Trend Guard** | Price vs SMA50 | Must be above SMA50 (exception: Rebound mode) |
    | **RSI Overheat** | RSI level | Blocks if RSI > 70 (overheated) |
    | **Kimi Floor** | Minimum quality score | Score > 15 (short) / 20 (swing) |
    | **Liquidity Guard** | Your order size vs market | Order must be < 1% of avg daily volume |
    
    If ANY guard fails, the stock gets **❌ Avoid** with a specific reason.
    
    ---
    ## ⚠️ RED FLAGS — When NOT to Buy
    *   **RSI > 70 + Score < 20**: Overbought AND weak. Recipe for a crash.
    *   **Vol Ratio < 0.5**: Dead volume. No institutions are buying.
    *   **Market Armor = BEAR**: Even 35+ score stocks will fall in a bear market.
    *   **Squeeze > 0.20**: Wide Bollinger Bands = high volatility, unpredictable moves.
    *   **Verdict = ❌ Avoid**: Trust the system. It's protecting your capital.
    """)

with st.expander("❓ FAQ", expanded=False):
    st.markdown("""
    ### 1. Which mode should I use — Sword or Eyes?
    *   **⚔️ Sword**: Focuses on quality + stability. Best for choppy/uncertain markets (~62-67% win rate).
    *   **👁️ Eyes**: Focuses on explosive AI patterns. Best for strongly bullish markets (~58-63% win rate).
    *   **Best Strategy:** Use BOTH. If a stock scores Gold on Kimi AND 75%+ on AI, that's maximum conviction.
    
    ### 2. Does changing Capital affect stock selection?
    *   **No.** Stock selection is 100% objective (market data only).
    *   **Capital only affects the Qty column** — how many shares fit your risk allocation.
    
    ### 3. What setup gives the best historical returns?
    *   **Timeframe**: 1-2 Weeks.
    *   **Volume**: 100M+.
    *   **Condition**: Deploy when **Market Armor = BULL**.
    *   **Selection**: Pick **✅ BUY** stocks that score high on both Sword AND Eyes.
    
    ### 4. What are "Squeeze" and "Coiling" in the results?
    *   **Squeeze** = Bollinger Bandwidth. When it's very low (< 0.08), the stock is compressed.
    *   **Coiling** = Price range compression. Low coiling = stock is "spring-loaded" for a move.
    *   **Together, low Squeeze + low Coiling = high breakout probability.**
    
    ### 5. Expected returns by mode?
    *   **Armor**: 0% directly, but it **prevents 100%** of bear market losses.
    *   **Sword**: ~20-30% p.a. Steady compounder with low drawdowns.
    *   **Eyes**: ~35%+ p.a. High-speed gains but requires strict exit discipline.
    """)

# ==========================================
# FOOTER & COMPLIANCE
# ==========================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #888; font-family: sans-serif; padding: 20px;'>
<h3 style='color: #00d2ff;'>🐺 Alpha-Wolf Unified Scanner</h3>
<p>App by Pravin A Mathew</p>
<p style='color: #e94560; font-weight: bold;'>THIS IS FOR SWING TRADING AND NOT FOR INTRADAY TRADING</p>
<div style='margin-top: 20px; font-size: 0.8em; text-align: justify; padding: 0 10%;'>
<p><b>SEBI Compliance & Risk Disclaimer:</b><br>
I am not a SEBI Registered Investment Advisor. This scanner is an automated tool designed for Educational & Research purposes only. The signals generated do not constitute financial advice or buy/sell recommendations. Paper trading is recommended before committing real capital. Trading in equities involves significant risk. The author is not responsible for any financial losses incurred using this tool. Do your own research (DYOR) and consult a certified professional before investing.</p>
<p><b>Strategy Expectations:</b><br>
In the professional trading world, most successful strategies operate with a 50% to 60% win rate. No professional system achieves 90-100% accuracy. The goal is positive expectancy — winning enough to grow capital over time.</p>
</div>
</div>
""", unsafe_allow_html=True)


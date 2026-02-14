# 🐺 Alpha-Wolf Unified (The Ultimate Engine)

## 🚀 Overview
**Alpha-Wolf Unified** is the **production-grade culmination** of the entire scanner ecosystem. It combines the strategic depth of Wolf-Pack with the robustness of Alpha-Zeta's data engine, using a validated V10 AI model.

**Best For:**
*   **Production Trading.**
*   Advanced users seeking maximum reliability.
*   Understanding the "Armor / Sword / Eyes" philosophy.

---

## 🛡️ The 3-Layer Architecture

### 1. The Armor (Market Health)
*   **Goal:** Protect capital.
*   **Logic:** Checks Nifty 50 vs SMA20 & SMA50.
*   **Verdict:** BULL (Green), CHOP (Yellow), BEAR (Red).

### 2. The Sword (Kimi Score)
*   **Goal:** Identify high-quality setups.
*   **Logic:** 0-100 Score based on 4 Pillars:
    1.  **Quality:** Price > SMA50.
    2.  **Value:** RSI Sweet Spot (40-60).
    3.  **Volatility:** Low volatility premium.
    4.  **Momentum:** Weighted returns.

### 3. The Eyes (V10 AI Model)
*   **Goal:** Predict probability of success.
*   **Logic:** Random Forest Classifier driven by **10 Technical Features**.

---

## 🛠️ Configuration & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard (Recommended)
```bash
streamlit run app.py
```
*   **Unified Interface:** See Market Health, Kimi Score, and AI Probability in one view.
*   **Surgical Verdict:** Automatically combines all layers into a final Buy/Skip signal.
*   **Data V2.0:** Toggle between Live and Cached data for speed or freshness.

### 3. Retrain the Wolf Brain
```bash
python trainer.py
```
*   **New Feature:** Generates `v10_model.pkl` with Squeeze + Coiling logic.
*   **Validation:** Uses Walk-Forward Validation to prevent overfitting.

### 4. Run the CLI Scanner
```bash
python cli.py
```

---

## ⚠️ Important Notes (Feb 2025 Update)

### 🔴 Critical Fixes Implemented
*   **Fixed AI Features:** Removed circular dependency. Now uses Bollinger Squeeze + Price Coiling.
*   **Fixed Engine:** Now calculates ALL 10 required features correctly.
*   **Fixed Scaling:** Kimi momentum score standardized to prevent capping strong movers.
*   **Validated Model:** Trainer script updated to use time-series cross-validation.

### 📉 Score Interpretation
*   **Surgical Entry:** High AI Prob (>70%) + Strong Kimi Score (>20) + Safe Market.
*   **Speculative Rebound:** Allowed in Aggressive Mode if Price < SMA50 but Momentum is high.

---

## 📜 License
Privately developed for high-probability swing trading.

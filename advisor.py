# advisor.py
import yfinance as yf
import pandas as pd
import re

# --- 1. UTILITIES ---
def normalize_ticker(user_input):
    """Clean ticker symbols (e.g., '  aapl ' -> 'AAPL')."""
    match = re.search(r'\b[A-Z]{1,5}\b', str(user_input).upper())
    if match and len(user_input.split()) < 3:
        return match.group(0).upper().strip()
    return user_input.strip()

def get_client_context(risk_profile, time_horizon):
    """Generates a text block for the AI prompt based on user settings."""
    return f"""
    *** ADVISORY MANDATE ***
    - CLIENT RISK PROFILE: {risk_profile}
    - TIME HORIZON: {time_horizon}
    - LEGAL STANDARD: SEC Regulation Best Interest (Reg BI)
    - INSTRUCTION: Recommendations must prioritize the client's best interest.
    """

# --- 2. FINANCIAL PLANNING LOGIC (GENERIC STANDARD) ---
def get_planning_prompt(fact_finder_data, user_request):
    """
    Titan 3.0 Planning Logic.
    Structure: Standard Comprehensive Financial Plan (7-Part Professional Format).
    Standards: Kitces (Strategy), Ed Slott (Tax), Reg BI (Fiduciary).
    """
    return f"""
    You are a Lead Financial Planner (CFP®). 
    Your task is to write a Comprehensive Financial Plan following industry-standard professional formatting.

    FULL DATASET PROVIDED:
    {fact_finder_data}

    USER'S PRIMARY OBJECTIVE:
    {user_request}

    *** REQUIRED REPORT STRUCTURE (STANDARD PROFESSIONAL FORMAT) ***
    
    1. **COVER LETTER & EXECUTIVE SUMMARY**
       - Personal address to the client.
       - 'Bottom Line' summary (On Track / Needs Attention).
       - Fiduciary Statement (Reg BI compliance).

    2. **NET WORTH STATEMENT (Balance Sheet)**
       - Assets (Liquid vs. Illiquid).
       - Liabilities (Mortgage, Debts).
       - Total Net Worth calculation.

    3. **CASH FLOW ANALYSIS**
       - Income Sources vs. Expenses (Needs & Wants).
       - Surplus/Deficit calculation.
       - Recommendation: Deployment of surplus or curing of deficit.

    4. **RETIREMENT PROJECTION**
       - Probability of Success Analysis (Simulated market conditions).
       - Longevity check (Does money last to target age?).
       - Gap Analysis: Shortfall amount if any.

    5. **TAX & DISTRIBUTION STRATEGY (Ed Slott Logic)**
       - Current Tax Bracket vs. Future Bracket Analysis.
       - Roth Conversion opportunities.
       - RMD (Required Minimum Distribution) planning.

    6. **RISK MANAGEMENT & INSURANCE**
       - Life Insurance Gap Analysis (Income Replacement).
       - Liability/Umbrella coverage check.
       - Estate Plan Status (Will, Trust, POA).

    7. **ACTION PLAN (Implementation)**
       - Immediate Steps (Next 30 Days).
       - Short-Term Steps (Next 6 Months).
       - Long-Term Goals (1-5 Years).
    """

# --- 3. TECHNICAL ANALYSIS ENGINE (FOR STOCK REQUESTS) ---
def calculate_technicals(ticker):
    """Calculates RSI, SMA, and Trend with safety checks."""
    try:
        clean_ticker = normalize_ticker(ticker)
        stock = yf.Ticker(clean_ticker)
        hist = stock.history(period="1y")
        if hist.empty: return "TECHNICAL DATA: N/A"
        
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        
        # RSI Calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        current_price = hist['Close'].iloc[-1]
        rsi = hist['RSI'].iloc[-1]
        sma_200 = hist['SMA_200'].iloc[-1]
        
        if len(hist) >= 200:
            trend = "BULLISH" if current_price > sma_200 else "BEARISH"
            sma_200_str = f"${sma_200:.2f}"
        else:
            sma_200_str, trend = "N/A", "NEUTRAL (Short History)"
            
        return f"""
        *** TECHNICAL ANALYSIS DATA ***
        - Price: ${current_price:.2f} | 200-SMA: {sma_200_str}
        - Trend: {trend} | RSI: {rsi:.2f}
        """
    except Exception as e:
        return f"TECHNICAL ERROR: {str(e)}"

def check_correlation(new_ticker, existing_portfolio_str):
    """Checks correlation to prevent over-concentration."""
    if not existing_portfolio_str: return "PORTFOLIO CONTEXT: No existing portfolio provided."
    try:
        existing = [normalize_ticker(x) for x in existing_portfolio_str.split(',') if x.strip()]
        target = normalize_ticker(new_ticker)
        tickers = list(set(existing + [target]))
        data = yf.download(tickers, period="1y", progress=False)['Close']
        
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close']
        
        corr_matrix = data.corr()
        if target not in corr_matrix.columns: return "PORTFOLIO CONTEXT: Correlation data missing."
        
        target_corrs = corr_matrix[target]
        summary = ["*** CONCENTRATION RISK CHECK ***"]
        for tick in existing:
            if tick in target_corrs.index and tick != target:
                val = target_corrs[tick]
                risk = "HIGH" if val > 0.7 else "LOW" if val < 0.3 else "MODERATE"
                summary.append(f"- vs {tick}: {val:.2f} ({risk} Correlation)")
        return "\n".join(summary)
    except Exception as e:
        return f"PORTFOLIO CONTEXT ERROR: {str(e)}"

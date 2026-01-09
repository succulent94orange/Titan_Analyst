import requests
import hashlib
import sys
import streamlit as st
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import asyncio
import random
import json
import re
import time
from datetime import datetime
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tempfile
import yfinance as yf
import markdown
from xhtml2pdf import pisa
from io import BytesIO
import advisor

# --- 1. SETUP & AUTHENTICATION ---
# Set Matplotlib to non-interactive mode
matplotlib.use('Agg')
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("🚨 CRITICAL ERROR: GOOGLE_API_KEY not found in .env file.")
    st.stop()

client = genai.Client(api_key=api_key)
MODELS = ["gemini-3-pro-preview"] 

HIDDEN_KEY = "67116e031024e38e146c9c61284d748f220376e109d941865c19d7d43f07a0e3"

def validate_access():
    url = "https://raw.githubusercontent.com/succulent94orange/Titan_Analyst/main/checksum.py"
    try:
        response = requests.get(url + "?v=1", timeout=5)
        response.raise_for_status()
        if HIDDEN_KEY not in response.text:
            st.error("🚨 SECURITY ALERT: Unauthorized modification detected.")
            st.stop()
        return True
    except: return True # Fail open for dev

# --- 2. STOCK ANALYSIS PROMPTS ---

def fetch_market_data(ticker):
    """Fetches Price, Name, and Treasury Yields."""
    clean_ticker = advisor.normalize_ticker(ticker)
    data_pkg = {"price": None, "yield": None, "is_valid_ticker": False, "name": ticker}
    try:
        stock = yf.Ticker(clean_ticker)
        hist = stock.history(period="5d")
        if not hist.empty:
            data_pkg["price"] = float(hist['Close'].iloc[-1])
            data_pkg["is_valid_ticker"] = True
            data_pkg["name"] = stock.info.get('longName', clean_ticker)
        
        treasury_data = yf.Ticker("^TNX").history(period="5d")
        if not treasury_data.empty:
            data_pkg["yield"] = float(treasury_data['Close'].iloc[-1])
    except: pass
    return data_pkg

def get_dynamic_prompts(subject, advisory_context, current_price=None, current_yield=None, is_ticker=True):
    now = datetime.now()
    current_date_str = now.strftime("%Y-%m-%d")
    
    context_parts = [
        f"CRITICAL REAL-TIME DATA (As of {current_date_str}):",
        f"- Subject of Analysis: {subject}",
        f"--------------------------------",
        f"{advisory_context}",
        f"--------------------------------"
    ]
    if is_ticker and current_price:
        context_parts.append(f"- CURRENT LIVE PRICE: ${current_price:.2f}")
        context_parts.append(f"- RULE: Price targets MUST be based on ${current_price:.2f}.")
    
    price_context = "\n".join(context_parts)
    prompts = {}

    prompts["MACRO"] = f"You are the Macro Council. Analyze global landscape for: {subject}.\n{price_context}"
    prompts["FUNDAMENTAL"] = f"You are the Fundamental Specialist. Analyze SWOT for: {subject}.\n{price_context}"
    prompts["CFA"] = f"You are a CFA Charterholder. Conduct risk analysis for: {subject}.\n{price_context}"
    prompts["QUANT"] = f"""You are the Quant Desk. Provide a JSON block for 12-month targets:
    ```json
    {{ "Bear": <number>, "Base": <number>, "Bull": <number> }}
    ```\n{price_context}"""
    prompts["RED_TEAM"] = f"You are the Red Team (Jim Chanos). Find fatal flaws in {subject}.\n{price_context}"
    prompts["PORTFOLIO"] = f"You are the CIO. Final verdict MUST end with VERDICT: [BUY/SELL/HOLD].\n{price_context}"
    
    return prompts

# --- 3. CHART & PERFORMANCE ENGINE ---

def generate_bar_chart(data_dict, title):
    """Generates high-resolution bar charts for price targets."""
    try:
        if not data_dict: return None
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.bar(data_dict.keys(), data_dict.values(), color='#0A1932')
        ax.set_title(title, fontweight='bold', color='#0A1932')
        ax.set_ylabel('Price ($)')
        plt.tight_layout()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name, format='png', dpi=300)
            plt.close(fig)
            return tmp.name
    except: return None

def generate_line_chart(df, title):
    """Generates 5-year relative performance line charts."""
    try:
        if df is None or df.empty: return None
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        for col in df.columns:
            ax.plot(df.index, df[col] * 100, label=col, linewidth=2)
        ax.set_title(title, fontweight='bold', color='#0A1932')
        ax.set_ylabel('Cumulative Return (%)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name, format='png', dpi=300)
            plt.close(fig)
            return tmp.name
    except: return None

def fetch_relative_returns(ticker, benchmark="SPY"):
    try:
        clean_ticker = advisor.normalize_ticker(ticker)
        data = yf.download([clean_ticker, benchmark], period="5y", progress=False)['Close']
        if data.empty: return None
        aligned_data = data.dropna()
        if aligned_data.empty: return None
        return (aligned_data / aligned_data.iloc[0]) - 1
    except: return None

# --- 4. PDF ENGINE (XHTML2PDF) ---

COLOR_NAVY = "#0A1932"
COLOR_GOLD = "#DAA520"
STYLE_CSS = f"""
    @page {{ size: letter; margin: 2cm; }}
    body {{ font-family: Helvetica; font-size: 11pt; color: #333; }}
    h1 {{ color: {COLOR_NAVY}; text-align: center; text-transform: uppercase; }}
    h2 {{ color: {COLOR_NAVY}; border-bottom: 2px solid {COLOR_GOLD}; text-transform: uppercase; margin-top: 20px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
    th {{ background-color: {COLOR_NAVY}; color: {COLOR_GOLD}; padding: 8px; font-weight: bold; }}
    td {{ padding: 8px; border: 1px solid #ddd; }}
    .center-img {{ text-align: center; margin: 20px 0; }}
"""

def generate_pdf_report(ticker, report_data, c_path=None, r_path=None):
    sections_html = ""
    # Define order for Stock Analysis
    order = [("1. EXECUTIVE THESIS", "Portfolio Manager"), ("2. MACRO", "Macro"), ("3. FUNDAMENTALS", "Fundamental"), ("4. CFA ANALYSIS", "CFA"), ("5. QUANTITATIVE", "Quant"), ("6. RED TEAM", "Red Team")]
    
    for title, key in order:
        text_content = report_data.get(key, "N/A")
        # Clean specific Quant JSON artifacts for readability
        if key == "Quant": 
            text_content = re.sub(r'```json.*?```', '', text_content, flags=re.DOTALL)
        
        content = markdown.markdown(text_content, extensions=['extra', 'tables'])
        sections_html += f"<h2>{title}</h2>{content}"
        
        # Inject Charts
        if key == "Quant" and c_path:
            sections_html += f'<div class="center-img"><h3>Valuation Targets</h3><img src="{c_path}" width="500" height="300"></div>'
        if key == "Fundamental" and r_path:
             sections_html += f'<div class="center-img"><h3>Relative Performance</h3><img src="{r_path}" width="500" height="300"></div>'

    full_html = f"""
    <html>
    <head><style>{STYLE_CSS}</style></head>
    <body>
        <div style="text-align:center; border-bottom: 2px solid {COLOR_GOLD}; padding-bottom: 10px;">
            <h1 style="margin:0;">TITAN ANALYST 3.0</h1>
            <p style="margin:0; color:#666;">FIDUCIARY WEALTH & EQUITY RESEARCH</p>
        </div>
        <br><br>
        <h1 style="font-size: 24pt;">REPORT: {ticker}</h1>
        <p style="text-align:center;">Generated: {datetime.now().strftime('%Y-%m-%d')}</p>
        <br>
        {sections_html}
    </body>
    </html>
    """
    pdf_file = BytesIO()
    pisa.CreatePDF(src=full_html, dest=pdf_file)
    return pdf_file.getvalue()

# --- 5. AGENT EXECUTION ---

async def run_agent(name, prompt, content):
    try:
        response = await client.aio.models.generate_content(
            model=MODELS[0],
            contents=f"{prompt}\nCONTEXT: {content}",
            config=types.GenerateContentConfig(
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_level="high"),
                tools=[types.Tool(google_search=types.GoogleSearch())],
                system_instruction="Strictly adhere to IRS.gov, Kitces.com, and Ed Slott logic. Cross-reference Reg BI."
            )
        )
        return name, response.text
    except Exception as e:
        return name, f"Error: {str(e)}"

async def execute_immersive_analysis(user_input, advisory_context, fact_finder):
    market_data = fetch_market_data(user_input)
    is_ticker = market_data["is_valid_ticker"]
    
    # PLANNING MODE DETECTION
    planning_keywords = ["plan", "retirement", "estate", "tax", "planning", "college", "wealth", "save", "budget"]
    is_planning_request = any(word in user_input.lower() for word in planning_keywords) and not is_ticker

    if is_planning_request:
        st.toast("📑 Titan 3.0: Constructing Fiduciary Wealth Plan...")
        prompt = advisor.get_planning_prompt(fact_finder, user_input)
        _, content = await run_agent("Portfolio Manager", prompt, user_input)
        
        # Structure for PDF (Single section for Plan)
        return {
            "Portfolio Manager": content, 
            "_is_ticker": False, 
            "_subject": "Comprehensive Wealth Plan",
            "_company_name": "Titan Wealth 3.0"
        }

    # STOCK MODE (Multi-Agent Swarm)
    prompts = get_dynamic_prompts(market_data.get("name", user_input), advisory_context, market_data["price"], market_data["yield"], is_ticker)
    
    # --- ERROR FIX: Map UI Names to Uppercase Prompt Keys ---
    agent_map = {
        "Macro": "MACRO",
        "Fundamental": "FUNDAMENTAL",
        "CFA": "CFA",
        "Quant": "QUANT"
    }
    
    tasks = [run_agent(name, prompts[key], user_input) for name, key in agent_map.items()]
    # -------------------------------------------------------
    
    results = await asyncio.gather(*tasks)
    data = {k: v for k, v in results}
    
    st.toast("⏳ Running Red Team Risk Analysis...")
    combined = "\n".join([v for k,v in data.items()])
    _, data["Red Team"] = await run_agent("Red Team", prompts["RED_TEAM"], combined)
    
    st.toast("⏳ Synthesizing Final Thesis...")
    combined += f"\n\nRED TEAM: {data['Red Team']}"
    _, data["Portfolio Manager"] = await run_agent("Portfolio Manager", prompts["PORTFOLIO"], combined)
    
    data.update({
        "_subject": user_input, "_is_ticker": is_ticker, "_price": market_data["price"],
        "_company_name": market_data.get("name", user_input)
    })
    return data

# --- 6. UI MAIN ---

def render_fact_finder():
    """Renders the comprehensive, immersive financial planning questionnaire."""
    st.sidebar.header("📋 Titan Immersive Fact Finder")
    
    with st.sidebar.expander("💼 Income & Cash Flow", expanded=True):
        gross_income = st.number_input("Annual Gross Income ($)", value=150000, step=5000)
        expenses = st.number_input("Monthly Living Expenses ($)", value=6000, step=500)
        filing_status = st.selectbox("Tax Filing Status", ["Single", "Married Filing Jointly", "Head of Household"])
        emergency_fund = st.number_input("Current Cash/Savings ($)", value=25000)

    with st.sidebar.expander("📉 Debt & Liabilities"):
        mortgage_bal = st.number_input("Mortgage Balance ($)", value=0)
        mortgage_rate = st.number_input("Mortgage Rate (%)", value=6.5, format="%.2f")
        consumer_debt = st.text_area("Other Debts (CC, Auto, Student)", 
                                     placeholder="Auto: $20k @ 5%, Student: $40k @ 4.5%...")

    with st.sidebar.expander("💰 Current Portfolio (Holdings)"):
        # NEW: Structured Data Editor for Portfolio
        st.caption("Enter your current investment holdings:")
        default_data = pd.DataFrame([
            {"Account Type": "Brokerage", "Ticker": "AAPL", "Value": 15000},
            {"Account Type": "Roth IRA", "Ticker": "VOO", "Value": 45000},
            {"Account Type": "401k", "Ticker": "VTI", "Value": 85000}
        ])
        
        portfolio_df = st.data_editor(
            default_data,
            num_rows="dynamic",
            column_config={
                "Account Type": st.column_config.SelectboxColumn(
                    "Account Type",
                    options=["Brokerage", "Roth IRA", "Traditional IRA", "401k", "HSA", "Crypto"],
                    required=True,
                ),
                "Ticker": st.column_config.TextColumn("Ticker", required=True),
                "Value": st.column_config.NumberColumn("Value ($)", min_value=0, format="$%d")
            },
            hide_index=True
        )
        
        total_value = portfolio_df["Value"].sum()
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")
        
        # Convert DF to string for AI
        portfolio_str = portfolio_df.to_string(index=False)
        tickers_list = ", ".join(portfolio_df["Ticker"].unique().tolist())

    with st.sidebar.expander("🛡️ Risk & Insurance"):
        life_ins = st.number_input("Total Life Insurance ($)", value=500000)
        health_ins = st.selectbox("Primary Health Coverage", ["PPO", "HDHP w/ HSA", "HMO", "Medicare"])
        disability = st.checkbox("Long-Term Disability Coverage?", value=True)

    with st.sidebar.expander("🎯 Retirement & Goals"):
        retire_age = st.slider("Target Retirement Age", 50, 75, 65)
        longevity = st.number_input("Planning Age (Longevity)", value=95)
        goals = st.text_area("Primary Wealth Goals", 
                             placeholder="1. Pay off home\n2. Fund children's 529\n3. Annual travel budget $15k")

    with st.sidebar.expander("⚖️ Risk & Estate"):
        risk_score = st.slider("Risk Tolerance (1-100)", 1, 100, 50)
        estate_plan = st.radio("Estate Documents Drafted?", ["Yes", "In Progress", "No"])

    return f"""
    --- IMMERSIVE FACT FINDER DATA ---
    CASH FLOW: Income ${gross_income}/yr, Expenses ${expenses}/mo, Status: {filing_status}, Cash: ${emergency_fund}
    DEBT: Mortgage ${mortgage_bal} at {mortgage_rate}%, Other: {consumer_debt}
    RISK: Life Ins ${life_ins}, Health: {health_ins}, Disability: {disability}
    PLANNING: Retire at {retire_age}, Longevity to {longevity}, Goals: {goals}
    GOVERNANCE: Risk Score {risk_score}/100, Estate Plan: {estate_plan}
    PORTFOLIO HOLDINGS:
    {portfolio_str}
    TOTAL ASSETS: ${total_value}
    """, tickers_list

def main():
    st.set_page_config(page_title="Titan 3.0", layout="wide")
    st.title("⚡ Titan Analyst 3.0: Immersive Wealth Engine")
    
    # 1. Capture Fact Finder Data
    fact_finder_data, portfolio_tickers = render_fact_finder()

    with st.form("titan_3_form"):
        user_input = st.text_input("Enter Ticker or Planning Request:", "NVDA")
        submitted = st.form_submit_button("Launch Titan 3.0 Engine")

    if submitted:
        with st.spinner("Titan 3.0 reasoning through Fact Finder data..."):
            validate_access()
            client_ctx = advisor.get_client_context("Moderate", "Long Term")
            
            # Technicals & Correlation only run if it's a ticker-like input
            techs = advisor.calculate_technicals(user_input)
            corrs = advisor.check_correlation(user_input, portfolio_tickers)
            full_context = f"{client_ctx}\n{techs}\n{corrs}"
            
            st.session_state.report = asyncio.run(execute_immersive_analysis(user_input, full_context, fact_finder_data))

    if "report" in st.session_state and st.session_state.report:
        rpt = st.session_state.report
        st.subheader(f"Strategic Output: {rpt['_subject']}")
        
        # Tabs for Stock Analysis
        if rpt.get("_is_ticker"):
            tabs = st.tabs(["Executive Thesis", "Macro", "Fundamental", "CFA", "Quant", "Red Team"])
            keys = ["Portfolio Manager", "Macro", "Fundamental", "CFA", "Quant", "Red Team"]
            for tab, key in zip(tabs, keys):
                with tab: st.markdown(rpt.get(key, "N/A"))
        else:
            # Single View for Wealth Plan
            st.markdown(rpt["Portfolio Manager"])

        # Chart Generation & PDF
        c_path, r_path = None, None
        if rpt.get("_is_ticker"):
            try:
                json_match = re.search(r'\{.*?"Bear".*?\}', rpt.get("Quant", ""), re.DOTALL)
                if json_match:
                    q_data = json.loads(json_match.group(0))
                    c_path = generate_bar_chart(q_data, "12-Month Price Targets")
                
                r_df = fetch_relative_returns(rpt["_subject"])
                if r_df is not None:
                    r_path = generate_line_chart(r_df, "Relative Performance (5Y)")
            except: pass

        pdf_bytes = generate_pdf_report(rpt["_subject"], rpt, c_path, r_path)
        st.download_button("📄 Download Titan 3.0 Fiduciary Report", data=pdf_bytes, file_name=f"{rpt['_subject']}_Report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()

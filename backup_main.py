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
from datetime import datetime, timedelta
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tempfile
import yfinance as yf
from fpdf import FPDF 
from io import BytesIO

# 1. SETUP
# Set Matplotlib to non-interactive mode to prevent crashes
matplotlib.use('Agg')

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("🚨 CRITICAL ERROR: GOOGLE_API_KEY not found in .env file.")
    st.stop()

# Initialize Client (New SDK)
client = genai.Client(api_key=api_key)

# Model Priority: STRICT MODE - GEMINI 3 PRO PREVIEW ONLY
MODELS = [
    "gemini-3-pro-preview"
]

# --- 2. DYNAMIC PROMPT GENERATION (ENHANCED WITH SEARCH) ---

def normalize_ticker(user_input):
    match = re.search(r'\b[A-Z]{1,5}\b', user_input)
    if match:
        if len(user_input.split()) > 1: return match.group(0)
        return user_input.upper().strip()
    return user_input.split()[0].upper().strip()

def fetch_market_data(ticker):
    """Fetches Stock Price, Name, and Treasury Yield."""
    clean_ticker = normalize_ticker(ticker)
    data_pkg = {"price": None, "yield": None, "is_valid_ticker": False, "name": ticker}
    
    try:
        # Fetch Stock Data
        stock = yf.Ticker(clean_ticker)
        hist = stock.history(period="5d")
        
        if not hist.empty:
            data_pkg["price"] = float(hist['Close'].iloc[-1])
            data_pkg["is_valid_ticker"] = True
            try:
                info = stock.info
                data_pkg["name"] = info.get('longName', clean_ticker)
            except:
                data_pkg["name"] = clean_ticker
        
        # Always fetch Treasury Yield for macro context
        treasury_data = yf.Ticker("^TNX").history(period="5d")
        if not treasury_data.empty:
            data_pkg["yield"] = float(treasury_data['Close'].iloc[-1])
            
    except:
        pass
    return data_pkg

def get_dynamic_prompts(subject, current_price=None, current_yield=None, is_ticker=True):
    now = datetime.now()
    current_date_str = now.strftime("%Y-%m-%d")
    
    # Build System Context
    context_parts = [
        f"CRITICAL REAL-TIME DATA (As of {current_date_str}):",
        f"- Subject of Analysis: {subject}"
    ]
    
    if is_ticker and current_price is not None:
        context_parts.append(f"- CURRENT LIVE PRICE: ${current_price:.2f}")
        context_parts.append(f"- RULE: All price targets MUST be based on ${current_price:.2f}. If Bullish, target > ${current_price:.2f}.")
    elif not is_ticker:
         context_parts.append(f"- NOTE: This is a General Finance Inquiry, not a specific stock ticker analysis.")
         
    if current_yield is not None:
        context_parts.append(f"- RISK-FREE RATE (10Y Treasury): {current_yield:.2f}%")
        context_parts.append(f"- RULE: Use {current_yield:.2f}% for WACC and Discount Rate calculations.")
        
    price_context = "\n".join(context_parts)
    
    prompts = {}
    
    # Shared Instructions
    guardrail = "GUARDRAIL: Use your Google Search Tool to find the latest data. Do not use 'Cannot determine' unless no data exists online."
    sec_instruction = "**SEC EDGAR PROTOCOL (MANDATORY):** Simulate accessing SEC.gov 10-K/10-Q filings." if is_ticker else "Use general financial knowledge and macroeconomic data sources."
    
    # 1. MACRO AGENT (UPDATED TO FORCE SEARCH)
    prompts["MACRO"] = f"""
    You are the Macro Council (Voices: Ray Dalio, Stanley Druckenmiller).
    {price_context}
    
    {guardrail}
    
    TASK: Analyze the global macro environment regarding: {subject}.
    
    **MANDATORY DATA SEARCH:**
    You MUST use your search tool to find the most recent available figures for the following. 
    Format this exactly as a markdown table:
    | Indicator | Current Rate/Value | Source/Date |
    | :--- | :--- | :--- |
    | Inflation Rate (CPI) | [Search] | [Search] |
    | GDP Growth Rate | [Search] | [Search] |
    | Unemployment Rate | [Search] | [Search] |
    | Consumer Confidence | [Search] | [Search] |
    | Digital Advertising Spend (Global/US) | [Search] | [Search] |
    | Interest Rates (Fed Funds) | [Search] | [Search] |

    REQUIRED ANALYSIS:
    1. **Data Synthesis:** How do the metrics above impact {subject}?
    2. **Tree of Thoughts:** Analyze 3 scenarios: Inflation Resurgence, Soft Landing, Deflationary Bust.
    3. **Macro Drag:** Identify specific headwinds/tailwinds.
    
    OUTPUT FORMAT (Text/Markdown):
    - Executive Summary
    - Key Indicators Table (Must be populated)
    - Scenario Analysis
    """

    # 2. FUNDAMENTAL AGENT
    prompts["FUNDAMENTAL"] = f"""
    You are the Fundamental Specialist (Voices: Peter Lynch, Warren Buffett).
    {price_context}
    
    {guardrail}
    
    TASK: Analyze the fundamentals of: {subject}.
    {sec_instruction}
    
    **PROTOCOL (Sell Side Handbook, Brown 2013, & IAIM Meta Example):**
    - **SWOT Analysis (IAIM Standard):** You MUST include a standard SWOT table:
      - **Strengths:** (e.g., Brand, Diversified Portfolio)
      - **Weaknesses:** (e.g., Reliance on specific revenue, Data risks)
      - **Opportunities:** (e.g., New revenue streams, Acquisitions)
      - **Threats:** (e.g., Competition, Economic Slowdown)
    - **Catalyst Identification:** Specific events (Earnings, Product Launch) expected in 6 months.
    - **Industry Knowledge:** Deep dive on competitive dynamics (e.g., Market Share %).
    
    REQUIRED ANALYSES:
    1. **SWOT Analysis:** (Strict Format).
    2. **Unit Economics:** LTV/CAC, Margins.
    3. **Moat Analysis:** Porter's 5 Forces.
    4. **Management Access:** Tone/Confidence from recent transcripts.
    
    OUTPUT FORMAT (Text/Markdown):
    - Fundamental Analysis
    - SWOT Analysis
    - Unit Economics
    - Economic Moat & Competitive Advantage
    - Management Scorecard
    """

    # 3. CFA AGENT
    prompts["CFA"] = f"""
    You are a CFA Charterholder.
    {price_context}
    
    {guardrail}
    
    TASK: Conduct a professional analysis adhering to **CFA Institute Research Report Essentials**.
    
    **CFA STANDARDS:**
    1. **Distinguish between Fact and Opinion.**
    2. **Quality of Earnings:** Analyze non-GAAP adjustments. Are they masking real costs?
    3. **Financial Strength:** Analyze Liquidity (Current Ratio) and Solvency (Debt/Equity).
    
    OUTPUT FORMAT (Text/Markdown):
    - CFA Analysis (Adhering to Institute Standards)
    - Key Insights (Quality of Earnings focus)
    - Risk Factors (Liquidity & Solvency)
    """

    # 4. QUANT AGENT
    prompts["QUANT"] = f"""
    You are the Quant Desk (Voices: Jim Simons).
    {price_context}
    
    {guardrail}
    
    TASK: Analyze valuation, risk, and sensitivity.
    
    **VALUATION PROTOCOL (Institute of Advanced Investment Management - IAIM):**
    - **Weighted Valuation Model:** Do not rely on one number. Calculate value based on a weighted average of:
      1. **Comparable Companies (Comps):** (e.g., 30% weight)
      2. **Precedent Transactions:** (e.g., 30% weight)
      3. **DCF (Levered/Unlevered):** (e.g., 40% weight)
    - **Explicit Assumptions:** You MUST state your assumptions for:
      - **WACC** (e.g., 10.3%)
      - **Terminal Growth Rate** (e.g., 3-4%)
      - **Exit Multiple** (e.g., EV/EBITDA 14x)
    
    REQUIRED CALCULATIONS:
    1. **Weighted Valuation:** Output the weighted target derived from Comps, Precedents, and DCF.
    2. **Stochastic DCF:** Mental Monte Carlo simulation.
    3. **Beneish M-Score:** Earnings manipulation check.
    
    If this is a specific stock, output:
    1. **Text Analysis**: Valuation Metrics (WACC, Exit Multiple, Weighted Logic).
    2. **CHART DATA**: JSON block for Price Targets: {{"Bear": 100, "Base": 150, "Bull": 200}}
    """

    # 5. RED TEAM AGENT
    prompts["RED_TEAM"] = f"""
    You are the Red Team (Voice: Jim Chanos).
    {price_context}
    
    {guardrail}
    
    TASK: Find fatal flaws. 
    **RISK PROTOCOLS (Brown 2013 & Sell Side Handbook):** - **The "Walk-Down":** Flag if analysts have recently lowered estimates to allow for an easy "beat".
    - **Consensus Herding:** If 90% of analysts are "Buy", assume the trade is crowded.
    - **Optimism Bias:** Strip out the investment banking optimism.
    
    RISK PROTOCOLS:
    1. **Backward Check:** Reverse-engineer the price. Is implied growth realistic?
    2. **The Grey Rhino:** Obvious, high-impact threats ignored by the market.
    3. **Pre-Mortem:** Assume it is 2030 and the company has failed. Write the obituary.
    4. **SEC Forensics:** Check "Legal Proceedings" for hidden risks.
    
    OUTPUT FORMAT (Text/Markdown):
    - Key Investment Risks (Stripping out Optimism Bias)
    - The Bear Case
    - Pre-Mortem
    """

    # 6. PORTFOLIO MANAGER AGENT
    prompts["PORTFOLIO"] = f"""
    You are the CIO.
    {price_context}
    
    {guardrail}
    
    TASK: Synthesize final thesis for: {subject}.
    
    **THE "MORNING MEETING" NOTE (Sell Side Handbook & IAIM):**
    - **Upside/Downside Analysis:** Explicitly state the Upside/Downside % to the Target Price.
    - **Structure:** Punchy, Actionable, Catalyst-focused.
    - **Variant Perception:** Why is the consensus wrong?
    
    **MANDATORY OUTPUT REQUIREMENT:**
    The VERY LAST line of your response MUST be exactly one of these three options to classify the trade:
    VERDICT: BUY
    VERDICT: SELL
    VERDICT: HOLD
    
    OUTPUT FORMAT (Text/Markdown):
    - Executive Thesis (Morning Note Format)
    - Variant Perception
    - Upside/Downside Scenarios
    - Final Verdict (Text explanation)
    - VERDICT: [BUY/SELL/HOLD]
    """
    
    return prompts

FOLLOWUP_PROMPT = "You are a Research Assistant. Answer user questions based ONLY on the provided report context."

# --- 3. FPDF2 REPORT ENGINE (LEGACY COMPATIBLE) ---

class PDFReport(FPDF):
    def header(self):
        # Logo or Header Text
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(10, 25, 50) # Navy
        self.cell(0, 10, 'TITAN FINANCIAL ANALYST // EQUITY RESEARCH', border=False, align='C', ln=1)
        
        # Gold underline
        self.set_draw_color(218, 165, 32) # Gold
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Strictly Confidential | Generated by Titan AI | Page {self.page_no()}', align='C')

    def chapter_title(self, label):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(10, 25, 50) # Navy
        self.cell(0, 10, label, ln=1)
        
        # Gold underline for section
        self.set_draw_color(218, 165, 32)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)

    def chapter_body(self, text):
        self.set_font('Helvetica', '', 11)
        self.set_text_color(50, 50, 50)
        # Sanitize text for Latin-1 (standard PDF font encoding)
        clean_text = text.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 6, clean_text)
        self.ln()

def generate_pdf_report(title_text, company_name, report_data, chart_path=None, return_path=None, return_df=None, verdict="HOLD", price_target="N/A"):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- TITLE PAGE ---
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(10, 25, 50)
    pdf.multi_cell(0, 15, title_text.upper(), align='C')
    
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, company_name.upper(), align='C', ln=1)
    
    # --- RECOMMENDATION BLOCK ---
    pdf.ln(20)
    pdf.set_draw_color(218, 165, 32) # Gold
    pdf.set_fill_color(245, 245, 245) # Light Grey
    pdf.rect(60, pdf.get_y(), 90, 35, 'FD') # Box for rating
    
    pdf.set_y(pdf.get_y() + 5)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "RECOMMENDATION", align='C', ln=1)
    
    # Verdict Color Logic
    if "BUY" in verdict.upper():
        pdf.set_text_color(0, 100, 0) # Green
    elif "SELL" in verdict.upper():
        pdf.set_text_color(150, 0, 0) # Red
    else:
        pdf.set_text_color(218, 165, 32) # Gold/Orange
        
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 10, verdict.upper(), align='C', ln=1)
    
    # Price Target
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(50, 50, 50)
    if price_target != "N/A":
        pdf.cell(0, 10, f"12-MONTH TARGET: ${price_target}", align='C', ln=1)
    else:
        pdf.cell(0, 10, "TARGET: N/A", align='C', ln=1)
        
    pdf.ln(20)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(218, 165, 32) # Gold
    pdf.cell(0, 10, "INSTITUTIONAL EQUITY RESEARCH", align='C', ln=1)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%B %d, %Y')}", align='C', ln=1)
    
    # --- CONTENT SECTIONS ---
    
    def clean_md(text):
        if not text: return "N/A"
        # Strip the strict output tag if it exists
        text = text.replace("VERDICT: BUY", "").replace("VERDICT: SELL", "").replace("VERDICT: HOLD", "")
        text = text.replace('**', '').replace('##', '').replace('###', '')
        return text.strip()

    # 1. Executive
    pdf.add_page()
    pdf.chapter_title("1. EXECUTIVE THESIS")
    pdf.chapter_body(clean_md(report_data.get("Portfolio Manager", "")))
    
    # 2. Macro
    pdf.chapter_title("2. MACRO-ECONOMIC LANDSCAPE")
    pdf.chapter_body(clean_md(report_data.get("Macro", "")))
    
    # 3. Fundamental
    pdf.add_page()
    pdf.chapter_title("3. FUNDAMENTAL DEEP DIVE")
    pdf.chapter_body(clean_md(report_data.get("Fundamental", "")))

    # 4. CFA
    pdf.chapter_title("4. CFA ANALYSIS")
    pdf.chapter_body(clean_md(report_data.get("CFA", "")))

    # 5. Quant (Text + Charts)
    pdf.add_page()
    pdf.chapter_title("5. QUANTITATIVE & RISK")
    
    # Remove JSON chart data from text before printing
    quant_text = report_data.get("Quant", "")
    quant_text = re.sub(r'(\*\*CHART DATA\*\*.*?)?\{.*?"Bear".*?"Bull".*?\}', '', quant_text, flags=re.DOTALL | re.IGNORECASE)
    quant_text = re.sub(r'```json', '', quant_text)
    quant_text = re.sub(r'```', '', quant_text)
    
    pdf.chapter_body(clean_md(quant_text))

    if chart_path:
        pdf.ln(5)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 10, "Valuation Scenarios (12mo Targets)", ln=1)
        try:
            # Centering image
            img_w = 150
            x_pos = (pdf.w - img_w) / 2
            pdf.image(chart_path, x=x_pos, w=img_w)
            pdf.ln(5)
        except:
            pdf.cell(0, 10, "[Chart Generation Error]", ln=1)

    if return_path:
        pdf.ln(5)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 10, "5-Year Relative Performance", ln=1)
        try:
            img_w = 150
            x_pos = (pdf.w - img_w) / 2
            pdf.image(return_path, x=x_pos, w=img_w)
            pdf.ln(10)
        except: pass

    # Table for Returns
    if return_df is not None:
        pdf.ln(5)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 8, "Historical Return Data (Relative)", ln=1)
        
        pdf.set_font('Courier', '', 9) # Monospace for table alignment
        
        # Header
        # Reset index to get Date column
        df_reset = return_df.reset_index()
        # Convert date to string
        df_reset.iloc[:, 0] = df_reset.iloc[:, 0].astype(str).str.slice(0, 10)
        
        # Simple list print for robustness (PDF tables are complex)
        headers = list(df_reset.columns)
        header_str = " | ".join(headers)
        pdf.cell(0, 6, header_str, ln=1)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        
        # Rows (Show last 15 entries to save space)
        subset = df_reset.tail(15) 
        for index, row in subset.iterrows():
            row_vals = [f"{str(val)[:10]}" for val in row.values]
            row_str = " | ".join(row_vals)
            pdf.cell(0, 6, row_str, ln=1)

    # 6. Red Team
    pdf.add_page()
    pdf.chapter_title("6. KEY RISKS (RED TEAM VERDICT)")
    pdf.set_text_color(150, 0, 0) # Dark Red for warnings
    pdf.chapter_body(clean_md(report_data.get("Red Team", "")))

    # Output
    return pdf.output(dest='S').encode('latin-1')

# --- 4. CHART GENERATORS ---
HEX_NAVY = '#0A1932'
HEX_GOLD = '#DAA520'

def generate_bar_chart(data_dict, title):
    try:
        if not data_dict: return None
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        bars = ax.bar(data_dict.keys(), data_dict.values(), color=HEX_NAVY)
        ax.set_title(title, fontsize=14, fontweight='bold', color=HEX_NAVY)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'${height:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color=HEX_GOLD)
        plt.tight_layout()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig)
            return tmp.name
    except: return None

def generate_line_chart(df, title):
    try:
        if df is None or df.empty: return None
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        df.index = pd.to_datetime(df.index)
        col_names = list(df.columns)
        ax.plot(df.index, df[col_names[0]] * 100, label=col_names[0], color=HEX_NAVY, linewidth=2.5)
        if len(col_names) > 1:
            ax.plot(df.index, df[col_names[1]] * 100, label=col_names[1], color='grey', linewidth=1.5, linestyle='--')
        ax.set_title(title, fontsize=14, fontweight='bold', color=HEX_NAVY)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        plt.xticks(rotation=45)
        plt.tight_layout()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig)
            return tmp.name
    except Exception as e: return None

# --- 5. DATA HELPERS ---
def fetch_relative_returns(ticker, benchmark="SPY"):
    try:
        clean_ticker = normalize_ticker(ticker)
        tickers = [clean_ticker, benchmark]
        data = yf.download(tickers, period="5y", progress=False)['Close']
        if data is None or data.empty: return None
        
        # Ensure numeric type for safe plotting/math
        data = data.apply(pd.to_numeric, errors='coerce')
        aligned_data = data.dropna()
        if aligned_data.empty: return None
        
        normalized = (aligned_data / aligned_data.iloc[0]) - 1
        return normalized
    except: return None

def extract_chart_data(text):
    data = {}
    try:
        matches = re.findall(r'"(Bear|Base|Bull)":\s*(\d+)', text, re.IGNORECASE)
        if not matches: matches = re.findall(r'(Bear|Base|Bull).*?\$(\d+)', text, re.IGNORECASE)
        for label, value in matches:
            data[label.capitalize()] = float(value)
        if len(data) >= 2: return data
    except: pass
    return None

def verify_and_correct_targets(chart_data, current_price):
    if not chart_data or current_price is None: return chart_data
    try:
        current_price_val = float(current_price) 
        bull_target = float(chart_data.get("Bull", 0))
        # Ensure we are comparing numbers, not None types
        if bull_target > 0 and bull_target < current_price_val:
            chart_data["Bear"] = round(current_price_val * 0.80, 2)
            chart_data["Base"] = round(current_price_val * 1.10, 2)
            chart_data["Bull"] = round(current_price_val * 1.30, 2)
    except: pass
    return chart_data

def extract_verdict(text):
    if not text: return "HOLD"
    match = re.search(r'VERDICT:\s*(BUY|SELL|HOLD)', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    return "HOLD"

# --- 6. AGENT ENGINE ---
async def run_agent(name, prompt, content):
    await asyncio.sleep(random.uniform(0.5, 2.0))
    # Retry Loop (3 Attempts per agent)
    for attempt in range(3):
        for model_name in MODELS:
            try:
                # ENABLE TOOLS for Search Grounding
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=f"{prompt}\nCONTEXT: {content}",
                    config=types.GenerateContentConfig(
                        temperature=0.2, 
                        tools=[types.Tool(google_search=types.GoogleSearch())], # SEARCH ENABLED
                        safety_settings=[
                            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
                            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
                            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
                            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH")
                        ]
                    )
                )
                if response.text: return name, response.text
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    await asyncio.sleep(10 * (attempt + 1))
                else:
                    await asyncio.sleep(5)
                continue 

    return name, f"Analysis Failed for {name}. Error: {error_msg}"

async def run_analysis(user_input):
    market_data = fetch_market_data(user_input)
    is_ticker = market_data["is_valid_ticker"]
    
    current_price = market_data["price"]
    current_yield = market_data["yield"]
    
    if is_ticker:
        subject = normalize_ticker(user_input)
    else:
        subject = user_input
        
    prompts = get_dynamic_prompts(subject, current_price, current_yield, is_ticker)
    
    tasks = [
        run_agent("Macro", prompts["MACRO"], subject),
        run_agent("Fundamental", prompts["FUNDAMENTAL"], subject),
        run_agent("CFA", prompts["CFA"], subject),
        run_agent("Quant", prompts["QUANT"], subject)
    ]
    results = await asyncio.gather(*tasks)
    data = {k: v for k, v in results}
    
    if any("Analysis Failed" in str(v) for v in data.values()): return data

    st.toast("⏳ Cooling down quota for Red Team analysis...")
    await asyncio.sleep(5)
    
    combined_reports = "\n".join([v for k,v in data.items() if not k.startswith("_")])
    _, data["Red Team"] = await run_agent("Red Team", prompts["RED_TEAM"], combined_reports)
    
    st.toast("⏳ Cooling down quota for Final Thesis...")
    await asyncio.sleep(5)
    
    combined_reports += f"\n\nRED TEAM VERDICT:\n{data['Red Team']}"
    _, data["Portfolio Manager"] = await run_agent("Portfolio Manager", prompts["PORTFOLIO"], combined_reports)
    
    data["_current_price"] = current_price
    data["_subject"] = subject
    data["_is_ticker"] = is_ticker
    data["_company_name"] = market_data.get("name", subject)
    
    # Extract Metadata for Cover Page
    data["_verdict"] = extract_verdict(data.get("Portfolio Manager", ""))
    quant_targets = extract_chart_data(data.get("Quant", ""))
    if quant_targets and "Base" in quant_targets:
        data["_price_target"] = quant_targets["Base"]
    else:
        data["_price_target"] = "N/A"
        
    return data

# --- 7. UI ---
def main():
    st.set_page_config(page_title="Titan 2.5", layout="wide")
    st.title("⚡ Titan Analyst 2.5")
    
    if "report" not in st.session_state: st.session_state.report = None
    if "history" not in st.session_state: st.session_state.history = []

    with st.form(key='analysis_form'):
        user_input = st.text_input("Enter Ticker or Question:", "NVDA")
        submit_button = st.form_submit_button(label='Run Analysis')
    
    if submit_button:
        with st.spinner("Initializing Titan Agents..."):
            st.session_state.report = asyncio.run(run_analysis(user_input))
    
    if st.session_state.report:
        rpt = st.session_state.report
        
        failed = [k for k, v in rpt.items() if "Analysis Failed" in str(v) and not k.startswith("_")]
        if failed:
            st.error(f"🚨 Analysis Incomplete. Failures in: {', '.join(failed)}")
            for f in failed:
                st.write(f"**{f} Debug Info**:")
                st.code(rpt[f])
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("🏆 Executive Thesis")
                st.info(rpt.get("Portfolio Manager"))
                st.metric("Titan Verdict", rpt.get("_verdict", "HOLD"))
            with c2:
                if rpt.get("_is_ticker"):
                    st.subheader("🎯 12-Month Targets")
                    chart_data = extract_chart_data(rpt.get("Quant", ""))
                    chart_data = verify_and_correct_targets(chart_data, rpt.get("_current_price"))
                    if chart_data: st.bar_chart(chart_data)
                    else: st.caption("No targets found.")
                else:
                    st.info("General Analysis Mode (No Price Targets)")

            t1, t2, t3, t4, t5 = st.tabs(["Macro", "Fundamental", "CFA", "Quant", "Red Team"])
            with t1: st.markdown(rpt.get("Macro", ""))
            with t2: st.markdown(rpt.get("Fundamental", ""))
            with t3: st.markdown(rpt.get("CFA", ""))
            with t4: 
                st.markdown(rpt.get("Quant", ""))
                if rpt.get("_is_ticker"):
                    st.divider()
                    st.subheader("📈 5-Year Relative Return")
                    ret_data = fetch_relative_returns(rpt["_subject"])
                    if ret_data is not None: st.line_chart(ret_data)
            with t5: st.error(rpt.get("Red Team", ""))
            
            st.divider()
            try:
                c_path = None
                r_path = None
                r_data = None
                
                if rpt.get("_is_ticker"):
                    c_data = extract_chart_data(rpt.get("Quant", ""))
                    c_path = generate_bar_chart(c_data, "Price Targets") if c_data else None
                    r_data = fetch_relative_returns(rpt["_subject"])
                    r_path = generate_line_chart(r_data, "5-Year Total Return vs Benchmark") if r_data is not None else None
                
                report_title = rpt["_subject"]
                company_name = rpt.get("_company_name", report_title)
                
                # EXTRACTED DATA FOR PDF
                verdict = rpt.get("_verdict", "HOLD")
                target = rpt.get("_price_target", "N/A")
                
                pdf_bytes = generate_pdf_report(report_title, company_name, rpt, chart_path=c_path, return_path=r_path, return_df=r_data, verdict=verdict, price_target=target)
                
                file_label = re.sub(r'[^A-Za-z0-9]', '_', rpt["_subject"])[:15]
                st.download_button("📄 Download Professional Report (PDF)", pdf_bytes, f"{file_label}_Titan_Report.pdf", "application/pdf")
            except Exception as e:
                st.error(f"PDF Gen Error: {e}")

            st.divider()
            st.subheader("💬 Chat with Analyst")
            for m in st.session_state.history:
                st.chat_message(m["role"]).write(m["content"])
                
            if q := st.chat_input("Ask a follow-up..."):
                st.session_state.history.append({"role": "user", "content": q})
                st.chat_message("user").write(q)
                with st.spinner("Thinking..."):
                    _, ans = asyncio.run(run_agent("Chat", FOLLOWUP_PROMPT, f"REPORT: {rpt}\nUSER: {q}"))
                    st.session_state.history.append({"role": "assistant", "content": ans})
                    st.chat_message("assistant").write(ans)

if __name__ == "__main__":
    main()

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
import markdown
from xhtml2pdf import pisa
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

# Model Priority: STRICT MODE - GEMINI 3 PRO PREVIEW
MODELS = [
    "gemini-3-pro-preview"
]

# --- 2. DYNAMIC PROMPT GENERATION ---

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
            data_pkg["price"] = hist['Close'].iloc[-1]
            data_pkg["is_valid_ticker"] = True
            # Get Company Name
            try:
                info = stock.info
                data_pkg["name"] = info.get('longName', clean_ticker)
            except:
                data_pkg["name"] = clean_ticker
        
        # Always fetch Treasury Yield for macro context
        treasury_data = yf.Ticker("^TNX").history(period="5d")
        if not treasury_data.empty:
            data_pkg["yield"] = treasury_data['Close'].iloc[-1]
            
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
    
    if is_ticker and current_price:
        context_parts.append(f"- CURRENT LIVE PRICE: ${current_price:.2f}")
        context_parts.append(f"- RULE: All price targets MUST be based on ${current_price:.2f}. If Bullish, target > ${current_price:.2f}.")
    elif not is_ticker:
         context_parts.append(f"- NOTE: This is a General Finance Inquiry, not a specific stock ticker analysis.")
         
    if current_yield:
        context_parts.append(f"- RISK-FREE RATE (10Y Treasury): {current_yield:.2f}%")
        context_parts.append(f"- RULE: Use {current_yield:.2f}% for WACC and Discount Rate calculations.")
        
    price_context = "\n".join(context_parts)
    
    prompts = {}
    
    sec_instruction = "**SEC EDGAR PROTOCOL (MANDATORY):** Simulate accessing SEC.gov." if is_ticker else "Use general financial knowledge and macroeconomic data sources."
    val_instruction = "CRITICAL OUTPUT: Markdown Table of Valuation Metrics and JSON CHART DATA." if is_ticker else "Provide relevant quantitative metrics if applicable."

    prompts["MACRO"] = f"""
    You are the Macro Council (Voices: Ray Dalio, Stanley Druckenmiller).
    {price_context}
    
    TASK: Analyze the global macro environment regarding: {subject}.
    1. **Step-Back Prompting:** Identify the abstract economic principle governing the current era.
    2. **Tree of Thoughts:** Analyze 3 scenarios: Inflation Resurgence, Soft Landing, Deflationary Bust.
    3. **Macro Drag:** Identify specific headwinds/tailwinds.
    
    OUTPUT FORMAT (Markdown):
    - ## Executive Summary
    - ### Key Indicators (Table)
    - ### Scenario Analysis
    """

    prompts["FUNDAMENTAL"] = f"""
    You are the Fundamental Specialist (Voices: Peter Lynch, Warren Buffett).
    {price_context}
    
    TASK: Analyze the fundamentals of: {subject}.
    {sec_instruction}
    
    REQUIRED ANALYSES:
    1. **Unit Economics:** LTV/CAC, Same-Store Sales, Margins.
    2. **Moat Analysis:** Porter's 5 Forces.
    3. **Capital Allocation:** Buybacks vs. Empire Building.
    4. **Executive Compensation:** Is pay tied to EPS (bad) or ROIC (good)?
    
    OUTPUT FORMAT (Markdown):
    - ## Fundamental Analysis
    - ### Unit Economics (Table)
    - ### Economic Moat & Competitive Advantage
    - ### Management Scorecard
    """

    prompts["CFA"] = f"""
    You are a CFA Charterholder.
    {price_context}
    
    TASK: Conduct a professional credit/financial analysis of: {subject}.
    Focus on Risks, Margins, Liquidity, and Quality of Earnings/Data.
    
    OUTPUT FORMAT (Markdown):
    - ## CFA Analysis
    - **Key Insights:** [Details]
    - **Risk Factors:** [Details]
    """

    prompts["QUANT"] = f"""
    You are the Quant Desk (Voices: Jim Simons).
    {price_context}
    
    TASK: Analyze valuation, risk, and sensitivity for: {subject}.
    {val_instruction}
    
    REQUIRED CALCULATIONS:
    1. **Valuation:** Compare P/E, PEG to historicals.
    2. **Stochastic DCF:** Mental Monte Carlo simulation (10k iterations).
    3. **Kelly Criterion:** Position sizing based on edge (Half-Kelly).
    4. **Beneish M-Score:** Check for earnings manipulation flags.
    
    If this is a specific stock, output:
    1. **Markdown Table**: Valuation Metrics (P/E, PEG, FCF Yield, Beneish M-Score).
    2. **CHART DATA**: JSON block for Price Targets: {{"Bear": 100, "Base": 150, "Bull": 200}}
    """

    prompts["RED_TEAM"] = f"""
    You are the Red Team (Voice: Jim Chanos).
    {price_context}
    
    TASK: Find fatal flaws, contradictions, and "Grey Rhino" risks in the bullish case for: {subject}.
    
    RISK PROTOCOLS:
    1. **Backward Check:** Reverse-engineer the price. Is implied growth realistic?
    2. **The Grey Rhino:** Obvious, high-impact threats ignored by the market.
    3. **Pre-Mortem:** Assume it is 2030 and the company has failed. Write the obituary explaining exactly why.
    4. **SEC Forensics:** Check "Legal Proceedings" for hidden risks.
    
    OUTPUT FORMAT (Markdown):
    - ## Key Investment Risks
    - ### The Bear Case
    - ### Pre-Mortem
    """

    prompts["PORTFOLIO"] = f"""
    You are the CIO.
    {price_context}
    
    TASK: Synthesize final thesis for: {subject}.
    
    GRAND UNIFICATION:
    Compare the Fundamental Core Value vs. Quant Price Targets.
    
    OUTPUT FORMAT (Markdown):
    - ## Executive Thesis (Chain of Density)
    - ### Variant Perception
    - ### Sell Triggers
    - ### Final Verdict
    """
    
    return prompts

FOLLOWUP_PROMPT = "You are a Research Assistant. Answer user questions based ONLY on the provided report context."

# --- 3. GRAPHIC DESIGN ENGINE (HTML/CSS -> PDF) ---

COLOR_NAVY = "#0A1932"
COLOR_GOLD = "#DAA520"
COLOR_LIGHT_BLUE = "#EBF0F5"
COLOR_DARK_GREY = "#323232"

STYLE_CSS = f"""
    @page {{
        size: letter;
        margin: 2cm;
        @frame header_frame {{
            -pdf-frame-content: header_content;
            left: 50pt; width: 512pt; top: 30pt; height: 40pt;
        }}
        @frame footer_frame {{
            -pdf-frame-content: footer_content;
            left: 50pt; width: 512pt; top: 750pt; height: 20pt;
        }}
    }}
    body {{
        font-family: Helvetica, Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: {COLOR_DARK_GREY};
    }}
    
    /* FORCE WRAPPING FOR ALL TEXT BLOCKS */
    pre, code, p, div, span, li {{
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word;
        max-width: 100%;
        font-family: Helvetica, Arial, sans-serif; 
        font-size: 11pt;
    }}
    
    h1 {{
        color: {COLOR_NAVY};
        font-size: 36pt; /* Larger Ticker */
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}
    .company-name {{
        color: {COLOR_NAVY};
        font-size: 18pt; /* Large Company Name */
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
        text-transform: uppercase;
    }}
    h2 {{
        color: {COLOR_NAVY};
        font-size: 16pt;
        border-bottom: 2px solid {COLOR_GOLD};
        padding-bottom: 5px;
        margin-top: 30px;
        text-transform: uppercase;
        font-weight: bold;
    }}
    h3 {{
        color: {COLOR_NAVY};
        font-size: 13pt;
        margin-top: 20px;
        font-weight: bold;
    }}
    blockquote {{
        background-color: #f9f9f9;
        border-left: 5px solid {COLOR_NAVY};
        margin: 1.5em 10px;
        padding: 0.5em 10px;
        font-style: italic;
        color: #555;
    }}
    
    /* TABLE STYLING */
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 10pt;
        border: 1px solid #ddd;
    }}
    th {{
        background-color: {COLOR_NAVY};
        color: {COLOR_GOLD};
        font-weight: bold;
        padding: 8px;
        text-align: center;
        border: 1px solid #ddd;
    }}
    td {{
        padding: 8px;
        border: 1px solid #ddd;
        text-align: right;
    }}
    td:first-child {{
        text-align: left;
    }}
    tr:nth-child(even) {{
        background-color: {COLOR_LIGHT_BLUE};
    }}
    /* Cleaned Executive Section (No Box, Just Header) */
    .executive-section {{
        margin-bottom: 30px;
        padding: 10px 0;
    }}
    
    .disclaimer {{
        font-size: 8pt;
        color: #888;
        margin-top: 50px;
        border-top: 1px solid #ccc;
        padding-top: 10px;
    }}
    .chart-container {{
        text-align: center;
        margin: 20px 0;
    }}
    img {{
        max-width: 100%;
        height: auto;
    }}
"""

def generate_pdf_report(title_text, company_name, report_data, chart_path=None, return_path=None, return_df=None):
    quant_text = report_data.get("Quant", "N/A")
    quant_text = re.sub(r'(\*\*CHART DATA\*\*.*?)?\{.*?"Bear".*?"Bull".*?\}', '', quant_text, flags=re.DOTALL | re.IGNORECASE)
    quant_text = re.sub(r'```json', '', quant_text)
    quant_text = re.sub(r'```', '', quant_text)

    pm_html = markdown.markdown(report_data.get("Portfolio Manager", "N/A"), extensions=['extra'])
    macro_html = markdown.markdown(report_data.get("Macro", "N/A"), extensions=['extra'])
    fund_html = markdown.markdown(report_data.get("Fundamental", "N/A"), extensions=['extra'])
    cfa_html = markdown.markdown(report_data.get("CFA", "N/A"), extensions=['extra'])
    quant_html = markdown.markdown(quant_text, extensions=['extra'])
    red_html = markdown.markdown(report_data.get("Red Team", "N/A"), extensions=['extra'])

    # Cleaned Executive Section (Matches Style)
    sections_html = f"""
    <div class="executive-section">
        <h2>1. EXECUTIVE THESIS (CHAIN OF DENSITY)</h2>
        {pm_html}
    </div>
    """
    
    sections_list = [
        ("2. MACRO-ECONOMIC LANDSCAPE", macro_html, None),
        ("3. FUNDAMENTAL DEEP DIVE", fund_html, None),
        ("4. CFA ANALYSIS: MD&A REVIEW", cfa_html, None),
        ("5. QUANTITATIVE & RISK ANALYSIS", quant_html, chart_path),
        ("6. KEY RISKS (RED TEAM VERDICT)", red_html, None)
    ]
    
    for title, html_content, img_path in sections_list:
        sections_html += f"<h2>{title}</h2>{html_content}"
        
        if "QUANTITATIVE" in title:
            if img_path:
                sections_html += f'<div class="chart-container"><h3>Valuation Scenarios</h3><img src="{img_path}" style="width: 15cm;" /></div>'
            if return_path:
                sections_html += f'<div class="chart-container"><h3>5-Year Performance</h3><img src="{return_path}" style="width: 15cm;" /></div>'
            if return_df is not None:
                sections_html += f"<h3>Historical Return Data</h3>{return_df.reset_index().to_html(index=False, classes='table')}"

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Titan Report</title><style>{STYLE_CSS}</style></head>
    <body>
        <div id="header_content" style="text-align: center; color: {COLOR_NAVY}; font-weight: bold; border-bottom: 2px solid {COLOR_GOLD}; padding-bottom: 5px;">TITAN FINANCIAL ANALYST // EQUITY RESEARCH</div>
        <div id="footer_content" style="text-align: center; color: #888; font-size: 8pt; border-top: 1px solid #ccc; padding-top: 5px;">Strictly Confidential | Generated by Titan AI</div>
        
        <!-- TITLE PAGE -->
        <div style="text-align: center; margin-top: 150px; margin-bottom: 150px;">
            <h1>{title_text}</h1>
            <div class="company-name">{company_name}</div>
            <div style="font-size: 16pt; color: {COLOR_GOLD}; font-weight: bold; margin-bottom: 20px;">INSTITUTIONAL EQUITY RESEARCH</div>
            <div style="font-size: 12pt; color: #555;">Date: {datetime.now().strftime('%B %d, %Y')}</div>
        </div>
        
        <pdf:nextpage />
        {sections_html}
        <div class="disclaimer"><strong>IMPORTANT DISCLOSURES</strong><br>This report is generated by AI (Titan Analyst) and is for informational purposes only.</div>
    </body>
    </html>
    """
    
    pdf_file = BytesIO()
    pisa_status = pisa.CreatePDF(src=full_html, dest=pdf_file)
    if pisa_status.err: return None
    return pdf_file.getvalue()

# --- 4. CHART GENERATORS ---
HEX_NAVY = '#0A1932'
HEX_GOLD = '#DAA520'

def generate_bar_chart(data_dict, title):
    try:
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
        # Check if valid
        data = yf.download(clean_ticker, period="5y", progress=False)['Close']
        if data is None or data.empty: return None

        tickers = [clean_ticker, benchmark]
        data = yf.download(tickers, period="5y", progress=False)['Close']
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
        if bull_target < current_price_val:
            chart_data["Bear"] = round(current_price_val * 0.80, 2)
            chart_data["Base"] = round(current_price_val * 1.10, 2)
            chart_data["Bull"] = round(current_price_val * 1.30, 2)
    except: pass
    return chart_data

def extract_return_data(text):
    return None

# --- 6. AGENT ENGINE (With Retries) ---
async def run_agent(name, prompt, content):
    await asyncio.sleep(random.uniform(0.5, 2.0))
    
    # Retry Loop (3 attempts)
    for attempt in range(3):
        for model_name in MODELS:
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=f"{prompt}\nCONTEXT: {content}",
                    config=types.GenerateContentConfig(temperature=0.2, safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
                    ])
                )
                if response.text: return name, response.text
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    await asyncio.sleep(10 * (attempt + 1))
                else:
                    await asyncio.sleep(5)
                continue # Retry loop
                
    return name, f"Analysis Failed for {name}. Error: {error_msg}"

async def run_analysis(user_input):
    market_data = fetch_market_data(user_input)
    is_ticker = market_data["is_valid_ticker"]
    
    if is_ticker:
        subject = normalize_ticker(user_input)
        current_price = market_data["price"]
        current_yield = market_data["yield"]
    else:
        subject = user_input
        current_price = None
        current_yield = market_data["yield"]
        
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
    await asyncio.sleep(8)
    
    combined_reports = "\n".join([v for k,v in data.items()])
    _, data["Red Team"] = await run_agent("Red Team", prompts["RED_TEAM"], combined_reports)
    
    st.toast("⏳ Cooling down quota for Final Thesis...")
    await asyncio.sleep(8)
    
    combined_reports += f"\n\nRED TEAM VERDICT:\n{data['Red Team']}"
    _, data["Portfolio Manager"] = await run_agent("Portfolio Manager", prompts["PORTFOLIO"], combined_reports)
    
    data["_current_price"] = current_price
    data["_subject"] = subject
    data["_is_ticker"] = is_ticker
    data["_company_name"] = market_data.get("name", subject)
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
                
                pdf_bytes = generate_pdf_report(report_title, company_name, rpt, chart_path=c_path, return_path=r_path, return_df=r_data)
                
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

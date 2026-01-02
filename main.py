import streamlit as st
import google.generativeai as genai
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

genai.configure(api_key=api_key)

# Model Priority: STRICT MODE - GEMINI 3 PRO ONLY
MODELS = [
    "gemini-3-pro-preview"
]

# SAFETY SETTINGS
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- 2. DYNAMIC PROMPT GENERATION ---

def get_dynamic_prompts(ticker):
    """Generates prompts with dynamic dates to ensure analysis is current."""
    now = datetime.now()
    current_date_str = now.strftime("%Y-%m-%d")
    start_date_str = (now - timedelta(days=5*365)).strftime("%Y-%m-%d")
    current_year = now.year
    
    # Fiscal Year Logic: If we are in Jan/Feb, the last full FY is usually 2 years ago in calendar terms until filing.
    # But for search, we just ask for the "Latest Available".
    
    prompts = {}

    prompts["MACRO"] = f"""
    You are the Macro Council.
    TASK: Analyze the global macro environment for {ticker} as of {current_date_str}.
    1. Identify the current economic regime (e.g., Inflationary, Stagflation, Growth).
    2. Analyze 3 scenarios (Bull/Bear/Base) for the next 12-24 months.
    OUTPUT FORMAT (Markdown):
    - ## Executive Summary
    - ### Key Indicators (Table)
    - ### Scenario Analysis
    """

    prompts["FUNDAMENTAL"] = f"""
    You are the Fundamental Specialist.
    TASK: Analyze {ticker}'s business health using the latest SEC filings available before {current_date_str}.
    **SEC EDGAR PROTOCOL:**
    1. Simulate accessing SEC.gov. Look for the latest 10-K and 10-Q filed closest to {current_date_str}.
    2. Analyze Item 1A (Risk Factors) and Item 7 (MD&A).
    OUTPUT FORMAT (Markdown):
    - ## Business Health Analysis
    - ### Unit Economics (Table)
    - ### Moat Analysis
    """

    prompts["CFA"] = f"""
    You are a CFA Charterholder.
    TASK: Forensic review of {ticker}'s MD&A section from the latest filings ({current_year} or {current_year-1}).
    Check for: Margin trends, Non-GAAP vs GAAP gaps, and Liquidity constraints.
    OUTPUT FORMAT (Markdown):
    - ## CFA Analysis: MD&A Review
    - **Margin Analysis:** [Details]
    - **Non-GAAP Reconciliations:** [Details]
    - **Liquidity:** [Details]
    """

    prompts["QUANT"] = f"""
    You are the Quant Desk.
    TASK: Analyze valuation and risk for {ticker} as of {current_date_str}.
    CRITICAL OUTPUT:
    1. **Markdown Table**: Valuation Metrics (P/E, PEG, FCF Yield).
    2. **CHART DATA**: Output a JSON block at the very end for 12-month Price Targets.
       Format: {{"Bear": 100, "Base": 150, "Bull": 200}}
    """

    prompts["RED_TEAM"] = f"""
    You are the Red Team.
    TASK: Review the reports for {ticker}. Identify fatal flaws and "Grey Rhino" risks as of {current_date_str}.
    OUTPUT FORMAT (Markdown):
    - ## Key Investment Risks
    - ### The Short Case
    - ### Pre-Mortem
    """

    prompts["PORTFOLIO"] = f"""
    You are the CIO.
    TASK: Synthesize a final Investment Thesis for {ticker} based on the reports.
    OUTPUT FORMAT (Markdown):
    - ## Executive Thesis
    - ### Variant Perception
    - ### Sell Triggers
    - ### Final Verdict
    """
    
    return prompts

FOLLOWUP_PROMPT = """
You are a Research Assistant. Answer user questions based ONLY on the provided report context.
"""

# --- 3. GRAPHIC DESIGN ENGINE (HTML/CSS -> PDF) ---

# Titan Color Palette (CSS Hex)
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
        font-family: 'Times New Roman', Times, serif;
        font-size: 11pt;
        line-height: 1.6;
        color: {COLOR_DARK_GREY};
    }}
    h1 {{
        color: {COLOR_NAVY};
        font-size: 24pt;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
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
    /* Blockquote for Personas */
    blockquote {{
        background-color: #f9f9f9;
        border-left: 5px solid {COLOR_NAVY};
        margin: 1.5em 10px;
        padding: 0.5em 10px;
        font-style: italic;
        color: #555;
    }}
    /* Table Styling */
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
    .executive-box {{
        background-color: #F4F7FA;
        border: 1px solid {COLOR_NAVY};
        padding: 15px;
        margin-bottom: 20px;
    }}
    .executive-title {{
        color: {COLOR_GOLD};
        font-weight: bold;
        font-size: 14pt;
        margin-bottom: 10px;
        display: block;
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

def generate_pdf_report(ticker, report_data, chart_path=None, return_path=None, return_df=None):
    # 1. Clean Quant Data (Remove Raw JSON)
    quant_text = report_data.get("Quant", "N/A")
    quant_text = re.sub(r'\{.*?"Bear".*?"Bull".*?\}', '', quant_text, flags=re.DOTALL | re.IGNORECASE)
    quant_text = re.sub(r'```json', '', quant_text)
    quant_text = re.sub(r'```', '', quant_text)

    # 2. Convert Markdown to HTML
    sections_html = ""
    
    # Portfolio Manager (Executive Box)
    pm_md = report_data.get("Portfolio Manager", "N/A")
    pm_html = markdown.markdown(pm_md, extensions=['tables', 'fenced_code'])
    sections_html += f"""
    <div class="executive-box">
        <span class="executive-title">1. EXECUTIVE THESIS</span>
        {pm_html}
    </div>
    """
    
    # Standard Sections
    section_map = [
        ("2. MACRO-ECONOMIC LANDSCAPE", "Macro"),
        ("3. FUNDAMENTAL DEEP DIVE", "Fundamental"),
        ("4. CFA ANALYSIS: MD&A REVIEW", "CFA"),
        ("5. QUANTITATIVE & RISK ANALYSIS", "Quant", quant_text),
        ("6. KEY RISKS (RED TEAM VERDICT)", "Red Team")
    ]
    
    for title, key, *rest in section_map:
        text_content = rest[0] if rest else report_data.get(key, "N/A")
        html_content = markdown.markdown(text_content, extensions=['tables', 'fenced_code'])
        sections_html += f"<h2>{title}</h2>{html_content}"
        
        # Inject Charts into Quant Section
        if key == "Quant":
            if chart_path:
                sections_html += f'<div class="chart-container"><h3>Valuation Scenarios (12-Month Targets)</h3><img src="{chart_path}" style="width: 15cm;" /></div>'
            if return_path:
                sections_html += f'<div class="chart-container"><h3>5-Year Total Return Comparison (Growth of $10k)</h3><img src="{return_path}" style="width: 15cm;" /></div>'

    # 3. Assemble Full HTML
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Titan Analyst Report - {ticker}</title>
        <style>{STYLE_CSS}</style>
    </head>
    <body>
        <!-- Header Frame Content -->
        <div id="header_content" style="text-align: center; color: {COLOR_NAVY}; font-weight: bold; border-bottom: 2px solid {COLOR_GOLD}; padding-bottom: 5px;">
            TITAN FINANCIAL ANALYST // INSTITUTIONAL RESEARCH
        </div>
        
        <!-- Footer Frame Content -->
        <div id="footer_content" style="text-align: center; color: #888; font-size: 8pt; border-top: 1px solid #ccc; padding-top: 5px;">
            Strictly Confidential | Generated by Titan AI
        </div>

        <!-- Title Page Content -->
        <div style="text-align: center; margin-top: 100px; margin-bottom: 100px;">
            <h1>{ticker}</h1>
            <div style="font-size: 16pt; color: {COLOR_GOLD}; font-weight: bold; margin-bottom: 20px;">
                INSTITUTIONAL EQUITY RESEARCH
            </div>
            <div style="font-size: 12pt; color: #555;">
                Date: {datetime.now().strftime('%B %d, %Y')}
            </div>
        </div>
        
        <pdf:nextpage />

        {sections_html}

        <div class="disclaimer">
            <strong>IMPORTANT DISCLOSURES & DISCLAIMER</strong><br>
            This report is generated by an AI system (Titan Analyst) and is for informational purposes only. It does not constitute financial advice.
        </div>
    </body>
    </html>
    """
    
    # 4. Generate PDF
    pdf_file = BytesIO()
    pisa_status = pisa.CreatePDF(src=full_html, dest=pdf_file)
    
    if pisa_status.err:
        return None
    
    return pdf_file.getvalue()

# --- 4. DATA & CHART LOGIC (Robust) ---
def fetch_relative_returns(ticker, benchmark="SPY"):
    try:
        tickers = [ticker, benchmark]
        # Download Data (5 Years default)
        data = yf.download(tickers, period="5y", progress=False)['Close']
        
        if data is None or data.empty:
            return None
            
        # Data Alignment (Inner Join via Dropna)
        aligned_data = data.dropna()
        
        if aligned_data.empty: return None

        # Normalize: (Price / Start_Price) - 1
        normalized = (aligned_data / aligned_data.iloc[0]) - 1
        return normalized
    except Exception as e:
        return None

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
            ax.text(bar.get_x() + bar.get_width()/2., height, f'${height:,.0f}', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color=HEX_GOLD)
        plt.tight_layout()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig)
            return tmp.name
    except: return None

def generate_line_chart(df, title):
    try:
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        col_names = list(df.columns)
        
        # Determine colors based on column names logic or index
        # We assume Target is col 0, Benchmark is col 1 if present
        
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

# --- 5. AGENT ENGINE ---
async def run_agent(name, prompt, content):
    await asyncio.sleep(random.uniform(0.5, 2.0))
    for model_name in MODELS:
        try:
            model = genai.GenerativeModel(model_name, tools='google_search', safety_settings=SAFETY_SETTINGS)
            response = await asyncio.wait_for(model.generate_content_async(f"{prompt}\nCONTEXT: {content}"), timeout=90.0)
            if response.text: return name, response.text
        except Exception as e:
            if "429" in str(e):
                await asyncio.sleep(15)
            # Fallback
            try:
                model_pure = genai.GenerativeModel(model_name, safety_settings=SAFETY_SETTINGS)
                response = await asyncio.wait_for(model_pure.generate_content_async(f"{prompt}\nCONTEXT: {content}"), timeout=90.0)
                if response.text: return name, response.text
            except: continue
    return name, f"Analysis Failed for {name}."

async def run_analysis(ticker):
    # GET DYNAMIC PROMPTS
    prompts = get_dynamic_prompts(ticker)
    
    tasks = [
        run_agent("Macro", prompts["MACRO"], ticker),
        run_agent("Fundamental", prompts["FUNDAMENTAL"], ticker),
        run_agent("CFA", prompts["CFA"], ticker),
        run_agent("Quant", prompts["QUANT"], ticker)
    ]
    results = await asyncio.gather(*tasks)
    data = {k: v for k, v in results}
    
    if any("Analysis Failed" in str(v) for v in data.values()): return data

    st.toast("⏳ Cooling down quota for Red Team analysis...")
    await asyncio.sleep(5)
    _, data["Red Team"] = await run_agent("Red Team", prompts["RED_TEAM"], str(data))
    
    st.toast("⏳ Cooling down quota for Final Thesis...")
    await asyncio.sleep(5)
    _, data["Portfolio Manager"] = await run_agent("Portfolio Manager", prompts["PORTFOLIO"], str(data))
    return data

# --- 6. UI ---
def main():
    st.set_page_config(page_title="Titan 2.0", layout="wide")
    st.title("⚡ Titan Analyst 2.0")
    
    if "report" not in st.session_state: st.session_state.report = None
    if "history" not in st.session_state: st.session_state.history = []

    with st.form(key='analysis_form'):
        ticker = st.text_input("Enter Ticker:", "NVDA")
        submit_button = st.form_submit_button(label='Run Analysis')
    
    if submit_button:
        with st.spinner("Initializing Titan Agents..."):
            st.session_state.report = asyncio.run(run_analysis(ticker))
    
    if st.session_state.report:
        rpt = st.session_state.report
        
        failed = [k for k, v in rpt.items() if "Analysis Failed" in str(v)]
        if failed:
            st.error(f"🚨 Analysis Incomplete. Failures in: {', '.join(failed)}")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("🏆 Executive Thesis")
                st.info(rpt.get("Portfolio Manager"))
            with c2:
                st.subheader("🎯 12-Month Targets")
                chart_data = extract_chart_data(rpt.get("Quant", ""))
                if chart_data: st.bar_chart(chart_data)
                else: st.caption("No targets found.")

            t1, t2, t3, t4, t5 = st.tabs(["Macro", "Fundamental", "CFA", "Quant", "Red Team"])
            with t1: st.markdown(rpt.get("Macro", ""))
            with t2: st.markdown(rpt.get("Fundamental", ""))
            with t3: st.markdown(rpt.get("CFA", ""))
            with t4: 
                st.markdown(rpt.get("Quant", ""))
                st.divider()
                st.subheader("📈 5-Year Relative Return")
                ret_data = fetch_relative_returns(ticker)
                if ret_data is not None: st.line_chart(ret_data)
            with t5: st.error(rpt.get("Red Team", ""))
            
            st.divider()
            try:
                c_data = extract_chart_data(rpt.get("Quant", ""))
                
                # Fetch Real Data for PDF Chart (Get fresh data for chart)
                r_data = fetch_relative_returns(ticker)
                
                # Generate Chart Images
                c_path = generate_bar_chart(c_data, "Price Targets") if c_data else None
                
                start_year = (datetime.now() - timedelta(days=5*365)).year
                current_year = datetime.now().year
                r_path = generate_line_chart(r_data, f"{start_year}-{current_year} Total Return vs Benchmark") if r_data is not None else None
                
                pdf_bytes = generate_pdf_report(ticker, rpt, chart_path=c_path, return_path=r_path)
                st.download_button("📄 Download Professional Report (PDF)", pdf_bytes, f"{ticker}_Titan_Report.pdf", "application/pdf")
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

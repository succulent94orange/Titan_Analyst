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
from fpdf import FPDF
from io import BytesIO

# --- 1. SETUP ---
# Set Matplotlib to non-interactive mode to prevent crashes
matplotlib.use('Agg')

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("🚨 CRITICAL ERROR: GOOGLE_API_KEY not found in .env file.")
    st.stop()

# Initialize Client
client = genai.Client(api_key=api_key)

# --- STRICT MODEL ENFORCEMENT ---
MODELS = ["gemini-3-pro-preview"] 

# --- 2. DYNAMIC PROMPT GENERATION ---

def normalize_ticker(user_input):
    # Try to find a ticker-like pattern (1-5 caps)
    match = re.search(r'\b[A-Z]{1,5}\b', user_input)
    if match and len(user_input.split()) < 3: # Assuming short inputs are tickers
        return match.group(0).upper().strip()
    return user_input.strip() # Return full string for generic queries

def fetch_market_data(ticker):
    """Fetches Stock Price, Name, and Treasury Yield."""
    clean_ticker = normalize_ticker(ticker)
    data_pkg = {"price": None, "yield": None, "is_valid_ticker": False, "name": ticker}
    
    try:
        # Fetch Stock Data
        stock = yf.Ticker(clean_ticker)
        # Check if it actually has history (validates if it's a ticker)
        hist = stock.history(period="5d")
        
        if not hist.empty:
            data_pkg["price"] = float(hist['Close'].iloc[-1])
            data_pkg["is_valid_ticker"] = True
            try:
                info = stock.info
                data_pkg["name"] = info.get('longName', clean_ticker)
            except:
                data_pkg["name"] = clean_ticker
        else:
            # If no history, treat as generic subject
            data_pkg["is_valid_ticker"] = False
            data_pkg["name"] = ticker
        
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
        context_parts.append(f"- RULE: All price targets MUST be based on ${current_price:.2f}.")
    else:
         context_parts.append(f"- NOTE: This is a General Finance Inquiry (Not a specific stock ticker).")
         context_parts.append(f"- RULE: Focus on conceptual analysis, strategies, and definitions.")
         
    if current_yield is not None:
        context_parts.append(f"- RISK-FREE RATE (10Y Treasury): {current_yield:.2f}%")
        
    price_context = "\n".join(context_parts)
    
    prompts = {}
    
    guardrail = "GUARDRAIL: Use your Google Search Tool to find the latest data."
    sec_instruction = "**SEC EDGAR PROTOCOL:** Simulate accessing SEC.gov 10-K/10-Q filings." if is_ticker else "Use general financial knowledge."
    
    # 1. MACRO AGENT
    prompts["MACRO"] = f"""
    You are the Macro Council.
    {price_context}
    {guardrail}
    
    TASK: Analyze the global macro environment regarding: {subject}.
    
    **MANDATORY FORMATTING:**
    Do NOT use Markdown Tables. Use a clean bulleted list for data.
    
    REQUIRED DATA (Find recent figures):
    - Inflation Rate (CPI): [Value]
    - GDP Growth Rate: [Value]
    - Interest Rates (Fed Funds): [Value]

    OUTPUT FORMAT:
    - Executive Summary
    - Key Indicators (Bulleted List)
    - Scenario Analysis (Inflation, Soft Landing, Deflation)
    """

    # 2. FUNDAMENTAL AGENT
    prompts["FUNDAMENTAL"] = f"""
    You are the Fundamental Specialist.
    {price_context}
    {guardrail}
    
    TASK: Analyze the fundamentals/core concepts of: {subject}.
    {sec_instruction}
    
    **CRITICAL FORMATTING INSTRUCTION:**
    Do NOT use Markdown Tables (no pipes '|').
    Format the output as **Structured Lists**.
    
    **SWOT Format (If applicable, or Pros/Cons):**
    **Strengths/Pros:**
    - [Point 1]
    
    **Weaknesses/Cons:**
    - [Point 1]
    
    **Opportunities/Use Cases:**
    - [Point 1]
    
    **Threats/Risks:**
    - [Point 1]
    
    OUTPUT FORMAT:
    - Fundamental Analysis
    - Pros & Cons (List Format)
    - Strategic Advantage
    """

    # 3. CFA AGENT
    prompts["CFA"] = f"""
    You are a CFA Charterholder.
    {price_context}
    {guardrail}
    
    TASK: Conduct a professional analysis adhering to CFA standards.
    Focus on Risks, Mechanics, and Financial Implications.
    
    OUTPUT FORMAT:
    - CFA Analysis
    - Key Insights
    - Risk Factors
    """

    # 4. QUANT AGENT
    prompts["QUANT"] = f"""
    You are the Quant Desk.
    {price_context}
    {guardrail}
    
    TASK: Analyze quantitative data/valuation.
    
    **CRITICAL INSTRUCTION FOR CHART DATA:**
    1. IF this is a specific STOCK TICKER, output a JSON block for price targets:
    ```json
    {{
        "Bear": <price_number>,
        "Base": <price_number>,
        "Bull": <price_number>
    }}
    ```
    2. IF this is a GENERIC QUESTION (e.g., "What is a bond?"), DO NOT output the JSON block. Just provide quantitative data/stats in text.
    
    OUTPUT FORMAT:
    - Quantitative Analysis (Text)
    - **CHART DATA** (JSON block ONLY if valid stock ticker)
    """

    # 5. RED TEAM AGENT
    prompts["RED_TEAM"] = f"""
    You are the Red Team (Jim Chanos).
    {price_context}
    {guardrail}
    
    TASK: Find fatal flaws, downsides, and "Grey Rhino" risks regarding {subject}.
    
    OUTPUT FORMAT:
    - Key Risks
    - The Bear Case / Downside Scenario
    - Pre-Mortem (Assume failure/loss in 5 years)
    """

    # 6. PORTFOLIO MANAGER AGENT
    prompts["PORTFOLIO"] = f"""
    You are the CIO.
    {price_context}
    {guardrail}
    
    TASK: Synthesize final thesis for: {subject}.
    
    **MANDATORY OUTPUT REQUIREMENT:**
    The VERY LAST line of your response MUST be exactly one of these:
    VERDICT: BUY
    VERDICT: SELL
    VERDICT: HOLD
    (If generic, use "VERDICT: HOLD" or closest equivalent action).
    
    OUTPUT FORMAT:
    - Executive Thesis
    - Variant Perception
    - Final Verdict (Text explanation)
    - VERDICT: [BUY/SELL/HOLD]
    """
    
    return prompts

FOLLOWUP_PROMPT = """
You are a Senior Research Analyst. 
The user is asking a follow-up question about the report you just generated.
Use the provided REPORT CONTEXT and your Search Tool to answer the user's question.
Be concise, direct, and helpful.
"""

# --- 3. FPDF REPORT ENGINE ---

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(10, 25, 50) # Navy
        self.cell(0, 10, 'TITAN FINANCIAL ANALYST // EQUITY RESEARCH', border=0, align='C', ln=1)
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
        self.set_text_color(10, 25, 50)
        self.cell(0, 10, label, ln=1)
        self.set_draw_color(218, 165, 32)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)

    def chapter_body(self, text):
        self.set_font('Helvetica', '', 11)
        self.set_text_color(50, 50, 50)
        
        # --- TEXT SANITIZATION ---
        replacements = {
            u'\u201c': '"', u'\u201d': '"',  # Smart quotes
            u'\u2018': "'", u'\u2019': "'",
            u'\u2013': '-', u'\u2014': '-',  # Dashes
            u'\u2026': '...',                # Ellipsis
            u'\u2022': '-',                  # Bullets
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        
        # Remove any remaining non-latin characters (like emojis)
        text = text.encode('latin-1', 'replace').decode('latin-1')
        
        self.multi_cell(0, 6, text)
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
    
    # Verdict Box
    pdf.ln(20)
    pdf.set_draw_color(218, 165, 32)
    pdf.set_fill_color(245, 245, 245)
    pdf.rect(60, pdf.get_y(), 90, 35, 'FD')
    
    pdf.set_y(pdf.get_y() + 5)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "RECOMMENDATION", align='C', ln=1)
    
    if "BUY" in verdict.upper(): pdf.set_text_color(0, 100, 0)
    elif "SELL" in verdict.upper(): pdf.set_text_color(150, 0, 0)
    else: pdf.set_text_color(218, 165, 32)
        
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 10, verdict.upper(), align='C', ln=1)
    
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(50, 50, 50)
    if price_target != "N/A":
        pdf.cell(0, 10, f"12-MONTH TARGET: ${price_target}", align='C', ln=1)
    else:
        pdf.cell(0, 10, "TARGET: N/A", align='C', ln=1)
        
    pdf.ln(20)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(218, 165, 32)
    pdf.cell(0, 10, "INSTITUTIONAL EQUITY RESEARCH", align='C', ln=1)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%B %d, %Y')}", align='C', ln=1)
    
    # --- SECTIONS ---
    def clean_md(text):
        if not text: return "N/A"
        # Remove bolding/headers for PDF cleanliness
        text = text.replace("VERDICT: BUY", "").replace("VERDICT: SELL", "").replace("VERDICT: HOLD", "")
        text = text.replace('**', '').replace('##', '').replace('###', '')
        return text.strip()

    sections = [
        ("1. EXECUTIVE THESIS", "Portfolio Manager"),
        ("2. MACRO-ECONOMIC LANDSCAPE", "Macro"),
        ("3. FUNDAMENTAL DEEP DIVE", "Fundamental"),
        ("4. CFA ANALYSIS", "CFA")
    ]

    for title, key in sections:
        pdf.add_page()
        pdf.chapter_title(title)
        pdf.chapter_body(clean_md(report_data.get(key, "")))

    # 5. Quant
    pdf.add_page()
    pdf.chapter_title("5. QUANTITATIVE & RISK")
    quant_text = report_data.get("Quant", "")
    quant_text = re.sub(r'```json.*?```', '', quant_text, flags=re.DOTALL)
    quant_text = re.sub(r'\{.*?"Bear".*?"Bull".*?\}', '', quant_text, flags=re.DOTALL)
    pdf.chapter_body(clean_md(quant_text))

    if chart_path:
        pdf.ln(5)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 10, "Valuation Scenarios (12mo Targets)", ln=1)
        try:
            pdf.image(chart_path, x=30, w=150)
            pdf.ln(5)
        except: pass

    if return_path:
        pdf.ln(5)
        pdf.cell(0, 10, "5-Year Relative Performance", ln=1)
        try:
            pdf.image(return_path, x=30, w=150)
            pdf.ln(10)
        except: pass

    # 6. Red Team
    pdf.add_page()
    pdf.chapter_title("6. KEY RISKS (RED TEAM)")
    pdf.set_text_color(150, 0, 0)
    pdf.chapter_body(clean_md(report_data.get("Red Team", "")))

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
    except: return None

# --- 5. DATA HELPERS ---
def fetch_relative_returns(ticker, benchmark="SPY"):
    try:
        clean_ticker = normalize_ticker(ticker)
        tickers = [clean_ticker, benchmark]
        data = yf.download(tickers, period="5y", progress=False)['Close']
        if data is None or data.empty: return None
        data = data.apply(pd.to_numeric, errors='coerce').dropna()
        if data.empty: return None
        normalized = (data / data.iloc[0]) - 1
        return normalized
    except: return None

def extract_chart_data(text):
    """Robust extraction of Bear/Base/Bull JSON."""
    data = {}
    try:
        json_match = re.search(r'\{.*?"Bear".*?"Bull".*?\}', text, re.DOTALL | re.IGNORECASE)
        if json_match:
            try:
                clean_json = json_match.group(0).replace("```json", "").replace("```", "")
                data = json.loads(clean_json)
                return {k.capitalize(): float(v) for k, v in data.items() if k.capitalize() in ["Bear", "Base", "Bull"]}
            except: pass
        
        matches = re.findall(r'"(Bear|Base|Bull)":\s*(\d+\.?\d*)', text, re.IGNORECASE)
        for label, value in matches:
            data[label.capitalize()] = float(value)
    except: pass
    return data if len(data) >= 2 else None

def verify_and_correct_targets(chart_data, current_price):
    if not chart_data or current_price is None: return chart_data
    try:
        cp = float(current_price) 
        bull = float(chart_data.get("Bull", 0))
        if bull == 0 or (bull < cp * 0.5) or (bull > cp * 3.0):
            chart_data["Bear"] = round(cp * 0.80, 2)
            chart_data["Base"] = round(cp * 1.10, 2)
            chart_data["Bull"] = round(cp * 1.30, 2)
    except: pass
    return chart_data

def extract_verdict(text):
    if not text: return "HOLD"
    match = re.search(r'VERDICT:\s*(BUY|SELL|HOLD)', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    return "HOLD"

# --- 6. ASYNC AGENT EXECUTION ---
async def run_agent(name, prompt, content):
    await asyncio.sleep(random.uniform(0.5, 2.0))
    for attempt in range(3):
        for model_name in MODELS:
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=f"{prompt}\nCONTEXT: {content}",
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        tools=[types.Tool(google_search=types.GoogleSearch())]
                    )
                )
                if response.text: return name, response.text
            except Exception as e:
                if "429" in str(e): await asyncio.sleep(5)
                else: await asyncio.sleep(2)
                continue 
    return name, f"Analysis Failed for {name}."

async def run_analysis(user_input):
    market_data = fetch_market_data(user_input)
    is_ticker = market_data["is_valid_ticker"]
    
    # 1. GENERATE PROMPTS
    prompts = get_dynamic_prompts(
        market_data.get("name", user_input), 
        market_data["price"], 
        market_data["yield"], 
        is_ticker
    )
    
    # 2. RUN PARALLEL AGENTS
    tasks = [
        run_agent("Macro", prompts["MACRO"], user_input),
        run_agent("Fundamental", prompts["FUNDAMENTAL"], user_input),
        run_agent("CFA", prompts["CFA"], user_input),
        run_agent("Quant", prompts["QUANT"], user_input)
    ]
    results = await asyncio.gather(*tasks)
    data = {k: v for k, v in results}
    
    # 3. RED TEAM & FINAL THESIS
    st.toast("⏳ Running Red Team Risk Analysis...")
    combined = "\n".join([v for k,v in data.items() if not k.startswith("_")])
    _, data["Red Team"] = await run_agent("Red Team", prompts["RED_TEAM"], combined)
    
    st.toast("⏳ Synthesizing Final Thesis...")
    combined += f"\n\nRED TEAM: {data['Red Team']}"
    _, data["Portfolio Manager"] = await run_agent("Portfolio Manager", prompts["PORTFOLIO"], combined)
    
    # 4. METADATA
    data["_current_price"] = market_data["price"]
    data["_subject"] = user_input
    data["_is_ticker"] = is_ticker
    data["_company_name"] = market_data.get("name", user_input)
    data["_verdict"] = extract_verdict(data["Portfolio Manager"])
    
    q_data = extract_chart_data(data["Quant"])
    data["_price_target"] = q_data.get("Base", "N/A") if q_data else "N/A"
    
    return data

# --- 7. UI MAIN ---
def main():
    st.set_page_config(page_title="Titan 2.5", layout="wide")
    st.title("⚡ Titan Analyst 2.5")
    
    if "report" not in st.session_state: st.session_state.report = None
    if "history" not in st.session_state: st.session_state.history = []
    
    # --- INPUT SECTION ---
    with st.form("input_form"):
        user_input = st.text_input("Enter Ticker or Financial Question:")
        submitted = st.form_submit_button("Run Analysis")
        
    if submitted:
        with st.spinner("Titan Agents Working..."):
            st.session_state.report = asyncio.run(run_analysis(user_input))
            
    # --- REPORT DISPLAY ---
    if st.session_state.report:
        rpt = st.session_state.report
        
        # Dashboard
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("🏆 Executive Thesis")
            st.info(rpt.get("Portfolio Manager", "N/A"))
            st.metric("Titan Verdict", rpt.get("_verdict"))
        with c2:
            if rpt.get("_is_ticker"):
                st.subheader("🎯 12-Month Targets")
                c_data = extract_chart_data(rpt.get("Quant", ""))
                c_data = verify_and_correct_targets(c_data, rpt.get("_current_price"))
                if c_data: st.bar_chart(c_data)
                else: st.warning("No Targets Generated")
            else:
                st.subheader("ℹ️ General Inquiry")
                st.caption("No Price Targets for Generic Questions")

        # Tabs
        tabs = st.tabs(["Macro", "Fundamental", "CFA", "Quant", "Red Team"])
        keys = ["Macro", "Fundamental", "CFA", "Quant", "Red Team"]
        for t, k in zip(tabs, keys):
            with t: st.markdown(rpt.get(k, ""))
            
        # PDF Generation
        st.divider()
        try:
            # Generate Assets
            c_data = extract_chart_data(rpt.get("Quant", ""))
            c_path = generate_bar_chart(c_data, "Price Targets") if c_data else None
            
            r_data = fetch_relative_returns(rpt["_subject"])
            r_path = generate_line_chart(r_data, "5-Year Performance") if r_data is not None else None
            
            pdf_bytes = generate_pdf_report(
                rpt["_subject"], 
                rpt["_company_name"], 
                rpt, 
                c_path, 
                r_path, 
                None, 
                rpt["_verdict"], 
                str(rpt["_price_target"])
            )
            # 1. Create the dynamic filename using the subject (User Input)
            # We use safe string formatting to ensure the file name is clean
            clean_subject = rpt["_subject"].strip().upper() 
            dynamic_filename = f"{clean_subject} - Titan Analyst.pdf"

            # 2. Pass the dynamic filename to the button
            st.download_button(
                label="📄 Download PDF Report", 
                data=pdf_bytes, 
                file_name=dynamic_filename, # <--- This is where the change happens
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"PDF Error: {e}")
            
    # --- CHAT / FOLLOW-UP SECTION ---
    st.divider()
    st.subheader("💬 Chat with Analyst")
    
    # Display Chat History
    for msg in st.session_state.history:
        st.chat_message(msg["role"]).write(msg["content"])
        
    # Input for Follow-up
    if q := st.chat_input("Ask a follow-up question..."):
        st.chat_message("user").write(q)
        st.session_state.history.append({"role": "user", "content": q})
        
        # Run Chat Agent
        with st.spinner("Analyzing..."):
            # Provide full report context + User Query
            context_str = f"FULL REPORT DATA: {st.session_state.report}\nUSER QUESTION: {q}"
            _, ans = asyncio.run(run_agent("Chat", FOLLOWUP_PROMPT, context_str))
            
            st.chat_message("assistant").write(ans)
            st.session_state.history.append({"role": "assistant", "content": ans})

if __name__ == "__main__":
    main()

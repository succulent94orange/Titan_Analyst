# --- START OF FILE main.py ---
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import asyncio
import random
import json
import re
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tempfile
from fpdf import FPDF

# 1. SETUP
# Set Matplotlib to non-interactive mode to prevent crashes
matplotlib.use('Agg')

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("🚨 CRITICAL ERROR: GOOGLE_API_KEY not found in .env file.")
    st.stop()

genai.configure(api_key=api_key)

# Model Priority: Try 3.0 -> 2.5 Pro -> 2.5 Flash -> 2.0 Flash
MODELS = ["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]

# --- 2. PROMPT DEFINITIONS (The Brains) ---
MACRO_PROMPT = """
You are the Macro Council (Voices: Ray Dalio, Stanley Druckenmiller).
TASK: Analyze the global macro environment for the target asset.
OUTPUT FORMAT:
- Executive Summary
- **Markdown Table**: Key Indicators (Inflation, Rates, GDP)
- Scenario Probabilities (Bull/Bear/Base)
"""

FUNDAMENTAL_PROMPT = """
You are the Fundamental Specialist (Voices: Peter Lynch, Warren Buffett).
TASK: Analyze the business health, revenue quality, and moat.
**SEC EDGAR PROTOCOL:**
1. Simulate accessing SEC.gov to check 10-K Risk Factors & MD&A.
2. Verify management claims.
OUTPUT FORMAT:
- Business Health Analysis
- **Markdown Table**: Unit Economics & KPIs
- Moat Analysis (Porter's 5 Forces)
"""

QUANT_PROMPT = """
You are the Quant Desk (Voices: Jim Simons).
TASK: Analyze valuation and risk.
CRITICAL OUTPUT:
1. **Markdown Table**: Valuation Metrics (P/E, PEG).
2. **CHART DATA** (Price Targets). Output a JSON block at the end:
   {"Bear": 100, "Base": 150, "Bull": 200}
3. **RETURN DATA** (5-Year Comparison vs SPY). Output a JSON block:
   {"Years": ["2020", "2021", "2022", "2023", "2024"], "Ticker": [10, 20, -5, 15, 30], "SPY": [15, 25, -18, 24, 12]}
"""

RED_TEAM_PROMPT = """
You are the Red Team (Voice: Jim Chanos).
TASK: Review the reports below. Find contradictions, hallucinations, and fatal risks.
SEC FORENSICS: Check Legal Proceedings and Related Party Transactions.
"""

PORTFOLIO_PROMPT = """
You are the CIO.
TASK: Synthesize the Red Team's verdict into a final Executive Thesis (Chain of Density).
"""

FOLLOWUP_PROMPT = """
You are a Research Assistant. Answer user questions based ONLY on the provided report context.
"""

# --- 3. GRAPHIC DESIGN ENGINE (Professional Sell-Side PDF) ---

COLOR_PRIMARY = (10, 25, 50)      # Deep Navy
COLOR_ACCENT = (218, 165, 32)     # Goldenrod
COLOR_BG_HEADER = (235, 240, 245) # Light Blue Grey
COLOR_TEXT_MAIN = (50, 50, 50)    # Dark Grey

def clean_text_for_pdf(text):
    """Sanitizes text to prevent PDF crashes."""
    if not isinstance(text, str): return str(text)
    replacements = {"’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."}
    for k, v in replacements.items(): text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

class TitanPDF(FPDF):
    def header(self):
        self.set_fill_color(*COLOR_PRIMARY)
        self.rect(0, 0, 210, 20, 'F')
        self.set_font('Arial', 'B', 12)
        self.set_text_color(*COLOR_ACCENT)
        self.set_xy(10, 5)
        self.cell(0, 10, 'TITAN FINANCIAL ANALYST // EQUITY RESEARCH', 0, 0, 'L')
        self.ln(25)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Strictly Confidential - For Professional Investors Only | Page {self.page_no()}', 0, 0, 'C')

    def add_title_page(self, ticker):
        self.add_page()
        self.ln(30)
        self.set_font('Arial', 'B', 48)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 20, clean_text_for_pdf(ticker), 0, 1, 'C')
        self.set_font('Arial', '', 16)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'INITIATION OF COVERAGE', 0, 1, 'C')
        self.ln(10)
        
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(*COLOR_PRIMARY)
        self.rect(65, 100, 80, 25, 'DF')
        self.set_xy(65, 105)
        self.set_font('Arial', 'B', 12)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(80, 8, "RATING: NEUTRAL", 0, 1, 'C')
        self.set_xy(65, 115)
        self.cell(80, 8, "TARGET: SEE RPT", 0, 1, 'C')
        
        self.ln(40)
        self.set_font('Arial', 'B', 12)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 10, f"Date: {time.strftime('%B %d, %Y')}", 0, 1, 'C')
        self.add_page()

    def section_header(self, title):
        self.ln(8)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(*COLOR_PRIMARY)
        self.set_fill_color(*COLOR_BG_HEADER)
        self.cell(0, 10, f"  {clean_text_for_pdf(title).upper()}", 0, 1, 'L', 1)
        self.ln(4)

    def body_text(self, text):
        self.set_font('Arial', '', 10)
        self.set_text_color(*COLOR_TEXT_MAIN)
        self.multi_cell(0, 5, clean_text_for_pdf(text))
        self.ln(3)

    def executive_box(self, text):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(*COLOR_ACCENT)
        self.cell(0, 10, "INVESTMENT THESIS", 0, 1, 'L')
        self.ln(2)
        self.set_font('Arial', 'I', 10)
        self.set_text_color(*COLOR_TEXT_MAIN)
        self.set_fill_color(250, 250, 250)
        self.set_draw_color(*COLOR_ACCENT)
        self.set_line_width(0.5)
        self.multi_cell(0, 6, clean_text_for_pdf(text), 1, 'L', 1)
        self.ln(5)

    def add_chart_image(self, image_path, title):
        if not image_path: return
        self.ln(5)
        self.set_font('Arial', 'B', 10)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 10, title, 0, 1, 'C')
        x_centered = (210 - 150) / 2
        try:
            self.image(image_path, x=x_centered, w=150)
        except:
            self.cell(0, 10, "[Chart Image Error]", 0, 1, 'C')
        self.ln(5)

    def draw_table(self, df):
        self.set_font('Arial', 'B', 9)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(240, 240, 240)
        cols = list(df.columns)
        if not cols: return
        col_width = 190 / len(cols)
        for col in cols:
            self.cell(col_width, 8, str(col), 1, 0, 'C', 1)
        self.ln()
        self.set_font('Arial', '', 9)
        for _, row in df.iterrows():
            for val in row:
                display_val = str(val)
                if isinstance(val, (float, int)): display_val = f"{val:,.2f}"
                self.cell(col_width, 8, display_val, 1, 0, 'C')
            self.ln()
        self.ln(5)
    
    def add_disclaimer(self):
        self.ln(10)
        self.set_font('Arial', 'B', 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, "IMPORTANT DISCLOSURES & DISCLAIMER", 0, 1, 'L')
        self.set_font('Arial', '', 7)
        disclaimer = "This report is generated by an AI system (Titan Analyst) and is for informational purposes only. It does not constitute financial advice."
        self.multi_cell(0, 3, clean_text_for_pdf(disclaimer))

def generate_bar_chart(data_dict, title):
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        bars = ax.bar(data_dict.keys(), data_dict.values(), color=colors[:len(data_dict)])
        ax.set_title(title)
        ax.set_ylabel('Price ($)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'${height:,.0f}', ha='center', va='bottom')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name, format='png', bbox_inches='tight')
            plt.close(fig)
            return tmp.name
    except: return None

def generate_line_chart(df, title):
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        for col in df.columns:
            ax.plot(df.index, df[col], marker='o', label=col)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name, format='png', bbox_inches='tight')
            plt.close(fig)
            return tmp.name
    except: return None

def create_pdf(ticker, data, chart_data=None, return_data=None):
    pdf = TitanPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_title_page(ticker)
    pdf.executive_box(data.get("Portfolio Manager", "N/A"))
    
    pdf.section_header("Company Overview & Industry Trends")
    pdf.body_text(data.get("Macro", ""))
    
    pdf.section_header("Valuation & Financial Analysis")
    pdf.body_text(data.get("Fundamental", ""))
    pdf.body_text(data.get("Quant", ""))
    
    if chart_data:
        chart_path = generate_bar_chart(chart_data, "12-Month Price Targets")
        pdf.add_chart_image(chart_path, "Valuation Scenarios")
    
    if return_data is not None:
        ret_path = generate_line_chart(return_data, "Performance Comparison")
        pdf.add_chart_image(ret_path, "5-Year Total Return Comparison")
        pdf.ln(5)
        pdf.draw_table(return_data.reset_index())

    pdf.section_header("Key Investment Risks")
    pdf.body_text(data.get("Red Team", ""))
    pdf.add_disclaimer()
        
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# --- 4. LOGIC HELPERS ---
def extract_chart_data(text):
    data = {}
    try:
        matches = re.findall(r'"(Bear|Base|Bull)":\s*(\d+)', text, re.IGNORECASE)
        if not matches:
             matches = re.findall(r'(Bear|Base|Bull).*?\$(\d+)', text, re.IGNORECASE)
        for label, value in matches:
            data[label.capitalize()] = float(value)
        if len(data) >= 2: return data
    except: pass
    return None

def extract_return_data(text):
    try:
        match = re.search(r"(\{.*?(?:Years|SPY).*?\})", text, re.IGNORECASE | re.DOTALL)
        if match:
            json_str = match.group(1).replace("```", "").replace("json", "").strip()
            json_str = json_str.replace("'", '"')
            data = json.loads(json_str)
            if "Years" in data and "Ticker" in data and "SPY" in data:
                df = pd.DataFrame(data)
                df["Ticker ($10k)"] = 10000 * (1 + pd.to_numeric(df["Ticker"])/100).cumprod()
                df["SPY ($10k)"] = 10000 * (1 + pd.to_numeric(df["SPY"])/100).cumprod()
                return df.set_index("Years")[["Ticker ($10k)", "SPY ($10k)"]]
    except: pass
    return None

def extract_markdown_tables(text):
    lines = text.split('\n')
    table_lines = []
    for line in lines:
        if line.strip().startswith('|'):
            if "---" in line: continue
            table_lines.append([x.strip() for x in line.strip().split('|') if x])
    if len(table_lines) > 1:
        try:
            headers = table_lines[0]
            data = table_lines[1:]
            return pd.DataFrame(data, columns=headers)
        except: return None
    return None

# --- 5. AGENT ENGINE ---
async def run_agent(name, prompt, content):
    # SLOW DOWN: Add clear stagger to avoid hitting QPS limit and show status
    delay = random.uniform(2.0, 5.0)
    st.toast(f"⏳ {name}: Queued. Starting in {delay:.1f}s...")
    await asyncio.sleep(delay)
    
    st.toast(f"🚀 {name}: Analyzing...")
    
    for model_name in MODELS:
        try:
            model = genai.GenerativeModel(model_name, tools='google_search') # Enabled tools
            response = await asyncio.wait_for(model.generate_content_async(f"{prompt}\nCONTEXT: {content}"), timeout=90.0)
            
            if response.text:
                st.toast(f"✅ {name}: Complete!")
                return name, response.text
        except Exception as e:
            # If rate limit, wait longer (10-15s) and notify user
            if "429" in str(e):
                st.toast(f"⚠️ {name}: Rate limit hit. Pausing 15s...")
                await asyncio.sleep(15)
            continue
            
    return name, f"Analysis Failed for {name}. Please try again later."

async def run_analysis(ticker):
    tasks = [
        run_agent("Macro", MACRO_PROMPT, ticker),
        run_agent("Fundamental", FUNDAMENTAL_PROMPT, ticker),
        run_agent("Quant", QUANT_PROMPT, ticker)
    ]
    results = await asyncio.gather(*tasks)
    data = {k: v for k, v in results}
    
    # Check for critical failures, but proceed if partial data exists
    failed = [k for k, v in data.items() if "Analysis Failed" in str(v)]
    if failed:
        st.warning(f"Note: Some agents ({', '.join(failed)}) timed out. Report may be partial.")

    # Second Phase: Debate & Synthesis (Sequential)
    st.toast("⚔️ Red Team: Checking for risks...")
    await asyncio.sleep(2)
    _, data["Red Team"] = await run_agent("Red Team", RED_TEAM_PROMPT, str(data))
    
    st.toast("👔 Portfolio Manager: Drafting thesis...")
    await asyncio.sleep(2)
    _, data["Portfolio Manager"] = await run_agent("Portfolio Manager", PORTFOLIO_PROMPT, str(data))
    
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
        with st.spinner("Initializing Titan Agents (Searching SEC EDGAR)..."):
            st.session_state.report = asyncio.run(run_analysis(ticker))
    
    if st.session_state.report:
        rpt = st.session_state.report
        
        # Dashboard
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("🏆 Executive Thesis")
            st.info(rpt.get("Portfolio Manager"))
        with c2:
            st.subheader("🎯 12-Month Targets")
            chart_data = extract_chart_data(rpt.get("Quant", ""))
            if chart_data: st.bar_chart(chart_data)
            else: st.caption("No targets found.")

        t1, t2, t3, t4 = st.tabs(["Macro", "Fundamental", "Quant", "Red Team"])
        with t1: st.markdown(rpt.get("Macro", ""))
        with t2: st.markdown(rpt.get("Fundamental", ""))
        with t3: 
            st.markdown(rpt.get("Quant", ""))
            st.divider()
            st.subheader("📈 5-Year Return (Growth of $10k)")
            ret_data = extract_return_data(rpt.get("Quant", ""))
            if ret_data is not None: st.line_chart(ret_data)
        with t4: st.error(rpt.get("Red Team", ""))
        
        st.divider()
        try:
            c_data = extract_chart_data(rpt.get("Quant", ""))
            r_data = extract_return_data(rpt.get("Quant", ""))
            pdf_bytes = create_pdf(ticker, rpt, chart_data=c_data, return_data=r_data)
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
# --- END OF FILE main.py ---

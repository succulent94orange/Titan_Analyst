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

# ReportLab Imports for Professional PDFs
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.units import inch

# 1. SETUP
# Set Matplotlib to non-interactive mode to prevent crashes
matplotlib.use('Agg')

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("🚨 CRITICAL ERROR: GOOGLE_API_KEY not found in .env file.")
    st.stop()

genai.configure(api_key=api_key)

# Model Priority: Try 3.0 -> 2.0 Flash -> 1.5 Pro (Safety Net)
MODELS = ["gemini-3-pro-preview", "gemini-2.0-flash", "gemini-1.5-pro"]

# SAFETY SETTINGS
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- 2. PROMPT DEFINITIONS ---
MACRO_PROMPT = """
You are the Macro Council (Voices: Ray Dalio, Stanley Druckenmiller).
FRAMEWORK: Step-Back Prompting + Tree of Thoughts.
TASK: Analyze the global macro environment for the target asset.
OUTPUT FORMAT:
- Executive Summary (Text)
- **Markdown Table**: Key Indicators (Inflation, Rates, GDP)
- Scenario Probabilities (Bull/Bear/Base)
"""

FUNDAMENTAL_PROMPT = """
You are the Fundamental Specialist (Voices: Peter Lynch, Warren Buffett, Michael Porter).
FRAMEWORK: ReAct (Reason+Act) + Unit Economics Deep-Dive.
TASK: Analyze the business health, revenue quality, and moat.
**SEC EDGAR PROTOCOL:**
1. Simulate accessing SEC.gov to check 10-K Risk Factors & MD&A.
2. Verify management claims.
OUTPUT FORMAT:
- Business Health Analysis (citing 10-K findings)
- **Markdown Table**: Unit Economics & KPIs
- Moat Analysis & Management Scorecard
"""

QUANT_PROMPT = """
You are the Quant Desk (Voices: Jim Simons, Nassim Taleb).
FRAMEWORK: Program-Aided Language (PAL) logic.
TASK: Analyze valuation, risk, and sensitivity.
**CRITICAL OUTPUT:**
1. **Markdown Table**: Valuation Metrics (P/E, PEG, FCF Yield).
2. **CHART DATA** (Price Targets). Format exactly like this:
   Bear Case: 100
   Base Case: 150
   Bull Case: 200
3. **RETURN DATA** (5-Year Comparison vs SPY). Output a JSON block:
   {"Years": ["2020", "2021", "2022", "2023", "2024"], "Ticker": [10, 20, -5, 15, 30], "SPY": [15, 25, -18, 24, 12]}
"""

RED_TEAM_PROMPT = """
You are the Red Team (Voice: Jim Chanos, Gary Klein).
FRAMEWORK: Bayesian Network Synthesis + Pre-Mortem + Forward-Backward Reasoning.
TASK: Review the reports below. Find contradictions, hallucinations, and fatal risks.
SEC FORENSICS: Check Legal Proceedings and Related Party Transactions.
"""

PORTFOLIO_PROMPT = """
You are the CIO (The Chair).
FRAMEWORK: Chain of Density + Sell Discipline.
TASK: Synthesize the Red Team's verdict into a final Executive Thesis.
"""

FOLLOWUP_PROMPT = """
You are a Research Assistant. Answer user questions based ONLY on the provided report context.
"""

# --- 3. GRAPHIC DESIGN ENGINE (ReportLab) ---

class TitanReportGen:
    def __init__(self, filename):
        self.filename = filename
        self.styles = getSampleStyleSheet()
        self.story = []
        
        # Define Custom Styles
        self.colors = {
            "navy": colors.Color(10/255, 25/255, 50/255),
            "gold": colors.Color(218/255, 165/255, 32/255),
            "light_blue": colors.Color(235/255, 240/255, 245/255),
            "grey": colors.Color(50/255, 50/255, 50/255)
        }
        
        # Custom Title Style
        self.styles.add(ParagraphStyle(
            name='TitanTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=self.colors["navy"],
            spaceAfter=20,
            alignment=1 # Center
        ))
        
        # Custom Header Style
        self.styles.add(ParagraphStyle(
            name='TitanHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=self.colors["navy"],
            backColor=self.colors["light_blue"],
            borderPadding=5,
            spaceAfter=10,
            spaceBefore=10
        ))
        
        # Body Text
        self.styles.add(ParagraphStyle(
            name='TitanBody',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=self.colors["grey"],
            leading=14, # Line spacing
            spaceAfter=10
        ))

    def add_header_footer(self, canvas, doc):
        canvas.saveState()
        
        # Header
        canvas.setFillColor(self.colors["navy"])
        canvas.rect(0, LETTER[1] - 50, LETTER[0], 50, fill=1)
        canvas.setFillColor(self.colors["gold"])
        canvas.setFont('Helvetica-Bold', 12)
        canvas.drawString(30, LETTER[1] - 30, "TITAN FINANCIAL ANALYST // INSTITUTIONAL RESEARCH")
        
        # Footer
        canvas.setFillColor(colors.grey)
        canvas.setFont('Helvetica-Oblique', 8)
        canvas.drawString(30, 30, f"Confidential Report | Generated by Titan AI | Page {doc.page}")
        
        canvas.restoreState()

    def clean_text(self, text):
        # ReportLab supports basic XML tags like <b>, <i>. 
        # We need to sanitize Markdown or unsupported chars.
        text = str(text)
        text = text.replace('**', '') # Remove bold markdown for now
        text = text.replace('#', '') 
        # Escape XML characters
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # Re-add basic formatting if needed, but simple clean is safer
        return text

    def add_title_page(self, ticker):
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph(f"Investment Memorandum: {ticker}", self.styles['TitanTitle']))
        self.story.append(Paragraph(f"Date: {time.strftime('%B %d, %Y')}", self.styles['TitanBody']))
        self.story.append(Spacer(1, 0.5*inch))
        
        # Executive Summary Box Logic (simulated with a Table)
        # We will add this later when processing sections
        self.story.append(PageBreak())

    def add_section(self, title, content):
        self.story.append(Paragraph(title, self.styles['TitanHeader']))
        
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        for p in paragraphs:
            if p.strip():
                self.story.append(Paragraph(self.clean_text(p), self.styles['TitanBody']))
    
    def add_image(self, image_path, width=6*inch):
        if image_path:
            img = Image(image_path)
            # Aspect ratio check could go here, but simple scaling for now
            img.drawHeight = 3*inch
            img.drawWidth = width
            self.story.append(img)
            self.story.append(Spacer(1, 0.2*inch))

    def add_dataframe_table(self, df):
        # Convert DataFrame to list of lists for ReportLab Table
        data = [df.columns.tolist()] + df.values.tolist()
        
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors["navy"]),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), self.colors["light_blue"]),
            ('GRID', (0, 0), (-1, -1), 1, colors.white)
        ]))
        self.story.append(t)
        self.story.append(Spacer(1, 0.2*inch))

    def build(self):
        doc = SimpleDocTemplate(self.filename, pagesize=LETTER)
        doc.build(self.story, onFirstPage=self.add_header_footer, onLaterPages=self.add_header_footer)

def generate_pdf_report(ticker, report_data, chart_path=None, return_path=None, return_df=None):
    pdf_file = f"{ticker}_Titan_Report.pdf"
    
    # Use a temp file for safety in Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        gen = TitanReportGen(tmp.name)
        
        # 1. Title Page
        gen.add_title_page(ticker)
        
        # 2. Executive Thesis (Priority)
        gen.add_section("1. Executive Thesis", report_data.get("Portfolio Manager", "N/A"))
        
        # 3. Macro
        gen.add_section("2. Macro-Economic View", report_data.get("Macro", "N/A"))
        
        # 4. Fundamental
        gen.add_section("3. Fundamental Deep Dive", report_data.get("Fundamental", "N/A"))
        
        # 5. Quant & Charts
        gen.add_section("4. Quantitative & Risk Analysis", report_data.get("Quant", "N/A"))
        if chart_path:
            gen.add_image(chart_path)
        
        if return_path:
            gen.add_section("Performance Comparison (5-Year)", "")
            gen.add_image(return_path)
            
        if return_df is not None:
             gen.add_dataframe_table(return_df.reset_index())

        # 6. Red Team
        gen.add_section("5. Key Risks (Red Team)", report_data.get("Red Team", "N/A"))
        
        gen.build()
        
        # Read back the bytes
        with open(tmp.name, "rb") as f:
            return f.read()

# --- 4. DATA & CHART LOGIC ---
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
            json_str = match.group(1).replace("```", "").replace("json", "").replace("'", '"').strip()
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
    await asyncio.sleep(random.uniform(0.5, 2.0))
    last_error = "Unknown"
    
    for model_name in MODELS:
        # ATTEMPT 1: Try with Tools (Search)
        try:
            model = genai.GenerativeModel(model_name, tools='google_search', safety_settings=SAFETY_SETTINGS)
            response = await asyncio.wait_for(model.generate_content_async(f"{prompt}\nCONTEXT: {content}"), timeout=90.0)
            if response.text: return name, response.text
        except Exception as e:
            last_error = f"{model_name} (Tools): {str(e)}"
            if "429" in str(e):
                await asyncio.sleep(5) # Wait for quota reset
            
            # ATTEMPT 2: Fallback to Pure Reasoning (No Tools)
            # This fixes "Internal Error" or "404" when search isn't supported
            try:
                model_pure = genai.GenerativeModel(model_name, safety_settings=SAFETY_SETTINGS)
                response = await asyncio.wait_for(model_pure.generate_content_async(f"{prompt}\nCONTEXT: {content}"), timeout=90.0)
                if response.text: return name, response.text
            except Exception as e2:
                last_error = f"{model_name} (Pure): {str(e2)}"
                continue

    return name, f"Analysis Failed. Last Error: {last_error}"

async def run_analysis(ticker):
    tasks = [
        run_agent("Macro", MACRO_PROMPT, ticker),
        run_agent("Fundamental", FUNDAMENTAL_PROMPT, ticker),
        run_agent("Quant", QUANT_PROMPT, ticker)
    ]
    results = await asyncio.gather(*tasks)
    data = {k: v for k, v in results}
    
    # Check for failures
    failed = [k for k, v in data.items() if "Analysis Failed" in str(v)]
    if failed:
        return data # Return partial data so user can see the error

    # COOL-DOWN: Wait 5 seconds before hitting the API again for Red Team
    st.toast("⏳ Cooling down quota for Red Team analysis...")
    await asyncio.sleep(5)
    
    _, data["Red Team"] = await run_agent("Red Team", RED_TEAM_PROMPT, str(data))
    
    # COOL-DOWN: Wait another 5 seconds for Portfolio Manager
    st.toast("⏳ Cooling down quota for Final Thesis...")
    await asyncio.sleep(5)
    
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
        
        failed_agents = [k for k, v in rpt.items() if "Analysis Failed" in str(v)]
        if failed_agents:
            st.error(f"🚨 Analysis Incomplete. Failures in: {', '.join(failed_agents)}")
            for agent in failed_agents:
                st.error(f"**{agent} Debug Info**: {rpt[agent]}")
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
                # Generate Charts for PDF
                c_data = extract_chart_data(rpt.get("Quant", ""))
                c_path = generate_bar_chart(c_data, "Price Targets") if c_data else None
                
                r_data = extract_return_data(rpt.get("Quant", ""))
                r_path = generate_line_chart(r_data, "5-Year Performance") if r_data is not None else None
                
                pdf_bytes = generate_pdf_report(ticker, rpt, chart_path=c_path, return_path=r_path, return_df=r_data)
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

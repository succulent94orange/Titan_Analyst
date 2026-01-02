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

# --- 2. PROMPT DEFINITIONS (TITAN V3) ---
MACRO_PROMPT = """
You are the Macro Council (Voices: Ray Dalio, Stanley Druckenmiller).
FRAMEWORK: Step-Back Prompting + Tree of Thoughts.

TASK: Analyze the global macro environment for the target asset.
1. "Step Back" and identify the abstract economic principle governing the current era.
2. Apply Tree of Thoughts (ToT) to branch into three scenarios: Inflation Resurgence, Soft Landing, Deflationary Bust.
3. Identify "Macro Drag": Specific headwinds for the specific sector.

OUTPUT FORMAT:
- Executive Summary (Text)
- **Markdown Table**: Key Indicators (Inflation, Rates, GDP)
- Scenario Probabilities (Bull/Bear/Base)
"""

FUNDAMENTAL_PROMPT = """
You are the Fundamental Specialist (Voices: Peter Lynch, Warren Buffett, Michael Porter).
FRAMEWORK: ReAct (Reason+Act) + Unit Economics Deep-Dive.

TASK: Analyze the business health, revenue quality, and moat.

**SEC EDGAR PROTOCOL (MANDATORY):**
1. You must simulate accessing [https://www.sec.gov/edgar/search/](https://www.sec.gov/edgar/search/)
2. Locate the latest **10-K** and **10-Q**.
3. Specifically analyze **Item 1A. Risk Factors** and **Item 7. MD&A**.

**DEEP DIVE TASKS:**
1. **Unit Economics Check:** Analyze LTV/CAC, Same-Store Sales, or relevant efficiency metrics.
2. **Moat Analysis (Porter):** Analyze Supplier Power, Buyer Power, and Threat of Substitutes.
3. **Capital Allocation Scorecard:** Grade management on Buybacks vs. Empire Building.
4. **Executive Compensation Audit:** Analyze the Proxy Statement. Are executives paid on EPS (Risk of Manipulation) or ROIC/Free Cash Flow (Shareholder Alignment)?

OUTPUT FORMAT:
- Business Health Analysis (citing 10-K findings)
- **Markdown Table**: Unit Economics & KPIs
- Moat Analysis & Management Scorecard
"""

QUANT_PROMPT = """
You are the Quant Desk (Voices: Jim Simons, Nassim Taleb).
FRAMEWORK: Program-Aided Language (PAL) logic.

TASK: Analyze valuation, risk, and sensitivity.

**QUANT TASKS:**
1. **Valuation Risk:** Is the stock priced for perfection? (Compare P/E to Historical Average).
2. **Factor Exposure:** Is the asset driven by Quality, Momentum, or Value factors?
3. **Sensitivity Analysis:** How does price change if WACC increases by 1%?
4. **Stochastic DCF:** Perform a mental Monte Carlo simulation (10k iterations) for 5th/95th percentile outcomes.

**CRITICAL OUTPUT:**
1. **Markdown Table**: Valuation Metrics (P/E, PEG, FCF Yield).
2. **CHART DATA** (Price Targets). Format exactly like this:
   {"Bear": 100, "Base": 150, "Bull": 200}
3. **RETURN DATA** (5-Year Comparison vs SPY). Output a JSON block:
   {"Years": ["2020", "2021", "2022", "2023", "2024"], "Ticker": [10, 20, -5, 15, 30], "SPY": [15, 25, -18, 24, 12]}
"""

RED_TEAM_PROMPT = """
You are the Red Team (Voice: Jim Chanos, Gary Klein).
FRAMEWORK: Bayesian Network Synthesis + Pre-Mortem + Forward-Backward Reasoning.

TASK: Review the reports below. Find contradictions, hallucinations, and fatal risks.

**RISK PROTOCOLS:**
1. **Backward Check:** Take the current stock price and reverse-engineer the required growth. Is it realistic?
2. **The Grey Rhino:** Identify high-probability, high-impact threats everyone is ignoring (e.g., Debt Maturity Walls).
3. **Pre-Mortem:** Assume the investment failed 3 years from now. Write the obituary.
4. **SEC Forensics:** Check "Legal Proceedings" and "Related Party Transactions" for red flags.
"""

PORTFOLIO_PROMPT = """
You are the CIO (The Chair).
FRAMEWORK: Chain of Density + Sell Discipline.

TASK: Synthesize the Red Team's verdict into a final Executive Thesis.

**CIO TASKS:**
1. **Generate Executive Thesis:** Use Chain of Density to maximize information per word.
2. **Variant Perception:** Explicitly state how your view differs from Consensus.
3. **Sell Triggers:** Define 3 hard rules for selling this position (e.g., "If ROIC drops below 15%").
4. **Historical Audit:** Briefly mention if this model would have failed in the last crisis (2020/2022) based on current logic.
"""

FOLLOWUP_PROMPT = """
You are a Research Assistant. Answer user questions based ONLY on the provided report context.
"""

# --- 3. GRAPHIC DESIGN ENGINE (Titan Professional PDF - HTML/CSS) ---

def generate_pdf_report(ticker, report_data, chart_path=None, return_path=None, return_df=None):
    # 1. Prepare HTML Content
    
    # Convert Markdown to HTML
    pm_html = markdown.markdown(report_data.get("Portfolio Manager", "N/A"), extensions=['tables'])
    macro_html = markdown.markdown(report_data.get("Macro", "N/A"), extensions=['tables'])
    fund_html = markdown.markdown(report_data.get("Fundamental", "N/A"), extensions=['tables'])
    quant_html = markdown.markdown(report_data.get("Quant", "N/A"), extensions=['tables'])
    red_html = markdown.markdown(report_data.get("Red Team", "N/A"), extensions=['tables'])
    
    # Prepare Images
    chart_html = ""
    if chart_path:
        # Use file protocol for local images if needed, but xhtml2pdf handles paths
        chart_html += f'<div class="chart-container"><h3>Valuation Scenarios (12-Month Targets)</h3><img src="{chart_path}" style="width: 15cm;" /></div>'
    
    if return_path:
        chart_html += f'<div class="chart-container"><h3>5-Year Total Return Comparison (Growth of $10k)</h3><img src="{return_path}" style="width: 15cm;" /></div>'
        
    # Prepare Data Table
    table_html = ""
    if return_df is not None:
        # Simple HTML table for the return data
        table_html = f"<h3>Historical Return Data</h3>{return_df.reset_index().to_html(index=False)}"

    # 2. Construct Full HTML Document with CSS
    # Using @frame for static headers/footers to avoid NotImplementedError in older reportlab/xhtml2pdf combos
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Titan Analyst Report - {ticker}</title>
        <style>
            @page {{
                size: letter;
                margin: 2.5cm;
                @frame header_frame {{           /* Static Frame */
                    -pdf-frame-content: header_content;
                    left: 20pt; width: 572pt; top: 20pt; height: 40pt;
                }}
                @frame footer_frame {{           /* Static Frame */
                    -pdf-frame-content: footer_content;
                    left: 20pt; width: 572pt; top: 750pt; height: 20pt;
                }}
            }}
            
            body {{
                font-family: Helvetica, sans-serif;
                font-size: 11pt;
                line-height: 1.5;
                color: #333333;
            }}
            h1 {{
                color: #0A1932; /* Navy */
                font-size: 26pt;
                text-align: center;
                margin-top: 0;
                margin-bottom: 5px;
            }}
            .subtitle {{
                color: #DAA520; /* Gold */
                font-size: 14pt;
                text-align: center;
                font-weight: bold;
                margin-bottom: 20px;
            }}
            .date {{
                text-align: center;
                font-size: 12pt;
                margin-bottom: 40px;
                color: #555;
            }}
            h2 {{
                color: #0A1932; /* Navy */
                font-size: 16pt;
                border-bottom: 2px solid #DAA520; /* Gold Underline */
                padding-bottom: 5px;
                margin-top: 30px;
                background-color: #EBF0F5;
                padding: 5px;
            }}
            h3 {{
                color: #0A1932;
                font-size: 13pt;
                margin-top: 20px;
            }}
            
            /* Executive Box Style */
            .executive-box {{
                background-color: #F4F7FA;
                border: 1px solid #0A1932;
                padding: 15px;
                margin-bottom: 20px;
            }}
            .executive-title {{
                color: #DAA520;
                font-weight: bold;
                font-size: 14pt;
                margin-bottom: 10px;
                display: block;
            }}
            
            /* Table Styling */
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
                font-size: 10pt;
                border: 1pt solid #dddddd;
            }}
            th {{
                background-color: #0A1932;
                color: #DAA520; /* Gold Text */
                font-weight: bold;
                text-align: center;
                padding: 8px;
            }}
            td {{
                padding: 8px;
                border-bottom: 1pt solid #dddddd;
            }}
            
            .disclaimer {{
                font-size: 8pt;
                color: #888;
                margin-top: 50px;
                text-align: justify;
                border-top: 1px solid #ccc;
                padding-top: 10px;
            }}
            .chart-container {{
                text-align: center;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <!-- Header Content -->
        <div id="header_content" style="text-align: center; color: #0A1932; font-weight: bold;">
            TITAN FINANCIAL ANALYST // EQUITY RESEARCH
        </div>
        
        <!-- Footer Content -->
        <div id="footer_content" style="text-align: center; color: #888; font-size: 8pt;">
            Strictly Confidential | Generated by Titan AI
        </div>

        <h1>{ticker}</h1>
        <div class="subtitle">INSTITUTIONAL EQUITY RESEARCH</div>
        <div class="date">Date: {time.strftime('%B %d, %Y')}</div>
        
        <div class="executive-box">
            <span class="executive-title">1. EXECUTIVE THESIS</span>
            {pm_html}
        </div>

        <h2>2. MACRO-ECONOMIC LANDSCAPE</h2>
        {macro_html}

        <h2>3. FUNDAMENTAL DEEP DIVE</h2>
        {fund_html}

        <h2>4. QUANTITATIVE & RISK ANALYSIS</h2>
        {quant_html}
        
        {chart_html}
        {table_html}

        <h2>5. KEY RISKS (RED TEAM VERDICT)</h2>
        {red_html}

        <div class="disclaimer">
            <strong>IMPORTANT DISCLOSURES & DISCLAIMER</strong><br>
            This report is generated by an AI system (Titan Analyst) and is for informational purposes only. It does not constitute financial advice, an offer to sell, or a solicitation of an offer to buy any securities. The information contained herein is based on data available at the time of generation and may not be accurate or complete. Past performance is not indicative of future results. Investment involves risk, including the possible loss of principal.
        </div>
    </body>
    </html>
    """
    
    # 3. Generate PDF
    pdf_file = BytesIO()
    pisa_status = pisa.CreatePDF(
        src=html_template,
        dest=pdf_file
    )
    
    if pisa_status.err:
        return None
    
    return pdf_file.getvalue()

# --- 4. CHART GENERATORS (Matched to Titan Theme) ---
def generate_bar_chart(data_dict, title):
    try:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        # Use Titan Colors: Navy for bars, Gold for highlight if needed
        bars = ax.bar(data_dict.keys(), data_dict.values(), color=HEX_NAVY)
        ax.set_title(title, fontsize=14, fontweight='bold', color=HEX_NAVY)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add values on top in Gold
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
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        # Plot lines with Titan Colors
        colors = [HEX_NAVY, HEX_GOLD, 'grey']
        for i, col in enumerate(df.columns):
            color = colors[i % len(colors)]
            ax.plot(df.index, df[col], marker='o', linewidth=2, label=col, color=color)
            
        ax.set_title(title, fontsize=14, fontweight='bold', color=HEX_NAVY)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig)
            return tmp.name
    except: return None

# --- 5. LOGIC HELPERS ---
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

# --- 6. AGENT ENGINE ---
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
    tasks = [
        run_agent("Macro", MACRO_PROMPT, ticker),
        run_agent("Fundamental", FUNDAMENTAL_PROMPT, ticker),
        run_agent("Quant", QUANT_PROMPT, ticker)
    ]
    results = await asyncio.gather(*tasks)
    data = {k: v for k, v in results}
    
    if any("Analysis Failed" in str(v) for v in data.values()):
        return data

    st.toast("⏳ Cooling down quota for Red Team analysis...")
    await asyncio.sleep(5)
    
    _, data["Red Team"] = await run_agent("Red Team", RED_TEAM_PROMPT, str(data))
    
    st.toast("⏳ Cooling down quota for Final Thesis...")
    await asyncio.sleep(5)
    
    _, data["Portfolio Manager"] = await run_agent("Portfolio Manager", PORTFOLIO_PROMPT, str(data))
    return data

# --- 7. UI ---
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
                # Extract data again for PDF generation context
                c_data = extract_chart_data(rpt.get("Quant", ""))
                r_data = extract_return_data(rpt.get("Quant", ""))
                pdf_bytes = generate_pdf_report(ticker, rpt, chart_path=generate_bar_chart(c_data, "Price Targets") if c_data else None, return_path=generate_line_chart(r_data, "Performance") if r_data is not None else None, return_df=r_data)
                if pdf_bytes:
                    st.download_button("📄 Download Professional Report (PDF)", pdf_bytes, f"{ticker}_Titan_Report.pdf", "application/pdf")
                else:
                    st.error("PDF generation failed.")
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

import os
import streamlit as st
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Initialize secure configuration environment vectors
load_dotenv()

# =====================================================================
# ENTERPRISE DESIGN SYSTEM (STANDARD COMPATIBLE ARCHITECTURE)
# =====================================================================
st.set_page_config(
    page_title="Luna AI",
    page_icon="🔮",
    layout="wide"
)

# Render Headings without complex overlapping divs
st.title("Luna Virtual Intelligence Platform")
st.caption("An enterprise-grade, high-performance computational agent engineered by Laiba Qamar. Optimized for technical scripting, code generation, and multi-turn reasoning.")
st.markdown("---")

# Multi-Turn Conversational Context Memory Init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================================================
# CORE QUANTUM ROUTER & MODEL ORCHESTRATION ENGINE (BACKEND)
# =====================================================================
def generate_luna_response(user_message: str, provider: str, auth_token: str) -> str:
    """
    Assembles structural payloads, manages thread context sequences,
    and executes synchronous API handshakes with global cloud clusters.
    """
    
    system_personality = (
        "You are Luna, an advanced, hyper-complex virtual intelligence engine engineered by Laiba Qamar. "
        "You operate strictly under enterprise performance benchmarks to assist technical teams and international clients. "
        "Your cognitive processing is structurally optimized for three core deployment workflows:\n"
        "1. ADVANCED SOFTWARE ARCHITECT: Generate clean, optimized, production-ready code across all software engineering stacks. "
        "Provide algorithmic breakdowns and comprehensive error analysis without parameter hallucination.\n"
        "2. TECHNICAL SPECIFICATION AUTHOR: Draft pristine software requirement documents, academic literature reviews, "
        "and algorithmic proofs with rigorous clarity.\n"
        "3. HIGH-INTELLECT DIALOGUE AGENT: Maintain stateful, contextual interaction patterns with advanced user tracking, "
        "blending logical depth with exceptional analytical efficiency.\n\n"
        "Formatting Directive: Output answers with structured Markdown formatting, precise bullet markers, and clean syntax-highlighted code blocks."
    )
    
    messages_payload = [{"role": "system", "content": system_personality}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history
    ]

    try:
        if provider == "Luna Free-Tier Server":
            response = requests.post(
                "https://api.airforce/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini", 
                    "messages": messages_payload, 
                    "temperature": 0.3,
                    "max_tokens": 4096
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return f"System Gateway Error ({response.status_code}). Please re-route traffic to a commercial cloud node."

        else:
            if not auth_token or auth_token.strip() == "":
                return "Initialization Halted: An active structural key token is mandatory to interface with this cluster."
            
            base_url = "https://api.deepseek.com" if provider == "DeepSeek Cluster" else "https://api.openai.com/v1"
            model_target = "deepseek-chat" if provider == "DeepSeek Cluster" else "gpt-4o-mini"
            
            client = OpenAI(api_key=auth_token, base_url=base_url)
            completion = client.chat.completions.create(
                model=model_target,
                messages=messages_payload,
                temperature=0.3,
                max_tokens=4096
            )
            return completion.choices[0].message.content.strip()

    except Exception as runtime_exception:
        return f"Pipeline Interruption: Transaction serialization failed. Diagnostics: {runtime_exception}"

# =====================================================================
# SYSTEM CONTROL INTERFACE & DASHBOARD METRICS (FRONTEND)
# =====================================================================
with st.sidebar:
    st.subheader("Infrastructure Control")
    st.caption("Enterprise Platform Build v3.6")
    st.markdown("---")
    
    engine_choice = st.selectbox(
        "Infrastructure Routing Node",
        ["Luna Free-Tier Server", "OpenAI Core Cluster", "DeepSeek Cluster"]
    )
    
    selected_key = ""
    if engine_choice == "OpenAI Core Cluster":
        selected_key = os.getenv("OPENAI_API_KEY") or st.text_input("OpenAI Production Token", type="password")
    elif engine_choice == "DeepSeek Cluster":
        selected_key = os.getenv("DEEPSEEK_API_KEY") or st.text_input("DeepSeek Production Token", type="password")
        
    st.markdown("---")
    
    # Telemetry Status Informational Block
    st.markdown(f"""
        <div style="background-color: #1e293b; padding: 12px; border-radius: 6px; border: 1px solid #334155;">
            <span style="color: #6366f1; font-weight: bold; font-size: 0.85rem;">SYSTEM STATUS</span><br>
            <small style="color: #94a3b8;">• Architect: Laiba Qamar</small><br>
            <small style="color: #94a3b8;">• Environment: Production-Ready</small><br>
            <small style="color: #94a3b8;">• Active Context Memory: {len(st.session_state.chat_history)} Logs</small>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Flush Context Memory", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Loop and render active threads onto layout
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# User Prompt Data Ingestion Gateway Loop
if user_input := st.chat_input("Enter parameters or request technical output..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Resolving response matrices..."):
            response_string = generate_luna_response(user_input, engine_choice, selected_key)
            st.markdown(response_string)
            
    st.session_state.chat_history.append({"role": "assistant", "content": response_string})
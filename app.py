import os
import streamlit as st
from openai import (
    OpenAI,
    APIError,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    NotFoundError,
)
from dotenv import load_dotenv

# Load local .env file (for testing on your own machine).
# On Streamlit Cloud, use the "Secrets" section instead - see notes below.
load_dotenv()

# =====================================================================
# ENTERPRISE DESIGN SYSTEM
# =====================================================================
st.set_page_config(
    page_title="Luna AI",
    page_icon="🔮",
    layout="wide"
)

st.title("Luna Virtual Intelligence Platform")
st.caption(
    "An enterprise-grade, high-performance computational agent engineered by "
    "Laiba Qamar. Optimized for technical scripting, code generation, and "
    "multi-turn reasoning."
)
st.markdown("---")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================================================
# PROVIDER REGISTRY
#
# Every provider below exposes an OpenAI-compatible /chat/completions
# endpoint, so a single client + function handles all of them. To add a
# new provider later, just add another entry here.
#
# "Luna (Free)" is powered by YOUR OWN free Groq key, stored in Streamlit
# secrets as GROQ_API_KEY. Visitors never see or need this key - it just
# works for them automatically.
# =====================================================================
PROVIDERS = {
    "Luna (Free - No Key Needed)": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "secret_name": "GROQ_API_KEY",
        "needs_user_key": False,
        "signup_url": None,
        "note": "Powered by a fast open-source model. No setup required for visitors.",
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "secret_name": "GEMINI_API_KEY",
        "needs_user_key": True,
        "signup_url": "https://aistudio.google.com/app/apikey",
        "note": "Free tier available - no credit card required.",
    },
    "Groq (Your Own Key)": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "secret_name": "GROQ_API_KEY_USER",
        "needs_user_key": True,
        "signup_url": "https://console.groq.com/keys",
        "note": "Free tier available - no credit card required.",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "secret_name": "OPENAI_API_KEY",
        "needs_user_key": True,
        "signup_url": "https://platform.openai.com/api-keys",
        "note": "Requires a payment method and account credit (no free tier).",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "secret_name": "DEEPSEEK_API_KEY",
        "needs_user_key": True,
        "signup_url": "https://platform.deepseek.com/api_keys",
        "note": "Very low cost - requires a small account top-up.",
    },
}

SYSTEM_PERSONALITY = (
    "You are Luna, an advanced virtual intelligence engine engineered by Laiba Qamar. "
    "You operate strictly under enterprise performance benchmarks to assist technical "
    "teams and international clients. Your cognitive processing is structurally "
    "optimized for three core deployment workflows:\n"
    "1. ADVANCED SOFTWARE ARCHITECT: Generate clean, optimized, production-ready code "
    "across all software engineering stacks, with algorithmic breakdowns and error "
    "analysis.\n"
    "2. TECHNICAL SPECIFICATION AUTHOR: Draft clear software requirement documents, "
    "literature reviews, and well-structured technical explanations.\n"
    "3. HIGH-INTELLECT DIALOGUE AGENT: Maintain stateful, contextual conversation, "
    "blending logical depth with clear, friendly communication.\n\n"
    "If a user asks how to get an API key for an AI provider (OpenAI, Google Gemini, "
    "Groq, or DeepSeek), explain that they should go to that provider's developer "
    "platform website, sign in or sign up, open the 'API Keys' section, create a new "
    "key, and paste it into Luna's sidebar under 'Infrastructure Control'. Mention that "
    "Google Gemini and Groq currently offer free tiers with no credit card required, "
    "while OpenAI and DeepSeek require adding billing or credits first.\n\n"
    "Formatting Directive: Use structured Markdown, clear bullet points where helpful, "
    "and syntax-highlighted code blocks."
)


# =====================================================================
# CORE ROUTER & MODEL ORCHESTRATION ENGINE (BACKEND)
# =====================================================================
def get_secret(name: str):
    """
    Safely read a value from Streamlit secrets or environment variables.
    Streamlit raises StreamlitSecretNotFoundError if NO secrets.toml exists
    anywhere - even when just calling .get() - so this wraps that in a
    try/except and falls back to .env / environment variables.
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def resolve_api_key(provider_name: str, user_key: str):
    """Return the API key to use for this provider, plus its config."""
    config = PROVIDERS[provider_name]

    if config["needs_user_key"]:
        # Professional / bring-your-own-key path
        api_key = user_key.strip() if user_key else None
    else:
        # Built-in free path: pulled from secrets, set by the developer
        api_key = get_secret(config["secret_name"])

    return api_key, config


def key_guidance_message(provider_name: str, config: dict) -> str:
    """Friendly guidance shown when a professional hasn't entered a key yet."""
    return (
        f"To use **{provider_name}**, you'll need your own API key.\n\n"
        f"{config['note']}\n\n"
        f"1. Visit **{config['signup_url']}**\n"
        f"2. Sign up or log in\n"
        f"3. Create a new API key\n"
        f"4. Paste it into the **\"{provider_name} API Key\"** box in the sidebar\n\n"
        f"Once it's entered, send your message again and Luna will respond using "
        f"{provider_name}."
    )


def generate_luna_response(provider_name: str, user_key: str) -> str:
    """
    Assembles the conversation payload and sends it to the selected provider
    via its OpenAI-compatible endpoint. Returns clean, user-friendly text -
    never raw exception dumps.
    """
    api_key, config = resolve_api_key(provider_name, user_key)

    if not api_key:
        if config["needs_user_key"]:
            return key_guidance_message(provider_name, config)
        else:
            # This means the developer hasn't configured the free key yet.
            return (
                "⚙️ **Setup needed (for the app owner):** the built-in free engine "
                "isn't configured yet. Add a `GROQ_API_KEY` to this app's Streamlit "
                "secrets (get one free at https://console.groq.com/keys), or pick "
                "another engine from the sidebar."
            )

    messages_payload = [{"role": "system", "content": SYSTEM_PERSONALITY}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history
    ]

    try:
        client = OpenAI(api_key=api_key, base_url=config["base_url"])
        completion = client.chat.completions.create(
            model=config["model"],
            messages=messages_payload,
            temperature=0.5,
            max_tokens=2048,
        )
        return completion.choices[0].message.content.strip()

    except AuthenticationError:
        return (
            f"🔑 That **{provider_name}** API key was rejected. Double-check you "
            f"copied it correctly with no extra spaces, and that it hasn't been "
            f"revoked. Get a new one at {config['signup_url']}."
        )
    except RateLimitError:
        return (
            f"⏳ **{provider_name}** says this account has hit its usage limit or "
            f"run out of credits. Wait a bit and try again, check your usage/billing "
            f"dashboard, or switch engines in the sidebar."
        )
    except APIConnectionError:
        return (
            f"📡 Couldn't reach **{provider_name}**'s servers right now. This is "
            f"usually temporary - please try again in a moment."
        )
    except NotFoundError:
        return (
            f"⚠️ The model `{config['model']}` isn't available for this "
            f"**{provider_name}** account. It may need to be enabled, or the model "
            f"name may have changed on the provider's side."
        )
    except APIError as e:
        return f"⚠️ **{provider_name}** returned an error: {e}"
    except Exception as e:
        return f"⚠️ Unexpected error: {e}"


# =====================================================================
# SYSTEM CONTROL INTERFACE & DASHBOARD METRICS (FRONTEND)
# =====================================================================
with st.sidebar:
    st.subheader("Infrastructure Control")
    st.caption("Enterprise Platform Build v3.7")
    st.markdown("---")

    provider_name = st.selectbox("AI Engine", list(PROVIDERS.keys()))
    config = PROVIDERS[provider_name]

    user_key = ""
    if config["needs_user_key"]:
        user_key = st.text_input(
            f"{provider_name} API Key",
            type="password",
            help=f"Get a free key: {config['signup_url']}",
        )
        if not user_key:
            with st.expander("📘 How to get an API key for this engine", expanded=True):
                st.markdown(
                    f"{config['note']}\n\n"
                    f"1. Visit [{config['signup_url']}]({config['signup_url']})\n"
                    f"2. Sign up or log in\n"
                    f"3. Create a new API key\n"
                    f"4. Paste it into the box above"
                )
    else:
        st.success("✅ Ready to chat — no setup needed.")

    st.markdown("---")

    st.markdown(f"""
        <div style="background-color: #1e293b; padding: 12px; border-radius: 6px; border: 1px solid #334155;">
            <span style="color: #6366f1; font-weight: bold; font-size: 0.85rem;">SYSTEM STATUS</span><br>
            <small style="color: #94a3b8;">• Architect: Laiba Qamar</small><br>
            <small style="color: #94a3b8;">• Active Engine: {provider_name}</small><br>
            <small style="color: #94a3b8;">• Active Context Memory: {len(st.session_state.chat_history)} Logs</small>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Flush Context Memory", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Render chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# User input
if user_input := st.chat_input("Enter parameters or request technical output..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Resolving response matrices..."):
            response_string = generate_luna_response(provider_name, user_key)
            st.markdown(response_string)

    st.session_state.chat_history.append({"role": "assistant", "content": response_string})
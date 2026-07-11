"""
AutoMind — Streamlit demo frontend.

Run:
    streamlit run frontend/app.py

This is a thin UI layer only — all the actual reasoning happens in the
FastAPI backend (src/api/main.py), which must be running separately on
port 8000 before you start this.

Visual identity: instrument-cluster / cockpit theme — deep navy background,
dashboard-amber for "needs attention" states, diagnostic teal for resolved
answers, monospace readouts for diagnostic data. Fits an in-vehicle AI
copilot better than a generic chat-app look.
"""

import os
import uuid
import requests
import streamlit as st


def _get_api_base_url() -> str:
    env_value = os.getenv("API_BASE_URL")
    if env_value:
        return env_value
    try:
        return st.secrets.get("API_BASE_URL", "http://localhost:8000")
    except Exception:
        return "http://localhost:8000"


API_BASE_URL = _get_api_base_url()

# Must match src/config.py VEHICLE_MANUAL_MAP exactly
VEHICLE_MANUAL_MAP = {
    "Honda Civic (2025)": "2025 Civic sedan.pdf",
    "Toyota Corolla (2025)": "TOYOTA 2025(Corolla).pdf",
    "Tata Harrier BS6 (2026)": "harrier-bs6-owners-manual-april-2026.pdf",
    "Maruti Suzuki Jimny": "NEXA-Jimny-Petrol-Manual-latestpdf.pdf",
    "Not sure / Other": None,
}

VEHICLE_ICONS = {
    "Honda Civic (2025)": "🚗",
    "Toyota Corolla (2025)": "🚘",
    "Tata Harrier BS6 (2026)": "🚙",
    "Maruti Suzuki Jimny": "🛻",
    "Not sure / Other": "❓",
}

st.set_page_config(page_title="AutoMind — Vehicle AI Copilot", page_icon="🚗", layout="centered")

# ---------------- Cockpit theme ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --am-bg: #0B1220;
    --am-bg-elevated: #121B2E;
    --am-bg-card: #16213A;
    --am-amber: #F2A93C;
    --am-teal: #2DD4BF;
    --am-text: #EDEFF3;
    --am-text-dim: #8B97AC;
    --am-border: #1E2A42;
}

.stApp {
    background: var(--am-bg);
    color: var(--am-text);
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--am-bg-elevated);
    border-right: 1px solid var(--am-border);
}
[data-testid="stSidebar"] * { color: var(--am-text) !important; }

/* Headings use the display face */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

/* Hero */
.am-hero {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 0.15rem;
}
.am-hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: var(--am-text);
    margin: 0;
}
.am-hero-accent { color: var(--am-amber); }
.am-hero-tagline {
    color: var(--am-text-dim);
    font-size: 0.95rem;
    margin-top: 0.1rem;
    margin-bottom: 0.9rem;
}
.am-status-row { display: flex; gap: 1.1rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
.am-status-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--am-text-dim);
    display: flex;
    align-items: center;
    gap: 0.4rem;
    letter-spacing: 0.02em;
}
.am-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--am-teal);
    box-shadow: 0 0 6px var(--am-teal);
}

/* Cards (login / vehicle select) */
.am-card {
    background: var(--am-bg-card);
    border: 1px solid var(--am-border);
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
}

/* Buttons */
.stButton > button {
    background: var(--am-bg-card);
    color: var(--am-text) !important;
    border: 1px solid var(--am-border);
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    transition: all 0.18s ease;
}
.stButton > button:hover {
    border-color: var(--am-amber);
    box-shadow: 0 0 0 1px var(--am-amber), 0 0 14px rgba(242,169,60,0.25);
    color: var(--am-amber) !important;
}
.stButton > button:active { transform: scale(0.98); }

/* Primary CTA (form submit / continue buttons) get the amber treatment */
.stFormSubmitButton > button, button[kind="primary"] {
    background: var(--am-amber) !important;
    color: #0B1220 !important;
    border: none !important;
    font-weight: 600 !important;
}
.stFormSubmitButton > button:hover { box-shadow: 0 0 16px rgba(242,169,60,0.45); }

/* Inputs */
.stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background: var(--am-bg-card) !important;
    color: var(--am-text) !important;
    border: 1px solid var(--am-border) !important;
    border-radius: 8px !important;
}

/* Chat messages: settle-in animation, like a gauge needle finding its mark */
[data-testid="stChatMessage"] {
    background: var(--am-bg-card);
    border: 1px solid var(--am-border);
    border-radius: 12px;
    animation: am-settle 0.32s cubic-bezier(0.22, 1, 0.36, 1);
}
@keyframes am-settle {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Diagnostic loading readout */
.am-diagnostic {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--am-teal);
    font-size: 0.88rem;
    padding: 0.6rem 0;
}
.am-cursor { animation: am-blink 1s step-end infinite; }
@keyframes am-blink { 50% { opacity: 0; } }

/* Sidebar captions as section eyebrows */
[data-testid="stSidebar"] .stCaption {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em;
    color: var(--am-amber) !important;
}

hr { border-color: var(--am-border) !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- Session state ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": ...}
if "vehicle_name" not in st.session_state:
    st.session_state.vehicle_name = None


# ---------------- Auth ----------------
def login(username: str, password: str) -> bool:
    try:
        resp = requests.post(
            f"{API_BASE_URL}/auth/login",
            data={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            st.session_state.access_token = resp.json()["access_token"]
            return True
        return False
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Can't reach the API. Is `uvicorn src.api.main:app --port 8000` running?")
        return False


def send_message(message: str) -> dict:
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    vehicle_manual = VEHICLE_MANUAL_MAP.get(st.session_state.vehicle_name)
    resp = requests.post(
        f"{API_BASE_URL}/chat",
        json={
            "session_id": st.session_state.session_id,
            "message": message,
            "vehicle_manual": vehicle_manual,
        },
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def reset_conversation():
    if st.session_state.access_token:
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        try:
            requests.delete(
                f"{API_BASE_URL}/chat/{st.session_state.session_id}",
                headers=headers,
                timeout=10,
            )
        except requests.exceptions.ConnectionError:
            pass
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []


def render_hero():
    st.markdown("""
    <div class="am-hero">
        <p class="am-hero-title">🚗 AutoMind<span class="am-hero-accent">.</span></p>
    </div>
    <p class="am-hero-tagline">Agentic in-vehicle AI copilot — describe a fault, or ask how to use a feature from your owner's manual.</p>
    <div class="am-status-row">
        <div class="am-status-pill"><span class="am-dot"></span> MANUAL RAG</div>
        <div class="am-status-pill"><span class="am-dot"></span> KNOWLEDGE GRAPH</div>
        <div class="am-status-pill"><span class="am-dot"></span> LLM REASONING</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- UI ----------------
render_hero()

# --- Login gate ---
if not st.session_state.access_token:
    st.markdown('<div class="am-card">', unsafe_allow_html=True)
    st.subheader("Sign in")
    with st.form("login_form"):
        username = st.text_input("Username", value="demo")
        password = st.text_input("Password", value="automind123", type="password")
        submitted = st.form_submit_button("Sign in")
        if submitted:
            if login(username, password):
                st.rerun()
            else:
                st.error("Invalid credentials.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# --- Vehicle selection gate ---
if not st.session_state.vehicle_name:
    st.markdown('<div class="am-card">', unsafe_allow_html=True)
    st.subheader("Which vehicle are we working on?")
    st.caption("This lets AutoMind search the correct owner's manual first — "
               "if it can't find what you need there, it automatically checks all manuals.")

    vehicle_names = list(VEHICLE_MANUAL_MAP.keys())
    cols = st.columns(2)
    for i, name in enumerate(vehicle_names):
        with cols[i % 2]:
            label = f"{VEHICLE_ICONS.get(name, '🚗')}  {name}"
            if st.button(label, key=f"vehicle_{name}", use_container_width=True):
                st.session_state.vehicle_name = name
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# --- Chat UI ---
with st.sidebar:
    st.markdown("### Vehicle")
    st.caption(f"{VEHICLE_ICONS.get(st.session_state.vehicle_name, '🚗')} {st.session_state.vehicle_name}")
    if st.button("Change vehicle", use_container_width=True):
        st.session_state.vehicle_name = None
        st.rerun()

    st.markdown("---")
    st.markdown("### Session")
    st.code(st.session_state.session_id[:8], language=None)
    if st.button("Reset conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.markdown("---")
    st.markdown("### Try these")

    st.caption("Fault diagnosis")
    diagnosis_examples = [
        "My check engine light is on and the car is misfiring",
        "The brakes are squealing when I stop",
        "My car won't start, just a clicking noise",
        "Temperature gauge is in the red and steam is coming from the hood",
        "Slipping gears — RPM goes up but the car doesn't speed up",
        "One tire is wearing unevenly on the inside edge",
    ]
    for ex in diagnosis_examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": ex})
            result = send_message(ex)
            st.session_state.chat_history.append({"role": "assistant", "content": result["reply"]})
            st.rerun()

    st.caption("Manual lookup")
    manual_examples = [
        "How do I reset the tire pressure monitoring system?",
        "How do I pair my phone via Bluetooth?",
        "What's the recommended engine oil type?",
        "How do I use adaptive cruise control?",
    ]
    for ex in manual_examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": ex})
            result = send_message(ex)
            st.session_state.chat_history.append({"role": "assistant", "content": result["reply"]})
            st.rerun()

    st.caption("General")
    if st.button("Hey, what can you help me with?", key="general_ex", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": "Hey, what can you help me with?"})
        result = send_message("Hey, what can you help me with?")
        st.session_state.chat_history.append({"role": "assistant", "content": result["reply"]})
        st.rerun()

for msg in st.session_state.chat_history:
    avatar = "🧑" if msg["role"] == "user" else "🛠️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

user_input = st.chat_input("Describe an issue or ask about a feature...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.write(user_input)

    with st.chat_message("assistant", avatar="🛠️"):
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="am-diagnostic">&gt; Running diagnostics<span class="am-cursor">_</span></div>',
            unsafe_allow_html=True,
        )
        try:
            result = send_message(user_input)
            placeholder.empty()
            st.write(result["reply"])
            if result.get("needs_clarification"):
                st.caption("🔍 Narrowing down the cause — answer above to continue.")
        except requests.exceptions.ConnectionError:
            placeholder.empty()
            st.error("⚠️ Can't reach the API. Is the FastAPI server running on port 8000?")
            result = {"reply": ""}

    st.session_state.chat_history.append({"role": "assistant", "content": result["reply"]})
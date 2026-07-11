"""
AutoMind — Streamlit demo frontend.

Run:
    streamlit run frontend/app.py

This is a thin UI layer only — all the actual reasoning happens in the
FastAPI backend (src/api/main.py), which must be running separately on
port 8000 before you start this.
"""

import os
import uuid
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Must match src/config.py VEHICLE_MANUAL_MAP exactly
VEHICLE_MANUAL_MAP = {
    "Honda Civic (2025)": "2025 Civic sedan.pdf",
    "Toyota Corolla (2025)": "TOYOTA 2025(Corolla).pdf",
    "Tata Harrier BS6 (2026)": "harrier-bs6-owners-manual-april-2026.pdf",
    "Maruti Suzuki Jimny": "NEXA-Jimny-Petrol-Manual-latestpdf.pdf",
    "Not sure / Other": None,
}

st.set_page_config(page_title="AutoMind — Vehicle AI Copilot", page_icon="🚗", layout="centered")

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


# ---------------- UI ----------------
st.title("🚗 AutoMind")
st.caption("Agentic in-vehicle AI copilot — ask about a fault, or how to use a feature from your owner's manual.")

# --- Login gate ---
if not st.session_state.access_token:
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
    st.stop()

# --- Vehicle selection gate ---
if not st.session_state.vehicle_name:
    st.subheader("Which vehicle are we working on?")
    st.caption("This lets AutoMind search the correct owner's manual first. "
               "If it can't find what you need there, it'll automatically check all manuals.")
    choice = st.selectbox("Select your vehicle", list(VEHICLE_MANUAL_MAP.keys()))
    if st.button("Continue"):
        st.session_state.vehicle_name = choice
        st.rerun()
    st.stop()

# --- Chat UI ---
with st.sidebar:
    st.markdown("### Vehicle")
    st.caption(st.session_state.vehicle_name)
    if st.button("🚗 Change vehicle"):
        st.session_state.vehicle_name = None
        st.rerun()

    st.markdown("---")
    st.markdown("### Session")
    st.code(st.session_state.session_id[:8], language=None)
    if st.button("🔄 Reset conversation"):
        reset_conversation()
        st.rerun()

    st.markdown("---")
    st.markdown("### Try these")

    st.caption("🔧 Fault diagnosis")
    diagnosis_examples = [
        "My check engine light is on and the car is misfiring",
        "The brakes are squealing when I stop",
        "My car won't start, just a clicking noise",
        "Temperature gauge is in the red and steam is coming from the hood",
        "Slipping gears — RPM goes up but the car doesn't speed up",
        "One tire is wearing unevenly on the inside edge",
    ]
    for ex in diagnosis_examples:
        if st.button(ex, key=ex):
            st.session_state.chat_history.append({"role": "user", "content": ex})
            result = send_message(ex)
            st.session_state.chat_history.append({"role": "assistant", "content": result["reply"]})
            st.rerun()

    st.caption("📖 Manual lookup")
    manual_examples = [
        "How do I reset the tire pressure monitoring system?",
        "How do I pair my phone via Bluetooth?",
        "What's the recommended engine oil type?",
        "How do I use adaptive cruise control?",
    ]
    for ex in manual_examples:
        if st.button(ex, key=ex):
            st.session_state.chat_history.append({"role": "user", "content": ex})
            result = send_message(ex)
            st.session_state.chat_history.append({"role": "assistant", "content": result["reply"]})
            st.rerun()

    st.caption("💬 General")
    if st.button("Hey, what can you help me with?", key="general_ex"):
        st.session_state.chat_history.append({"role": "user", "content": "Hey, what can you help me with?"})
        result = send_message("Hey, what can you help me with?")
        st.session_state.chat_history.append({"role": "assistant", "content": result["reply"]})
        st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Describe an issue or ask about a feature...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = send_message(user_input)
                st.write(result["reply"])
                if result.get("needs_clarification"):
                    st.caption("🔍 Narrowing down the cause — answer above to continue.")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Can't reach the API. Is the FastAPI server running on port 8000?")
                result = {"reply": ""}

    st.session_state.chat_history.append({"role": "assistant", "content": result["reply"]})
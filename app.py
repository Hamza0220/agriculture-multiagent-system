import base64
import os
import sys
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from dotenv import load_dotenv

# Force reload of environment variables on every rerun
load_dotenv(override=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.input_validator import validate_input
from utils.location_handler import resolve_location
from utils.weather_interpreter import interpret_weather
from utils.response_formatter import format_response
from utils.error_handler import handle_error
from utils.tts_handler import text_to_speech, get_audio_html, _clean_for_speech

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Kisaan Dost — Pakistan Agricultural AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&display=swap');

:root {
    --green-800: #166016;
    --green-700: #1e7b1e;
    --green-600: #28a428;
    --green-500: #36b936;
    --green-400: #52cc52;
    --green-300: #80db80;
    --green-200: #b3ecb3;
    --green-100: #d9f5d9;
    --green-50 : #f0faf0;
    --green-25 : #f8fdf8;
    --white     : #ffffff;
    --gray-50   : #f9fafb;
    --gray-100  : #f3f4f6;
    --gray-200  : #e5e7eb;
    --gray-300  : #d1d5db;
    --gray-400  : #9ca3af;
    --gray-500  : #6b7280;
    --gray-600  : #4b5563;
    --gray-700  : #374151;
    --gray-800  : #1f2937;
    --blue-50   : #eff6ff; --blue-100 : #dbeafe; --blue-600 : #2563eb; --blue-700 : #1d4ed8;
    --yellow-50 : #fffbeb; --yellow-100: #fef3c7; --yellow-600: #d97706; --yellow-700: #92400e;
    --red-50    : #fef2f2; --red-100  : #fee2e2; --red-600  : #dc2626; --red-700  : #991b1b;
    --teal-50   : #f0fdfa; --teal-100 : #ccfbf1; --teal-600 : #0d9488; --teal-700 : #0f766e;
    --shadow-sm : 0 1px 3px rgba(0,0,0,0.08);
    --shadow-md : 0 4px 16px rgba(0,0,0,0.10);
    --r-sm:8px; --r-md:12px; --r-lg:18px;
}

* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; -webkit-font-smoothing: antialiased; color: var(--gray-800); }
.stApp { background: linear-gradient(160deg,#f0faf0 0%,#e8f7e8 50%,#f5fcf5 100%) !important; }
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--green-100) !important;
    box-shadow: 2px 0 12px rgba(22,96,22,0.06) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
.sb-brand {
    background: linear-gradient(135deg, var(--green-600), var(--green-800));
    padding: 20px 18px 16px; text-align: center;
}
.sb-brand-title { color:#fff; font-family:'Poppins',sans-serif; font-size:1.35rem; font-weight:700; }
.sb-brand-sub   { color:rgba(255,255,255,0.8); font-size:0.72rem; margin-top:3px; }
.sb-section { padding:12px 14px 4px; }
.sb-label { color:var(--gray-400); font-size:0.63rem; font-weight:600; letter-spacing:1.2px; text-transform:uppercase; margin-bottom:5px; padding:0 4px; }
.sb-divider { height:1px; background:var(--gray-100); margin:8px 14px; }
.sb-card {
    background: var(--green-25); border:1px solid var(--green-100);
    border-radius: var(--r-md); padding:10px 13px; margin:5px 14px;
}
.sb-card-title { color:var(--gray-400); font-size:0.63rem; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px; }
.sb-card-val   { color:var(--gray-800); font-size:0.88rem; font-weight:600; }
.sb-card-sub   { color:var(--gray-400); font-size:0.72rem; margin-top:1px; }

/* Nav buttons */
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.stButton) {
    margin-bottom: -10px !important;
}
.stButton > button {
    background: transparent !important; color: var(--gray-600) !important;
    border: none !important; border-radius: var(--r-sm) !important;
    font-family:'Inter',sans-serif !important; font-weight:500 !important;
    font-size:0.9rem !important; padding:8px 14px !important;
    display: inline-flex !important; justify-content: flex-start !important;
    text-align:left !important; width:100% !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}
.stButton > button div {
    display: flex !important; justify-content: flex-start !important; width: 100% !important;
}
.stButton > button p {
    text-align: left !important; margin: 0 !important;
}
.stButton > button:hover {
    background: var(--green-50) !important; color:var(--green-700) !important;
    transform: translateX(4px) !important;
}

/* ── PAGE HEADER ── */
.page-hdr {
    background: var(--white); border:1px solid var(--green-100);
    border-radius:var(--r-lg); padding:20px 26px; margin-bottom:18px;
    box-shadow:var(--shadow-sm); display:flex; align-items:center;
    justify-content:space-between; flex-wrap:wrap; gap:12px;
}
.page-hdr h1 { font-family:'Poppins',sans-serif; font-size:1.5rem; font-weight:700; color:var(--green-800); line-height:1.2; margin-bottom:3px; }
.page-hdr p  { color:var(--gray-500); font-size:0.85rem; }
.badge { padding:4px 11px; border-radius:20px; font-size:0.7rem; font-weight:600; border:1px solid; display:inline-block; margin:2px; }
.badge-g { background:var(--green-50); color:var(--green-700); border-color:var(--green-200); }
.badge-y { background:var(--yellow-50); color:var(--yellow-700); border-color:#fde68a; }
.badge-b { background:var(--blue-50); color:var(--blue-700); border-color:var(--blue-100); }
.badge-r { background:var(--red-50); color:var(--red-700); border-color:var(--red-100); }
.badge-t { background:var(--teal-50); color:var(--teal-700); border-color:var(--teal-100); }

/* ── STAT CARDS ── */
.stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:18px; }
.stat-card {
    background:var(--white); border:1px solid var(--gray-100);
    border-radius:var(--r-md); padding:16px 18px;
    box-shadow:var(--shadow-sm); transition:all 0.2s ease;
}
.stat-card:hover { box-shadow:var(--shadow-md); transform:translateY(-2px); }
.stat-card.bl { border-left:4px solid var(--green-400); }
.stat-card.by { border-left:4px solid #f59e0b; }
.stat-card.bb { border-left:4px solid #3b82f6; }
.stat-card.bt { border-left:4px solid #14b8a6; }
.stat-icon  { font-size:1.4rem; margin-bottom:5px; }
.stat-val   { font-size:1.25rem; font-weight:700; color:var(--gray-800); line-height:1.2; }
.stat-lbl   { font-size:0.68rem; color:var(--gray-400); font-weight:500; text-transform:uppercase; letter-spacing:0.5px; margin-top:2px; }

/* ── CARD WRAPPER ── */
.card {
    background:var(--white); border:1px solid var(--green-100);
    border-radius:var(--r-lg); padding:18px 20px;
    box-shadow:var(--shadow-sm); margin-bottom:14px;
}
.card-title { font-size:0.72rem; font-weight:600; color:var(--gray-400); text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid var(--gray-100); }

/* ── AGENT EXECUTION PANEL ── */
.agent-exec-wrap {
    background:var(--white); border:1px solid var(--green-100);
    border-radius:var(--r-lg); overflow:hidden; box-shadow:var(--shadow-sm); margin-bottom:14px;
}
.agent-exec-header {
    background:linear-gradient(135deg,var(--green-600),var(--green-700));
    padding:12px 18px; display:flex; align-items:center; gap:10px;
}
.agent-exec-header-title { color:#fff; font-size:0.88rem; font-weight:600; }
.agent-exec-header-sub   { color:rgba(255,255,255,0.75); font-size:0.73rem; margin-top:1px; }

.agent-row {
    padding:14px 18px; border-bottom:1px solid var(--gray-100);
    display:flex; align-items:flex-start; gap:14px;
}
.agent-row:last-child { border-bottom:none; }

.agent-icon-wrap {
    width:42px; height:42px; border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-size:1.25rem; flex-shrink:0;
}
.icon-green  { background:var(--green-50); }
.icon-blue   { background:var(--blue-50); }
.icon-yellow { background:var(--yellow-50); }
.icon-teal   { background:var(--teal-50); }
.icon-gray   { background:var(--gray-100); }

.agent-info { flex:1; min-width:0; }
.agent-name { font-size:0.9rem; font-weight:600; color:var(--gray-800); }
.agent-model { font-size:0.72rem; color:var(--gray-400); margin-top:1px; }
.agent-desc { font-size:0.8rem; color:var(--gray-500); margin-top:4px; line-height:1.5; }

.agent-status-wrap { flex-shrink:0; text-align:right; }
.status-pill {
    display:inline-flex; align-items:center; gap:5px;
    padding:3px 10px; border-radius:20px;
    font-size:0.7rem; font-weight:600;
}
.status-idle    { background:var(--gray-100); color:var(--gray-500); }
.status-running { background:#fef3c7; color:#92400e; }
.status-done    { background:var(--green-50); color:var(--green-700); }
.status-skip    { background:var(--gray-100); color:var(--gray-400); }
.status-error   { background:var(--red-50); color:var(--red-700); }

.pulse-dot {
    width:7px; height:7px; border-radius:50%;
    display:inline-block; animation:pulse 1.2s infinite;
}
.pulse-green  { background:var(--green-500); }
.pulse-orange { background:#f59e0b; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.75)} }

/* ── AGENT OUTPUT CARD ── */
.agent-out-card {
    border-radius:var(--r-md); padding:14px 16px; margin-top:8px;
}
.ao-green  { background:var(--green-50); border:1px solid var(--green-200); }
.ao-blue   { background:var(--blue-50); border:1px solid var(--blue-100); }
.ao-yellow { background:var(--yellow-50); border:1px solid #fde68a; }
.ao-gray   { background:var(--gray-50); border:1px solid var(--gray-200); }

.ao-section { margin-bottom:10px; }
.ao-section:last-child { margin-bottom:0; }
.ao-label { font-size:0.67rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px; }
.ao-label-green  { color:var(--green-700); }
.ao-label-blue   { color:var(--blue-700); }
.ao-label-yellow { color:var(--yellow-600); }
.ao-value { font-size:0.85rem; color:var(--gray-700); line-height:1.55; }
.ao-highlight {
    background:var(--white); border-radius:var(--r-sm);
    padding:6px 10px; display:inline-block; font-weight:600;
    color:var(--gray-800); font-size:0.88rem;
}

/* Confidence badge */
.conf-high   { background:#dcfce7; color:#15803d; padding:2px 9px; border-radius:12px; font-size:0.72rem; font-weight:700; }
.conf-medium { background:#fef3c7; color:#92400e; padding:2px 9px; border-radius:12px; font-size:0.72rem; font-weight:700; }
.conf-low    { background:#fee2e2; color:#991b1b; padding:2px 9px; border-radius:12px; font-size:0.72rem; font-weight:700; }

.severity-critical { background:#fee2e2; color:#991b1b; padding:2px 9px; border-radius:12px; font-size:0.72rem; font-weight:700; }
.severity-serious  { background:#fef3c7; color:#92400e; padding:2px 9px; border-radius:12px; font-size:0.72rem; font-weight:700; }
.severity-moderate { background:#fef9c3; color:#713f12; padding:2px 9px; border-radius:12px; font-size:0.72rem; font-weight:700; }
.severity-mild     { background:#dcfce7; color:#15803d; padding:2px 9px; border-radius:12px; font-size:0.72rem; font-weight:700; }

/* ── CHAT ── */
.chat-wrap { background:var(--white); border:1px solid var(--green-100); border-radius:var(--r-lg); padding:18px; box-shadow:var(--shadow-sm); margin-bottom:14px; }
.chat-title { font-size:0.72rem; font-weight:600; color:var(--gray-400); text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid var(--gray-100); }
[data-testid="stChatMessage"] { background:transparent !important; border:none !important; padding:5px 0 !important; }
[data-testid="stChatMessageContent"] {
    background:var(--green-25) !important; border:1px solid var(--green-100) !important;
    border-radius:var(--r-md) !important; padding:12px 16px !important;
    color:var(--gray-700) !important; font-size:0.88rem !important; line-height:1.7 !important;
    box-shadow:var(--shadow-sm) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background:linear-gradient(135deg,var(--green-50),var(--white)) !important;
    border-color:var(--green-200) !important;
}
.stChatInputContainer {
    background:var(--white) !important; border:1.5px solid var(--green-200) !important;
    border-radius:var(--r-md) !important; box-shadow:var(--shadow-sm) !important;
}
.stChatInputContainer:focus-within { border-color:var(--green-500) !important; box-shadow: 0 0 0 3px rgba(54,185,54,0.1) !important; }
.stChatInput textarea { background:transparent !important; color:var(--gray-800) !important; font-family:'Inter',sans-serif !important; font-size:0.88rem !important; }
.stChatInput textarea::placeholder { color:var(--gray-400) !important; }

/* ── WEATHER ── */
.wx-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.wx-item { text-align:center; padding:10px; background:var(--green-25); border-radius:var(--r-sm); border:1px solid var(--green-100); }
.wx-item-val { font-size:1.1rem; font-weight:700; color:var(--gray-800); }
.wx-item-lbl { font-size:0.68rem; color:var(--gray-400); margin-top:2px; }
.fc-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:9px; }
.fc-card { background:var(--green-25); border:1px solid var(--green-100); border-radius:var(--r-sm); padding:11px 8px; text-align:center; }
.fc-day  { font-size:0.68rem; font-weight:600; color:var(--gray-400); text-transform:uppercase; }
.fc-temp { font-size:0.95rem; font-weight:700; color:var(--gray-800); margin:4px 0; }
.fc-rain { font-size:0.68rem; color:#3b82f6; font-weight:500; }
.fc-note { font-size:0.63rem; color:var(--gray-500); margin-top:4px; line-height:1.3; }

/* ── DASHBOARD GRID ── */
.dash-grid { display:grid; grid-template-columns:1.4fr 1fr; gap:14px; }
.dash-grid-3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }

/* ── ACTIVITY LOG ── */
.act-item { display:flex; align-items:flex-start; gap:10px; padding:8px 0; border-bottom:1px solid var(--gray-100); }
.act-item:last-child { border-bottom:none; }
.act-time { color:var(--gray-400); font-size:0.7rem; white-space:nowrap; padding-top:2px; min-width:52px; }
.act-text { color:var(--gray-600); font-size:0.82rem; line-height:1.4; }
.act-dot  { width:8px; height:8px; border-radius:50%; margin-top:5px; flex-shrink:0; }
.dot-g { background:var(--green-400); }
.dot-b { background:#3b82f6; }
.dot-y { background:#f59e0b; }
.dot-r { background:#ef4444; }

/* ── CROP TAG ── */
.crop-tags-wrap { display:flex; flex-wrap:wrap; gap:6px; }
.crop-tag { background:var(--green-50); border:1px solid var(--green-200); color:var(--green-700); font-size:0.73rem; font-weight:500; padding:4px 11px; border-radius:20px; }

/* ── FORM / INPUT ── */
.stTextInput > div > div > input {
    background:var(--white) !important; border:1.5px solid var(--green-200) !important;
    border-radius:var(--r-sm) !important; color:var(--gray-800) !important;
    font-family:'Inter',sans-serif !important; font-size:0.87rem !important; padding:8px 12px !important;
}
.stTextInput > div > div > input:focus { border-color:var(--green-500) !important; box-shadow:0 0 0 3px rgba(54,185,54,0.1) !important; outline:none !important; }
.stTextInput > label { color:var(--gray-600) !important; font-size:0.8rem !important; font-weight:500 !important; }

[data-testid="stMetric"] { background:var(--white) !important; border:1px solid var(--gray-100) !important; border-radius:var(--r-md) !important; padding:12px 14px !important; box-shadow:var(--shadow-sm) !important; }
[data-testid="stMetricLabel"] { color:var(--gray-500) !important; font-size:0.72rem !important; }
[data-testid="stMetricValue"] { color:var(--gray-800) !important; font-size:1.1rem !important; font-weight:700 !important; }

[data-testid="stFileUploader"] { background:var(--green-25) !important; border:2px dashed var(--green-200) !important; border-radius:var(--r-md) !important; }
[data-testid="stFileUploader"]:hover { border-color:var(--green-400) !important; }
[data-testid="stFileUploader"] label { color:var(--gray-600) !important; font-size:0.82rem !important; }

.stSuccess > div { background:#f0fdf4 !important; color:#15803d !important; border-radius:var(--r-sm) !important; border-color:#bbf7d0 !important; font-size:0.82rem !important; }
.stInfo    > div { background:var(--blue-50) !important; color:var(--blue-700) !important; border-radius:var(--r-sm) !important; border-color:var(--blue-100) !important; font-size:0.82rem !important; }
.stWarning > div { background:var(--yellow-50) !important; color:var(--yellow-700) !important; border-radius:var(--r-sm) !important; border-color:#fde68a !important; font-size:0.82rem !important; }
.stCaption, small { color:var(--gray-400) !important; font-size:0.76rem !important; }
hr { border:none; border-top:1px solid var(--gray-100) !important; margin:12px 0 !important; }
details { background:var(--white) !important; border:1px solid var(--green-100) !important; border-radius:var(--r-md) !important; }
details > summary { color:var(--green-700) !important; font-weight:600 !important; font-size:0.87rem !important; padding:11px 15px !important; }
::-webkit-scrollbar { width:5px; } ::-webkit-scrollbar-track { background:var(--gray-100); } ::-webkit-scrollbar-thumb { background:var(--green-300); border-radius:3px; }
.tts-btn button {
    background: linear-gradient(135deg, var(--green-500), var(--green-600)) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    padding: 8px 16px !important; font-size: 0.82rem !important;
    font-weight: 600 !important; transition: all 0.3s ease !important;
}
.tts-btn button:hover { background: var(--green-800) !important; transform: scale(1.03) !important; }
@media(max-width:900px){ .stat-grid{grid-template-columns:1fr 1fr;} .dash-grid{grid-template-columns:1fr;} .fc-grid{grid-template-columns:1fr;} }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
DEFAULTS = {
    "messages": [], "user_location": "Pakistan", "user_crop": None,
    "history": [], "agents_last_called": [], "total_queries": 0,
    "last_urgency": "MEDIUM", "active_page": "📊 Dashboard",
    "activity_log": [],
    # Agent execution tracking
    "agent_statuses": {},     # {agent_name: "idle"|"running"|"done"|"skip"|"error"}
    "agent_outputs": {},      # {agent_name: raw dict output}
    "orchestration_data": {}, # last orchestrator output
    "pipeline_running": False,
    "last_query_time": None,
    "last_response": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def process_image(f):
    return base64.b64encode(f.getvalue()).decode("utf-8") if f else None

def get_season():
    m = datetime.now().month
    if m in [12,1,2]:    return "❄️ Winter (Rabi)"
    if m in [3,4,5]:     return "🌸 Spring"
    if m in [6,7,8,9]:   return "☀️ Kharif (Monsoon)"
    return "🍂 Autumn"

def add_log(msg, dot="dot-g"):
    t = datetime.now().strftime("%H:%M:%S")
    st.session_state.activity_log.insert(0, {"time": t, "msg": msg, "dot": dot})
    st.session_state.activity_log = st.session_state.activity_log[:20]

import re as _re
def _strip(val) -> str:
    """Strip HTML tags that LLMs sometimes embed inside JSON string values."""
    if not val:
        return "—"
    return _re.sub(r"<[^>]+>", "", str(val)).strip() or "—"

AGENT_META = {
    "ORCHESTRATOR":        {"icon":"🧠","name":"Orchestrator","color":"icon-teal", "model":"OpenAI GPT-4o-mini", "desc":"Query analyze karke decide karta hai kaunse agents call honge"},
    "CROP_DOCTOR":         {"icon":"🩺","name":"Crop Doctor","color":"icon-green","model":"OpenAI GPT-4o-mini (Vision)","desc":"Fasal ki bimarion, keeron aur nutrient ki pehchan"},
    "IRRIGATION_ADVISOR":  {"icon":"💧","name":"Irrigation Advisor","color":"icon-blue","model":"OpenAI GPT-4o-mini + OpenWeatherMap","desc":"Paani ki zaroorat aur irrigation schedule"},
    "MARKET_PRICE":        {"icon":"📈","name":"Market Price","color":"icon-yellow","model":"OpenAI GPT-4o-mini + Tavily Search","desc":"Mandi rates aur selling advice"},
    "SYNTHESIZER":         {"icon":"✍️","name":"Response Synthesizer","color":"icon-teal","model":"OpenAI GPT-4o-mini","desc":"Saare agents ki output ko ek jawab mein combine karta hai"},
}

CROPS = ["Wheat","Rice","Cotton","Sugarcane","Maize","Tomato","Onion","Potato","Mango","Chilli","Mustard"]
NAV = [("📊","Dashboard"),("💬","Chat"),("🌤️","Weather"),("🌱","Crops & Info"),("❓","Help")]

# ─────────────────────────────────────────────
#  PIPELINE (with tracking) — thread-safe
# ─────────────────────────────────────────────
def run_pipeline_with_tracking(user_query, image_base64, location, crop_name, conversation_history):
    """
    Runs the full pipeline.
    IMPORTANT: st.session_state must NOT be accessed inside ThreadPoolExecutor
    worker threads — Streamlit is not thread-safe. We use local thread-safe
    dicts + threading.Lock, then copy everything to session_state from main thread.
    """
    from agents.orchestrator import orchestrate
    from agents.crop_doctor import diagnose
    from agents.irrigation_advisor import advise_irrigation
    from agents.market_price_agent import get_market_advice
    from agents.response_synthesizer import synthesize_response

    try:
        from rag.retriever import retrieve_crop_knowledge
    except ImportError:
        def retrieve_crop_knowledge(query, crop_name=None, category=None, top_k=5):
            return "RAG knowledge base not yet available."

    # ── Thread-safe local state (NO st.session_state inside threads!) ──
    _lock         = threading.Lock()
    _statuses     = {k: "idle" for k in AGENT_META}   # local copy
    _outputs      = {}                                  # local copy
    _logs         = []                                  # local log buffer

    def _local_log(msg, dot="dot-g"):
        t = datetime.now().strftime("%H:%M:%S")
        with _lock:
            _logs.append({"time": t, "msg": msg, "dot": dot})

    # ── Reset session state (main thread — safe) ──
    st.session_state.agent_statuses = {k: "idle" for k in AGENT_META}
    st.session_state.agent_outputs  = {}
    st.session_state.pipeline_running = True
    add_log("Orchestrator: query routing started", "dot-b")

    # ── 1. ORCHESTRATOR (main thread) ──
    _statuses["ORCHESTRATOR"] = "running"
    orchestration = orchestrate(
        user_query=user_query,
        has_image=image_base64 is not None,
        location=location,
        conversation_history=conversation_history,
    )
    _statuses["ORCHESTRATOR"] = "done"
    _outputs["ORCHESTRATOR"]  = orchestration
    _local_log(f"Orchestrator done → agents: {orchestration.get('agents_to_call', [])}", "dot-g")

    agents_to_call = ["CROP_DOCTOR", "IRRIGATION_ADVISOR", "MARKET_PRICE"]

    # Initialize all agent statuses to idle/skip first
    for ag in ["CROP_DOCTOR", "IRRIGATION_ADVISOR", "MARKET_PRICE"]:
        if ag not in agents_to_call:
            _statuses[ag] = "skip"

    crop_doctor_result = irrigation_result = market_price_result = None

    # ── 2. PARALLEL AGENTS — ONLY local vars, NO st.session_state ──
    def run_crop_doctor():
        _local_log("Crop Doctor: analyzing...", "dot-y")
        with _lock: _statuses["CROP_DOCTOR"] = "running"
        ctx = orchestration.get("context_for_agents", "")
        rag = retrieve_crop_knowledge(ctx, orchestration.get("crop_detected"), "DISEASE", 5)
        res = diagnose(
            farmer_description=ctx,
            crop_name=orchestration.get("crop_detected"),
            image_base64=image_base64,
            location=orchestration.get("location", "Pakistan"),
            season=orchestration.get("season"),
            weather_summary="See weather page",
            rag_context=rag,
        )
        with _lock:
            _statuses["CROP_DOCTOR"] = "done"
            _outputs["CROP_DOCTOR"]  = res
        _local_log("Crop Doctor: diagnosis complete", "dot-g")
        return res

    def run_irrigation():
        _local_log("Irrigation Advisor: calculating...", "dot-y")
        with _lock: _statuses["IRRIGATION_ADVISOR"] = "running"
        ctx = orchestration.get("context_for_agents", "")
        rag = retrieve_crop_knowledge(ctx, orchestration.get("crop_detected"), "IRRIGATION", 5)
        res = advise_irrigation(
            crop_name=orchestration.get("crop_detected"),
            crop_urdu=orchestration.get("crop_urdu"),
            location=orchestration.get("location", "Pakistan"),
            field_size_acres=1,
            irrigation_type="tube well",
            rag_irrigation_context=rag,
        )
        with _lock:
            _statuses["IRRIGATION_ADVISOR"] = "done"
            _outputs["IRRIGATION_ADVISOR"]  = res
        _local_log("Irrigation Advisor: recommendation ready", "dot-g")
        return res

    def run_market():
        _local_log("Market Price: fetching mandi data...", "dot-y")
        with _lock: _statuses["MARKET_PRICE"] = "running"
        res = get_market_advice(
            crop_name=orchestration.get("crop_detected", "crop"),
            crop_urdu=orchestration.get("crop_urdu"),
            location=orchestration.get("location", "Pakistan"),
        )
        with _lock:
            _statuses["MARKET_PRICE"] = "done"
            _outputs["MARKET_PRICE"]  = res
        _local_log("Market Price: mandi data received", "dot-g")
        return res

    tasks = {}
    if "CROP_DOCTOR"        in agents_to_call: tasks["CROP_DOCTOR"]        = run_crop_doctor
    if "IRRIGATION_ADVISOR" in agents_to_call: tasks["IRRIGATION_ADVISOR"] = run_irrigation
    if "MARKET_PRICE"       in agents_to_call: tasks["MARKET_PRICE"]       = run_market

    if tasks:
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {ex.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if   name == "CROP_DOCTOR":        crop_doctor_result = result
                    elif name == "IRRIGATION_ADVISOR": irrigation_result  = result
                    elif name == "MARKET_PRICE":       market_price_result = result
                except Exception as e:
                    with _lock:
                        _statuses[name] = "error"
                    _local_log(f"{name} failed: {str(e)[:80]}", "dot-r")

    # ── 3. SYNTHESIZER (main thread — safe) ──
    _statuses["SYNTHESIZER"] = "running"
    _local_log("Synthesizer: combining responses...", "dot-b")

    detected_crop = orchestration.get("crop_detected") or crop_name or "crop"
    response_text = synthesize_response(
        original_query=user_query,
        location=orchestration.get("location", location),
        crop_name=detected_crop,
        language=orchestration.get("farmer_language", "roman_urdu"),
        crop_doctor_output=crop_doctor_result,
        irrigation_output=irrigation_result,
        market_price_output=market_price_result,
        agents_called=agents_to_call,
    )
    _statuses["SYNTHESIZER"] = "done"
    formatted = format_response(
        synthesizer_output=response_text,
        agents_called=agents_to_call,
        urgency=orchestration.get("urgency", "MEDIUM"),
    )
    _local_log("Response synthesized ✓", "dot-g")

    # ── Copy all local state to session_state (MAIN THREAD ONLY) ──
    st.session_state.agent_statuses   = dict(_statuses)
    st.session_state.agent_outputs    = dict(_outputs)
    st.session_state.orchestration_data = orchestration
    st.session_state.pipeline_running = False
    st.session_state.last_query_time  = datetime.now().strftime("%H:%M:%S")
    st.session_state.last_response    = formatted

    # Flush buffered logs (in chronological order)
    for entry in reversed(_logs):
        st.session_state.activity_log.insert(0, entry)
    st.session_state.activity_log = st.session_state.activity_log[:20]

    return {
        "response_text": response_text,
        "formatted":     formatted,
        "agents_called": agents_to_call,
        "urgency":       orchestration.get("urgency", "MEDIUM"),
        "crop_detected": detected_crop,
    }

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div class="sb-brand">
        <div class="sb-brand-title">🌾 Kisaan Dost</div>
        <div class="sb-brand-sub">Pakistan Ka AI Agriculture Assistant</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">Navigation</div></div>', unsafe_allow_html=True)
    for icon, label in NAV:
        key = f"{icon} {label}"
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.active_page = key
            st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section"><div class="sb-label">Location</div></div>', unsafe_allow_html=True)
    loc_val = st.session_state.user_location if st.session_state.user_location != "Pakistan" else ""
    location_input = st.text_input("Location", value=loc_val, placeholder="Shehar ya Zila (e.g. Lahore)", key="loc_in", label_visibility="collapsed")

    # Resolve the freshest location for weather display
    if location_input:
        li = resolve_location(location_input)
        current_loc = li["city"]
        st.session_state.user_location = current_loc
        if li["province"]:
            st.success(f"📍 {current_loc}, {li['province']}")
        else:
            st.caption(f"📍 {current_loc}")
    else:
        current_loc = st.session_state.user_location
        st.caption(f"📍 {current_loc}")

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    crop_v = st.session_state.user_crop or "—"
    st.markdown(f"""<div class="sb-card">
        <div class="sb-card-title">🌿 Detected Crop</div>
        <div class="sb-card-val">{crop_v}</div>
        <div class="sb-card-sub">{get_season()}</div>
    </div>""", unsafe_allow_html=True)

    try:
        # Always use current_loc — the freshly-resolved location this render cycle
        w = interpret_weather(current_loc)
        c = w.get("current", {})
        temp = f"{c['temp_c']}°C" if c.get("temp_c") else "N/A"
        hum  = f"{c['humidity_pct']}%" if c.get("humidity_pct") else "N/A"
        cond = (c.get("condition") or "N/A").title()
        st.markdown(f"""<div class="sb-card">
            <div class="sb-card-title">🌤️ {current_loc} Weather</div>
            <div class="sb-card-val">{temp} · {hum}</div>
            <div class="sb-card-sub">{cond}</div>
        </div>""", unsafe_allow_html=True)
    except:
        pass


    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown("""<div class="sb-card">
        <div class="sb-card-title">📞 Helpline</div>
        <div class="sb-card-val">0800-15000</div>
        <div class="sb-card-sub">Free · Mon–Sat 8am–5pm</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("")
    if st.button("🗑️  Clear All Data", use_container_width=True, key="clear"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v if not isinstance(v, list) else []
        st.session_state.active_page = "📊 Dashboard"
        st.rerun()

# ─────────────────────────────────────────────
#  REUSABLE AGENT STATUS PANEL
# ─────────────────────────────────────────────
def render_agent_panel(show_outputs=True):
    statuses = st.session_state.agent_statuses
    if not statuses:
        return

    running_any = any(v == "running" for v in statuses.values())
    done_count  = sum(1 for v in statuses.values() if v == "done")
    total_ag    = len([k for k in statuses if k != "ORCHESTRATOR"])

    st.markdown(f"""
    <div class="agent-exec-wrap">
        <div class="agent-exec-header">
            <div>
                <div class="agent-exec-header-title">
                    {'⚡ ' if running_any else '✅ '}Agent Execution Pipeline
                </div>
                <div class="agent-exec-header-sub">
                    {'Processing...' if running_any else f'Completed — {done_count} agents ran'} · {st.session_state.user_location}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    agent_order = ["ORCHESTRATOR","CROP_DOCTOR","IRRIGATION_ADVISOR","MARKET_PRICE","SYNTHESIZER"]

    for ag_key in agent_order:
        if ag_key not in statuses:
            continue
        meta   = AGENT_META[ag_key]
        status = statuses.get(ag_key, "idle")

        # Status pill
        if   status == "running": pill = '<span class="status-pill status-running"><span class="pulse-dot pulse-orange"></span>Running</span>'
        elif status == "done":    pill = '<span class="status-pill status-done"><span class="pulse-dot pulse-green"></span>Done</span>'
        elif status == "skip":    pill = '<span class="status-pill status-skip">Skipped</span>'
        elif status == "error":   pill = '<span class="status-pill status-error">Error</span>'
        else:                     pill = '<span class="status-pill status-idle">Idle</span>'

        st.markdown(f"""
        <div class="agent-row">
            <div class="agent-icon-wrap {meta['color']}">{meta['icon']}</div>
            <div class="agent-info">
                <div class="agent-name">{meta['name']}</div>
                <div class="agent-model">🔧 {meta['model']}</div>
                <div class="agent-desc">{meta['desc']}</div>
            </div>
            <div class="agent-status-wrap">{pill}</div>
        </div>
        """, unsafe_allow_html=True)

        # Show output details if done and outputs available
        if show_outputs and status == "done" and ag_key in st.session_state.agent_outputs:
            out = st.session_state.agent_outputs[ag_key]
            _render_agent_output(ag_key, out)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_agent_output(ag_key, out):
    """Render structured output cards per agent."""
    if not out or not isinstance(out, dict):
        return

    if ag_key == "ORCHESTRATOR":
        agents_called = out.get("agents_to_call", [])
        crop = out.get("crop_detected","—") or "—"
        lang = out.get("farmer_language","—")
        urgency = out.get("urgency","—")
        prob = ", ".join(out.get("problem_type",[]) or []) or "—"
        season = out.get("season","—") or "—"
        call_order = out.get("call_order","parallel")

        urg_cls = {"CRITICAL":"conf-low","HIGH":"conf-medium","MEDIUM":"conf-high","LOW":"conf-high"}.get(urgency,"conf-high")
        st.markdown(f"""
        <div style="padding:0 18px 14px 74px;">
        <div class="agent-out-card ao-gray">
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:0.8rem;">
                <div><span style="color:#9ca3af;font-size:0.65rem;text-transform:uppercase;font-weight:700;">Crop</span><br><b>{crop}</b></div>
                <div><span style="color:#9ca3af;font-size:0.65rem;text-transform:uppercase;font-weight:700;">Season</span><br><b>{season}</b></div>
                <div><span style="color:#9ca3af;font-size:0.65rem;text-transform:uppercase;font-weight:700;">Urgency</span><br><span class="{urg_cls}">{urgency}</span></div>
                <div><span style="color:#9ca3af;font-size:0.65rem;text-transform:uppercase;font-weight:700;">Language</span><br><b>{lang}</b></div>
                <div><span style="color:#9ca3af;font-size:0.65rem;text-transform:uppercase;font-weight:700;">Problem Type</span><br><b>{prob}</b></div>
                <div><span style="color:#9ca3af;font-size:0.65rem;text-transform:uppercase;font-weight:700;">Call Order</span><br><b>{call_order}</b></div>
            </div>
            <div style="margin-top:8px;font-size:0.72rem;color:#6b7280;">
                <b>Agents Activated:</b> {"  ·  ".join([f"<span class='badge-g badge'>{a}</span>" for a in agents_called]) if agents_called else "None"}
            </div>
        </div></div>
        """, unsafe_allow_html=True)

    elif ag_key == "CROP_DOCTOR":
        diag = out.get("diagnosis", {})
        primary_name = diag.get("primary","—")
        urdu_name  = diag.get("urdu_name","")
        confidence = diag.get("confidence","—")
        severity   = diag.get("severity","—")
        time_act   = diag.get("time_to_act","—")
        alts       = diag.get("alternatives", [])

        conf_cls = {"HIGH":"conf-high","MEDIUM":"conf-medium","LOW":"conf-low"}.get(confidence,"conf-medium")
        sev_cls  = {"CRITICAL":"severity-critical","SERIOUS":"severity-serious","MODERATE":"severity-moderate","MILD":"severity-mild"}.get(severity,"severity-moderate")

        treat = out.get("treatment",{})
        opt_a = treat.get("option_a",{})
        opt_b = treat.get("option_b",{})

        vis = out.get("visual_observations",{})
        affected = ", ".join(vis.get("affected_parts",[]) or []) or "—"
        symptoms = "; ".join(vis.get("symptoms_seen",[]) or []) or "—"
        spread   = vis.get("spread_estimate","—") or "—"

        alts_str = ", ".join(alts) if alts else "None"
        prevention = out.get("prevention_next_season",[]) or []

        st.markdown(f"""
<div style="padding:0 18px 14px 74px;">
<div class="agent-out-card ao-green">
<div style="display:flex;justify-content:space-between;margin-bottom:12px;border-bottom:1px solid #d9f5d9;padding-bottom:12px;">
<div>
<div style="font-size:0.75rem;color:#1e7b1e;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">🩺 Primary Diagnosis</div>
<div style="font-size:1.15rem;font-weight:700;color:#1f2937;">{primary_name}</div>
{f'<span style="color:#6b7280;font-size:0.8rem;">({urdu_name})</span>' if urdu_name else ''}
<span class="{conf_cls}">{confidence} Confidence</span>
<span class="{sev_cls}">{severity}</span>
</div>
<div style="font-size:0.8rem;color:#4b5563;">⏰ Act: {time_act}</div>
{f'<div style="font-size:0.78rem;color:#9ca3af;margin-top:3px;">Alternatives: {alts_str}</div>' if alts else ''}
</div>
<div class="ao-section">
<div class="ao-label ao-label-green">👁️ Visual Observations</div>
<div style="font-size:0.8rem;color:#4b5563;line-height:1.6;">
Affected: <b>{affected}</b> · Spread: <b>{spread}</b><br>
Symptoms: {symptoms[:120]}{'...' if len(symptoms)>120 else ''}
</div>
</div>
{'<div class="ao-section"><div class="ao-label ao-label-green">💊 Treatment Option A</div><div style="font-size:0.8rem;color:#4b5563;line-height:1.6;"><b>'+opt_a.get("product","—")+'</b> · Brands: '+", ".join(opt_a.get("local_brands",[]) or [])+'<br>Dose/Acre: '+opt_a.get("dose_per_acre","—")+'  ·  Cost: '+opt_a.get("cost_per_acre","—")+'</div></div>' if opt_a.get("product") else ''}
{'<div class="ao-section"><div class="ao-label ao-label-green">💊 Treatment Option B (Budget)</div><div style="font-size:0.8rem;color:#4b5563;line-height:1.6;"><b>'+opt_b.get("product","—")+'</b> · '+", ".join(opt_b.get("local_brands",[]) or [])+'  ·  Cost: '+opt_b.get("cost_per_acre","—")+'</div></div>' if opt_b.get("product") else ''}
{('<div class="ao-section"><div class="ao-label ao-label-green">🛡️ Prevention</div><ul style="font-size:0.79rem;color:#4b5563;padding-left:16px;line-height:1.7;">'+''.join(f"<li>{p}</li>" for p in prevention[:3])+'</ul></div>') if prevention else ''}
</div></div>
""", unsafe_allow_html=True)

    elif ag_key == "IRRIGATION_ADVISOR":
        stress   = out.get("water_stress_level", "—")
        irrigate = out.get("irrigate_today", False)
        delay    = _strip(out.get("delay_reason", ""))
        advice   = out.get("irrigation_advice", {})
        cost     = out.get("cost_saving", {})
        et0      = out.get("et0_mm_per_day", "—")
        wr       = out.get("weather_raw", {})

        # Strip any LLM-injected HTML from JSON values
        amount      = _strip(advice.get("amount", "—"))
        timing      = _strip(advice.get("timing", "—"))
        next_irr    = _strip(advice.get("next_irrigation", "—"))
        std_cost    = _strip(cost.get("standard_farmer_cost", "—"))
        monthly_sav = _strip(cost.get("monthly_saving", "—"))
        opt_tip     = _strip(cost.get("optimization_tip", "—"))

        stress_cls  = {"HIGH": "conf-low", "MODERATE": "conf-medium", "LOW": "conf-high"}.get(str(stress).upper(), "conf-medium")
        irr_str = "✅ YES — Irrigate Today" if irrigate else "⏳ WAIT — Delay Irrigation"
        irr_col = "#15803d" if irrigate else "#92400e"
        irr_bg  = "#f0fdf4" if irrigate else "#fffbeb"

        temp_d = f"{wr.get('temp_c', '?')}°C" if wr.get("success") else "N/A"
        hum_d  = f"{wr.get('humidity', '?')}%" if wr.get("success") else "N/A"
        delay_html = f'<div style="font-size:0.78rem;color:#6b7280;margin-top:4px;">Reason: {delay}</div>' if delay and delay != "—" else ""

        st.markdown(f"""
<div style="padding:0 18px 14px 74px;">
<div class="agent-out-card ao-blue">
<div class="ao-section">
<div class="ao-label ao-label-blue">💧 Water Analysis</div>
<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
<span>Stress Level: <span class="{stress_cls}">{stress}</span></span>
<span>ET₀: <b>{et0} mm/day</b></span>
<span>Temp: <b>{temp_d}</b></span>
<span>Humidity: <b>{hum_d}</b></span>
</div>
<div style="background:{irr_bg};border-radius:8px;padding:7px 11px;display:inline-block;font-size:0.85rem;font-weight:600;color:{irr_col};">
{irr_str}
</div>
{delay_html}
</div>
<div class="ao-section">
<div class="ao-label ao-label-blue">📋 Irrigation Schedule</div>
<div style="font-size:0.8rem;color:#1e40af;line-height:1.7;">
Amount: <b>{amount}</b><br>
Timing: <b>{timing}</b><br>
Next: <b>{next_irr}</b>
</div>
</div>
<div class="ao-section">
<div class="ao-label ao-label-blue">💰 Cost Saving</div>
<div style="font-size:0.8rem;color:#1e40af;line-height:1.6;">
Current Cost: {std_cost}<br>
Monthly Saving: <b>{monthly_sav}</b><br>
Tip: {opt_tip}
</div>
</div>
</div></div>
""", unsafe_allow_html=True)

    elif ag_key == "MARKET_PRICE":
        pd_  = out.get("price_data",{})
        sa   = out.get("selling_advice",{})
        tp   = out.get("transportation",{})
        tips = out.get("mandi_tips",[]) or []

        price_40 = pd_.get("price_per_maund_40kg","—")
        price_kg = pd_.get("price_per_kg","—")
        trend    = pd_.get("trend","—")
        vs_week  = pd_.get("compared_to_last_week","—")
        mandi    = pd_.get("best_mandi","—")
        fresh    = pd_.get("data_freshness","—")
        rec      = sa.get("recommendation","—")
        reason   = sa.get("reason","—")
        trans    = tp.get("estimated_cost","—")

        trend_col = "#15803d" if "RISING" in (trend or "") else ("#dc2626" if "FALLING" in (trend or "") else "#92400e")
        rec_col   = "#15803d" if rec=="SELL NOW" else "#92400e"
        rec_bg    = "#f0fdf4" if rec=="SELL NOW" else "#fffbeb"
        fresh_cls = "conf-high" if fresh=="CURRENT" else "conf-low"

        st.markdown(f"""
<div style="padding:0 18px 14px 74px;">
<div class="agent-out-card ao-yellow">
<div class="ao-section">
<div class="ao-label ao-label-yellow">💰 Mandi Prices</div>
<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:6px;">
<div style="text-align:center;background:#fff;border-radius:8px;padding:8px 14px;border:1px solid #fde68a;">
<div style="font-size:1rem;font-weight:700;color:#1f2937;">{price_40}</div>
<div style="font-size:0.65rem;color:#9ca3af;text-transform:uppercase;">Per Maund (40kg)</div>
</div>
<div style="text-align:center;background:#fff;border-radius:8px;padding:8px 14px;border:1px solid #fde68a;">
<div style="font-size:1rem;font-weight:700;color:#1f2937;">{price_kg}</div>
<div style="font-size:0.65rem;color:#9ca3af;text-transform:uppercase;">Per KG</div>
</div>
</div>
<div style="font-size:0.79rem;color:#4b5563;line-height:1.65;">
Trend: <b style="color:{trend_col};">{trend}</b>  ·  vs Last Week: <b>{vs_week}</b>  ·  Best Mandi: <b>{mandi}</b><br>
Data: <span class="{fresh_cls}">{fresh}</span>
</div>
</div>
<div class="ao-section">
<div class="ao-label ao-label-yellow">📊 Selling Advice</div>
<div style="background:{rec_bg};border-radius:8px;padding:7px 11px;font-size:0.87rem;font-weight:700;color:{rec_col};margin-bottom:5px;">{rec}</div>
<div style="font-size:0.8rem;color:#4b5563;line-height:1.55;">{reason[:150]}{'...' if len(reason)>150 else ''}</div>
</div>
{'<div class="ao-section"><div class="ao-label ao-label-yellow">🚛 Transport Cost</div><div style="font-size:0.8rem;color:#4b5563;">'+trans+'</div></div>' if trans and trans != "—" else ''}
{('<div class="ao-section"><div class="ao-label ao-label-yellow">💡 Mandi Tips</div><ul style="font-size:0.79rem;color:#4b5563;padding-left:16px;line-height:1.7;">'+''.join(f"<li>{t}</li>" for t in tips[:3])+'</ul></div>') if tips else ''}
</div></div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════
page = st.session_state.active_page

if "Dashboard" in page:
    st.markdown("""<div class="page-hdr">
        <div>
            <h1>📊 Live Dashboard</h1>
            <p>Saari cheezain ek jagah — real-time overview with agent activity</p>
        </div>
        <div>
            <span class="badge badge-g">🤖 3 AI Agents</span>
            <span class="badge badge-b">🧠 RAG Powered</span>
            <span class="badge badge-y">📸 Vision AI</span>
            <span class="badge badge-t">📡 Live Weather</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Stats ──
    total = st.session_state.total_queries
    crop_d = st.session_state.user_crop or "—"
    loc_d  = st.session_state.user_location
    ag_run = len([v for v in st.session_state.agent_statuses.values() if v=="done"])

    st.markdown(f"""<div class="stat-grid">
        <div class="stat-card bl"><div class="stat-icon">💬</div><div class="stat-val">{total}</div><div class="stat-lbl">Total Queries</div></div>
        <div class="stat-card by"><div class="stat-icon">🌿</div><div class="stat-val">{crop_d}</div><div class="stat-lbl">Detected Crop</div></div>
        <div class="stat-card bb"><div class="stat-icon">📍</div><div class="stat-val">{loc_d}</div><div class="stat-lbl">Location</div></div>
        <div class="stat-card bt"><div class="stat-icon">🤖</div><div class="stat-val">{ag_run}</div><div class="stat-lbl">Agents Completed</div></div>
    </div>""", unsafe_allow_html=True)

    # ── Main Dashboard Grid ──
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        # Agent Execution Panel
        if st.session_state.agent_statuses:
            render_agent_panel(show_outputs=True)
        else:
            st.markdown("""<div class="card">
                <div class="card-title">🤖 Agent Execution Pipeline</div>
                <div style="text-align:center;padding:30px 20px;color:#9ca3af;">
                    <div style="font-size:2rem;margin-bottom:8px;">⚡</div>
                    <div style="font-size:0.9rem;font-weight:600;color:#6b7280;">Koi query abhi tak nahi ki</div>
                    <div style="font-size:0.8rem;margin-top:4px;">Chat page par jakar sawaal karein — agents yahan dikhenge</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Last Response
        if st.session_state.last_response:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📝 Last AI Response</div>', unsafe_allow_html=True)
            st.markdown(st.session_state.last_response[:1500] + ("..." if len(st.session_state.last_response)>1500 else ""),
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        # Weather quick panel
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🌤️ Weather Now</div>', unsafe_allow_html=True)
        try:
            w = interpret_weather(st.session_state.user_location)
            c = w.get("current",{})
            if c.get("temp_c"):
                col_t, col_h = st.columns(2)
                with col_t: st.metric("🌡️ Temp", f"{c['temp_c']}°C")
                with col_h: st.metric("💧 Humidity", f"{c['humidity_pct']}%")
                st.caption(f"💨 Wind: {c.get('wind_kmh','?')} km/h · {(c.get('condition') or '').title()}")
                fc = w.get("forecast_3day",[])
                if fc:
                    st.markdown('<div class="fc-grid">', unsafe_allow_html=True)
                    fc_html = '<div class="fc-grid">'
                    for d in fc[:3]:
                        fc_html += f"""<div class="fc-card">
                            <div class="fc-day">{d['date'][:3]}</div>
                            <div class="fc-temp">{d['high']}°/{d['low']}°</div>
                            <div class="fc-rain">🌧 {d['rain_probability']}</div>
                        </div>"""
                    fc_html += "</div>"
                    st.markdown(fc_html, unsafe_allow_html=True)
                for al in w.get("alerts",[])[:2]:
                    st.warning(f"⚠️ {al}")
            else:
                st.info("API key configure karein")
        except: st.info("Weather unavailable")
        st.markdown("</div>", unsafe_allow_html=True)

        # Activity Log
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 Activity Log</div>', unsafe_allow_html=True)
        if st.session_state.activity_log:
            log_html = ""
            for entry in st.session_state.activity_log[:10]:
                log_html += f"""<div class="act-item">
                    <div class="act-dot {entry['dot']}"></div>
                    <div class="act-time">{entry['time']}</div>
                    <div class="act-text">{entry['msg']}</div>
                </div>"""
            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.caption("Activity yahan dikhegi...")
        st.markdown("</div>", unsafe_allow_html=True)

        # Orchestrator details (if available)
        orch = st.session_state.orchestration_data
        if orch:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🧠 Orchestrator Decision</div>', unsafe_allow_html=True)
            ctx = orch.get("context_for_agents","")
            st.markdown(f"""
            <div style="font-size:0.82rem;color:#374151;line-height:1.75;">
                <b>Problem Type:</b> {", ".join(orch.get("problem_type",[]) or []) or "—"}<br>
                <b>Urgency:</b> {orch.get("urgency","—")}<br>
                <b>Call Order:</b> {orch.get("call_order","—")}<br>
                <b>Image Analysis:</b> {'Yes' if orch.get("image_must_analyze") else 'No'}<br>
                <b>Weather Needed:</b> {'Yes' if orch.get("weather_data_needed") else 'No'}<br>
                <b>Context:</b> <span style="color:#6b7280;">{ctx[:120]}{'...' if len(ctx)>120 else ''}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: CHAT
# ═══════════════════════════════════════════════════════════
elif "Chat" in page:
    st.markdown("""<div class="page-hdr">
        <div><h1>💬 AI Agri Chat</h1><p>Apni fasal ke baare mein poochein — Roman Urdu, Urdu ya English mein</p></div>
        <div>
            <span class="badge badge-g">🤖 3 Agents</span>
            <span class="badge badge-b">📸 Image Support</span>
        </div>
    </div>""", unsafe_allow_html=True)

    total = st.session_state.total_queries
    crop_d = st.session_state.user_crop or "—"
    loc_d  = st.session_state.user_location
    ag_run = len([v for v in st.session_state.agent_statuses.values() if v=="done"])
    st.markdown(f"""<div class="stat-grid">
        <div class="stat-card bl"><div class="stat-icon">💬</div><div class="stat-val">{total}</div><div class="stat-lbl">Queries</div></div>
        <div class="stat-card by"><div class="stat-icon">🌿</div><div class="stat-val">{crop_d}</div><div class="stat-lbl">Crop</div></div>
        <div class="stat-card bb"><div class="stat-icon">📍</div><div class="stat-val">{loc_d}</div><div class="stat-lbl">Location</div></div>
        <div class="stat-card bt"><div class="stat-icon">🤖</div><div class="stat-val">{ag_run}</div><div class="stat-lbl">Agents Run</div></div>
    </div>""", unsafe_allow_html=True)

    chat_col, agent_col = st.columns([3, 2])

    with chat_col:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="chat-title">Conversation</div>', unsafe_allow_html=True)
        if not st.session_state.messages:
            st.markdown("""<div style="text-align:center;padding:36px 20px;color:#9ca3af;">
                <div style="font-size:2.2rem;margin-bottom:10px;">🌾</div>
                <div style="font-size:0.95rem;font-weight:600;color:#6b7280;margin-bottom:5px;">Kisaan Dost mein Khush Aamdeed!</div>
                <div style="font-size:0.82rem;max-width:340px;margin:0 auto;line-height:1.6;">
                    Fasal ki koi masla share karein — disease, irrigation, ya mandi prices
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg.get("image") and msg["role"]=="user":
                        st.image(msg["image"], caption="📸 Fasal ki Photo", width=260)
        st.markdown("</div>", unsafe_allow_html=True)

        # Upload
        uploaded_file = st.file_uploader("📸 Fasal ki photo (optional)", type=["jpg","jpeg","png"], key="img_up", label_visibility="visible")

        # Input
        if prompt := st.chat_input("Sawaal likhein... (e.g. 'Mere gehun ke patte peele ho rahe hain Lahore mein')"):
            image_base64 = process_image(uploaded_file)
            st.session_state.total_queries += 1
            add_log(f"New query: {prompt[:60]}...", "dot-b")

            st.session_state.messages.append({"role":"user","content":prompt,"image":uploaded_file})
            with st.chat_message("user"):
                st.markdown(prompt)
                if uploaded_file: st.image(uploaded_file, caption="📸", width=260)

            with st.chat_message("assistant"):
                with st.spinner("🌾 AI agents analyze kar rahe hain..."):
                    try:
                        validated = validate_input(raw_input=prompt, has_image=image_base64 is not None,
                                                   known_location=st.session_state.user_location, known_crop=st.session_state.user_crop)
                        if validated.get("crop_detected"):
                            st.session_state.user_crop = validated["crop_detected"]
                        resolved = resolve_location(validated.get("location", st.session_state.user_location), st.session_state.user_location)
                        st.session_state.user_location = resolved["city"]

                        if not validated.get("proceed_to_orchestrator", True):
                            clarification = validated.get("location_question","Kya aap apna shehar bata sakte hain?")
                            response_text = f"👨‍🌾 **Assalam-o-Alaikum!**\n\nBehtar jawab ke liye:\n\n❓ {clarification}"
                            st.session_state.agents_last_called = []
                            st.session_state.last_response = response_text
                        else:
                            result = run_pipeline_with_tracking(
                                user_query=prompt, image_base64=image_base64,
                                location=st.session_state.user_location,
                                crop_name=validated.get("crop_detected"),
                                conversation_history=st.session_state.history,
                            )
                            response_text = result["formatted"]
                            st.session_state.agents_last_called = result["agents_called"]
                            st.session_state.last_urgency = result["urgency"]
                            if result.get("crop_detected"): st.session_state.user_crop = result["crop_detected"]
                            st.session_state.history.append({"role":"user","content":prompt})
                            st.session_state.history.append({"role":"assistant","content":response_text[:500]})

                    except Exception as e:
                        response_text = handle_error("groq_rate_limit","pipeline",prompt,{},st.session_state.user_crop,st.session_state.user_location)
                        st.session_state.agents_last_called = []
                        add_log(f"Error: {str(e)[:60]}", "dot-r")

                st.markdown(response_text, unsafe_allow_html=True)

                # ── TTS Play Button ──
                clean_text = _clean_for_speech(response_text)
                if len(clean_text) > 80:
                    tts_key = f"tts_{st.session_state.total_queries}"
                    if tts_key not in st.session_state:
                        st.session_state[tts_key] = None

                    if st.button("🔊 Sunain (Play Audio)", key=f"play_{st.session_state.total_queries}"):
                        with st.spinner("🔊 Audio generate ho raha hai..."):
                            lang = "en" if "english" in str(validated.get('language','')).lower() else "ur"
                            audio_b64 = text_to_speech(clean_text[:2000], lang)
                            st.session_state[tts_key] = audio_b64

                    if st.session_state.get(tts_key):
                        st.markdown(get_audio_html(st.session_state[tts_key]), unsafe_allow_html=True)

            st.session_state.messages.append({"role":"assistant","content":response_text})
            st.rerun()

    with agent_col:
        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
        if st.session_state.agent_statuses:
            render_agent_panel(show_outputs=True)
        else:
            st.markdown("""<div class="card">
                <div class="card-title">🤖 Agent Pipeline</div>
                <div style="text-align:center;padding:24px 16px;color:#9ca3af;">
                    <div style="font-size:1.8rem;margin-bottom:8px;">⚡</div>
                    <div style="font-size:0.85rem;color:#6b7280;">Query karo to agents yahan dikhenge</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Log
        if st.session_state.activity_log:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📋 Pipeline Log</div>', unsafe_allow_html=True)
            log_html = ""
            for e in st.session_state.activity_log[:8]:
                log_html += f'<div class="act-item"><div class="act-dot {e["dot"]}"></div><div class="act-time">{e["time"]}</div><div class="act-text">{e["msg"]}</div></div>'
            st.markdown(log_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: WEATHER
# ═══════════════════════════════════════════════════════════
elif "Weather" in page:
    st.markdown(f"""<div class="page-hdr">
        <div><h1>🌤️ Weather Dashboard</h1><p>Live mausam aur farming conditions — {st.session_state.user_location}</p></div>
        <div><span class="badge badge-b">📡 OpenWeatherMap</span><span class="badge badge-g">🌱 Farming Insights</span></div>
    </div>""", unsafe_allow_html=True)

    col_ref, _ = st.columns([1,4])
    with col_ref:
        if st.button("🔄  Refresh", use_container_width=True): st.rerun()

    try:
        weather = interpret_weather(st.session_state.user_location)
        cur = weather.get("current",{})
        if cur.get("temp_c") is not None:
            st.markdown('<div class="card"><div class="card-title">Current Conditions</div>', unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("🌡️ Temperature", f"{cur['temp_c']}°C")
            with c2: st.metric("💧 Humidity",    f"{cur['humidity_pct']}%")
            with c3: st.metric("💨 Wind",         f"{cur.get('wind_kmh','?')} km/h")
            with c4: st.metric("🌧️ Rain 24h",    f"{cur.get('rainfall_24h_mm',0)} mm")
            st.markdown(f'<div style="margin-top:10px;padding:10px 14px;background:var(--green-50);border:1px solid var(--green-200);border-radius:8px;font-size:0.84rem;color:#166016;line-height:1.6;">☁️ {cur.get("condition","").title()} · 🌾 {cur.get("farming_summary","")}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            fc = weather.get("forecast_3day",[])
            if fc:
                st.markdown('<div class="card"><div class="card-title">3-Day Forecast</div>', unsafe_allow_html=True)
                fc_html = '<div class="fc-grid">'
                for d in fc[:3]:
                    fc_html += f"""<div class="fc-card">
                        <div class="fc-day">{d['date']}</div>
                        <div class="fc-temp">{d['high']}° / {d['low']}°</div>
                        <div class="fc-rain">🌧 {d['rain_probability']} · {d['rain_mm']}mm</div>
                        <div class="fc-note">{d.get('farming_note','')}</div>
                    </div>"""
                fc_html += "</div>"
                st.markdown(fc_html + "</div>", unsafe_allow_html=True)

            spray = weather.get("spray_window",""); irrig = weather.get("irrigation_decision","")
            st.markdown('<div class="card"><div class="card-title">Farming Recommendations</div>', unsafe_allow_html=True)
            cs, ci = st.columns(2)
            with cs: st.markdown(f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:13px 15px;"><div style="font-size:0.68rem;font-weight:700;color:#15803d;text-transform:uppercase;margin-bottom:5px;">💊 Spray Window</div><div style="color:#166016;font-size:0.84rem;line-height:1.6;">{spray}</div></div>', unsafe_allow_html=True)
            with ci: st.markdown(f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:13px 15px;"><div style="font-size:0.68rem;font-weight:700;color:#1d4ed8;text-transform:uppercase;margin-bottom:5px;">🚿 Irrigation</div><div style="color:#1e40af;font-size:0.84rem;line-height:1.6;">{irrig}</div></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            for al in weather.get("alerts",[])[:2]: st.warning(f"⚠️ {al}")
        else:
            st.info("🔑 OPENWEATHER_API_KEY configure karein .env file mein")
    except Exception as e:
        st.error(f"❌ Weather load nahi hua: {str(e)[:100]}")


# ═══════════════════════════════════════════════════════════
#  PAGE: CROPS & INFO
# ═══════════════════════════════════════════════════════════
elif "Crops" in page:
    st.markdown("""<div class="page-hdr">
        <div><h1>🌱 Crops & Seasonal Guide</h1><p>Supported crops, AI agents aur seasonal farming calendar</p></div>
        <div><span class="badge badge-g">11 Crops</span><span class="badge badge-y">3 Agents</span></div>
    </div>""", unsafe_allow_html=True)

    col_c, col_a = st.columns([3,2])
    with col_c:
        st.markdown('<div class="card"><div class="card-title">Supported Crops (Faslain)</div>', unsafe_allow_html=True)
        tags = "".join(f'<span class="crop-tag">{c}</span>' for c in ["Wheat (Gehun)","Rice (Chawal)","Cotton (Kapas)","Sugarcane (Ganna)","Maize (Makki)","Tomato (Tamatar)","Onion (Pyaz)","Potato (Aloo)","Mango (Aam)","Chilli (Mirch)","Mustard (Sarson)"])
        st.markdown(f'<div class="crop-tags-wrap">{tags}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">Seasonal Calendar</div>', unsafe_allow_html=True)
        for s,p,c,bg,br,tc in [("❄️ Rabi (Winter)","Oct–Mar","Wheat, Mustard, Potato, Onion, Peas","#eff6ff","#bfdbfe","#1e40af"),("☀️ Kharif (Summer)","Apr–Sep","Rice, Cotton, Sugarcane, Maize, Tomato","#f0fdf4","#bbf7d0","#15803d"),("🍂 Zaid (Short)","Mar–Jun","Mango, Watermelon, Cucumber","#fffbeb","#fde68a","#92400e")]:
            st.markdown(f'<div style="background:{bg};border:1px solid {br};border-radius:8px;padding:11px 14px;margin-bottom:8px;"><b style="color:{tc};font-size:0.87rem;">{s}</b> <span style="color:#9ca3af;font-size:0.73rem;">📅 {p}</span><br><span style="color:{tc};font-size:0.8rem;">🌿 {c}</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_a:
        st.markdown('<div class="card"><div class="card-title">AI Agents System</div>', unsafe_allow_html=True)
        for key,meta in AGENT_META.items():
            st.markdown(f"""<div style="display:flex;gap:10px;align-items:flex-start;padding:10px 0;border-bottom:1px solid #f3f4f6;">
                <div class="agent-icon-wrap {meta['color']}" style="width:36px;height:36px;font-size:1rem;flex-shrink:0;">{meta['icon']}</div>
                <div><div style="font-size:0.87rem;font-weight:600;color:#1f2937;">{meta['name']}</div>
                <div style="font-size:0.72rem;color:#9ca3af;">🔧 {meta['model']}</div>
                <div style="font-size:0.78rem;color:#6b7280;margin-top:2px;">{meta['desc']}</div></div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: HELP
# ═══════════════════════════════════════════════════════════
elif "Help" in page:
    st.markdown("""<div class="page-hdr">
        <div><h1>❓ Help & Guide</h1><p>Step-by-step guide aur FAQs</p></div>
        <div><span class="badge badge-g">📖 Guide</span><span class="badge badge-y">❓ FAQs</span></div>
    </div>""", unsafe_allow_html=True)

    h1, h2 = st.columns(2)
    def hcard(title, content):
        return f'<div style="background:#fff;border:1px solid #d9f5d9;border-radius:12px;padding:18px 20px;box-shadow:0 1px 3px rgba(0,0,0,0.07);margin-bottom:12px;"><h3 style="color:#166016;font-size:0.95rem;font-weight:600;margin-bottom:8px;">{title}</h3><div style="color:#4b5563;font-size:0.85rem;line-height:1.7;">{content}</div></div>'

    with h1:
        st.markdown(hcard("🚀 Shuru Kaise Karein?","1. Sidebar mein shehar ya zila type karein<br>2. Chat page par sawaal likhein<br>3. Photo ho to upload karein<br>4. Send karein — AI agents jawab denge"), unsafe_allow_html=True)
        st.markdown(hcard("💬 Sample Queries","• <i>Mere gehun ke patte peele ho rahe hain Lahore mein</i><br>• <i>Tamatar mein safed powder aagaya hai</i><br>• <i>Faisalabad mein cotton ka rate kya hai?</i><br>• <i>Meri rice fasal ko kitna paani chahiye?</i>"), unsafe_allow_html=True)
        st.markdown(hcard("📊 Dashboard Kya Dikhata Hai?","Dashboard par yeh sab ek saath dekhein:<br>• Real-time agent execution status<br>• Har agent ki detailed output<br>• Weather snapshot<br>• Activity log aur pipeline history"), unsafe_allow_html=True)

    with h2:
        st.markdown(hcard("🤖 Agents Kya Karte Hain?","🩺 <b>Crop Doctor</b> — Disease & pest diagnosis<br>💧 <b>Irrigation Advisor</b> — Paani ka schedule<br>📈 <b>Market Price</b> — Mandi rates & advice<br>🧠 <b>Orchestrator</b> — Routes query to right agents"), unsafe_allow_html=True)
        st.markdown(hcard("🌤️ Weather Page","• Live temperature, humidity, wind<br>• 3-din forecast<br>• Spray karne ka sahi waqt<br>• Irrigation kab karni chahiye<br>• Alerts (heat wave, frost risk)"), unsafe_allow_html=True)
        st.markdown(hcard("📞 Madad Chahiye?","<b>0800-15000</b> (Bilkul Free)<br>Mon–Sat, 8am–5pm<br><br>Ya apne Local Agriculture Extension Officer se raabta karein."), unsafe_allow_html=True)

    st.markdown('<div style="background:#fff;border:1px solid #d9f5d9;border-radius:12px;padding:18px 22px;box-shadow:0 1px 3px rgba(0,0,0,0.07);">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">❓ Frequently Asked Questions</div>', unsafe_allow_html=True)
    for q,a in [("Kya ye service free hai?","Haan, Kisaan Dost bilkul free hai."),("Weather kaam nahi kar raha?","OPENWEATHER_API_KEY .env mein set karein — openweathermap.org se free milti hai."),("Agents kab activated hote hain?","Orchestrator automatically decide karta hai — aapko manually select nahi karna."),("Kya photo zaroori hai?","Nahi, lekin disease ke liye photo se accuracy barh jaati hai."),("Dashboard mein outputs kab dikhte hain?","Query karne ke baad agents ka detailed output dashboard aur chat page dono par dikhta hai.")]:
        with st.expander(f"**{q}**"):
            st.markdown(f"<p style='color:#374151;font-size:0.87rem;line-height:1.65;'>{a}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

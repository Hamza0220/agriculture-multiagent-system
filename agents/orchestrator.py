import json
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Master Orchestrator for "Kisaan Dost" — Pakistan's AI
Agricultural Assistant that helps Pakistani farmers with crop diseases,
irrigation, and market prices.

You receive every user query and decide which specialist agents to activate.

AVAILABLE SPECIALIST AGENTS:
1. CROP_DOCTOR — Analyzes crop diseases and pests from photos and descriptions.
   Uses vision AI to examine crop images. Has RAG database of diseases.
   Call when: User mentions diseased plants, pests, crop damage, abnormal appearance.

2. IRRIGATION_ADVISOR — Provides water management recommendations.
   Uses real-time weather data + crop water requirement knowledge.
   Call when: User asks about watering schedule, water stress, irrigation timing.

3. MARKET_PRICE — Provides current market prices from Pakistan mandis.
   Uses live web search for today's prices.
   Call when: User asks about prices, selling time, mandi rates, bhav.

ROUTING RULES:
- IMPORTANT DEFAULT: For any farming or crop-related question, ALWAYS activate ALL THREE agents ("CROP_DOCTOR", "IRRIGATION_ADVISOR", "MARKET_PRICE") to provide a comprehensive response!
- The user wants a complete 360-degree analysis for every query, so default to ["CROP_DOCTOR", "IRRIGATION_ADVISOR", "MARKET_PRICE"].
- Even if the user only asks about a disease, ALSO activate IRRIGATION_ADVISOR and MARKET_PRICE to give them extra helpful information.
- Only omit an agent if the user's query is completely unrelated to farming.

MULTI-AGENT SCENARIOS (call all relevant):
- "is fasal ko dekho" → ALL THREE ("CROP_DOCTOR", "IRRIGATION_ADVISOR", "MARKET_PRICE")
- "Meri wheat mein disease hai" → ALL THREE
- "Irrigation schedule batao" → ALL THREE
- "Complete farm report" → ALL THREE

CROP DETECTION (identify from query):
wheat=gandum, rice=chawal, cotton=kapas, sugarcane=ganna,
maize=makki, tomato=tamatar, onion=pyaaz, potato=aloo,
mango=aam, chilli=mirch, mustard=sarson, lentil=masoor/dal

SEASON DETECTION:
Rabi (Oct-Apr): wheat, mustard, chickpea, lentil, potato
Kharif (Apr-Oct): rice, cotton, sugarcane, maize, mango, tomato

OUTPUT (JSON only, no other text):
{
  "agents_to_call": ["CROP_DOCTOR", "IRRIGATION_ADVISOR", "MARKET_PRICE"],
  "call_order": "parallel",
  "crop_detected": "wheat",
  "crop_urdu": "gandum",
  "location": "Lahore, Punjab",
  "season": "Rabi",
  "urgency": "HIGH",
  "problem_type": ["disease"],
  "weather_data_needed": false,
  "image_must_analyze": true,
  "context_for_agents": "Farmer reports yellow stripes on wheat leaves, wants to know treatment and current market price",
  "farmer_language": "roman_urdu",
  "respond_in": "roman_urdu"
}

LANGUAGE DETECTION:
- If query contains Urdu/Roman Urdu words (hai, hain, mein, ko, kya, kaise, fasal, bimari etc.) → "roman_urdu"
- If query is purely in English → "english"
- Mixed Roman Urdu + English → "roman_urdu" (default to Roman Urdu)

Respond with ONLY valid JSON. No explanations, no markdown code blocks."""


def detect_season(current_date: str = None) -> str:
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")

    month = datetime.strptime(current_date[:10], "%Y-%m-%d").month
    if month in [10, 11, 12, 1, 2, 3, 4]:
        return "Rabi"
    else:
        return "Kharif"


def clean_json_response(raw_text: str) -> str:
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    return raw_text.strip()


def orchestrate(
    user_query: str,
    has_image: bool = False,
    location: str = "Pakistan",
    conversation_history: list = None,
    current_date: str = None,
) -> dict:
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")

    if conversation_history is None:
        conversation_history = []

    history_str = ""
    if conversation_history:
        recent = conversation_history[-5:]
        history_str = "\n".join(
            [f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content']}"
             for h in recent]
        )

    user_prompt = f"""USER INPUT: {user_query}
HAS_CROP_IMAGE: {has_image}
USER_LOCATION: {location}
CURRENT_DATE: {current_date}
CONVERSATION_HISTORY: {history_str if history_str else "None"}"""

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    raw_output = response.choices[0].message.content
    cleaned = clean_json_response(raw_output)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {
            "agents_to_call": ["CROP_DOCTOR"],
            "call_order": "parallel",
            "crop_detected": None,
            "crop_urdu": None,
            "location": location,
            "season": detect_season(current_date),
            "urgency": "MEDIUM",
            "problem_type": ["general"],
            "weather_data_needed": False,
            "image_must_analyze": has_image,
            "context_for_agents": user_query,
            "farmer_language": "roman_urdu",
            "respond_in": "roman_urdu",
        }

    result.setdefault("season", detect_season(current_date))
    result.setdefault("location", location)
    result.setdefault("image_must_analyze", has_image)

    detected_lang = result.get("farmer_language", "roman_urdu")
    if detected_lang in ("english", "roman_urdu", "mixed"):
        result["farmer_language"] = detected_lang
        result["respond_in"] = detected_lang
    else:
        is_english = _is_primarily_english(user_query)
        lang = "english" if is_english else "roman_urdu"
        result["farmer_language"] = lang
        result["respond_in"] = lang

    return result


def _is_primarily_english(text: str) -> bool:
    import re
    urdu_markers = [
        r"\bhai\b", r"\bhain\b", r"\bhun\b", r"\bho\b", r"\btha\b", r"\bthi\b",
        r"\bmein\b", r"\bko\b", r"\bki\b", r"\bka\b", r"\bke\b", r"\bsy\b", r"\bse\b", r"\bpar\b",
        r"\bkarein\b", r"\bkarun\b", r"\bkya\b", r"\bkaise\b", r"\bkab\b", r"\bkitna\b",
        r"\bfasal\b", r"\bbimari\b", r"\bzang\b", r"\bkeera\b", r"\bpaani\b", r"\bmandi\b",
        r"\bbhav\b", r"\bkhad\b", r"\bbechen\b",
        r"\blag\s+gai\b", r"\blag\s+gayi\b",
        r"\baa\s+rahi\b", r"\baa\s+raha\b", r"\bho\s+rahi\b", r"\bho\s+raha\b",
        r"\bdekh\b", r"\bkaro\b", r"\bbolo\b", r"\bjaldi\b", r"\bzaroor\b",
        r"\bgandum\b", r"\bchawal\b", r"\bkapas\b", r"\bganna\b", r"\bmakki\b",
        r"\btamatar\b", r"\bpyaaz\b", r"\baloo\b", r"\baam\b", r"\bmirch\b", r"\bsarson\b",
    ]
    text_lower = text.lower()
    urdu_count = sum(1 for m in urdu_markers if re.search(m, text_lower))
    return urdu_count == 0

import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

SYNTHESIZER_SYSTEM_PROMPT = """You are the Response Synthesizer for Kisaan Dost — Pakistan Agricultural
Assistant. You receive outputs from multiple specialist agents and combine
them into one coherent, helpful response for the farmer.

SYNTHESIS RULES:

1. ONLY include sections from agents that were actually called.
   Do not fabricate data from agents not called.

2. PRIORITIZE BY URGENCY:
   - CRITICAL disease → Disease treatment comes FIRST
   - Urgent irrigation need → Irrigation advice comes FIRST
   - Market timing → Price advice comes first if harvest is ready

3. CONNECT THE DOTS:
   Find connections between agent outputs that the farmer will value:
   Example: "Treat disease first, THEN sell in 3 weeks when you'll also
   get better mandi prices"
   Example: "Rain is coming Thursday, delay irrigation AND this will help
   spread the fungicide you apply"

4. RESOLVE CONFLICTS:
   If agents give conflicting advice, state both and explain trade-off:
   Example: Market agent says sell now, but crop doctor says disease
   needs treatment first — resolve this for the farmer.

5. LANGUAGE AND TONE:
   - Roman Urdu if farmer_language is roman_urdu
   - Simple English if farmer_language is english
   - Always warm and respectful — farmers deserve dignity
   - Use "aap" not "tu" in Urdu
   - Use analogies from Pakistani farming life

6. STRUCTURE (in order of urgency):
   🌾 FASAL KI SITUATION (Crop Situation Summary — 2 sentences)
   [Most urgent section first]
   🔬 BIMARI / KEERA (if crop doctor called)
   💧 PAANI KA SCHEDULE (if irrigation advisor called)
   💰 MANDI PRICE (if market agent called)
   📋 AAP KA ACTION PLAN (numbered, most urgent first)
   📞 MADAD KE LIYE (helpline numbers)

7. ACTION PLAN FORMAT:
   Must be numbered, specific, and time-bound:
   ✗ "Apply pesticide on your plants"
   ✓ "1. Kal subah (Thursday) 5 bajay Tilt 250EC spray karein —
      0.5ml per liter paani — poore 100 liter per acre laga den"

MANDATORY FOOTER (always include):
---
📞 Madad ke liye: Pakistan Agriculture Helpline: 0800-15000 (Free)
ℹ️ Ye advice aapki batai hui information par based hai. Agar problem
   zyada serious lage to apne local Agriculture Extension Officer se
   milein.
---

OUTPUT: Formatted markdown text for Streamlit display (not JSON).
Maximum 600 words. Scannable format with emojis for section headers."""


def synthesize_response(
    original_query: str,
    location: str,
    crop_name: str,
    language: str = "roman_urdu",
    crop_doctor_output: dict = None,
    irrigation_output: dict = None,
    market_price_output: dict = None,
    agents_called: list = None,
) -> str:
    if agents_called is None:
        agents_called = []

    def agent_or_none(agent_name, output):
        return json.dumps(output, ensure_ascii=False) if output else "NOT CALLED"

    user_prompt = f"""FARMER'S ORIGINAL QUERY: {original_query}
FARMER'S LOCATION: {location}
CROP: {crop_name}
FARMER'S LANGUAGE: {language}

SPECIALIST AGENT OUTPUTS:
Crop Doctor Output: {agent_or_none('CROP_DOCTOR', crop_doctor_output)}
Irrigation Advisor Output: {agent_or_none('IRRIGATION_ADVISOR', irrigation_output)}
Market Price Output: {agent_or_none('MARKET_PRICE', market_price_output)}
Agents Called: {', '.join(agents_called) if agents_called else 'NONE'}

Synthesize these outputs into one coherent, farmer-friendly response.
Follow all synthesis rules from your system prompt. Use {language} for the response.
Focus on being practical, actionable, and respectful."""

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=1200,
    )

    synthesized = response.choices[0].message.content.strip()

    if "📞 Madad ke liye" not in synthesized:
        synthesized += (
            "\n\n---\n📞 Madad ke liye: Pakistan Agriculture Helpline: 0800-15000 (Free)\n"
            "ℹ️ Ye advice aapki batai hui information par based hai. Agar problem\n"
            "   zyada serious lage to apne local Agriculture Extension Officer se milein."
        )

    return synthesized

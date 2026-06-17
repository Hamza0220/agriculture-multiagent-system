import json
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

INPUT_VALIDATOR_SYSTEM_PROMPT = """You are the input processor for Kisaan Dost — Pakistan Agricultural Assistant.
You process every farmer's message to extract structured information.

EXTRACTION TASKS:

1. LANGUAGE DETECTION:
   - Roman Urdu (Urdu written in English letters): most common
   - Urdu (Arabic script)
   - English
   - Mixed (use "mixed")
   - Local dialect indicators (Punjabi: "kidon", "kinne"; Sindhi: different vocabulary)

2. LOCATION EXTRACTION:
   Extract city, district, or province mentioned.
   Common Pakistani farming locations:
   Punjab: Lahore, Faisalabad, Multan, Sialkot, Gujranwala, Sahiwal, Okara,
           Sheikhupura, Kasur, Pakpattan, Vehari, Lodhran, Bahawalpur
   Sindh: Hyderabad, Sukkur, Larkana, Nawabshah, Mirpurkhas, Sanghar
   KPK: Peshawar, Mardan, Swat, Charsadda, Nowshera
   Balochistan: Quetta, Turbat, Khuzdar

   If no location mentioned: use PREVIOUSLY_KNOWN_LOCATION
   If completely unknown: set to "Pakistan" and flag for asking

3. CROP EXTRACTION:
   Detect crop from Urdu/Roman Urdu/English names:
   gandum/wheat, chawal/rice/dhan, kapas/cotton,
   ganna/sugarcane, makki/maize/corn, tamatar/tomatoes,
   pyaaz/onion, aloo/potato, aam/mango, mirch/chilli,
   sarson/mustard, dal/masoor/lentil, chanay/chickpea

4. PROBLEM TYPE DETECTION:
   DISEASE: zang, bimari, kala, peela, safed, dag, rot, wilt, fungus
   PEST: keera, insect, makhi, tiddas, sundi, locust, aphid, mite
   IRRIGATION: paani, sookha, flood, drain, nali, tube well
   MARKET: bhav, price, mandi, bikri, sell, rate
   FERTILIZER: khad, urea, DAP, potash, fertilizer
   GENERAL: general farming advice or multiple issues

5. IMAGE ASSESSMENT (if has_image):
   What does the image likely show?
   If it appears to be: crop photo, plant disease photo, field photo → agricultural_image: true
   If it appears to be: food, selfie, unrelated → agricultural_image: false

6. URGENCY DETECTION:
   CRITICAL: "fasal khatam ho rahi hai", "sab mar raha hai", "aaj hi bikna hai"
   HIGH: "2-3 din se", "tezi se phaila raha hai", "kal mandi jana hai"
   MEDIUM: "kuch din se", "thodi si problem", "janna chahta hun"
   LOW: General information request, planning for next season

OUTPUT (JSON only, no other text):
{
  "is_valid": true,
  "language": "roman_urdu",
  "crop_detected": "wheat",
  "crop_urdu": "gandum",
  "location": "Faisalabad",
  "province": "Punjab",
  "problem_types": ["disease"],
  "urgency": "HIGH",
  "agricultural_image": true,
  "needs_location_clarification": false,
  "location_question": null,
  "cleaned_input": "[sanitized input]",
  "proceed_to_orchestrator": true,
  "key_terms_extracted": ["paton par zard dhariyan", "tezi sy phaili"]
}

Respond with ONLY valid JSON. No explanations, no markdown code blocks."""


def clean_json_response(raw_text: str) -> str:
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    return raw_text.strip()


def validate_input(
    raw_input: str,
    has_image: bool = False,
    known_location: str = None,
    known_crop: str = None,
) -> dict:
    user_prompt = f"""RAW USER INPUT: {raw_input}
HAS_IMAGE: {has_image}
PREVIOUSLY_KNOWN_LOCATION: {known_location or 'None'}
PREVIOUSLY_KNOWN_CROP: {known_crop or 'None'}"""

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": INPUT_VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )

    raw_output = response.choices[0].message.content
    cleaned = clean_json_response(raw_output)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {
            "is_valid": True,
            "language": "roman_urdu",
            "crop_detected": known_crop,
            "crop_urdu": None,
            "location": known_location or "Pakistan",
            "province": None,
            "problem_types": ["general"],
            "urgency": "MEDIUM",
            "agricultural_image": has_image,
            "needs_location_clarification": not known_location,
            "location_question": "Aap kis shehar/kasba se hain?" if not known_location else None,
            "cleaned_input": raw_input.strip(),
            "proceed_to_orchestrator": True,
            "key_terms_extracted": [],
        }

    return result

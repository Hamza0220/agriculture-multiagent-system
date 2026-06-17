import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
VISION_MODEL = os.getenv("VISION_MODEL", "google/gemini-2.0-flash-exp:free")
# Working OpenRouter vision fallbacks (in order of preference)
VISION_FALLBACK_MODELS = [
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "qwen/qwen-vl-plus:free",
]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

CROP_DOCTOR_SYSTEM_PROMPT = """You are Dr. Kisaan — an expert plant pathologist and agricultural scientist
who has spent 20 years working in Pakistan's farming regions of Punjab,
Sindh, KPK and Balochistan. You specialize in diagnosing crop diseases,
pest infestations, nutrient deficiencies, and water stress in Pakistani crops.

DIAGNOSIS PROTOCOL:

STEP 1 — VISUAL ANALYSIS (if image provided):
Examine the image systematically:
□ Leaf color: Normal green / Yellow / Brown / Black spots / White powder
□ Leaf texture: Holes / Curling / Wilting / Burning edges
□ Stem condition: Healthy / Rotting / Lesions / Color change
□ Fruit/grain: Normal / Deformed / Discolored / Damaged
□ Root (if visible): Healthy / Rotting / Pest damage
□ Pattern: Whole plant / Random leaves / Bottom leaves / Top leaves
□ Spread: Single plant / Patch / Whole field

STEP 2 — IDENTIFY THE PROBLEM:
Based on visual analysis + farmer description + RAG knowledge:
Primary diagnosis: [Most likely issue]
Confidence: HIGH (>80%) / MEDIUM (50-80%) / LOW (<50%)
Alternative possibilities: [If confidence is not HIGH]

Pakistan-specific context to consider:
- Current season diseases prevalent in this region
- Recent weather patterns favoring certain diseases
- Common mistakes farmers make in this area

STEP 3 — SEVERITY ASSESSMENT:
CRITICAL: Will destroy crop within days if untreated
SERIOUS: Significant yield loss expected, treat within 3-5 days
MODERATE: Manageable, treat within 2 weeks
MILD: Monitor and apply preventive measures

STEP 4 — TREATMENT PLAN:
Give THREE options in order of preference:
Option A (Most Effective): Chemical treatment with local brand names
Option B (Budget-Friendly): Cheaper chemical alternative
Option C (Organic/Natural): If available and effective enough

For each option specify EXACTLY:
- Product name (generic) + common Pakistani brand name
- Dose: per liter of water AND per acre
- Time of application: Morning/Evening (avoid peak sun)
- Number of applications and interval
- Cost estimate in Pakistani Rupees
- Where to buy (agri shop / local dealer)

STEP 5 — PREVENTION:
2-3 specific steps to prevent this problem next season.

CRITICAL RULES:
- Never give a confident diagnosis when image is unclear
- Always mention if professional visit is needed (severe cases)
- Include safety warnings for all chemical recommendations
- Convert all measurements to what farmers use: acre not hectare
- If disease is new or unusual: recommend contacting local agriculture extension officer
- Pakistan Agriculture Helpline: 0800-15000 (free)

OUTPUT (JSON only, no other text):
{
  "diagnosis": {
    "primary": "Yellow Rust (Puccinia striiformis)",
    "urdu_name": "Zard Zang",
    "confidence": "HIGH",
    "alternatives": [],
    "severity": "SERIOUS",
    "time_to_act": "Within 3-5 days"
  },
  "visual_observations": {
    "affected_parts": ["leaves"],
    "symptoms_seen": ["yellow stripes parallel to leaf veins", "yellow powder"],
    "spread_estimate": "30% of visible plants affected",
    "image_clarity": "CLEAR/UNCLEAR"
  },
  "treatment": {
    "option_a": {
      "type": "Chemical",
      "product": "Propiconazole 25EC",
      "local_brands": ["Tilt 250EC", "Bumper 25EC"],
      "dose_per_liter": "0.5ml",
      "dose_per_acre": "200ml in 100 liters water",
      "timing": "Early morning or evening, avoid rain for 6 hours after",
      "applications": "2 sprays, 10 days apart",
      "cost_per_acre": "Rs. 800-1200",
      "availability": "Available at any agri shop"
    },
    "option_b": {
      "type": "Budget Chemical",
      "product": "Mancozeb 80WP",
      "local_brands": ["Dithane M-45", "Indofil M-45"],
      "dose_per_liter": "2.5g",
      "dose_per_acre": "250g in 100 liters water",
      "cost_per_acre": "Rs. 300-500"
    },
    "option_c": {
      "type": "Organic",
      "product": "Not effective enough for severe yellow rust",
      "note": "Organic options only work for very mild early infection"
    }
  },
  "prevention_next_season": [
    "Use rust-resistant wheat variety: Punjab-2011 or Faisalabad-2008",
    "Apply preventive fungicide at booting stage before symptoms appear",
    "Avoid late sowing — early planted wheat has higher rust risk"
  ],
  "safety_warnings": [
    "Wear mask and gloves when spraying",
    "Do not spray in strong wind",
    "Keep children away from treated field for 24 hours"
  ],
  "seek_expert_if": "Disease spreads to >50% of field within 3 days despite treatment",
  "helpline": "Pakistan Agriculture Helpline: 0800-15000 (free, Monday-Saturday)"
}

Respond with ONLY valid JSON. No explanations, no markdown code blocks."""


def analyze_image_with_vision(image_base64: str, crop_name: str = None) -> str:
    crop_context = f" of {crop_name} crop plants" if crop_name else ""

    prompt_text = f"""You are a plant disease expert analyzing a crop photo{crop_context}.
Examine this image carefully and describe what you see. Focus on:

1. LEAF CONDITION: Color (green/yellow/brown/black), texture (holes/curling/wilting), any spots, powder, or stripes
2. STEM CONDITION: Healthy, rotting, lesions, or color changes
3. FRUIT/GRAIN (if visible): Normal, deformed, discolored, or damaged
4. PATTERN: Is the problem on whole plant, random leaves, bottom leaves, or top leaves?
5. SPREAD: Single plant, patch, or seems to be whole field?
6. IMAGE CLARITY: Is the photo clear enough for confident diagnosis?

Be specific and detailed. Note any disease symptoms you recognize.
IMPORTANT: If the image is NOT a crop/plant photo, say so clearly."""

    # ── Try 1: OpenAI GPT-4o / gpt-4o-mini (primary — best quality) ──
    if OPENAI_API_KEY and OPENAI_API_KEY not in ("your_openai_key_here", ""):
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=OPENAI_VISION_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ]}],
                max_tokens=500,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Vision] OpenAI failed ({type(e).__name__}): {e}")

    # ── Fallback: no vision available — text-only diagnosis ──
    print("[Vision] Vision models failed — proceeding with text-only diagnosis")
    return (
        "Image analysis unavailable (vision API not accessible or out of quota). "
        "Diagnosis will be based on farmer's text description only. "
        "For better accuracy, describe symptoms in detail."
    )



def clean_json_response(raw_text: str) -> str:
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    return raw_text.strip()


def diagnose(
    farmer_description: str,
    crop_name: str = None,
    image_base64: str = None,
    location: str = "Pakistan",
    season: str = None,
    weather_summary: str = "No weather data available",
    rag_context: str = "No RAG context available",
) -> dict:
    image_analysis = "No image provided."

    if image_base64:
        image_analysis = analyze_image_with_vision(image_base64, crop_name)

    if season is None:
        season = "Unknown"

    user_prompt = f"""INPUTS:
- Crop Image: {'Provided' if image_base64 else 'Not provided'} → Image Analysis: {image_analysis}
- Farmer's Description: {farmer_description}
- Crop Name: {crop_name if crop_name else 'Not specified'}
- Location: {location}
- Season: {season}
- Current Weather: {weather_summary}
- RAG Knowledge Context: {rag_context}

Diagnose this crop problem based on all available information and provide treatment recommendations.
Follow the full diagnosis protocol in your system prompt."""

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": CROP_DOCTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1500,
    )

    raw_output = response.choices[0].message.content
    cleaned = clean_json_response(raw_output)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {
            "diagnosis": {
                "primary": "Unable to determine from available information",
                "urdu_name": "",
                "confidence": "LOW",
                "alternatives": [],
                "severity": "MODERATE",
                "time_to_act": "Consult local expert",
            },
            "visual_observations": {
                "affected_parts": [],
                "symptoms_seen": [image_analysis] if image_base64 else [],
                "spread_estimate": "Unknown",
                "image_clarity": "UNCLEAR" if image_base64 else "NO_IMAGE",
            },
            "treatment": {
                "option_a": {
                    "type": "Unknown",
                    "product": "Unable to recommend — please provide more details",
                    "local_brands": [],
                    "dose_per_liter": "",
                    "dose_per_acre": "",
                    "timing": "",
                    "applications": "",
                    "cost_per_acre": "",
                    "availability": "Consult local agri shop",
                }
            },
            "prevention_next_season": [],
            "safety_warnings": ["Always consult local agriculture extension officer for proper diagnosis"],
            "seek_expert_if": "Diagnosis could not be confirmed",
            "helpline": "Pakistan Agriculture Helpline: 0800-15000 (free, Monday-Saturday)",
        }

    if image_base64 and "visual_observations" in result:
        result["visual_observations"]["raw_vision_analysis"] = image_analysis

    return result

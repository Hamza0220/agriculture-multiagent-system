"""
Agricultural Query Expander (M1-2)
Expands vague farmer queries (Roman Urdu/Urdu/English) into precise
search queries for maximum vector DB retrieval quality.

Uses Groq Llama-3.3-70b for intelligent interpretation.
Falls back to rule-based expansion if API is unavailable.
"""

import json
import re
import os
from typing import List, Dict, Optional, Tuple

try:
    from groq import Groq
except ImportError:
    Groq = None


# ── Urdu keywords map for rule-based fallback ──────────────────────────

URDU_PROBLEM_MAP = {
    # Disease indicators (Urdu)
    "bimari": "DISEASE", "bemari": "DISEASE", "marz": "DISEASE",
    "zang": "DISEASE", "rust": "DISEASE", "jali": "DISEASE",
    "phaphundi": "DISEASE", "phafhindi": "DISEASE", "safed": "DISEASE",
    "kala": "DISEASE", "peela": "DISEASE", "peeli": "DISEASE",
    "dhariyan": "DISEASE", "dhabbay": "DISEASE", "dag": "DISEASE",
    "rot": "DISEASE", "gala": "DISEASE", "saran": "DISEASE",
    "surkh": "DISEASE", "jala": "DISEASE",
    # Disease indicators (English)
    "disease": "DISEASE", "infection": "DISEASE", "fungus": "DISEASE",
    "fungal": "DISEASE", "blight": "DISEASE", "mildew": "DISEASE",
    "spots": "DISEASE", "lesions": "DISEASE", "yellowing": "DISEASE",
    "wilting": "DISEASE", "stripe": "DISEASE", "blotch": "DISEASE",

    # Pest indicators (Urdu)
    "keera": "PEST", "kera": "PEST", "makhi": "PEST",
    "sundi": "PEST", "tana sundi": "PEST", "phali sundi": "PEST",
    "safed makhi": "PEST", "deemak": "PEST", "tidda": "PEST",
    # Pest indicators (English)
    "whitefly": "PEST", "aphid": "PEST", "termite": "PEST",
    "locust": "PEST", "mite": "PEST", "thrips": "PEST",
    "armyworm": "PEST", "borer": "PEST", "mealybug": "PEST",
    "fruit fly": "PEST", "pest": "PEST", "insect": "PEST",
    "caterpillar": "PEST", "worm": "PEST", "hopper": "PEST",
    "grasshopper": "PEST", "beetle": "PEST", "weevil": "PEST",

    # Irrigation indicators
    "paani": "IRRIGATION", "pani": "IRRIGATION",
    "irrigation": "IRRIGATION", "sookha": "IRRIGATION", "sukha": "IRRIGATION",
    "tube well": "IRRIGATION", "tubewell": "IRRIGATION", "nahar": "IRRIGATION",
    "kanal": "IRRIGATION", "drain": "IRRIGATION",
    "water": "IRRIGATION", "moisture": "IRRIGATION", "barish": "IRRIGATION",
    "flood": "IRRIGATION", "drought": "IRRIGATION", "watering": "IRRIGATION",

    # Fertilizer indicators
    "khad": "FERTILIZER", "urea": "FERTILIZER", "dap": "FERTILIZER",
    "fertilizer": "FERTILIZER", "potash": "FERTILIZER", "nitrogen": "FERTILIZER",
    "phosphorus": "FERTILIZER", "zinc": "FERTILIZER", "gobar": "FERTILIZER",
    "npk": "FERTILIZER", "manure": "FERTILIZER", "compost": "FERTILIZER",

    # Soil indicators
    "mitti": "SOIL", "soil": "SOIL", "zameen": "SOIL",
    "namak": "SOIL", "saline": "SOIL", "ph": "SOIL",

    # Harvest indicators
    "katai": "HARVESTING", "harvest": "HARVESTING", "tod": "HARVESTING",
    "pukka": "HARVESTING", "ripe": "HARVESTING",

    # Market indicators
    "bhav": "MARKET", "mandi": "MARKET", "bikri": "MARKET",
    "bechen": "MARKET", "mol": "MARKET", "arhi": "MARKET",
    "price": "MARKET", "market": "MARKET", "rate": "MARKET",
    "sell": "MARKET", "sale": "MARKET",
}

CROP_KEYWORDS = {
    "gandum": "wheat", "wheat": "wheat", "kanak": "wheat",
    "chawal": "rice", "rice": "rice", "dhan": "rice", "paddy": "rice",
    "kapas": "cotton", "cotton": "cotton", "rui": "cotton",
    "ganna": "sugarcane", "sugarcane": "sugarcane",
    "makki": "maize", "maize": "maize", "corn": "maize",
    "tamatar": "tomato", "tomato": "tomato", "tamatar": "tomato",
    "pyaaz": "onion", "onion": "onion",
    "aloo": "potato", "potato": "potato",
    "aam": "mango", "mango": "mango", "amba": "mango",
    "mirch": "chilli", "chilli": "chilli", "chili": "chilli", "mirchain": "chilli",
    "sarson": "mustard", "mustard": "mustard", "raya": "mustard",
}

SEASON_MAP = {
    "october": "Rabi", "november": "Rabi", "december": "Rabi",
    "january": "Rabi", "february": "Rabi", "march": "Rabi",
    "april": "Kharif", "may": "Kharif", "june": "Kharif",
    "july": "Kharif", "august": "Kharif", "september": "Kharif",
}

CROP_SEASON = {
    "wheat": "Rabi", "mustard": "Rabi", "potato": "Rabi", "onion": "Rabi",
    "rice": "Kharif", "cotton": "Kharif", "sugarcane": "Kharif",
    "maize": "Kharif", "mango": "Kharif", "chilli": "Kharif",
    "tomato": "Both",
}


# ── M1-2 System Prompt ─────────────────────────────────────────────────

M1_2_SYSTEM_PROMPT = """You are an agricultural query interpreter for Pakistani farmers.
Farmers often describe problems vaguely in Roman Urdu or Urdu.
Your job is to expand their query into precise searchable terms
that will retrieve the best information from our knowledge base.

TASK 1 — INTERPRET THE QUERY:
What is the farmer ACTUALLY asking about?
Common vague queries and their real meaning:
- "fasal kharab ho rahi hai" → Could be disease/pest/water stress/nutrient deficiency
- "patta peela ho raha hai" → Yellow leaf = could be nitrogen deficiency OR rust disease
- "fal nahi aa raha" → Poor fruiting = pollination issue OR nutrient issue OR variety problem
- "paani kitna doon" → Irrigation scheduling query
- "mandi mein kya bhav hai" → Market price query
- "keera lag gaya" → Pest infestation — which pest?

TASK 2 — GENERATE SEARCH QUERIES:
Create EXACTLY 4 search queries for the vector database:
- Query 1: English technical terms (disease/pest/technique name)
- Query 2: Symptoms-based search (what farmer can see/observe)
- Query 3: Pakistan region-specific search
- Query 4: Urdu/Roman Urdu terms (how the farmer described it)

TASK 3 — CATEGORIZE:
Which knowledge base categories are relevant?
DISEASE / PEST / IRRIGATION / FERTILIZER / SOIL / HARVESTING / MARKET

TASK 4 — IDENTIFY URGENCY:
- IMMEDIATE (crop dying now, harvest imminent)
- THIS WEEK (spreading disease, weather window closing)
- THIS SEASON (planning for upcoming weeks)
- GENERAL (information gathering)

OUTPUT ONLY VALID JSON (no markdown, no code fences):
{
  "original_query": "...",
  "interpreted_meaning": "Farmer is asking about X...",
  "crop_name": "wheat",
  "crop_urdu": "gandum",
  "query_category": ["DISEASE"],
  "search_queries": [
    "english technical search query",
    "symptoms-based search query",
    "Pakistan region-specific query",
    "urdu/roman urdu search query"
  ],
  "urgency": "IMMEDIATE",
  "requires_image_analysis": false,
  "weather_data_needed": false,
  "market_price_needed": false,
  "followup_questions": []
}"""


# ── Prompt Builder ─────────────────────────────────────────────────────

def build_m1_2_prompt(
    user_query: str,
    crop_name: Optional[str] = None,
    user_location: Optional[str] = None,
    current_season: Optional[str] = None,
    has_image: bool = False,
) -> str:
    """Build the M1-2 user message for the LLM."""
    crop_info = crop_name or "not specified"
    location_info = user_location or "not specified"
    season_info = current_season or "auto-detect"

    return f"""USER QUERY: {user_query}
CROP MENTIONED (if any): {crop_info}
LOCATION: {location_info}
SEASON: {season_info}
IMAGE UPLOADED: {"yes" if has_image else "no"}"""


def parse_llm_response(text: str) -> Dict:
    """Parse LLM response, extracting JSON even from markdown-wrapped output."""
    # Remove markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: return raw text as error
    return {"error": f"Could not parse LLM response as JSON", "raw": text[:500]}


# ── Rule-based fallback ────────────────────────────────────────────────

def rule_based_expand(
    user_query: str,
    crop_name: Optional[str] = None,
    user_location: Optional[str] = None,
    current_season: Optional[str] = None,
    has_image: bool = False,
) -> Dict:
    """
    Rule-based query expansion when LLM is unavailable.
    Less accurate but always works — essential for hackathon reliability.
    """
    query_lower = user_query.lower()

    # Detect crop
    detected_crop = crop_name
    detected_crop_urdu = ""
    if not detected_crop:
        for keyword, standard in CROP_KEYWORDS.items():
            if keyword in query_lower:
                detected_crop = standard
                # Get Urdu name
                for k, v in CROP_KEYWORDS.items():
                    if v == standard:
                        detected_crop_urdu = k
                        break
                break

    # Detect categories
    categories = set()
    for keyword, category in URDU_PROBLEM_MAP.items():
        if keyword in query_lower:
            categories.add(category)
    if not categories:
        categories.add("GENERAL")

    # Detect urgency
    urgency = "GENERAL"
    urgency_keywords_high = [
        "khatam", "mar raha", "barbaad", "aaj", "kal",
        "tezi se", "phail", "emergency", "critical"
    ]
    urgency_keywords_medium = [
        "kuch din", "thodi", "halka", "shuru",
        "ho raha", "lag gaya", "lag gayi", "ho rahi",
        "ho rahe", "peela", "peeli", "peele",
    ]
    for kw in urgency_keywords_high:
        if kw in query_lower:
            urgency = "IMMEDIATE"
            break
    if urgency == "GENERAL":
        for kw in urgency_keywords_medium:
            if kw in query_lower:
                urgency = "THIS WEEK"
                break

    # Detect season
    season = current_season
    if not season and detected_crop:
        season = CROP_SEASON.get(detected_crop, "")
    if not season:
        for month, s in SEASON_MAP.items():
            if month in query_lower:
                season = s
                break

    # Build 4 diverse search queries
    search_queries = []
    
    # Q1: English technical terms
    category_label = list(categories)[0] if categories else "farming"
    if categories & {"DISEASE", "PEST"}:
        if detected_crop:
            search_queries.append(f"{detected_crop} {list(categories)[0].lower()} Pakistan treatment control")
        else:
            search_queries.append(f"crop {list(categories)[0].lower()} treatment Pakistan")
    elif "IRRIGATION" in categories:
        search_queries.append(f"{detected_crop or 'crop'} irrigation schedule water requirement Pakistan")
    elif "FERTILIZER" in categories:
        search_queries.append(f"{detected_crop or 'crop'} fertilizer dose per acre NPK recommendation")
    elif "MARKET" in categories:
        search_queries.append(f"{detected_crop or 'crop'} mandi price Pakistan today {user_location or ''}")
    else:
        search_queries.append(f"{detected_crop or 'agriculture'} farming Pakistan practices")

    # Q2: Symptoms-based (extract key symptoms from query)
    # Remove common filler words to extract the core issue
    filler_words = ["meri", "mujhe", "hai", "ho", "raha", "rahi", "rahe",
                     "kya", "karun", "karein", "ka", "ki", "ke", "mein", "ko",
                     "aaj", "kal", "aur", "toh", "to"]
    words = query_lower.split()
    core_words = [w for w in words if w not in filler_words and len(w) > 2]
    symptom_query = " ".join(core_words[:10]) if len(core_words) >= 3 else query_lower[:100]
    search_queries.append(symptom_query)

    # Q3: Pakistan region-specific
    location = user_location or "Pakistan"
    if detected_crop:
        search_queries.append(f"{detected_crop} {location} {category_label.lower()} advice")
    else:
        search_queries.append(f"farming {location} {category_label.lower()} help")

    # Q4: Urdu/Roman Urdu (extract key Urdu terms)
    # Keep the original query as-is for the Urdu version
    search_queries.append(query_lower[:150])

    # Ensure exactly 4 unique-ish queries
    seen = set()
    unique_queries = []
    for sq in search_queries:
        if sq[:40] not in seen:
            unique_queries.append(sq)
            seen.add(sq[:40])
    while len(unique_queries) < 4:
        unique_queries.append(query_lower[:80])
    search_queries = unique_queries[:4]

    result = {
        "original_query": user_query,
        "interpreted_meaning": f"Farmer query about {detected_crop or 'crop'} related to {', '.join(categories)}",
        "crop_name": detected_crop or "general",
        "crop_urdu": detected_crop_urdu or "",
        "query_category": list(categories),
        "search_queries": search_queries[:4],
        "urgency": urgency,
        "requires_image_analysis": has_image,
        "weather_data_needed": "IRRIGATION" in categories,
        "market_price_needed": "MARKET" in categories,
        "followup_questions": [],
    }
    return result


# ── Public API ─────────────────────────────────────────────────────────

def expand_query(
    user_query: str,
    crop_name: Optional[str] = None,
    user_location: Optional[str] = None,
    current_season: Optional[str] = None,
    has_image: bool = False,
    use_llm: bool = True,
) -> Dict:
    """
    Expand a vague farmer query into 4 precise search queries + metadata.

    Uses Groq Llama-3.3-70b by default. Falls back to rule-based if
    API key is missing, API call fails, or use_llm=False.

    Args:
        user_query: Farmer's original query (Roman Urdu, Urdu, or English)
        crop_name: Known crop (None = auto-detect from query)
        user_location: Farmer's city/region (None = unknown)
        current_season: Rabi/Kharif (None = auto-detect)
        has_image: Whether user uploaded a crop photo
        use_llm: If True, try Groq first. If False, use rules only.

    Returns:
        Dict with keys: original_query, interpreted_meaning, crop_name, crop_urdu,
                        query_category, search_queries, urgency, requires_image_analysis,
                        weather_data_needed, market_price_needed, followup_questions
    """
    if not use_llm:
        return rule_based_expand(
            user_query, crop_name, user_location, current_season, has_image
        )

    # Try LLM path
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        # Fallback to rule-based
        return rule_based_expand(
            user_query, crop_name, user_location, current_season, has_image
        )

    if Groq is None:
        return rule_based_expand(
            user_query, crop_name, user_location, current_season, has_image
        )

    try:
        client = Groq(api_key=groq_api_key)
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        user_prompt = build_m1_2_prompt(
            user_query, crop_name, user_location, current_season, has_image
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": M1_2_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=800,
        )

        raw = response.choices[0].message.content
        result = parse_llm_response(raw)

        if "error" in result:
            # LLM returned unparseable JSON — fallback to rules
            return rule_based_expand(
                user_query, crop_name, user_location, current_season, has_image
            )

        # Validate required fields
        if "search_queries" not in result or not result["search_queries"]:
            result["search_queries"] = [user_query] * 4

        if "crop_name" not in result or not result["crop_name"]:
            result["crop_name"] = crop_name or "general"

        # Ensure exactly 4 search queries
        while len(result.get("search_queries", [])) < 4:
            result["search_queries"].append(user_query[:100])

        result["search_queries"] = result["search_queries"][:4]

        return result

    except Exception as e:
        # Any API failure — fallback to rules
        return rule_based_expand(
            user_query, crop_name, user_location, current_season, has_image
        )


def expand_and_retrieve(
    user_query: str,
    crop_name: Optional[str] = None,
    user_location: Optional[str] = None,
    current_season: Optional[str] = None,
    has_image: bool = False,
    top_k: int = 3,
    use_llm: bool = True,
) -> Tuple[str, Dict]:
    """
    Chain expand_query → retrieve_crop_knowledge for each expanded query.
    Returns combined RAG context + expansion metadata.

    Args:
        Same as expand_query() + top_k results per query

    Returns:
        (combined_context_string, expansion_metadata_dict)
    """
    from .retriever import retrieve_crop_knowledge

    # Step 1: Expand the query
    expansion = expand_query(
        user_query=user_query,
        crop_name=crop_name,
        user_location=user_location,
        current_season=current_season,
        has_image=has_image,
        use_llm=use_llm,
    )

    # Step 2: Retrieve for each search query
    all_results = []
    seen_chunks = set()

    for sq in expansion.get("search_queries", [user_query]):
        categories = expansion.get("query_category", [])
        for cat in categories:
            if cat == "MARKET":
                continue  # Market queries go to Tavily, not RAG
            if cat == "GENERAL":
                continue
            context = retrieve_crop_knowledge(
                query=sq,
                crop_name=expansion.get("crop_name", crop_name),
                category=cat if cat in ("DISEASE", "PEST", "IRRIGATION", "FERTILIZER",
                                         "SOIL", "HARVESTING", "STORAGE", "VARIETIES") else None,
                top_k=top_k,
            )
            # Deduplicate by content preview
            preview = context[:50]
            if preview not in seen_chunks and context not in ("No relevant agricultural knowledge found for this query.",
                                                               ""):
                all_results.append(context)
                seen_chunks.add(preview)

    # Also do a general (no-category) search
    general_context = retrieve_crop_knowledge(
        query=user_query,
        crop_name=expansion.get("crop_name", crop_name),
        top_k=top_k,
    )
    preview = general_context[:50]
    if preview not in seen_chunks and general_context not in (
        "No relevant agricultural knowledge found for this query.", ""
    ):
        all_results.append(general_context)

    # Combine all results
    if all_results:
        combined = "\n\n".join(all_results)
    else:
        combined = retrieve_crop_knowledge(query=user_query, top_k=top_k)

    return combined, expansion


# ── CLI entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    test_query = " ".join(sys.argv[1:]) or "meri gandum ki fasal peeli ho rahi hai paton par zard dhariyan"
    
    # Test with rule-based (no API key needed)
    print("=== Rule-based expansion ===")
    result = expand_query(test_query, use_llm=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n=== Expand + Retrieve ===")
    context, meta = expand_and_retrieve(test_query, use_llm=False)
    print(f"\nCategories: {meta['query_category']}")
    print(f"Urgency: {meta['urgency']}")
    print(f"Search queries: {meta['search_queries']}")
    print(f"\nContext ({len(context)} chars):")
    print(context[:500])

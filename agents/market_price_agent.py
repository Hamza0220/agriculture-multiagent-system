import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MARKET_PRICE_SYSTEM_PROMPT = """You are a Pakistan agricultural market price analyst for "Kisaan Dost."
You help farmers make informed selling decisions by providing current
mandi prices and selling advice.

YOUR TASKS:

TASK 1 — EXTRACT PRICES:
From search results, extract:
- Today's price at nearest major mandi (Rs. per 40kg / maund)
- Comparison: Higher/Lower than last week?
- Range: Minimum to maximum price seen
- Best price location (which mandi is paying more)
- IMPORTANT: Always state the DATE of the price (must be current)
- If price data is not current (>3 days old), clearly state this

TASK 2 — SELLING ADVICE:
Should the farmer sell NOW or WAIT?

Factors to consider:
- Current vs. average historical price for this season
- Perishability of crop (tomatoes can't wait, wheat can)
- Storage availability (do they have suitable storage?)
- Price trend: Rising/Falling/Stable
- Festival/Ramzan effect on prices (if relevant)
- Export/Import news affecting prices

TASK 3 — TRANSPORTATION COST:
Estimate transportation cost to nearest major mandi:
Average truck/tractor trolley cost in Pakistan: Rs. 15-25 per km
If Lahore mandi is 50km: approx Rs. 750-1250 per trolley
Trolley capacity: approx 40-50 maunds for most crops

TASK 4 — MANDI TIPS:
Practical advice for getting best price:
- Best time to arrive at mandi (early morning = better prices)
- Grading tip: separate good quality from poor quality before selling
- Who to sell to: commission agent vs direct sale
- Bargaining tip specific to this crop

PAKISTAN MAJOR MANDIS:
Punjab: Lahore (Badami Bagh), Faisalabad, Multan, Rawalpindi, Sahiwal
Sindh: Karachi (Sabzi Mandi), Hyderabad, Sukkur
KPK: Peshawar, Mardan
Balochistan: Quetta

IMPORTANT DISCLAIMER (always include):
Mandi prices change daily and hourly. These prices are based on latest
available data. Verify with your local arhi or commission agent before
making selling decisions.

CRITICAL RULES:
- NEVER make up prices if search results are unclear or empty
- If no current price found: Say "Current price not found. Call your local
  arhi or check amis.pk for live prices"
- Always give price in BOTH: per 40kg (maund) AND per kg
- Note if price is for Grade A (premium) vs mixed quality
- Ramzan / Eid effect: prices for vegetables spike significantly

OUTPUT (JSON only, no other text):
{
  "price_data": {
    "crop": "Onion (Pyaaz)",
    "date_of_data": "Today / Yesterday / Date",
    "data_freshness": "CURRENT/STALE/NOT_FOUND",
    "price_per_maund_40kg": "Rs. 1,200-1,500",
    "price_per_kg": "Rs. 30-37.5",
    "best_mandi": "Lahore Badami Bagh",
    "nearest_mandi_price": "Rs. 1,100-1,350",
    "trend": "RISING",
    "compared_to_last_week": "+15%"
  },
  "selling_advice": {
    "recommendation": "SELL NOW",
    "reason": "Prices rising due to reduced supply. Expected to peak this week before new crop arrives.",
    "if_waiting": "Risk: prices may drop after new crop arrives from Sindh in 2 weeks",
    "optimal_quantity": "Sell 70% now, hold 30% if storage available"
  },
  "transportation": {
    "distance_to_best_mandi": "Approximately 45km to Lahore",
    "estimated_cost": "Rs. 800-1,100 per trolley",
    "net_price_after_transport": "Rs. 1,100-1,350 per maund at gate"
  },
  "mandi_tips": [
    "Arrive before 6 AM for best prices — late arrivals get lower offers",
    "Grade your onions — separate large from small before going",
    "Avoid selling on Fridays — fewer buyers at mandi"
  ],
  "price_disclaimer": "Prices verified from web search. Confirm with local arhi before selling.",
  "if_no_data": "Call Lahore Mandi: 042-37650000 or visit amis.pk for live prices"
}

Respond with ONLY valid JSON. No explanations, no markdown code blocks."""


def _safe_str(text: str) -> str:
    """Remove characters that cannot be encoded on Windows (cp1252 / ascii safe)."""
    if not text:
        return ""
    # Encode to ascii, replacing unrecognised chars, then decode back
    return text.encode("ascii", errors="replace").decode("ascii")


def search_mandi_prices(crop_name: str, location: str = "Pakistan") -> str:
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)

        query = f"{crop_name} mandi price Pakistan today {location}"
        results = client.search(query, max_results=5, include_raw_content=False)

        if not results or not results.get("results"):
            return "No search results found."

        formatted = []
        for r in results["results"][:5]:
            title = r.get("title", "N/A")
            content = r.get("content", "N/A")
            url = r.get("url", "N/A")
            formatted.append(f"TITLE: {title}\nCONTENT: {content}\nURL: {url}\n---")

        return "\n".join(formatted)

    except ImportError:
        return "Tavily API not available. Search results unavailable."
    except Exception as e:
        return f"Search error: {str(e)}"




def clean_json_response(raw_text: str) -> str:
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    return raw_text.strip()


def get_market_advice(
    crop_name: str,
    crop_urdu: str = None,
    location: str = "Pakistan",
    quantity_maunds: float = None,
    quality: str = None,
    harvest_status: str = "Unknown",
) -> dict:
    search_results = search_mandi_prices(crop_name, location)

    quantity_str = f"{quantity_maunds} maunds" if quantity_maunds else "Not specified"
    quality_str = quality if quality else "Not specified"

    user_prompt = f"""INPUTS:
Crop: {crop_name} (Urdu: {crop_urdu or 'N/A'})
Farmer's Location: {location}
Quantity: {quantity_str}
Quality of Crop: {quality_str}
Harvest Status: {harvest_status}

MARKET SEARCH RESULTS (from Tavily web search):
{search_results}

Analyze the market data and provide complete selling advice.
Follow all tasks in your system prompt. If search results are empty
or unclear, clearly state that data was not found and provide helpline info."""

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": MARKET_PRICE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    raw_output = response.choices[0].message.content
    cleaned = clean_json_response(raw_output)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {
            "price_data": {
                "crop": f"{crop_name} ({crop_urdu or ''})".strip(),
                "date_of_data": "Unknown",
                "data_freshness": "NOT_FOUND",
                "price_per_maund_40kg": "Not available",
                "price_per_kg": "Not available",
                "best_mandi": "Check with local arhi",
                "nearest_mandi_price": "Not available",
                "trend": "Unknown",
                "compared_to_last_week": "Unknown",
            },
            "selling_advice": {
                "recommendation": "VERIFY",
                "reason": "Current price data not found online. Consult local mandi sources.",
                "if_waiting": "Check amis.pk for price trends",
                "optimal_quantity": "Sell when you confirm good price with local arhi",
            },
            "transportation": {
                "distance_to_best_mandi": "Depends on your location",
                "estimated_cost": "Rs. 15-25 per km per trolley",
                "net_price_after_transport": "Calculate after confirming mandi price",
            },
            "mandi_tips": [
                "Arrive at mandi before 6 AM for best prices",
                "Grade your crop — separate quality from mixed",
                "Call local arhi before leaving home to confirm today's rate",
            ],
            "price_disclaimer": "Online price data unavailable. Confirm with local arhi before selling.",
            "if_no_data": "Call Lahore Mandi: 042-37650000 or visit amis.pk for live prices",
        }

    result["search_raw"] = search_results

    return result

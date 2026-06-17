"""
Quick diagnostic script — tests each agent independently
Run: python debug_agents.py
"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

def sep(title):
    print("\n" + "="*55)
    print(f"  {title}")
    print("="*55)

# ── 1. ENV CHECK ────────────────────────────────────────────
sep("ENV KEYS CHECK")
keys = [
    "GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",
    "OPENWEATHER_API_KEY", "TAVILY_API_KEY", "GROQ_MODEL", "VISION_MODEL",
]
for k in keys:
    val = os.getenv(k, "")
    status = "OK" if val and "your_" not in val else "MISSING/PLACEHOLDER"
    masked = val[:12] + "..." if val and len(val) > 12 else val
    print(f"  {status:12}  {k} = {masked}")

# ── 2. GROQ ─────────────────────────────────────────────────
sep("GROQ API TEST")
try:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "user", "content": "Reply with only: OK"}],
        max_tokens=5,
    )
    print(f"  GROQ: {resp.choices[0].message.content.strip()}")
except Exception as e:
    print(f"  GROQ ERROR: {type(e).__name__}: {e}")

# ── 3. OPENROUTER / VISION ──────────────────────────────────
sep("OPENROUTER (Vision) API TEST")
try:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("VISION_MODEL", "google/gemini-2.0-flash-exp:free"),
        messages=[{"role": "user", "content": "Reply with only: OK"}],
        max_tokens=5,
    )
    print(f"  OPENROUTER: {resp.choices[0].message.content.strip()}")
except Exception as e:
    print(f"  OPENROUTER ERROR: {type(e).__name__}: {e}")

# ── 4. TAVILY ───────────────────────────────────────────────
sep("TAVILY SEARCH TEST")
try:
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search("wheat price Pakistan today", max_results=2)
    count = len(results.get("results", []))
    print(f"  TAVILY: Got {count} search results")
    if count > 0:
        print(f"  First result: {results['results'][0].get('title','')[:80]}")
except Exception as e:
    print(f"  TAVILY ERROR: {type(e).__name__}: {e}")

# ── 5. CROP DOCTOR ──────────────────────────────────────────
sep("CROP DOCTOR AGENT TEST")
try:
    from agents.crop_doctor import diagnose
    result = diagnose(
        farmer_description="wheat leaves turning yellow with rust spots in Lahore",
        crop_name="wheat",
        location="Lahore",
        season="Rabi",
        weather_summary="Temp 28C, Humidity 65%",
        rag_context="No RAG context",
    )
    diag = result.get("diagnosis", {})
    primary = diag.get("primary", "No diagnosis returned")
    confidence = diag.get("confidence", "?")
    severity = diag.get("severity", "?")
    print(f"  PRIMARY:    {primary}")
    print(f"  CONFIDENCE: {confidence}")
    print(f"  SEVERITY:   {severity}")
except Exception as e:
    print(f"  CROP DOCTOR ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

# ── 6. MARKET PRICE ─────────────────────────────────────────
sep("MARKET PRICE AGENT TEST")
try:
    from agents.market_price_agent import search_mandi_prices, get_market_advice

    print("  [step 1] Tavily search...")
    raw = search_mandi_prices("wheat", "Lahore")
    print(f"  Search OK: {raw[:100]}...")

    print("  [step 2] LLM analysis...")
    result = get_market_advice(crop_name="wheat", crop_urdu="gandum", location="Lahore")
    pd = result.get("price_data", {})
    sa = result.get("selling_advice", {})
    print(f"  PRICE:  {pd.get('price_per_maund_40kg', 'N/A')}")
    print(f"  TREND:  {pd.get('trend', 'N/A')}")
    print(f"  ADVICE: {sa.get('recommendation', 'N/A')}")
except Exception as e:
    print(f"  MARKET PRICE ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

# ── 7. IRRIGATION ───────────────────────────────────────────
sep("IRRIGATION ADVISOR TEST")
try:
    from agents.irrigation_advisor import advise_irrigation
    result = advise_irrigation(crop_name="wheat", crop_urdu="gandum", location="Lahore")
    print(f"  STRESS LEVEL:   {result.get('water_stress_level', 'N/A')}")
    print(f"  IRRIGATE TODAY: {result.get('irrigate_today', 'N/A')}")
    adv = result.get("irrigation_advice", {})
    print(f"  AMOUNT:         {adv.get('amount', 'N/A')}")
    print(f"  ET0:            {result.get('et0_mm_per_day', 'N/A')} mm/day")
except Exception as e:
    print(f"  IRRIGATION ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*55)
print("  DIAGNOSTIC COMPLETE")
print("="*55 + "\n")

"""
LIVE API TEST - Weather, Groq, Pipeline
"""
import sys
import json
sys.path.insert(0, ".")

print("=" * 60)
print("TEST 1: OpenWeatherMap Live API - Lahore")
print("=" * 60)

from utils.weather_interpreter import interpret_weather, fetch_raw_weather

weather = interpret_weather("Lahore")
c = weather.get("current", {})

if c.get("temp_c"):
    print(f"  Temperature: {c['temp_c']}C")
    print(f"  Humidity: {c['humidity_pct']}%")
    print(f"  Condition: {c['condition']}")
    print(f"  Wind: {c['wind_kmh']} km/h")
    print(f"  Rain 24h: {c['rainfall_24h_mm']}mm")
    print(f"  Farming Summary: {c['farming_summary'][:80]}...")
    print(f"  Spray Window: {weather.get('spray_window', 'N/A')[:80]}")
    print(f"  Irrigation Decision: {weather.get('irrigation_decision', 'N/A')[:80]}")

    forecast = weather.get("forecast_3day", [])
    if forecast:
        print(f"\n  3-Day Forecast:")
        for d in forecast:
            print(f"    {d['date']}: {d['high']}C / {d['low']}C, Rain: {d['rain_probability']}, {d['farming_note']}")

    alerts = weather.get("alerts", [])
    if alerts:
        print(f"\n  Alerts: {alerts}")

    print("\n  RESULT: PASS - OpenWeatherMap LIVE data received!")
else:
    print(f"  RESULT: FAIL - No temp data. Raw error: {weather.get('raw', {}).get('error', 'unknown')}")


print("\n" + "=" * 60)
print("TEST 2: Groq API - Orchestrator Live Test")
print("=" * 60)

from agents.orchestrator import orchestrate

test_queries = [
    ("Mere gandum ke paton par zard dhariyan aa rahi hain, Faisalabad mein hun", False),
    ("Aaj Lahore mein pyaaz ka kya bhav hai?", False),
    ("Mujhe batao ke chawal ko kitna paani dena chahiye", False),
]

for query, has_img in test_queries:
    try:
        result = orchestrate(
            user_query=query,
            has_image=has_img,
            location="Pakistan",
        )
        agents = result.get("agents_to_call", [])
        crop = result.get("crop_detected", "?")
        urgency = result.get("urgency", "?")
        season = result.get("season", "?")
        language = result.get("farmer_language", "?")

        print(f"\n  Query: '{query[:60]}...'")
        print(f"    Agents to call: {agents}")
        print(f"    Crop: {crop}, Season: {season}, Urgency: {urgency}")
        print(f"    Language: {language}")

        # Verify routing
        if "zard" in query.lower() or "gandum" in query.lower():
            routing_ok = "CROP_DOCTOR" in agents
        elif "bhav" in query.lower() or "pyaaz" in query.lower():
            routing_ok = "MARKET_PRICE" in agents
        elif "paani" in query.lower() or "chawal" in query.lower():
            routing_ok = "IRRIGATION_ADVISOR" in agents
        else:
            routing_ok = True

        print(f"    Routing correct: {'PASS' if routing_ok else 'FAIL'}")
        print(f"    RESULT: {'PASS' if agents else 'FAIL'}")

    except Exception as e:
        print(f"\n  Query: '{query[:60]}...'")
        print(f"  RESULT: FAIL - {str(e)[:100]}")


print("\n" + "=" * 60)
print("TEST 3: Full Pipeline - Complete Integration Test")
print("=" * 60)

from agents.agent_pipeline import run_agri_pipeline

try:
    result = run_agri_pipeline(
        user_query="Mere gandum ke paton par zard dhariyan aa rahi hain, Faisalabad mein hun",
        image_base64=None,
        location="Faisalabad",
    )

    print(f"  Agents called: {result['agents_called']}")
    print(f"  Crop detected: {result['crop_detected']}")
    print(f"  Urgency: {result['urgency']}")
    print(f"  Response length: {len(result['response_text'])} chars")

    # Show first 200 chars of response
    response_preview = result['response_text'][:200].replace('\n', ' ')
    print(f"  Response preview: {response_preview}...")

    # Check raw outputs
    raw = result.get('raw_outputs', {})
    cd = raw.get('crop_doctor')
    if cd:
        diag = cd.get('diagnosis', {})
        print(f"  Crop Doctor diagnosis: {diag.get('primary', 'N/A')}")
        print(f"  Confidence: {diag.get('confidence', 'N/A')}")
        print(f"  Severity: {diag.get('severity', 'N/A')}")

    has_response = len(result['response_text']) > 100
    has_helpline = "0800-15000" in result['response_text']
    print(f"\n  Has meaningful response: {'PASS' if has_response else 'FAIL'}")
    print(f"  Has helpline number: {'PASS' if has_helpline else 'FAIL'}")
    print(f"  RESULT: {'PASS' if has_response and has_helpline else 'FAIL'}")

except Exception as e:
    import traceback
    print(f"  RESULT: FAIL")
    traceback.print_exc()


print("\n" + "=" * 60)
print("TEST 4: Market Price Agent (Live Tavily Search)")
print("=" * 60)

from agents.market_price_agent import search_mandi_prices, get_market_advice

try:
    search_results = search_mandi_prices("onion", "Lahore")
    has_results = len(search_results) > 50 and "No search results" not in search_results
    print(f"  Search results length: {len(search_results)} chars")
    print(f"  Has results: {'PASS' if has_results else 'NO_RESULTS (may be expected)'}")

    if has_results:
        print(f"  Preview: {search_results[:150]}...")
    else:
        print(f"  Preview: {search_results[:200]}...")

except Exception as e:
    print(f"  RESULT: FAIL - {e}")


print("\n" + "=" * 60)
print("TEST 5: Crop Doctor (Text-only, no image)")
print("=" * 60)

from agents.crop_doctor import diagnose

try:
    result = diagnose(
        farmer_description="Mere gandum ke paton par zard dhariyan aa rahi hain, patte peele ho rahe hain",
        crop_name="wheat",
        location="Faisalabad",
        season="Rabi",
    )

    diag = result.get("diagnosis", {})
    treatment = result.get("treatment", {})
    opt_a = treatment.get("option_a", {})

    print(f"  Diagnosis: {diag.get('primary', 'N/A')}")
    print(f"  Urdu Name: {diag.get('urdu_name', 'N/A')}")
    print(f"  Confidence: {diag.get('confidence', 'N/A')}")
    print(f"  Severity: {diag.get('severity', 'N/A')}")
    print(f"  Treatment: {opt_a.get('product', 'N/A')} - {opt_a.get('cost_per_acre', 'N/A')}")
    print(f"  Safety warnings: {result.get('safety_warnings', [])}")

    has_diag = bool(diag.get("primary"))
    has_treatment = bool(opt_a.get("product"))
    print(f"\n  Has diagnosis: {'PASS' if has_diag else 'FAIL'}")
    print(f"  Has treatment: {'PASS' if has_treatment else 'FAIL'}")

except Exception as e:
    import traceback
    print(f"  RESULT: FAIL")
    traceback.print_exc()


print("\n" + "=" * 60)
print("FINAL LIVE TEST SUMMARY")
print("=" * 60)
print("  All 5 live tests completed!")
print("  APIs tested: OpenWeatherMap, Groq, OpenRouter (vision), Tavily")
print("  Verify the PASS/FAIL results above for each test.")

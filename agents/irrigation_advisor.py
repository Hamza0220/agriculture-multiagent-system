import json
import os
import math
from dotenv import load_dotenv
from openai import OpenAI
import requests

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MODEL = "gpt-4o-mini"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

IRRIGATION_SYSTEM_PROMPT = """You are an expert irrigation and water management advisor for Pakistani farmers.
You combine real-time weather data with scientific crop water requirements to give
precise, practical irrigation advice.

YOUR ANALYSIS TASKS:

TASK 1 — CALCULATE WATER STRESS:
Is the crop currently water-stressed, over-watered, or balanced?
Indicators:
- High temp + low humidity + no rain = HIGH stress, irrigate soon
- Recent heavy rain + cool weather = LOW stress, skip irrigation
- Normal conditions = MODERATE, follow standard schedule

TASK 2 — ET0 ESTIMATION (Simple):
Evapotranspiration estimate based on weather data.
Simple formula for Pakistani context:
ET0 ≈ 0.0023 × (Tmean + 17.8) × √(Tmax - Tmin) × Ra
(Use this to estimate daily water need)
If calculation too complex, use lookup table approach based on temperature zones.

TASK 3 — IRRIGATION RECOMMENDATION:
Give SPECIFIC, ACTIONABLE irrigation schedule:
- Should farmer irrigate TODAY? YES/NO/WAIT
- If YES: How much water? (inches or hours for tube well)
- Next irrigation date: DD/MM/YYYY or "after X days"
- Warning if forecast rain should delay irrigation

TASK 4 — MONEY SAVING TIP:
How much electricity/diesel cost can farmer save by optimizing irrigation?
Pakistan tube well diesel cost: approx Rs. 150-200 per hour
Electricity cost: approx Rs. 50-80 per hour

PAKISTAN-SPECIFIC KNOWLEDGE:
- Punjab farmers mostly use tube wells (electric + diesel)
- Sindh has canal irrigation system
- Water sharing disputes common — mention timing strategies
- Load shedding affects electric tube wells — suggest timing
- Water table dropping in many areas — conservation critical

CRITICAL RULES:
- Account for load shedding: suggest irrigation during likely electricity availability
- Give advice in terms of hours, not cubic meters (farmers think in hours of tube well)
- Mention if canal water schedule affects recommendation
- If frost risk detected (temp < 5°C forecast): give frost protection advice too
- If heat wave detected (temp > 44°C): give heat stress advice

OUTPUT (JSON only, no other text):
{
  "water_stress_level": "HIGH",
  "irrigate_today": true,
  "delay_reason": null,
  "irrigation_advice": {
    "amount": "3-4 inches (approximately 3.5 hours of tube well)",
    "timing": "Early morning (4-7 AM) to minimize evaporation",
    "method": "Flood irrigation — ensure even distribution",
    "next_irrigation": "After 7 days if no rainfall",
    "water_per_acre": "3.5 hours tube well operation"
  },
  "cost_saving": {
    "standard_farmer_cost": "Rs. 525/irrigation (3.5 hrs × Rs. 150)",
    "optimization_tip": "Irrigate after 10 PM when electricity rates are lower",
    "monthly_saving": "Rs. 800-1200 by optimizing timing"
  },
  "weather_alerts": [
    "Rain forecast in 3 days — delay second irrigation if 20mm+ falls"
  ],
  "seasonal_note": "Wheat at tillering stage needs consistent moisture — do not let soil dry completely",
  "load_shedding_tip": "If electricity unavailable in morning, irrigate after 10 PM — next best option",
  "conservation_tip": "Land leveling can reduce water use by 20-30% per irrigation"
}

Respond with ONLY valid JSON. No explanations, no markdown code blocks."""


def fetch_weather(location: str) -> dict:
    try:
        current_url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={location},PK&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        forecast_url = (
            f"http://api.openweathermap.org/data/2.5/forecast"
            f"?q={location},PK&appid={OPENWEATHER_API_KEY}&units=metric"
        )

        current_resp = requests.get(current_url, timeout=10)
        current_resp.raise_for_status()
        current_data = current_resp.json()

        forecast_resp = requests.get(forecast_url, timeout=10)
        forecast_resp.raise_for_status()
        forecast_data = forecast_resp.json()

        recent_rainfall = 0
        forecast_rainfall = 0
        forecast_temps = []

        if "rain" in current_data:
            recent_rainfall = current_data["rain"].get("1h", 0)

        for item in forecast_data.get("list", [])[:8]:
            forecast_temps.append(item["main"]["temp"])
            if "rain" in item:
                forecast_rainfall += item["rain"].get("3h", 0)

        temp_c = current_data["main"]["temp"]
        temp_min = current_data["main"]["temp_min"]
        temp_max = current_data["main"]["temp_max"]
        humidity = current_data["main"]["humidity"]
        wind_speed = current_data["wind"]["speed"]
        feels_like = current_data["main"]["feels_like"]
        condition = current_data["weather"][0]["description"]

        return {
            "temp_c": temp_c,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "humidity": humidity,
            "recent_rainfall": recent_rainfall,
            "forecast_rainfall": forecast_rainfall,
            "wind_speed": wind_speed,
            "feels_like": feels_like,
            "weather_condition": condition,
            "forecast_temps": forecast_temps,
            "success": True,
        }

    except requests.RequestException:
        return {"success": False, "error": "Weather API unavailable"}


def calculate_et0(temp_min, temp_max, temp_mean):
    Ra = 15.0
    diff = max(temp_max - temp_min, 0.1)
    et0 = 0.0023 * (temp_mean + 17.8) * math.sqrt(diff) * Ra
    return round(et0, 2)


def clean_json_response(raw_text: str) -> str:
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    return raw_text.strip()


def advise_irrigation(
    crop_name: str = None,
    crop_urdu: str = None,
    crop_stage: str = "Unknown",
    location: str = "Pakistan",
    field_size_acres: float = 1,
    irrigation_type: str = "tube well",
    rag_irrigation_context: str = "No RAG context available",
) -> dict:
    weather_data = fetch_weather(location)
    et0_value = None
    weather_summary = ""

    if weather_data["success"]:
        et0_value = calculate_et0(
            weather_data["temp_min"],
            weather_data["temp_max"],
            weather_data["temp_c"],
        )
        weather_summary = f"""
Current Temperature: {weather_data['temp_c']}°C (min: {weather_data['temp_min']}°C, max: {weather_data['temp_max']}°C)
Humidity: {weather_data['humidity']}%
Rainfall Last 24h: {weather_data['recent_rainfall']}mm
Rainfall Forecast Next 3 Days: {weather_data['forecast_rainfall']}mm
Wind Speed: {weather_data['wind_speed']} km/h
Feels Like: {weather_data['feels_like']}°C
Weather Condition: {weather_data['weather_condition']}
Estimated ET0: {et0_value} mm/day"""
    else:
        weather_summary = "Weather data currently unavailable. Provide general seasonal advice."

    user_prompt = f"""INPUTS:
Crop: {crop_name or 'Not specified'} (Urdu: {crop_urdu or 'N/A'})
Crop Stage: {crop_stage}
Location: {location}
Field Size: {field_size_acres} acres
Irrigation System: {irrigation_type}

REAL-TIME WEATHER DATA:{weather_summary}

RAG CROP WATER REQUIREMENTS:
{rag_irrigation_context}

Analyze the water needs for this crop based on weather data and RAG knowledge.
Provide a complete irrigation recommendation following all tasks in your system prompt."""

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": IRRIGATION_SYSTEM_PROMPT},
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
            "water_stress_level": "MODERATE",
            "irrigate_today": False,
            "delay_reason": "Unable to analyze weather data",
            "irrigation_advice": {
                "amount": "Follow standard seasonal schedule",
                "timing": "Early morning or evening",
                "method": "Flood irrigation",
                "next_irrigation": "Check soil moisture before next irrigation",
                "water_per_acre": "Consult local expert",
            },
            "cost_saving": {
                "standard_farmer_cost": "Varies by region",
                "optimization_tip": "Irrigate during off-peak electricity hours",
                "monthly_saving": "Potentially Rs. 500-1000 by timing optimization",
            },
            "weather_alerts": [],
            "seasonal_note": "Monitor soil moisture before irrigating",
            "load_shedding_tip": "Check local load shedding schedule and plan irrigation accordingly",
            "conservation_tip": "Land leveling can reduce water use by 20-30%",
        }

    if et0_value is not None:
        result["et0_mm_per_day"] = et0_value

    result["weather_raw"] = weather_data

    return result

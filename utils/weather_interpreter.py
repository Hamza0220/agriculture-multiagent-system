import json
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


def fetch_raw_weather(location: str) -> dict:
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

        return {
            "success": True,
            "current": current_data,
            "forecast": forecast_data,
        }

    except requests.RequestException as e:
        return {"success": False, "error": str(e)}


def interpret_weather(location: str, crop_name: str = None, date: str = None) -> dict:
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    raw = fetch_raw_weather(location)

    if not raw["success"]:
        return {
            "current": {
                "temp_c": None,
                "humidity_pct": None,
                "condition": "Weather data unavailable",
                "rainfall_24h_mm": 0,
                "wind_kmh": None,
                "farming_summary": "Mausam ka data abhi nahi mil raha. Apne area ka mausam dekh kar faisla karein.",
            },
            "forecast_3day": [],
            "alerts": ["Weather API unavailable — using seasonal defaults"],
            "spray_window": "Check local weather before spraying",
            "irrigation_decision": "Check soil moisture before irrigating",
            "raw": raw,
        }

    current_data = raw["current"]
    forecast_data = raw["forecast"]

    temp_c = current_data["main"]["temp"]
    humidity = current_data["main"]["humidity"]
    condition = current_data["weather"][0]["description"]
    wind_speed = current_data["wind"]["speed"]
    rain_24h = current_data.get("rain", {}).get("1h", 0) if "rain" in current_data else 0

    farming_friendly = _build_farming_summary(temp_c, humidity, condition, wind_speed)

    forecast_3day = _build_3day_forecast(forecast_data)

    spray_window, irrigation_decision = _assess_farming_conditions(
        temp_c, humidity, wind_speed, forecast_3day
    )

    alerts = []
    if temp_c > 42:
        alerts.append("Heat wave! Crops under stress — provide shade or extra irrigation if possible")
    if any(d.get("low", 100) < 5 for d in forecast_3day):
        alerts.append("Frost risk in coming days — protect sensitive crops")
    if any(d.get("rain_probability", "0%") > "60" for d in forecast_3day):
        alerts.append("Heavy rain expected — delay irrigation and chemical spraying")

    return {
        "current": {
            "temp_c": temp_c,
            "humidity_pct": humidity,
            "condition": condition,
            "rainfall_24h_mm": rain_24h,
            "wind_kmh": wind_speed,
            "farming_summary": farming_friendly,
        },
        "forecast_3day": forecast_3day,
        "alerts": alerts if alerts else [],
        "spray_window": spray_window,
        "irrigation_decision": irrigation_decision,
        "raw": raw,
    }


def _build_farming_summary(temp_c, humidity, condition, wind_speed) -> str:
    remarks = []

    if temp_c < 10:
        remarks.append("thand hai")
    elif temp_c < 25:
        remarks.append("mausam theek hai")
    elif temp_c < 35:
        remarks.append("garmi hai")
    else:
        remarks.append("bohat garmi hai")

    if "rain" in condition.lower():
        remarks.append("barish ho rahi hai — spray na karein")
    elif humidity > 80:
        remarks.append("bohat humidity hai — bimari ka risk zyada hai")
    elif humidity < 30:
        remarks.append("bohat khushki hai — paani ki zaroorat hai")

    if wind_speed > 20:
        remarks.append("hawa tez hai — spray safe nahi hai")
    else:
        remarks.append("hawa bhi theek hai")

    return ". ".join(remarks).capitalize()


def _build_3day_forecast(forecast_data: dict) -> list:
    today = datetime.now().date()
    daily = {}

    for item in forecast_data.get("list", []):
        dt = datetime.fromtimestamp(item["dt"])
        day_key = dt.date().isoformat()

        if day_key not in daily:
            daily[day_key] = {
                "temps": [],
                "rains": [],
                "conditions": [],
                "pop": [],
            }

        daily[day_key]["temps"].append(item["main"]["temp"])
        if "rain" in item:
            daily[day_key]["rains"].append(item["rain"].get("3h", 0))
        daily[day_key]["conditions"].append(item["weather"][0]["description"])
        daily[day_key]["pop"].append(item.get("pop", 0))

    results = []
    for i, (day_key, data) in enumerate(sorted(daily.items())):
        if i >= 3:
            break

        dt = datetime.strptime(day_key, "%Y-%m-%d")
        if dt.date() < today:
            continue

        high = round(max(data["temps"]), 1)
        low = round(min(data["temps"]), 1)
        rain_total = sum(data["rains"])
        max_pop = max(data["pop"]) * 100
        rain_probability = f"{round(max_pop)}%"
        main_condition = max(set(data["conditions"]), key=data["conditions"].count)

        label = "Today" if dt.date() == today else "Tomorrow" if dt.date() == today + timedelta(days=1) else dt.strftime("%A")

        farming_note = _farming_note_for_day(label, rain_probability, max_pop, high)

        results.append({
            "date": label,
            "high": high,
            "low": low,
            "rain_probability": rain_probability,
            "rain_mm": round(rain_total, 1),
            "farming_note": farming_note,
            "condition": main_condition,
        })

    return results


def _farming_note_for_day(label, rain_prob_str, pop, temp) -> str:
    rain_pct = float(rain_prob_str.replace("%", ""))

    if rain_pct > 50:
        return "Barish ka chance hai — spray na karein"
    elif rain_pct > 20:
        return "Barish ho sakti hai — spray ka plan soch samajh kar banayein"
    elif temp > 35:
        return "Bohat garmi — subah jaldi ya raat ko spray karein"
    else:
        return "Mausam theek hai — farming activities ke liye acha din hai"


def _assess_farming_conditions(temp_c, humidity, wind_speed, forecast_3day) -> tuple:
    spray_ok = True
    spray_reasons = []

    if wind_speed > 15:
        spray_ok = False
        spray_reasons.append("hawa tez")
    if humidity > 85:
        spray_ok = False
        spray_reasons.append("bohat zyada humidity")
    if any(float(d.get("rain_probability", "0%").replace("%", "")) > 40 for d in forecast_3day):
        spray_ok = False
        spray_reasons.append("barish ka chance")

    if spray_ok:
        spray_window = "Aaj spray ke liye acha din hai — subah ya shaam karein"
    else:
        spray_window = f"Aaj spray na karein ({', '.join(spray_reasons)}). Agle achy din ka wait karein"

    should_irrigate = True
    irrigate_reasons = []

    if temp_c < 15 and humidity > 60:
        should_irrigate = False
        irrigate_reasons.append("thanda mausam aur humidity hai")
    if any(float(d.get("rain_probability", "0%").replace("%", "")) > 60 for d in forecast_3day):
        should_irrigate = False
        irrigate_reasons.append("kal barish ka strong chance")

    if should_irrigate:
        irrigation_decision = "Aaj irrigation kar sakte hain — subah ya raat ka time best hai"
    else:
        irrigation_decision = f"Aaj irrigation na karein ({', '.join(irrigate_reasons)})"

    return spray_window, irrigation_decision


def get_weather_summary_for_agent(location: str) -> dict:
    interpreted = interpret_weather(location)
    c = interpreted.get("current", {})

    forecast_rain = 0
    for d in interpreted.get("forecast_3day", []):
        forecast_rain += d.get("rain_mm", 0)

    return {
        "temp_c": c.get("temp_c"),
        "temp_min": None,
        "temp_max": None,
        "humidity": c.get("humidity_pct"),
        "recent_rainfall": c.get("rainfall_24h_mm", 0),
        "forecast_rainfall": forecast_rain,
        "wind_speed": c.get("wind_kmh"),
        "feels_like": None,
        "weather_condition": c.get("condition", "Unknown"),
        "success": interpreted.get("raw", {}).get("success", False),
    }

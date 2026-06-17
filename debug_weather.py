import os
from dotenv import load_dotenv
load_dotenv()
from utils.weather_interpreter import fetch_raw_weather

cities = ["Lahore", "Karachi", "Multan", "Faisalabad", "Peshawar", "Islamabad", "Quetta"]
for city in cities:
    raw = fetch_raw_weather(city)
    if raw.get("success"):
        t    = raw["current"]["main"]["temp"]
        h    = raw["current"]["main"]["humidity"]
        cond = raw["current"]["weather"][0]["description"]
        name = raw["current"].get("name", "?")
        print(f"{city:12} -> API city: {name:12} | Temp: {t}C | Hum: {h}% | {cond}")
    else:
        print(f"{city:12} -> ERROR: {raw.get('error', 'unknown')}")

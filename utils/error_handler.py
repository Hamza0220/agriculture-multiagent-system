from datetime import datetime


def handle_error(
    error_type: str,
    failed_component: str,
    original_query: str,
    partial_data: dict = None,
    crop_name: str = None,
    location: str = "Pakistan",
) -> str:
    if partial_data is None:
        partial_data = {}

    crop_label = crop_name or "fasal"
    location_label = location or "aapka area"

    error_responses = {
        "weather_api_failed": _weather_api_failed(crop_label, location_label),
        "vision_unclear": _vision_unclear(),
        "market_price_not_found": _market_price_not_found(crop_label),
        "rag_no_results": _rag_no_results(crop_label, original_query),
        "groq_rate_limit": _groq_rate_limit(partial_data),
    }

    default_response = (
        f"⚠️ Kisaan Dost mein abhi masla aa gaya hai.\n\n"
        f"Aap {crop_label} ke baare mein pooch rahe hain. "
        f"Thodi der baad dobara try karein.\n\n"
        f"📞 Is dauraan aap Pakistan Agriculture Helpline: 0800-15000 par call kar sakte hain "
        f"(Monday-Saturday, free hai).\n\n"
        f"Ya apne local Agriculture Extension Officer se milein."
    )

    response = error_responses.get(error_type, default_response)

    helpline_block = (
        "\n\n---\n📞 **Zaroorat Pari To:**\n"
        "• Agriculture Helpline: 0800-15000 (Free, Mon-Sat)\n"
        "• PARC: 051-9255012\n"
        "• Agriculture Extension Punjab: 042-99200762\n"
        "• Pest Warning Punjab: 042-99200763"
    )

    if "0800-15000" not in response:
        response += helpline_block

    return response


def _weather_api_failed(crop_label: str, location_label: str) -> str:
    current_month = datetime.now().month

    if current_month in [10, 11, 12, 1, 2, 3, 4]:
        season_note = (
            "Rabi season chal raha hai. Punjab mein sardiyon mein fasal ko "
            "har 15-20 din baad paani dene ki zaroorat hoti hai. "
            "Sindh mein thoda zyada pani lag sakta hai."
        )
    else:
        season_note = (
            "Kharif season chal raha hai. Monsoon ke mausam mein "
            "barish ka andaza laga kar hi irrigation karein. "
            "Zyada pani fasal ko nuksan de sakta hai."
        )

    return (
        f"⚠️ **Weather Data Abhi Nahi Mil Raha**\n\n"
        f"OpenWeatherMap se real-time data nahi aa raha. "
        f"Ye {location_label} ke liye general seasonal advice hai "
        f"— apne area ka mausam dekh kar adjust karein.\n\n"
        f"🌤️ **Seasonal Guidance:**\n"
        f"{season_note}\n\n"
        f"💧 **Irrigation Tip:**\n"
        f"Subah jaldi ya raat ko irrigation karein taake paani ka evaporation kam ho.\n"
        f"Tube well ho to 2-3 ghante per acre kaafi hota hai (soil moisture check karein)."
    )


def _vision_unclear() -> str:
    return (
        f"📸 **Tasveer Thodi Unclear Hai**\n\n"
        f"Kya aap ye kar sakte hain:\n\n"
        f"1. **Dhoop mein close-up photo lein** — patte ke 30cm nazdik se\n"
        f"2. **Symptoms describe karein:**\n"
        f"   - Kya rang hai? (peela/bhoora/kala/safed)\n"
        f"   - Patte ke upar ya neeche?\n"
        f"   - Powder hai ya dhabb?\n"
        f"   - Kab se shuru hua?\n\n"
        f"Aap description dein to main behtar diagnosis kar sakta hoon.\n"
        f"Ya phir dobara photo lein — subah 8-10 baje ki dhoop best hoti hai."
    )


def _market_price_not_found(crop_label: str) -> str:
    return (
        f"💰 **Aaj Ka Mandi Bhav Online Nahi Mila**\n\n"
        f"{crop_label} ka latest price online nahi mila. Ye resources try karein:\n\n"
        f"📱 **Live Prices Ke Liye:**\n"
        f"• **AMIS Pakistan:** amis.pk (government live mandi prices)\n"
        f"• **Lahore Mandi:** 042-37650000\n"
        f"• **Zarai Baithak:** zarai-baithak app check karein\n\n"
        f"👨‍🌾 **Sab Se Aasan Tareeqa:**\n"
        f"Apne local arhi (commission agent) ko call karein — "
        f"woh aaj ka exact bhav bata dein ge.\n\n"
        f"💡 **Tip:** Mandi jane se pehle arhi se bhav confirm zaroor karein, "
        f"warna be-wajah transport ka kharcha ho sakta hai."
    )


def _rag_no_results(crop_label: str, original_query: str) -> str:
    return (
        f"📚 **Database Mein Is Specific Problem Ka Solution Nahi Mila**\n\n"
        f"Meri database mein {crop_label} ke liye exactly ye problem nahi mili.\n\n"
        f"Ye general guidance hai — **local agriculture officer se confirm karein:**\n\n"
        f"🔍 **Kya Karein:**\n"
        f"1. Apne nazdeeki agri shop ya pesticide dealer se sample dikha kar poochein\n"
        f"2. Pakistan Agriculture Helpline: 0800-15000 par call karein\n"
        f"3. Local Agriculture Extension Officer se field visit ki request karein\n\n"
        f"⚠️ Galat treatment se fasal kharab ho sakti hai — "
        f"expert ki raye zaroori hai."
    )


def _groq_rate_limit(partial_data: dict) -> str:
    partial_text = ""
    if partial_data:
        if "response_text" in partial_data:
            return partial_data["response_text"]
        if "crop_doctor" in partial_data and partial_data["crop_doctor"]:
            cd = partial_data["crop_doctor"]
            diag = cd.get("diagnosis", {})
            if diag:
                partial_text = (
                    f"🔬 **Diagnosis (Partial):** {diag.get('primary', 'Unknown')}\n"
                    f"⚠️ Complete advice ke liye dobara try karein."
                )

    return (
        f"⏳ **Abhi System Thoda Busy Hai**\n\n"
        f"{partial_text or 'Aapka sawal process ho raha hai.'}\n\n"
        f"Thodi der baad dobara try karein.\n"
        f"Ya apne local agri expert se raabta karein.\n\n"
        f"📞 Agriculture Helpline: 0800-15000 (Free)"
    )

PROVINCE_CITIES = {
    "Punjab": [
        "Lahore", "Faisalabad", "Multan", "Sialkot", "Gujranwala",
        "Sahiwal", "Okara", "Sheikhupura", "Kasur", "Pakpattan",
        "Vehari", "Lodhran", "Bahawalpur", "Rahim Yar Khan", "Jhang",
        "Sargodha", "Gujrat", "Mandi Bahauddin", "DG Khan", "Muzaffargarh",
    ],
    "Sindh": [
        "Hyderabad", "Sukkur", "Larkana", "Nawabshah", "Mirpurkhas",
        "Sanghar", "Karachi", "Dadu", "Shikarpur", "Jacobabad",
    ],
    "KPK": [
        "Peshawar", "Mardan", "Swat", "Charsadda", "Nowshera",
        "Abbottabad", "Mansehra", "Bannu", "Kohat", "DI Khan",
    ],
    "Balochistan": [
        "Quetta", "Turbat", "Khuzdar", "Gwadar", "Sibi",
        "Zhob", "Loralai", "Chaman",
    ],
}

CITY_TO_PROVINCE = {}
for province, cities in PROVINCE_CITIES.items():
    for city in cities:
        CITY_TO_PROVINCE[city.lower()] = province


def get_province(location: str) -> str:
    if not location:
        return None

    loc_lower = location.lower().strip()

    if loc_lower in CITY_TO_PROVINCE:
        return CITY_TO_PROVINCE[loc_lower]

    for province in PROVINCE_CITIES:
        if province.lower() in loc_lower:
            return province

    return None


def resolve_location(raw_location: str, known_location: str = None) -> dict:
    location = raw_location.strip() if raw_location else (known_location or "Pakistan")
    province = get_province(location)

    return {
        "city": location if province else ("Pakistan" if not known_location else known_location),
        "province": province,
        "needs_clarification": province is None and location.lower() != "pakistan",
        "question": "Aap kis shehar ya zile se hain?" if province is None and location.lower() != "pakistan" else None,
    }

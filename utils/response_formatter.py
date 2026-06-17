import re


def format_response(
    synthesizer_output: str,
    agents_called: list = None,
    urgency: str = "MEDIUM",
    language: str = "roman_urdu",
) -> str:
    if agents_called is None:
        agents_called = []

    urgency_header = {
        "CRITICAL": "🔴 **CRITICAL ALERT — Turant Action Chahiye** 🔴\n\n",
        "HIGH": "🟡 **Important — Jald Az Jald Action Lein**\n\n",
        "MEDIUM": "🟢 **Aapki Fasal Ke Liye Advice**\n\n",
        "LOW": "🟢 **Aapki Fasal Ke Liye Advice**\n\n",
    }

    text = synthesizer_output.strip()

    header = urgency_header.get(urgency, urgency_header["MEDIUM"])

    for old, new in {
        "🌾 **": "🌾 **",
        "🔬 **": "🔬 **",
        "💧 **": "💧 **",
        "💰 **": "💰 **",
        "📋 **": "📋 **",
        "📞 **": "📞 **",
    }.items():
        pass

    if not text:
        text = "*Koi jawab generate nahi ho saka. Dobara try karein.*"

    text = _format_action_plan(text)
    text = _ensure_mobile_friendly(text)
    text = _add_missing_sections(text, agents_called)

    disclaimer_block = (
        "\n\n---\n"
        "*ℹ️ Ye advice aapki batai hui information par based hai. Local expert se verify karein.*\n"
        "*📞 Agriculture Helpline: 0800-15000 (Free)*"
    )

    if "0800-15000" not in text:
        text += disclaimer_block

    if urgency == "CRITICAL":
        text = (
            '<div style="border:2px solid red; padding:10px; border-radius:8px; margin-bottom:10px;">\n\n'
            + text
            + "\n\n</div>"
        )

    return text


def _format_action_plan(text: str) -> str:
    lines = text.split("\n")
    result = []
    in_action = False
    number = 1

    for line in lines:
        stripped = line.strip()
        if "📋" in stripped and ("ACTION PLAN" in stripped.upper() or "AAP KA" in stripped.upper()):
            in_action = True
            number = 1
            result.append(line)
            continue
        if in_action and stripped and (stripped.startswith("#") or any(
            emoji in stripped for emoji in ["🌾", "🔬", "💧", "💰", "📞"]
        )):
            in_action = False

        if in_action and stripped:
            if re.match(r"^\d+[\.\)]", stripped):
                result.append(f"**{stripped}**")
            elif not stripped[0].isdigit():
                if any(
                    kw in stripped.lower()
                    for kw in ["kal", "today", "aaj", "subah", "shaam", "is hafte", "this week"]
                ):
                    result.append(f"**{number}. {stripped.strip('- ')}**")
                    number += 1
                else:
                    result.append(line)
            else:
                result.append(line)
        else:
            result.append(line)

    return "\n".join(result)


def _ensure_mobile_friendly(text: str) -> str:
    lines = text.split("\n")
    result = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("---"):
            result.append(line)
            continue
        if len(stripped) > 120 and not stripped.startswith("#") and not stripped.startswith("*"):
            midpoint = stripped.rfind("। ", 0, 120)
            if midpoint == -1:
                midpoint = stripped.rfind(". ", 0, 120)
            if midpoint == -1:
                midpoint = stripped.rfind(", ", 0, 120)
            if midpoint > 40:
                result.append(stripped[: midpoint + 1])
                result.append(stripped[midpoint + 1 :].strip())
                continue
        result.append(line)

    return "\n".join(result)


def _add_missing_sections(text: str, agents_called: list) -> str:
    has_crop_situation = "🌾" in text or "fasal" in text.lower()
    has_action_plan = "📋" in text or "action plan" in text.lower()
    has_helpline = "0800-15000" in text

    if not has_crop_situation:
        prefix = "🌾 **Aapki Fasal Ki Situation**\n\n"
        idx = text.find("🔬")
        if idx == -1:
            idx = text.find("💧")
        if idx == -1:
            idx = text.find("💰")
        if idx > 0:
            text = text[:idx] + prefix + text[idx:]
        elif not text.strip().startswith("🌾"):
            text = prefix + text

    if not has_action_plan and agents_called:
        text += "\n\n📋 **Aapka Action Plan**\n"
        text += "1. Upar di gayi advice ko follow karein\n"
        text += "2. Apni fasal ki rozana monitoring karein\n"
        text += "3. Kisi bhi badlav par local expert se raabta karein\n"

    if not has_helpline:
        text += "\n\n---\n📞 **Zaroorat Pari To**\n"
        text += "📞 Agriculture Helpline: 0800-15000 (Free)\n"
        text += "ℹ️ Local Agriculture Extension Officer se bhi raabta kar sakte hain\n"

    return text

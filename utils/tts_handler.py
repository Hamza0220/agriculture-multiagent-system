import base64
import io
import os
import tempfile
from gtts import gTTS


def text_to_speech(text: str, lang: str = "ur") -> str:
    """
    Convert text to speech audio and return base64 encoded audio data.

    Args:
        text: Text to speak (Roman Urdu or English)
        lang: Language code — 'ur' for Urdu, 'en' for English

    Returns:
        Base64-encoded audio data string, or empty string on failure
    """
    try:
        clean_text = _clean_for_speech(text)

        if not clean_text:
            return ""

        tts_lang = _detect_tts_language(clean_text, lang)

        tts = gTTS(text=clean_text, lang=tts_lang, slow=False)

        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
        return audio_base64

    except Exception:
        return ""


def get_audio_html(audio_base64: str) -> str:
    """Generate HTML audio player tag with auto-generated audio."""
    if not audio_base64:
        return ""

    return f"""
    <audio controls style="width:100%; margin-top:8px;">
        <source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg">
        Your browser does not support audio.
    </audio>
    """


def _clean_for_speech(text: str) -> str:
    """Remove emojis and markdown that gTTS can't pronounce well."""
    import re

    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE,
    )
    clean = emoji_pattern.sub("", text)
    clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
    clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
    clean = re.sub(r"__([^_]+)__", r"\1", clean)
    clean = re.sub(r"`([^`]+)`", r"\1", clean)
    clean = re.sub(r"#+ ", "", clean)
    clean = re.sub(r"[-*] ", "", clean)
    clean = re.sub(r"---", "", clean)
    clean = re.sub(r"📞|ℹ️|🌾|🔬|💧|💰|📋|⚠️|🟡|🔴|🟢|👨‍🌾|🌤️|📍", "", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    clean = clean.strip()

    return clean


def _detect_tts_language(text: str, preferred_lang: str) -> str:
    """Determine best gTTS language for the text."""
    if preferred_lang in ("en", "english"):
        return "en"

    return "ur"

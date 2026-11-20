"""Multilingual language detection, cross-lingual processing, and prompt formatting for Visual RAG Studio."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Supported languages in Visual RAG Studio
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "hinglish": "Hinglish (Hindi in Roman script)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
}

# Standard English abstention message
DEFAULT_ABSTENTION = "I could not find enough retrieved evidence to answer that question."

# Localized abstention messages per language
LOCALIZED_ABSTENTIONS: dict[str, str] = {
    "en": DEFAULT_ABSTENTION,
    "hi": "मुझे इस प्रश्न का उत्तर देने के लिए पर्याप्त साक्ष्य नहीं मिले।",
    "hinglish": "Mujhe is prashn ka uttar dene ke liye paryapt evidence nahi mila.",
    "es": "No pude encontrar suficiente evidencia para responder a esa pregunta.",
    "fr": "Je n'ai pas trouvé suffisamment de preuves pour répondre à cette question.",
    "de": "Ich konnte nicht genügend Belege finden, um diese Frage zu beantworten.",
}

# Common Hinglish marker words (case-insensitive, avoiding English stopword collisions)
HINGLISH_MARKERS = {
    "kya", "kaise", "kaisa", "kaisi", "kyu", "kyon", "kahan", "kidhar", "kab", "kaun",
    "kitna", "kitne", "kitni", "karta", "karti", "karte", "karna", "kare", "karenge",
    "kiya", "kiye", "hai", "hain", "hoon", "hun", "tha", "thi",
    "mein", "se", "ke", "ki", "ko", "ka", "par", "pe", "nahi", "nahin", "mat",
    "aur", "bhi", "hum", "hume", "aap", "aapko", "tum", "tumko", "ye", "yeh",
    "vo", "woh", "unhe", "unko", "inka", "unka", "chahiye", "bataye", "bataiye",
    "samjhaiye", "hota", "hoti", "hote", "baare", "iske", "uske", "kuch",
    "baat", "kaam", "dastavej", "jankari", "bheja", "gaya", "shamil",
}

# Common English markers to disambiguate Latin-script queries
ENGLISH_MARKERS = {
    "the", "is", "are", "was", "were", "what", "which", "where", "when", "why", "how",
    "who", "whom", "this", "that", "these", "those", "have", "has", "had", "with",
    "from", "about", "into", "through", "city", "year", "planet", "capital",
}

# Devanagari Unicode range: \u0900 - \u097F
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")


def detect_language(text: str) -> str:
    """Detect the language of text with zero external dependencies.

    Returns:
        'hi' for Devanagari script (Hindi)
        'hinglish' for Hindi written in Roman/Latin script
        'es', 'fr', 'de' for European Latin languages with clear markers
        'en' default for English / Latin script
    """
    if not text or not text.strip():
        return "en"

    clean_text = unicodedata.normalize("NFC", text.strip())

    # Check for Devanagari characters
    devanagari_chars = len(DEVANAGARI_REGEX.findall(clean_text))
    alpha_chars = sum(1 for c in clean_text if c.isalpha())

    if alpha_chars > 0 and (devanagari_chars / alpha_chars) >= 0.25:
        return "hi"

    # Analyze Latin-script words
    words = [w.lower() for w in re.findall(r"\b[^\W\d_]+\b", clean_text, flags=re.UNICODE)]
    if not words:
        return "en"

    # Check for Hinglish markers vs English markers
    hinglish_count = sum(1 for w in words if w in HINGLISH_MARKERS)
    english_count = sum(1 for w in words if w in ENGLISH_MARKERS)

    if hinglish_count > 0 and hinglish_count > english_count:
        if len(words) <= 4:
            return "hinglish"
        if hinglish_count >= 2 or (hinglish_count / len(words)) >= 0.15:
            return "hinglish"

    lower_text = clean_text.lower()

    # Unique Spanish punctuation & characters
    if any(c in lower_text for c in ("¿", "¡", "ñ")):
        return "es"

    # Unique German characters (ä, ö, ü, ß)
    if any(c in lower_text for c in ("ä", "ö", "ü", "ß")):
        return "de"

    # Unique French characters (œ, ç, è, ê, ë, à, â, ù, û)
    if any(c in lower_text for c in ("œ", "ç", "è", "ê", "ë", "à", "â", "ù", "û")):
        return "fr"

    # Word-level score comparisons
    german_markers = {"der", "die", "das", "und", "in", "den", "von", "zu", "mit", "sich", "des", "auf", "ist", "nicht", "wie", "für", "ein", "eine"}
    french_markers = {"le", "la", "les", "des", "dans", "avec", "pour", "sur", "qui", "que", "c'est", "est", "comment", "une", "un"}
    spanish_markers = {"el", "la", "los", "las", "un", "una", "por", "para", "con", "como", "qué", "del", "al"}

    scores = {
        "es": sum(1 for w in words if w in spanish_markers) + (1 if any(c in lower_text for c in ("á", "í", "ó", "ú")) else 0),
        "fr": sum(1 for w in words if w in french_markers),
        "de": sum(1 for w in words if w in german_markers),
    }
    best_lang, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score >= 2:
        return best_lang

    return "en"


def get_multilingual_abstention(language: str) -> str:
    """Return the canonical abstention message for the specified language."""
    return LOCALIZED_ABSTENTIONS.get(language, DEFAULT_ABSTENTION)


def is_multilingual_abstention(text: str) -> bool:
    """Check if generated text represents an explicit abstention across languages."""
    if not text or not text.strip():
        return True

    lower = text.lower().strip()

    # English phrases
    english_phrases = (
        "could not find enough retrieved evidence",
        "cannot answer that question",
        "not enough evidence to answer",
        "insufficient evidence to answer",
        "i do not have enough information",
        "none of the provided sources contain",
        "provided evidence does not contain",
        "the provided text does not contain",
        "the provided document does not contain",
        "the provided documents do not contain",
        "provided documents do not contain",
        "provided document does not contain",
        "do not contain information",
        "does not contain information",
        "do not contain enough",
        "does not contain enough",
        "no evidence found to answer",
        "i don't have relevant info",
        "i do not have relevant info",
    )
    if any(p in lower for p in english_phrases) and len(lower) < 350:
        return True

    # Hindi phrases
    hindi_phrases = (
        "पर्याप्त साक्ष्य नहीं मिले",
        "पर्याप्त जानकारी नहीं मिली",
        "उत्तर देने के लिए पर्याप्त",
        "दिए गए साक्ष्य में जानकारी नहीं है",
        "दस्तावेज़ में जानकारी नहीं है",
        "कोई साक्ष्य नहीं मिला",
        "जानकारी उपलब्ध नहीं है",
        "उत्तर उपलब्ध नहीं है",
    )
    if any(p in lower for p in hindi_phrases) and len(lower) < 350:
        return True

    # Hinglish phrases
    hinglish_phrases = (
        "paryapt evidence nahi mila",
        "paryapt jankari nahi mili",
        "koi evidence nahi mila",
        "document me information nahi hai",
        "evidence me jankari nahi hai",
        "utter dene ke liye paryapt",
    )
    if any(p in lower for p in hinglish_phrases) and len(lower) < 350:
        return True

    # Exact matches against any localized abstention
    for abstention in LOCALIZED_ABSTENTIONS.values():
        if lower == abstention.lower() or abstention.lower() in lower:
            return True

    return False


def build_multilingual_instructions(
    response_language_mode: str,
    query_text: str,
    detected_query_lang: str | None = None,
) -> tuple[str, str]:
    """Generate system instructions and determine effective response language.

    Args:
        response_language_mode: Configured mode ('query', 'source', 'en', 'hi', 'es', 'fr', 'de', 'auto').
        query_text: The user question text.
        detected_query_lang: Pre-detected query language or None.

    Returns:
        (instruction_prompt_snippet, effective_language_code)
    """
    detected_lang = detected_query_lang or detect_language(query_text)

    # Resolve target language
    if response_language_mode in ("query", "auto"):
        target_lang = detected_lang
    elif response_language_mode in SUPPORTED_LANGUAGES:
        target_lang = response_language_mode
    else:
        target_lang = "en"

    if target_lang == "hi":
        instructions = (
            "Answer the user's question directly and thoroughly in Hindi (हिंदी). "
            "Base every statement strictly on the provided retrieved evidence passages. "
            "Cite all facts using evidence brackets (e.g. [1], [2], [chunk_id]). "
            "Keep technical terms, identifiers, and original citations intact. "
            f"If the provided evidence does not contain sufficient information, state: '{LOCALIZED_ABSTENTIONS['hi']}'."
        )
    elif target_lang == "hinglish":
        instructions = (
            "Answer the user's question directly in Hinglish (Hindi written in Roman script) or natural conversational Hindi/English. "
            "Base every statement strictly on the provided retrieved evidence passages. "
            "Cite all facts using evidence brackets (e.g. [1], [2], [chunk_id]). "
            "Keep technical terms and names clear and accurate. "
            f"If the provided evidence does not contain sufficient information, state: '{LOCALIZED_ABSTENTIONS['hinglish']}'."
        )
    elif target_lang == "es":
        instructions = (
            "Answer the user's question directly and thoroughly in Spanish (Español). "
            "Base every statement strictly on the provided retrieved evidence passages. "
            "Cite all facts using evidence brackets (e.g. [1], [2], [chunk_id]). "
            f"If the provided evidence does not contain sufficient information, state: '{LOCALIZED_ABSTENTIONS['es']}'."
        )
    elif target_lang == "fr":
        instructions = (
            "Answer the user's question directly and thoroughly in French (Français). "
            "Base every statement strictly on the provided retrieved evidence passages. "
            "Cite all facts using evidence brackets (e.g. [1], [2], [chunk_id]). "
            f"If the provided evidence does not contain sufficient information, state: '{LOCALIZED_ABSTENTIONS['fr']}'."
        )
    elif target_lang == "de":
        instructions = (
            "Answer the user's question directly and thoroughly in German (Deutsch). "
            "Base every statement strictly on the provided retrieved evidence passages. "
            "Cite all facts using evidence brackets (e.g. [1], [2], [chunk_id]). "
            f"If the provided evidence does not contain sufficient information, state: '{LOCALIZED_ABSTENTIONS['de']}'."
        )
    else:  # default 'en'
        instructions = (
            "Answer the user's question directly, factually, and thoroughly in English. "
            "Base every statement strictly on the provided retrieved evidence passages. "
            "Cite all facts using evidence brackets (e.g. [1], [2], [chunk_id]). "
            f"If the provided evidence does not contain sufficient information, state: '{DEFAULT_ABSTENTION}'."
        )

    return instructions, target_lang

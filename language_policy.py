"""
Central output-language policy for the whole Mojo agent.

Rules (product-wide, every reply / tool / future feature):
  1. Default → Roman Urdu (Latin letters). Never default to Devanagari/Arabic script.
  2. Explicit English ask → English.
  3. Explicit Urdu/Hindi *script* ask → Urdu (Arabic) or Hindi (Devanagari) letters.
  4. "urdu me" / "hindi me" without "script/letters/likhai" → Roman Urdu, not native script.

This is intentionally deterministic (no LangChain graph). A shared pure module
is lower latency and easier to apply at every boundary (prompts, tools, ASR).
"""

from __future__ import annotations

import re
from typing import Literal, Optional

OutputLang = Literal["en", "roman_urdu", "urdu_script", "hindi_script"]

# Explicit English
_EN_PHRASES = frozenset({
    "in english",
    "english me",
    "english mein",
    "english mien",
    "angrezi me",
    "angrezi mein",
    "english me likho",
    "english mein likho",
    "english me translate",
    "translate to english",
    "translate in english",
    "word to word english",
    "english word to word",
    "english me refine",
    "english mein refine",
    "reply in english",
    "answer in english",
    "respond in english",
})

# Explicit native-script asks (rare)
_URDU_SCRIPT_PHRASES = frozenset({
    "urdu script",
    "urdu letters",
    "urdu likhai",
    "urdu main likho",
    "urdu mein likho",
    "arabic script",
    "nastaliq",
    "اردو میں لکھو",
    "اردو لکھاو",
})

_HINDI_SCRIPT_PHRASES = frozenset({
    "hindi script",
    "hindi letters",
    "hindi likhai",
    "devanagari",
    "देवनागरी",
    "हिंदी में लिखो",
    "हिन्दी में लिखो",
})

# Roman Urdu / Hindi (spoken language, Latin letters)
_ROMAN_URDU_PHRASES = frozenset({
    "roman urdu",
    "roman hindi",
    "urdu me",
    "urdu mein",
    "hindi me",
    "hindi mein",
    "urdu me batao",
    "hindi me batao",
    "urdu me likho",
    "hindi me likho",  # without "script" → Roman, not Devanagari
})


def _norm(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"https?://\s*\S+", " ", t)
    t = re.sub(r"@\d[\d\s]*", " ", t)
    t = re.sub(r"\[quoted message\]:.*", " ", t, flags=re.I | re.S)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def detect_output_lang(user_text: str, default: OutputLang = "roman_urdu") -> OutputLang:
    """
    Infer desired *output* language/script from the user message.
    Default is always Roman Urdu unless explicitly overridden.
    """
    t = _norm(user_text)
    if not t:
        return default

    # Native scripts first (most specific)
    if any(p in t for p in _URDU_SCRIPT_PHRASES):
        return "urdu_script"
    if any(p in t for p in _HINDI_SCRIPT_PHRASES):
        return "hindi_script"

    # English
    if any(p in t for p in _EN_PHRASES):
        return "en"
    # bare token checks
    if re.search(r"\b(english|angrezi)\b", t) and re.search(
        r"\b(me|mein|mien|likho|translate|reply|answer|refine|word)\b", t
    ):
        return "en"

    # Roman Urdu / Hindi (Latin)
    if any(p in t for p in _ROMAN_URDU_PHRASES):
        return "roman_urdu"

    return default


def to_tool_language_arg(lang: OutputLang) -> str:
    """Map policy lang → transcribe_video `language` arg (fetch stays auto)."""
    if lang == "en":
        return "en"
    if lang == "urdu_script":
        return "ur"
    if lang == "hindi_script":
        return "hi"
    return "auto"  # roman_urdu / default → fetch auto, render Roman


def script_rule_for_llm(lang: OutputLang) -> str:
    """One block to inject into system prompts."""
    if lang == "en":
        return (
            "OUTPUT LANGUAGE: English only. Latin letters only. "
            "Translate any Urdu/Hindi speech into clear English."
        )
    if lang == "urdu_script":
        return (
            "OUTPUT SCRIPT: Urdu (Arabic/Nastaliq letters). "
            "User explicitly asked for Urdu script."
        )
    if lang == "hindi_script":
        return (
            "OUTPUT SCRIPT: Hindi Devanagari letters. "
            "User explicitly asked for Hindi script/writing."
        )
    return (
        "OUTPUT SCRIPT: Roman Urdu only (Latin letters). "
        "Transliterate any Devanagari/Arabic. Never output Urdu or Hindi script "
        "unless the user explicitly asked for script/letters/likhai."
    )


def has_non_latin_script(text: str, sample: int = 4000) -> bool:
    for ch in (text or "")[:sample]:
        o = ord(ch)
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x0900 <= o <= 0x097F:
            return True
    return False


def non_latin_ratio(text: str, sample: int = 8000) -> float:
    if not text:
        return 0.0
    s = text[:sample]
    letters = 0
    non_lat = 0
    for ch in s:
        if ch.isalpha() or (0x0900 <= ord(ch) <= 0x097F) or (0x0600 <= ord(ch) <= 0x06FF):
            letters += 1
            o = ord(ch)
            if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x0900 <= o <= 0x097F:
                non_lat += 1
    if letters < 20:
        return 0.0
    return non_lat / letters


def policy_blurb() -> str:
    """Short block for the main agent system prompt."""
    return (
        "LANGUAGE POLICY (always):\n"
        "- Default: Roman Urdu (Latin letters only).\n"
        "- Explicit English ask (in english / english me / angrezi) → English.\n"
        "- Explicit Urdu/Hindi *script* ask (urdu script, hindi letters, देवनागरी) "
        "→ that script only.\n"
        "- 'urdu me' / 'hindi me' without script/letters → Roman Urdu, NOT native script.\n"
        "- Never reply in Devanagari or Arabic script unless the user clearly asked for it."
    )

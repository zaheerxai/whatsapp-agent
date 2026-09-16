"""
Central output-language policy for the whole Mojo agent.

Rules (product-wide, every reply / tool / future feature):
  1. Match the user's query language when clear (English query → English reply).
  2. Default → Roman Urdu (Latin letters). Never default to Devanagari/Arabic script.
  3. Explicit English ask → English.
  4. Explicit Urdu/Hindi *script* ask → Urdu (Arabic) or Hindi (Devanagari) letters.
  5. "urdu me" / "hindi me" without "script/letters/likhai" → Roman Urdu, not native script.

Deterministic pure module (no LangChain graph) — low latency, applies everywhere.
"""

from __future__ import annotations

import re
from typing import Literal

OutputLang = Literal["en", "roman_urdu", "urdu_script", "hindi_script"]

# Explicit English switch phrases
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
    "hindi me likho",
})

# Tokens that strongly mark Roman Urdu / mixed Indo-Pak chat
_ROMAN_URDU_TOKENS = frozenset({
    "hai", "hain", "tha", "thi", "kya", "kyun", "kyu", "kaise", "kaisa",
    "kaisi", "kese", "yeh", "woh", "mein", "main", "hum",
    "aap", "tum", "bhai", "yaar", "nahi", "nahin", "haan",
    "karo", "karna", "dena", "doon", "batao", "batain",
    "likho", "samjhao", "samajh", "dekho", "suno",
    "chahiye", "chahye", "chahta", "chahti", "wala", "wali", "wale",
    "kiya", "kiye", "hua", "hui", "hoga", "hogi", "honge",
    "abhi", "phir", "lekin", "magar", "kyunki", "isliye",
    "paani", "khana", "kaam", "ghanta",
    "isme", "iski", "usme", "uski", "mujhe", "tumhe", "apko",
    "achha", "acha", "theek", "thik", "bilkul", "zaroor",
    "bhejo", "bhejna", "milte", "milna", "rakhna", "rakho",
    "karlo", "bata", "likh", "dekh",
})

# Common English function / content words (Latin queries)
_EN_TOKENS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "for", "on", "in", "at", "by", "with", "from",
    "this", "that", "these", "those", "it", "its", "as", "if", "then",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "can", "could", "will", "would", "should", "shall", "may", "might",
    "do", "does", "did", "done", "have", "has", "had",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their",
    "not", "no", "yes", "please", "thanks", "thank",
    "tell", "give", "show", "explain", "summarize", "summary", "translate",
    "transcript", "transcribe", "about", "video", "full", "detail", "detailed",
    "word", "reply", "answer", "message", "send", "list", "cancel", "set",
    "need", "want", "get", "make", "help", "check", "find", "search",
})


def _norm(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"https?://\s*\S+", " ", t)
    t = re.sub(r"@\d[\d\s]*", " ", t)
    t = re.sub(r"\[quoted message\]:.*", " ", t, flags=re.I | re.S)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokens(t: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", t)


def _looks_like_english_query(t: str) -> bool:
    """
    True when the user's message is primarily English (not Roman Urdu mix).
    Used so English queries get English replies without needing "in english".
    """
    toks = _tokens(t)
    if len(toks) < 2:
        return False
    ru_hits = sum(1 for x in toks if x in _ROMAN_URDU_TOKENS)
    en_hits = sum(1 for x in toks if x in _EN_TOKENS)
    if ru_hits >= 2:
        return False
    if ru_hits >= 1 and en_hits < ru_hits + 2:
        return False
    if en_hits >= 2 and ru_hits == 0:
        return True
    if len(toks) >= 4 and en_hits >= 3 and ru_hits <= 1:
        return True
    if len(toks) >= 3 and ru_hits == 0 and en_hits >= 1:
        return en_hits >= max(1, len(toks) // 3)
    # Majority English tokens even if a few unknown words
    if len(toks) >= 4 and en_hits >= (len(toks) + 1) // 2 and ru_hits == 0:
        return True
    return False
    # Strong Roman-Urdu markers → not pure English
    ru_hits = sum(1 for x in toks if x in _ROMAN_URDU_TOKENS)
    en_hits = sum(1 for x in toks if x in _EN_TOKENS)
    # Classic Roman-Urdu particles alone are enough
    if ru_hits >= 2:
        return False
    if ru_hits >= 1 and en_hits < 3:
        return False
    # Enough English function words and few RU markers
    if en_hits >= 2 and ru_hits == 0:
        return True
    if len(toks) >= 4 and en_hits >= 3 and ru_hits <= 1:
        return True
    # Short all-English phrases e.g. "give full transcript", "what is this"
    if len(toks) >= 3 and ru_hits == 0 and en_hits >= 1:
        # Require majority of tokens to be plain English letters words
        # already true if no RU tokens; avoid treating pure gibberish as EN
        return en_hits >= max(1, len(toks) // 3)
    return False


def detect_output_lang(user_text: str, default: OutputLang = "roman_urdu") -> OutputLang:
    """
    Infer desired *output* language/script from the user message.

    Priority:
      1. Explicit native-script ask
      2. Explicit English ask OR English-dominant query
      3. Explicit Roman Urdu / "urdu me" / "hindi me"
      4. Default Roman Urdu
    """
    t = _norm(user_text)
    if not t:
        return default

    # Native scripts first (most specific)
    if any(p in t for p in _URDU_SCRIPT_PHRASES):
        return "urdu_script"
    if any(p in t for p in _HINDI_SCRIPT_PHRASES):
        return "hindi_script"

    # Explicit English switch phrases
    if any(p in t for p in _EN_PHRASES):
        return "en"
    if re.search(r"\b(english|angrezi)\b", t) and re.search(
        r"\b(me|mein|mien|likho|translate|reply|answer|refine|word)\b", t
    ):
        return "en"

    # Explicit Roman Urdu preference
    if any(p in t for p in _ROMAN_URDU_PHRASES):
        return "roman_urdu"

    # Match query language: pure/mostly English message → English reply
    if _looks_like_english_query(t):
        return "en"

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
        "- Match the user's query language: English message → English reply; "
        "Roman Urdu message → Roman Urdu reply.\n"
        "- Default when mixed/unclear: Roman Urdu (Latin letters only).\n"
        "- Explicit English ask → English.\n"
        "- Explicit Urdu/Hindi *script* ask → that script only.\n"
        "- 'urdu me' / 'hindi me' without script/letters → Roman Urdu, NOT native script.\n"
        "- Never reply in Devanagari or Arabic script unless the user clearly asked for it."
    )

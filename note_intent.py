"""Shared note / image-describe intent detection.

Used by whatsapp_agent.handle_media_message and agent_loop.run_agent so phrase
lists cannot drift and reintroduce fake "noted" replies without note_down.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

# Whole-word / phrase forms for "save this to journal"
_NOTE_INTENT_RE = re.compile(
    r"(?<!\w)("
    r"note(?:\s+down|\s+kar(?:lo|o|do|dena)?|\s+this)?"
    r"|notes?"
    r"|save(?:\s+this|\s+kar(?:lo|o)?)"
    r"|onedrive\s*me\s*note"
    r"|one\s*drive\s*me\s*note"
    r"|journal"
    r"|likh\s+lo"
    r"|likh\s+do"
    r")(?!\w)",
    re.I,
)

_CONDENSED_NOTE_RE = re.compile(
    r"(?<!\w)("
    r"summary|summarize|summarise|khulasa|"
    r"key\s*points?|keypoints|bullets?|main\s+points?|"
    r"short\s+me|mukhtasir"
    r")(?!\w)",
    re.I,
)

# "ye photo kya hai" / describe — NOT journal
_DESCRIBE_INTENT_RE = re.compile(
    r"(?<!\w)("
    r"ye\s+photo\s+kya|"
    r"ye\s+image\s+kya|"
    r"ye\s+pic\s+kya|"
    r"photo\s+kya\s+hai|"
    r"image\s+kya\s+hai|"
    r"pic\s+kya\s+hai|"
    r"what(?:'s|\s+is)\s+(?:in\s+)?(?:this\s+)?(?:photo|image|pic|picture)|"
    r"describe\s+(?:this\s+)?(?:photo|image|pic)?|"
    r"is\s+(?:photo|image|pic)\s+me\s+kya|"
    r"isme\s+kya\s+(?:hai|dikha)|"
    r"kya\s+dikha\s+raha|"
    r"read\s+(?:this\s+)?(?:image|photo)|"
    r"ocr"
    r")(?!\w)",
    re.I,
)

_REACTION_WRAP_RE = re.compile(
    r"\[User is reacting to your previous message[^\]]*\]",
    re.I,
)


def strip_caption_noise(text: str) -> str:
    """Remove reaction wrappers / user-caption labels for intent matching."""
    t = text or ""
    t = _REACTION_WRAP_RE.sub(" ", t)
    t = re.sub(r"(?i)User caption:\s*", " ", t)
    t = re.sub(r"@\d[\d\s]*", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def intent_head(text: str) -> str:
    """User-side text only (before [Quoted Message] / media blocks)."""
    low = (text or "").lower()
    head = re.split(r"\[quoted message\]:", low, maxsplit=1, flags=re.I)[0]
    head = re.split(
        r"\[(?:document|image content|voice note transcript|"
        r"cached recent voice-note transcript)[^\]]*\]:",
        head,
        maxsplit=1,
        flags=re.I,
    )[0]
    return strip_caption_noise(head)


def is_note_intent(text: str) -> bool:
    """True when user wants content saved to the OneDrive journal."""
    head = intent_head(text)
    if not head:
        return False
    if re.fullmatch(r"noted[!?.\s]*", head, flags=re.I):
        return False
    return bool(_NOTE_INTENT_RE.search(head))


def wants_condensed_note(text: str) -> bool:
    """True when user asked for summary/key points instead of verbatim."""
    head = intent_head(text)
    return bool(_CONDENSED_NOTE_RE.search(head))


def is_describe_intent(text: str) -> bool:
    """True for 'ye photo kya hai' style asks (no journal write)."""
    if is_note_intent(text):
        return False
    head = intent_head(text)
    if not head:
        # Empty caption on image often means "what is this?"
        return True
    if _DESCRIBE_INTENT_RE.search(head):
        return True
    # Very short residual after strip → describe
    tokens = re.sub(r"[^\w\s]", "", head).split()
    greet = {
        "hi", "hello", "hey", "salam", "salaam", "ok", "okay", "thanks",
        "shukriya", "ji", "bro", "bhai",
    }
    if tokens and len(tokens) <= 4 and not all(t in greet for t in tokens):
        # "ye kya hai", "kya hai ye", "batao" on an image → describe
        if any(
            w in head
            for w in ("kya", "what", "batao", "batana", "dekho", "describe")
        ):
            return True
    return False


def content_fingerprint(body: str) -> str:
    """Short hash + length for tool_call history (no truncated body leakage)."""
    raw = (body or "").encode("utf-8", errors="replace")
    h = hashlib.sha256(raw).hexdigest()[:12]
    return f"sha256={h};chars={len(body or '')}"

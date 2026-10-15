"""
bot_voice.py — one place for what the bot SAYS (wording) and what it must never say (raw errors).

Languages: "en" (English) and "ur" (Roman Urdu, the way a Pakistani types on WhatsApp). Pick
with lang_of(user_text). Tone rules used across every string here:
  * warm, brief, one idea per line; no robotic "Operation completed successfully"
  * say what happened, then the next step ("Draft is safe — say *send* to try again")
  * light emoji only where it carries meaning (📧 ✅ ⚠️), never decorative spam
  * never expose exception text, URLs of APIs, status-code dumps
"""
from __future__ import annotations

import re
from typing import Dict

try:  # optional: same detector the rest of the bot uses
    import language_policy as _lp
except Exception:  # pragma: no cover
    _lp = None


def lang_of(text: str, default: str = "ur") -> str:
    """'en' if the user is writing English, else 'ur' (Roman Urdu)."""
    if not text:
        return default
    try:
        if _lp is not None:
            return "en" if _lp.detect_output_lang(text, default="roman_urdu") == "en" else "ur"
    except Exception:
        pass
    return default


# --------------------------------------------------------------------------- catalogue
_M: Dict[str, Dict[str, str]] = {
    # ---- email draft card
    "draft_title": {"en": "📧 *Email draft*", "ur": "📧 *Email draft*"},
    "draft_footer": {
        "en": "Say *send* to ship it — or tell me what to change (shorter, new subject, add a CC, different tone…).",
        "ur": "Bhejne ke liye *send* likho — ya jo badalna ho bata do (chhota karo, subject badlo, CC add karo, tone change…).",
    },
    "draft_footer_missing": {
        "en": "Tell me who to send it to and I'll take it from there.",
        "ur": "Bata do kisko bhejna hai, baqi main sambhal lunga.",
    },
    "more_words": {
        "en": "…(+{n} more words — the full text goes out in the email)",
        "ur": "…(+{n} aur words — poora text email mein jayega)",
    },
    "updated": {"en": "✏️ Updated: {what}", "ur": "✏️ Badal diya: {what}"},
    "what_to": {"en": "recipient", "ur": "recipient"},
    "what_cc": {"en": "CC", "ur": "CC"},
    "what_subject": {"en": "subject", "ur": "subject"},
    "what_body": {"en": "wording", "ur": "wording"},
    "what_attach": {"en": "attachment", "ur": "attachment"},
    "nothing_changed": {
        "en": "I couldn't tell what to change. Try: \"make it shorter\", \"change the subject to …\", \"add ali@x.com in CC\" or \"remove the resume\".",
        "ur": "Samajh nahi aaya kya badalna hai. Aise bolo: \"chhota karo\", \"subject … kar do\", \"ali@x.com ko CC mein add karo\" ya \"resume hata do\".",
    },
    # ---- recipient
    "ask_recipient": {
        "en": "Who should this go to? Send me {name}'s email address and I'll remember it for next time.",
        "ur": "Ye kisko bhejna hai? {name} ka email address bhej do — agli baar yaad rakhunga.",
    },
    "ask_recipient_anon": {
        "en": "Who should I send this to? Drop the email address and I'll prepare the rest.",
        "ur": "Kisko bhejna hai? Email address bhej do, baqi main taiyaar kar deta hun.",
    },
    "ask_which": {
        "en": "I know more than one \"{name}\": {options}. Which one?",
        "ur": "Mere paas \"{name}\" naam ke kai log hain: {options}. Kaunsa?",
    },
    "bad_email": {
        "en": "⚠️ \"{addr}\" doesn't look like a valid email address. Mind checking it?",
        "ur": "⚠️ \"{addr}\" sahi email address nahi lag raha. Ek dafa check kar lo?",
    },
    "typo_email": {
        "en": "⚠️ \"{addr}\" looks like a typo — did you mean *{fix}*? Say *use {fix}* to switch, or *send* again to keep it as is.",
        "ur": "⚠️ \"{addr}\" mein typo lag rahi hai — kya *{fix}* hona chahiye? Badalne ke liye *use {fix}* likho, ya jaisa hai waisa bhejna ho to dobara *send* likho.",
    },
    # ---- send
    "confirm_send": {
        "en": "Send this to *{to}* now? Reply *send* to confirm, or tell me what to change.",
        "ur": "*{to}* ko abhi bhej doon? Confirm karne ke liye *send* likho, ya jo badalna ho bata do.",
    },
    "sent_ok": {"en": "✅ *Sent* to {to}", "ur": "✅ *Bhej diya* — {to} ko"},
    "sent_attach": {"en": "📎 {name} attached", "ur": "📎 {name} laga hua hai"},
    "already_sent": {
        "en": "That one already went out at {when} to {to}. Want to write a new email?",
        "ur": "Ye email {when} par {to} ko pehle hi ja chuki hai. Nayi email likhni hai?",
    },
    "no_draft": {
        "en": "There's no draft right now. Tell me who to write to and what to say — or quote a message and say \"email this to …\".",
        "ur": "Abhi koi draft nahi hai. Bata do kisko kya likhna hai — ya koi message quote karke \"ye … ko email kar do\" bol do.",
    },
    "expired": {
        "en": "That draft is more than a day old, so I dropped it. Want me to write it again?",
        "ur": "Wo draft ek din se purana tha isliye hata diya. Dobara likh doon?",
    },
    "cancelled": {
        "en": "Draft cancelled. Say *undo* within a few minutes if that was a mistake.",
        "ur": "Draft cancel kar diya. Ghalti se hua ho to kuch minute mein *undo* likh do.",
    },
    "undone": {"en": "↩️ Back to the previous version.", "ur": "↩️ Pichla version wapas laga diya."},
    "nothing_undo": {"en": "Nothing to undo.", "ur": "Undo karne ke liye kuch nahi hai."},
    "resume_missing": {
        "en": "⚠️ I couldn't find your resume file ({why}), so I removed the attachment. Say *send* to go without it, or fix the resume path first.",
        "ur": "⚠️ Resume file nahi mili ({why}), isliye attachment hata diya. Bina attachment ke bhejna ho to *send* likho, ya pehle resume path theek kar lo.",
    },
    "not_configured": {
        "en": "Gmail isn't connected yet, so I can prepare drafts but not send them. The draft is saved.",
        "ur": "Gmail abhi connect nahi hai, isliye draft ban sakta hai magar send nahi. Draft save hai.",
    },
    "auth_expired": {
        "en": "Gmail's login has expired (refresh token). A new token is needed before I can send. Your draft is safe.",
        "ur": "Gmail ka login expire ho gaya (refresh token). Bhejne se pehle naya token chahiye. Draft safe hai.",
    },
    "send_scope": {
        "en": "Gmail refused the send — the token is missing the send permission. Your draft is safe.",
        "ur": "Gmail ne send reject kar diya — token mein send ki permission nahi hai. Draft safe hai.",
    },
    "send_bad_to": {
        "en": "Gmail rejected the recipient address. Check it and say *send* again — the draft is safe.",
        "ur": "Gmail ne recipient address reject kar diya. Check karke dobara *send* likho — draft safe hai.",
    },
    "send_quota": {
        "en": "Gmail's sending limit was hit. Try again in a bit — the draft is safe.",
        "ur": "Gmail ki sending limit lag gayi. Thori der baad try karo — draft safe hai.",
    },
    "send_big": {
        "en": "The email (or its attachment) is too large for Gmail. Remove the attachment or shorten it — the draft is safe.",
        "ur": "Email (ya attachment) Gmail ke liye bohat bari hai. Attachment hata do ya chhota karo — draft safe hai.",
    },
    "send_busy": {
        "en": "Gmail didn't respond properly. Say *send* again in a minute — the draft is safe.",
        "ur": "Gmail ne theek jawab nahi diya. Ek minute baad dobara *send* likho — draft safe hai.",
    },
    "send_failed": {
        "en": "I couldn't send it ({why}). The draft is safe — say *send* to retry.",
        "ur": "Send nahi ho saka ({why}). Draft safe hai — dobara try ke liye *send* likho.",
    },
    "owner_only": {
        "en": "Email is only available to the owner.",
        "ur": "Email sirf owner ke liye available hai.",
    },
    "ai_busy": {
        "en": "My writing brain is busy right now — send that again in a minute and I'll pick it up. Your draft is untouched.",
        "ur": "Abhi AI busy hai — ek minute baad ye dobara bhej do. Aapka draft jaisa tha waisa hai.",
    },
    "empty_forward": {
        "en": "I don't see any text to send. Quote the message you want emailed and say \"email this to …\".",
        "ur": "Bhejne ke liye koi text nahi dikha. Jo message bhejna hai use quote karke \"ye … ko email kar do\" likho.",
    },
    "empty_body": {
        "en": "The email body is empty — tell me what it should say.",
        "ur": "Email ki body khali hai — bata do kya likhna hai.",
    },
    # ---- voice notes
    "voice_off": {"en": "🎤 Voice notes are turned off for this chat.", "ur": "🎤 Is chat mein voice notes band hain."},
    "voice_too_long": {
        "en": "🎤 That voice note is {mins} min long — my limit is {max} min. Could you send it in shorter parts?",
        "ur": "🎤 Ye voice note {mins} minute ki hai — meri limit {max} minute hai. Chhote hisson mein bhej do?",
    },
    "voice_failed": {
        "en": "🎤 I couldn't process that voice note. Please resend it or type it out.",
        "ur": "🎤 Ye voice note process nahi ho saki. Dobara bhej do ya type kar do.",
    },
    "voice_empty": {
        "en": "🎤 I couldn't hear anything in that voice note — mind sending it again?",
        "ur": "🎤 Is voice note mein kuch sunai nahi diya — dobara bhej do?",
    },
    # ---- links / video
    "ig_blocked": {
        "en": "I couldn't read that Instagram link — it's private or behind a login. Send a public reel, or share the video itself.",
        "ur": "Ye Instagram link read nahi ho saka — private hai ya login maang raha hai. Public reel bhej do, ya video hi share kar do.",
    },
    "fb_blocked": {
        "en": "I couldn't read that Facebook link — it may be private. A public post should work.",
        "ur": "Ye Facebook link read nahi ho saka — shayad private hai. Public post chal jayegi.",
    },
    "link_failed": {
        "en": "I couldn't pull a title or caption from that link. Is it a public URL?",
        "ur": "Is link ka title/caption nahi nikal saka. Kya ye public URL hai?",
    },
    "video_no_content": {
        "en": "I couldn't get any audio or caption from that video. Try a public link.",
        "ur": "Is video ka audio ya caption nahi mil saka. Public link try karo.",
    },
    # ---- research
    "research_need_question": {
        "en": "What should I research? Tell me the topic plus any limits (budget, hardware, region…).",
        "ur": "Kya research karna hai? Topic ke saath limits bhi bata do (budget, hardware, region…).",
    },
    "research_no_results": {
        "en": "The search engines didn't return anything usable for that. Try again in a bit, or make the question a little more specific.",
        "ur": "Search engines ne is par kuch kaam ka nahi diya. Thori der baad try karo, ya sawal thora specific kar do.",
    },
    # ---- generic
    "oops": {
        "en": "Something hiccuped on my side — give me a moment and try again 🙏",
        "ur": "Thora sa masla aa gaya — ek minute baad dobara try karo 🙏",
    },
}


def say(key: str, lang: str = "en", **kw) -> str:
    entry = _M.get(key) or {}
    txt = entry.get(lang) or entry.get("en") or key
    try:
        return txt.format(**kw)
    except (KeyError, IndexError):
        return txt


# --------------------------------------------------------------------------- raw-error scrubber
_RAW_ERROR_RES = [
    re.compile(r"\b\d{3}\s+(?:Client|Server)\s+Error\b", re.I),
    re.compile(r"\bfor url:\s*https?://", re.I),
    re.compile(r"HTTPS?ConnectionPool|Max retries exceeded|NameResolutionError|ConnectTimeoutError", re.I),
    re.compile(r"^\s*Traceback \(most recent call last\)", re.I),
    re.compile(r"requests\.exceptions\.\w+", re.I),
]


def scrub_errors(text: str, lang: str = "en") -> str:
    """Remove raw exception debris from a user-bound reply (keeps any good text around it).
    If nothing meaningful is left, return a friendly one-liner. Code fences are left alone."""
    if not text or not isinstance(text, str):
        return text or ""
    if not any(r.search(text) for r in _RAW_ERROR_RES):
        return text
    kept, in_code, dropped = [], False, False
    for ln in text.split("\n"):
        if ln.strip().startswith("```"):
            in_code = not in_code
            kept.append(ln)
            continue
        if not in_code and any(r.search(ln) for r in _RAW_ERROR_RES):
            dropped = True
            continue
        kept.append(ln)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    if dropped and len(re.sub(r"\W+", "", cleaned)) < 12:
        return say("oops", lang)
    return cleaned

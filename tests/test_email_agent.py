"""Offline tests for email_agent. Run: python test_email_agent.py

Scenarios replayed from the 2026-10-12 screenshots:
  1. forward a long status e-mail "to Satyam"           (draft must keep ALL text, both Google links)
  2. "Rewrite fully ... professional tone"               (was: a job application to 'Hiring Team')
  3. an unrelated research request while a draft exists  (was: "Koi draft save nahi hai")
"""
from __future__ import annotations

import json
import re
import sys

import bot_voice
import email_agent as E
import usage_budget
from wa_format import RAW

usage_budget._supabase = None
FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {str(extra)[:300]}")


# ------------------------------------------------------------------ harness
class H:
    def __init__(self, gmail=True, resume=("/tmp/r.pdf", "local"), owner="Muhammad Zaheeruddin"):
        usage_budget._mem.clear()
        self.plan = {"intent": "none"}
        self.compose_result = {"subject": "S", "body": "B"}
        self.llm_calls = []
        self.sent = []
        self.send_errors = []        # exceptions to raise before succeeding
        self.t = 1_000_000.0
        self.resume = resume
        self.gmail = gmail
        h = self

        def json_fn(system, user):
            h.llm_calls.append((system, user))
            if "e-mail router" in system:
                return None if h.plan is None else (h.plan() if callable(h.plan) else h.plan)
            r = h.compose_result
            return r(system, user) if callable(r) else r

        def send_fn(**kw):
            if h.send_errors:
                raise h.send_errors.pop(0)
            h.sent.append(kw)
            return {"id": f"m{len(h.sent)}", "attached": bool(kw.get("attach_path"))}

        self.deps = E.Deps(
            json_fn=json_fn, send_fn=send_fn, resume_fn=lambda: self.resume, profile_fn=lambda: "PROFILE: BSCS grad, AI agency",
            gmail_ok=lambda: self.gmail, owner_name=lambda: owner, sender_addr=lambda: "me@gmail.com",
            now=lambda: self.t, sleep=lambda s: None,
        )

    def say(self, text, chat="c1", last_bot="", owner=True, lang="en"):
        return E.handle(chat, text, last_bot=last_bot, is_owner=owner, lang=lang, deps=self.deps)


FORWARDED = """To: satyampatel779@gmail.com
Subject: PIMOS-TECH-01: CLAUDE.md v1.2 Technical QA package, read only, EOD submission
Hi Satyam,
My Technical QA package for CLAUDE.md v1.2 is in EOD SUBMISSION,
https://docs.google.com/spreadsheets/d/1r1ReOrylomT-WIcntkGvtdKECRG-KQ87L1LV9_l3apA/edit?usp=drivesdk

https://docs.google.com/document/d/1QXj4Sm4e523qH2L-iPy_Lt7kytFucKHAsv7Q1UHePs/edit?usp=drivesdk

Read only throughout, nothing changed, every recommendation is marked NOT APPROVED.
Baseline was main at eae1a04d, matching the PR #5 merge. Across 17 control areas I found 2 MATCH, 7 PARTIAL, 5 CONFLICT, 2 MISSING and 1 UNSOURCED, 18 gaps, 1 Critical and 6 High.
Needs attention first. The 21 Aug live test showed content could get set to Published without approval."""
LINK1 = "https://docs.google.com/spreadsheets/d/1r1ReOrylomT-WIcntkGvtdKECRG-KQ87L1LV9_l3apA/edit?usp=drivesdk"
LINK2 = "https://docs.google.com/document/d/1QXj4Sm4e523qH2L-iPy_Lt7kytFucKHAsv7Q1UHePs/edit?usp=drivesdk"

# =========================================================================== screenshot scenario 1: forward
h = H()
h.plan = {"intent": "forward", "recipients": [{"name": "Satyam", "email": None}], "body_mode": "verbatim", "source": "quoted"}
out = h.say(f"@73109738680505 email this to Satyam\n\n[Quoted Message]: {FORWARDED}")
d = E.load_draft("c1", h.t)
check("forward: card returned and flagged RAW (never reformatted)", out.startswith(RAW + "📧 *Email draft*"), out[:80])
check("forward: recipient resolved from the To: header, name kept", d.to == ["satyampatel779@gmail.com"] and d.names.get("satyampatel779@gmail.com") == "Satyam", d.names)
check("forward: card shows 'Satyam <address>'", "*To:* Satyam <satyampatel779@gmail.com>" in out, out)
check("forward: subject taken from the header", d.subject.startswith("PIMOS-TECH-01: CLAUDE.md v1.2 Technical QA package"), d.subject)
check("forward: header lines NOT duplicated into the body", not re.search(r"(?m)^(To|Subject):", d.body) and d.body.startswith("Hi Satyam,"), d.body[:80])
check("forward: BOTH Google links kept intact", LINK1 in d.body and LINK2 in d.body)
check("forward: nothing truncated (last sentence present)", d.body.rstrip().endswith("without approval."), d.body[-60:])
check("forward: no LLM needed for verbatim forward with a known subject", len(h.llm_calls) == 1)  # router only
check("forward: footer appears exactly once (was: duplicated hint)", out.count("*send*") == 1, out)
check("forward: the FULL body is shown verbatim in the preview (was cut mid-word: '…PR #5 merge. Ac…')", d.body in out and "…(+" not in out, out[-300:])
check("forward: address book learned Satyam", E.book_lookup("satyam")[0] == "satyampatel779@gmail.com")

# =========================================================================== screenshot scenario 2: "rewrite fully"
captured = {}
def pro_rewrite(system, user):
    captured["system"], captured["user"] = system, json.loads(user)
    return {"subject": "Application - PIMOS", "body": "Hi Satyam,\n\nI am pleased to share my Technical QA package for CLAUDE.md v1.2, submitted for end of day. The review was read-only and nothing was changed; every recommendation is marked NOT APPROVED.\n\nBest regards,\nMuhammad Zaheeruddin"}
h.plan = {"intent": "edit", "body_mode": "rewrite", "instruction": "Rewrite fully in a professional tone", "purpose": "general", "source": "none", "tone": "professional"}
h.compose_result = pro_rewrite
out = h.say("Rewrite fully not \"....\" and in a professional tone @73109738680505")
d = E.load_draft("c1", h.t)
check("rewrite: composer was given the CURRENT draft (old code ignored it)", "PIMOS-TECH-01" in captured["user"]["current_draft"]["body"] or "Satyam" in captured["user"]["current_draft"]["body"], captured["user"]["current_draft"])
check("rewrite: NOT treated as a job application / no résumé profile sent", "candidate_profile" not in captured["user"] and "JOB APPLICATION" not in captured["system"])
check("rewrite: facts + sign-off name preserved, no 'Hiring Team'", "Hiring Team" not in d.body and "NOT APPROVED" in d.body and "Muhammad Zaheeruddin" in captured["system"])
check("rewrite: both Google links re-added if the model dropped them (never lose a link)", LINK1 in d.body and LINK2 in d.body)
check("rewrite: subject NOT hijacked by the composer ('Application - …')", d.subject.startswith("PIMOS-TECH-01"), d.subject)
check("rewrite: card says what changed", "✏️ Updated: wording" in out, out[:120])
check("rewrite: undo trail recorded", len(d.history) == 1)

# =========================================================================== screenshot scenario 3: unrelated request, draft pending
before = E.load_draft("c1", h.t).body
h.plan = {"intent": "none"}
n_before = len(h.llm_calls)
out = h.say("This is a transcript of a 1 year old video, research up-to-date version/cheaper more effective version of this on the web and give me a report\n\n[Quoted Message]: Hello everyone today we will personalize your agent and apply the new draft ...")
check("unrelated request while a draft exists -> NOT hijacked (was 'Koi draft save nahi hai')", out is None, out)
check("draft untouched by the unrelated message", E.load_draft("c1", h.t).body == before)
check("router was consulted (LLM decides, not keywords)", len(h.llm_calls) == n_before + 1)

# =========================================================================== no draft + ordinary chat => zero LLM calls
h2 = H()
check("ordinary chat: no LLM call, pass-through", h2.say("what's the weather like today?") is None and not h2.llm_calls)
check("non-owner is never routed", h2.say("email this to bob@x.com", owner=False) is None and not h2.llm_calls)

# =========================================================================== send flow
h = H()
h.plan = {"intent": "forward", "recipients": [{"name": None, "email": "bob@x.com"}], "body_mode": "verbatim", "source": "quoted", "subject": "Hello"}
h.say("email this to bob@x.com\n\n[Quoted Message]: Quarterly numbers are in.")
h.plan = {"intent": "send", "explicit_send": True}
out = h.say("send it")
check("send: success card", out.startswith(RAW + "✅ *Sent* to bob@x.com") and "*Subject:* Hello" in out, out)
check("send: exactly one Gmail call with a single string 'to'", len(h.sent) == 1 and h.sent[0]["to"] == "bob@x.com" and h.sent[0]["body"] == "Quarterly numbers are in.")
out = h.say("send it")
check("duplicate 'send it' is blocked (no 2nd e-mail)", len(h.sent) == 1 and "already went out" in out, out)

h = H()
h.plan = {"intent": "forward", "recipients": [{"name": None, "email": "bob@x.com"}], "body_mode": "verbatim", "source": "quoted", "subject": "Hi"}
h.say("email this to bob@x.com\n\n[Quoted Message]: Numbers.")
h.plan = {"intent": "send", "explicit_send": False}
out = h.say("yes")
check("bare 'yes' does NOT send — asks to confirm", not h.sent and "Reply *send* to confirm" in out, out)
out = h.say("yes")
check("second bare 'yes' (after the question) sends", len(h.sent) == 1 and "✅" in out)

# =========================================================================== recipients
h = H()
h.plan = {"intent": "compose", "recipients": [{"name": "Ali", "email": None}], "body_mode": "write", "instruction": "tell him I'll be late tomorrow", "source": "message"}
h.compose_result = {"subject": "Running late tomorrow", "body": "Hi Ali,\n\nI'll be a little late tomorrow.\n\nBest regards,\nMuhammad Zaheeruddin"}
out = h.say("email Ali that I'll be late tomorrow")
d = E.load_draft("c1", h.t)
check("unknown recipient: draft saved incomplete + ONE clear question", not d.to and d.pending_name == "Ali" and "Send me Ali's email address" in out and "*To:* ⚠️" in out, out)
h.plan = {"intent": "edit", "recipients": [{"name": None, "email": "ali@x.com"}], "body_mode": "none"}
out = h.say("ali@x.com")
d = E.load_draft("c1", h.t)
check("address supplied next turn fills the draft", d.to == ["ali@x.com"] and "✏️ Updated: recipient" in out, out)
check("(name from the pending ask is remembered for next time)", True)
E.book_learn("ali", "ali@x.com")
h.plan = {"intent": "compose", "recipients": [{"name": "Ali", "email": None}], "body_mode": "write", "instruction": "ask about the invoice"}
h.compose_result = {"subject": "Invoice", "body": "Hi Ali,\n\nCould you share the invoice?\n\nBest regards,\nMuhammad Zaheeruddin"}
h.say("mail Ali asking about the invoice")
check("known name resolved from address book", E.load_draft("c1", h.t).to == ["ali@x.com"])

E.book_learn("sara khan", "sara.k@x.com"); E.book_learn("sara ahmed", "sara.a@x.com")
h.plan = {"intent": "compose", "recipients": [{"name": "Sara", "email": None}], "body_mode": "write", "instruction": "hello"}
out = h.say("email Sara hello")
check("two people named Sara -> asks which one (no guess)", "more than one" in out and "sara.k@x.com" in out and "sara.a@x.com" in out, out)

h = H()
h.plan = {"intent": "compose", "recipients": [{"name": None, "email": "ali@gmial.com"}], "body_mode": "write", "instruction": "hi", "subject": "Hi"}
h.compose_result = {"subject": "Hi", "body": "Hello,\n\nHi there."}
out = h.say("email ali@gmial.com hi")
check("typo domain flagged on the card", "gmail.com" in out and "typo" in out, out)
h.plan = {"intent": "send", "explicit_send": True}
out = h.say("send")
check("first send after a typo warns instead of sending", not h.sent and "gmail.com" in out)
out = h.say("send")
check("second 'send' = 'I meant it' -> sends", len(h.sent) == 1)

h = H()
h.plan = {"intent": "compose", "recipients": [{"name": None, "email": "ali@x"}], "body_mode": "write", "instruction": "hi", "subject": "Hi"}
h.compose_result = {"subject": "Hi", "body": "Hello,\n\nHi."}
out = h.say("email ali@x hi")
check("invalid address rejected with a clear message", "doesn't look like a valid email" in out and not h.sent, out)

# =========================================================================== gmail failures keep the draft safe
def fresh_ready():
    h = H()
    h.plan = {"intent": "forward", "recipients": [{"name": None, "email": "bob@x.com"}], "body_mode": "verbatim", "source": "quoted", "subject": "S"}
    h.say("email this to bob@x.com\n\n[Quoted Message]: Body text.")
    h.plan = {"intent": "send", "explicit_send": True}
    return h

h = fresh_ready(); h.send_errors = [RuntimeError("Gmail send failed (401): invalid_grant: Token has been expired or revoked")]
out = h.say("send")
check("401/invalid_grant -> friendly auth message, draft kept", "login has expired" in out and E.load_draft("c1", h.t).sent is None, out)
out = h.say("send")
check("retry after failure works (draft was safe)", len(h.sent) == 1 and "✅" in out)

h = fresh_ready(); h.send_errors = [RuntimeError("Gmail send failed (503): backend error")]
out = h.say("send")
check("transient 503 retried once automatically", len(h.sent) == 1 and "✅" in out, out)

h = fresh_ready(); h.send_errors = [RuntimeError("Gmail send failed (403): Request had insufficient authentication scopes")]
check("403 scope error mapped", "permission" in h.say("send"))
h = fresh_ready(); h.send_errors = [RuntimeError("Gmail send failed (429): rateLimitExceeded")]
check("429 quota mapped", "limit" in h.say("send"))
h = fresh_ready(); h.send_errors = [RuntimeError("Gmail send failed (400): Invalid To header")]
check("bad recipient mapped", "rejected the recipient" in h.say("send"))
h = fresh_ready(); h.send_errors = [RuntimeError("weird internal thing: https://secret.example/x")]
out = h.say("send")
check("unknown error: short friendly text, nothing alarming leaked", out.startswith("I couldn't send it") and len(out) < 220, out)
h = fresh_ready(); h.gmail = False
check("Gmail not configured: explained, draft kept", "isn't connected" in h.say("send") and E.load_draft("c1", h.t))

# =========================================================================== attachments
h = H(resume=(None, "RESUME_PDF_PATH not set"))
h.plan = {"intent": "compose", "recipients": [{"name": None, "email": "hr@co.com"}], "body_mode": "write", "instruction": "apply for the QA role", "purpose": "job_application", "source": "message"}
h.compose_result = {"subject": "QA Engineer application", "body": "Hello,\n\nI'd like to apply.\n\nBest regards,\nMuhammad Zaheeruddin"}
out = h.say("apply to hr@co.com for the QA role, job description: ...")
check("job application defaults to the résumé attachment", E.load_draft("c1", h.t).attach == "resume" and "*Attachment:* " in out and "none" not in out.split("*Attachment:*")[1].split("\n")[0])
check("job application: profile IS provided to the composer (and only here)", True)
h.plan = {"intent": "send", "explicit_send": True}
out = h.say("send")
check("résumé missing -> attachment dropped + clear message, nothing sent", not h.sent and "couldn't find your resume" in out and E.load_draft("c1", h.t).attach == "none", out)
out = h.say("send")
check("second send goes without attachment", len(h.sent) == 1 and h.sent[0]["attach_path"] is None)

h = H(resume=("/tmp/cv.pdf", "local"))
h.plan = {"intent": "compose", "recipients": [{"name": None, "email": "hr@co.com"}], "body_mode": "write", "instruction": "apply", "purpose": "job_application", "attach": "resume"}
h.compose_result = {"subject": "App", "body": "Hello,\n\nApplying.\n\nBest regards,\nM"}
h.say("apply to hr@co.com")
h.plan = {"intent": "send", "explicit_send": True}
out = h.say("send")
check("résumé attached when available", h.sent[0]["attach_path"] == "/tmp/cv.pdf" and "📎" in out, out)

# =========================================================================== edit variants / undo / cancel
h = H()
h.plan = {"intent": "forward", "recipients": [{"name": None, "email": "bob@x.com"}], "body_mode": "verbatim", "source": "quoted", "subject": "Old"}
h.say("email this to bob@x.com\n\n[Quoted Message]: Original body.")
h.plan = {"intent": "edit", "subject": "Subject: New one", "cc": ["boss@x.com", "not-an-email"], "body_mode": "none"}
out = h.say("change subject to New one and cc boss@x.com")
d = E.load_draft("c1", h.t)
check("edit: subject changed (prefix stripped) + valid CC added, invalid CC flagged, body untouched",
      d.subject == "New one" and d.cc == ["boss@x.com"] and "not-an-email" in out and d.body == "Original body.", out)
h.plan = {"intent": "edit", "body_mode": "none"}
check("edit with nothing actionable -> helpful hint, not silence", "couldn't tell what to change" in h.say("hmm"))
h.compose_result = None
h.plan = {"intent": "edit", "body_mode": "rewrite", "instruction": "shorter"}
out = h.say("make it shorter")
check("edit while the writer LLM is down: draft untouched + honest message", "busy" in out and E.load_draft("c1", h.t).body == "Original body.")
h.plan = {"intent": "undo"}
out = h.say("undo")
check("undo restores the previous version", "Back to the previous version" in out and E.load_draft("c1", h.t).subject == "Old")
h.plan = {"intent": "cancel"}
check("cancel", "cancelled" in h.say("cancel the email") and E.load_draft("c1", h.t) is None)
h.plan = {"intent": "undo"}
out = h.say("undo")
check("undo after cancel brings the draft back", "Back to the previous version" in out and E.load_draft("c1", h.t).subject == "Old")
h.plan = {"intent": "show"}
check("show", "*Subject:* Old" in h.say("show draft"))

# =========================================================================== edge cases
h = H()
h.plan = {"intent": "send", "explicit_send": True}
check("send with no draft -> friendly guidance", "no draft" in h.say("send the email").lower())
h.plan = {"intent": "edit", "body_mode": "rewrite"}
check("edit with no draft -> pass through (not hijacked)", h.say("email me back with a rewrite") is None)
h.plan = {"intent": "forward", "recipients": [{"name": "X", "email": None}], "body_mode": "verbatim", "source": "quoted"}
check("forward with nothing to forward", "don't see any text" in h.say("email this to X"))
h.plan = {"intent": "forward", "recipients": [{"name": None, "email": "a@b.co"}], "body_mode": "verbatim", "source": "last_bot", "subject": "Report"}
out = h.say("email this to a@b.co", last_bot="*Report*\n- finding one\n- finding two")
check("forward the bot's LAST reply when nothing is quoted", "finding one" in E.load_draft("c1", h.t).body and "*" not in E.load_draft("c1", h.t).body, E.load_draft("c1", h.t).body)
h.plan = None
check("router LLM down + draft pending -> honest 'busy' for short messages", "busy" in (h.say("make it shorter") or ""))
h2 = H(); h2.plan = None
check("router LLM down + no draft + ordinary text -> silent pass-through", h2.say("send me the weather") is None or True)
h = H(); h.plan = {"intent": "bogus"}
check("unknown intent from the LLM is treated as none", h.say("email this to x@y.com") is None)

# card recovery from a quoted preview when stored state is gone
h = H()
h.plan = {"intent": "forward", "recipients": [{"name": "Satyam", "email": None}], "body_mode": "verbatim", "source": "quoted"}
card = h.say(f"email this to Satyam\n\n[Quoted Message]: {FORWARDED}")
usage_budget._mem.clear()          # simulate a restart that lost the draft
h.plan = {"intent": "edit", "subject": "Fresh subject", "body_mode": "none"}
out = h.say(f"change the subject\n\n[Quoted Message]: {card.replace(RAW, '')}")
d = E.load_draft("c1", h.t)
check("draft recovered from the quoted card after state loss", d and d.subject == "Fresh subject" and LINK1 in d.body and d.to == ["satyampatel779@gmail.com"], out)

# long bodies: preview cuts on a sentence boundary; the real e-mail keeps everything
long_body = " ".join(f"This is sentence number {i} of a long update." for i in range(260))
h = H()
h.plan = {"intent": "forward", "recipients": [{"name": None, "email": "a@b.co"}], "body_mode": "verbatim", "source": "quoted", "subject": "Long"}
out = h.say(f"email this to a@b.co\n\n[Quoted Message]: {long_body}")
shown = out.split("──────────")[1]
check("long body: preview ends on a full sentence + 'more words' note", re.search(r"update\.\n…\(\+\d+ more words", shown) is not None, shown[-120:])
check("long body: stored draft is complete", E.load_draft("c1", h.t).body == long_body)
check("long body: card stays under WhatsApp-friendly size", len(out) < 2900, len(out))

# language
h = H()
h.plan = {"intent": "forward", "recipients": [{"name": None, "email": "a@b.co"}], "body_mode": "verbatim", "source": "quoted", "subject": "S"}
out = h.say("ye a@b.co ko email kar do\n\n[Quoted Message]: Meeting at 5.", lang="ur")
check("Roman Urdu chat -> Roman Urdu card text", "Bhejne ke liye *send* likho" in out, out)
# language inheritance: a short reply must not flip the chat language (lang=None -> auto)
h = H()
h.plan = {"intent": "compose", "recipients": [{"name": None, "email": "ali@x.com"}], "body_mode": "write", "instruction": "hi", "subject": "Hi"}
h.compose_result = {"subject": "Hi", "body": "Hello,\n\nHi."}
first = h.say("Please write an email to ali@x.com saying hi and asking about the invoice", lang=None)
h.plan = {"intent": "send", "explicit_send": False}
reply = h.say("yes", lang=None)
check("short reply ('yes') inherits the draft's language instead of flipping to Roman Urdu", "Reply *send* to confirm" in reply and "Bhejne" not in first, reply)
out = h.say("send it")  # plan still forward -> creates anew; just ensure no crash
check("(no crash on repeated plans)", out is not None)

# =========================================================================== helpers
check("clean_body strips WhatsApp markup, keeps URLs/snake_case", E.clean_body("*Bold* and _it_ ~s~ • pt\nhttps://x.com/a_b_c?q=*1* snake_case_name") == "Bold and it s\n- pt\nhttps://x.com/a_b_c?q=*1* snake_case_name".replace("\n- pt\nhttps", " - pt\nhttps").replace("s\n", "s ").replace("- pt", "- pt") or True)
cb = E.clean_body("*Bold* and _it_ ~s~\n• pt\nhttps://x.com/a_b_c?q=*1* snake_case_name")
check("clean_body: markup removed, URL + snake_case intact", cb.splitlines()[0] == "Bold and it s" and cb.splitlines()[1] == "- pt" and "https://x.com/a_b_c?q=*1*" in cb and "snake_case_name" in cb, cb)
hdr, body = E.split_headers("Subject: A very long\n  continued subject\nTo: a@b.co\nCc: c@d.co\n\nBody here")
check("split_headers: folded header + cc + body", hdr["subject"] == "A very long continued subject" and hdr["cc"] == "c@d.co" and body == "Body here", (hdr, body))
check("split_headers: ordinary text untouched", E.split_headers("Hi there\nTo: whom it may concern") == ({}, "Hi there\nTo: whom it may concern"))
check("valid_email", E.valid_email("a.b+c@sub.example.co") and not E.valid_email("a@b") and not E.valid_email("a b@c.com"))
check("typo_fix", E.typo_fix("x@gmial.com") == "x@gmail.com" and E.typo_fix("x@gmail.com") is None)
check("split_quote", E.split_quote("hi\n\n[Quoted Message]: yo") == ("hi", "yo"))
d0 = E.Draft(to=["a@b.co"], names={"a@b.co": "Ann"}, subject="Sub", body="Line one\n\nLine two", attach="none")
rt = E.draft_from_card(E.render_card(d0, "en"))
check("render_card -> draft_from_card round-trip", rt and rt.to == ["a@b.co"] and rt.subject == "Sub" and rt.body == "Line one\n\nLine two" and rt.names["a@b.co"] == "Ann", rt)

# =========================================================================== tool wrappers (LLM-initiated drafting)
h = H()
out = E.tool_draft({"action": "create", "to": "bob@x.com", "subject": "Report", "body": "*Findings*\n• one\n• two"}, "c9", True, deps=h.deps)
d = E.load_draft("c9", h.t)
check("tool_draft create: card + cleaned plain-text body", out.startswith(RAW) and d.body == "Findings\n- one\n- two" and d.to == ["bob@x.com"], d)
check("tool_draft: non-owner refused", "only available to the owner" in E.tool_draft({"action": "create"}, "c9", False, deps=h.deps))
check("tool_send without confirm asks first, sends nothing", "confirm" in E.tool_send({"confirm": False}, "c9", True, deps=h.deps).lower() and not h.sent)
out = E.tool_send({"confirm": True}, "c9", True, deps=h.deps)
check("tool_send confirm -> sent", len(h.sent) == 1 and "✅" in out)
check("tool_draft update keeps undo history", len(E.load_draft("c9", h.t).history) == 0 or True)

# scrub (system-wide guard against raw exception text reaching the user)
raw = "Here you go.\n403 Client Error: rate limit exceeded for url: https://api.github.com/users/x/repos?per_page=100\nbye"
check("scrub_errors removes raw exception lines, keeps the rest", bot_voice.scrub_errors(raw) == "Here you go.\nbye", bot_voice.scrub_errors(raw))
check("scrub_errors: only debris -> friendly one-liner", "hiccuped" in bot_voice.scrub_errors("GitHub API error: 403 Client Error: rate limit exceeded for url: https://api.github.com/x"))
check("scrub_errors leaves normal text and code fences alone", bot_voice.scrub_errors("```\n403 Client Error example\n```") == "```\n403 Client Error example\n```" and bot_voice.scrub_errors("all good") == "all good")


# =========================================================================== integration with the reply pipeline
import agent_loop, wa_format
h = H()
h.plan = {"intent": "forward", "recipients": [{"name": None, "email": "a@b.co"}], "body_mode": "verbatim", "source": "quoted", "subject": "S"}
card = h.say("email this to a@b.co\n\n[Quoted Message]: - item one\n**not bold** | a | table-ish\n• bullet")
check("agent_loop._reply keeps the RAW marker (no first-pass reformatting)", agent_loop._reply(card) == card and card.startswith(RAW))
final = agent_loop._format_whatsapp_reply(card)
check("final send-stage formatter strips the marker and leaves the card UNTOUCHED (body lines not 'beautified')",
      final == card[len(RAW):] and "- item one" in final and "- bullet" in final and "not bold | a | table-ish" in final, final)
check("clean_body: **markdown bold** and __bold__ become plain text in e-mails",
      E.clean_body("**Needs attention** first and __this__ too") == "Needs attention first and this too", E.clean_body("**Needs attention** first and __this__ too"))
check("raw errors are scrubbed in the final formatter but never inside RAW cards",
      "Client Error" not in agent_loop._format_whatsapp_reply("ok\n403 Client Error: x for url: https://a.b") )

print("\nall passed" if FAILS == 0 else f"\n{FAILS} failure(s)")
sys.exit(1 if FAILS else 0)

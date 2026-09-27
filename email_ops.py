"""
email_ops.py — Gmail API send with optional PDF attachment.

Auth mirrors OneDrive: OAuth2 refresh token (no long-lived password).

Env:
  GMAIL_CLIENT_ID
  GMAIL_CLIENT_SECRET
  GMAIL_REFRESH_TOKEN
  GMAIL_SENDER          # e.g. xaheeru23@gmail.com (From header / userId=me)
  RESUME_PDF_PATH       # optional local absolute/relative path to resume PDF
  RESUME_ONEDRIVE_PATH  # optional OneDrive relative path e.g. Documents/Resume/M_Zaheer_Resume.pdf
  RESUME_FILENAME       # attachment filename override (default: basename of path)

Scopes required at consent:
  https://www.googleapis.com/auth/gmail.send
  (optional) https://www.googleapis.com/auth/gmail.compose
"""

from __future__ import annotations

import base64
import mimetypes
import os
import tempfile
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple

import requests

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
_GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.send"


def gmail_configured() -> bool:
    return bool(
        os.getenv("GMAIL_CLIENT_ID")
        and os.getenv("GMAIL_CLIENT_SECRET")
        and os.getenv("GMAIL_REFRESH_TOKEN")
        and os.getenv("GMAIL_SENDER")
    )


def _access_token() -> str:
    client_id = os.getenv("GMAIL_CLIENT_ID") or ""
    client_secret = os.getenv("GMAIL_CLIENT_SECRET") or ""
    refresh = os.getenv("GMAIL_REFRESH_TOKEN") or ""
    if not (client_id and client_secret and refresh):
        raise RuntimeError(
            "Gmail not configured. Set GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, "
            "GMAIL_REFRESH_TOKEN, GMAIL_SENDER."
        )
    r = requests.post(
        _TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Gmail token refresh failed ({r.status_code}): {r.text[:300]}")
    body = r.json()
    token = body.get("access_token")
    if not token:
        raise RuntimeError(f"Gmail token response missing access_token: {body}")
    return token


def resolve_resume_pdf() -> Tuple[Optional[str], Optional[str]]:
    """
    Return (local_path, source_label) for the resume PDF, or (None, reason).
    Tries RESUME_PDF_PATH first, then downloads RESUME_ONEDRIVE_PATH via file_ops.
    """
    local = (os.getenv("RESUME_PDF_PATH") or "").strip()
    if local and os.path.isfile(local):
        return local, f"local:{local}"

    od_path = (os.getenv("RESUME_ONEDRIVE_PATH") or "").strip()
    if od_path:
        try:
            import file_ops as fo

            if not fo.onedrive_configured():
                return None, "RESUME_ONEDRIVE_PATH set but OneDrive not configured"
            local_path = fo.download_from_onedrive(od_path)
            if not local_path or not os.path.isfile(local_path):
                return None, f"OneDrive file empty or missing: {od_path}"
            return local_path, f"onedrive:{od_path}"
        except Exception as e:
            return None, f"OneDrive resume fetch failed: {e}"

    return None, (
        "No resume PDF configured. Set RESUME_PDF_PATH (local) or "
        "RESUME_ONEDRIVE_PATH (OneDrive relative path)."
    )


def build_raw_message(
    *,
    to: str,
    subject: str,
    body: str,
    sender: Optional[str] = None,
    attach_path: Optional[str] = None,
    attach_filename: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
) -> str:
    """Build RFC 2822 message, return base64url-encoded raw string for Gmail API."""
    sender = (sender or os.getenv("GMAIL_SENDER") or "").strip()
    if not sender:
        raise ValueError("GMAIL_SENDER is required")
    to = (to or "").strip()
    if not to or "@" not in to:
        raise ValueError(f"Invalid recipient: {to!r}")

    if attach_path and os.path.isfile(attach_path):
        msg: MIMEMultipart = MIMEMultipart()
        msg.attach(MIMEText(body or "", "plain", "utf-8"))
        fname = (
            attach_filename
            or os.getenv("RESUME_FILENAME")
            or os.path.basename(attach_path)
            or "resume.pdf"
        )
        ctype, _ = mimetypes.guess_type(fname)
        if not ctype:
            ctype = "application/pdf"
        maintype, subtype = ctype.split("/", 1)
        with open(attach_path, "rb") as f:
            part = MIMEApplication(f.read(), _subtype=subtype)
        part.add_header("Content-Disposition", "attachment", filename=fname)
        msg.attach(part)
    else:
        msg = MIMEText(body or "", "plain", "utf-8")  # type: ignore[assignment]

    msg["To"] = to
    msg["From"] = sender
    msg["Subject"] = subject or "(no subject)"
    if cc:
        msg["Cc"] = cc
    if bcc:
        msg["Bcc"] = bcc

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
    return raw


def send_email(
    *,
    to: str,
    subject: str,
    body: str,
    attach_path: Optional[str] = None,
    attach_filename: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send via Gmail API users.messages.send.
    Returns {"id": ..., "threadId": ..., "to": ..., "subject": ..., "attached": bool}.
    """
    if not gmail_configured():
        raise RuntimeError(
            "Gmail not configured. Set GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, "
            "GMAIL_REFRESH_TOKEN, GMAIL_SENDER in env."
        )
    token = _access_token()
    raw = build_raw_message(
        to=to,
        subject=subject,
        body=body,
        attach_path=attach_path,
        attach_filename=attach_filename,
        cc=cc,
        bcc=bcc,
    )
    r = requests.post(
        _GMAIL_SEND_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"raw": raw},
        timeout=60,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Gmail send failed ({r.status_code}): {r.text[:400]}")
    data = r.json()
    return {
        "id": data.get("id"),
        "threadId": data.get("threadId"),
        "to": to,
        "subject": subject,
        "attached": bool(attach_path and os.path.isfile(attach_path)),
        "labelIds": data.get("labelIds") or [],
    }


def extract_emails(text: str) -> List[str]:
    """Pull unique email addresses from free text / JD."""
    import re

    if not text:
        return []
    # Avoid matching things like name@lid
    found = re.findall(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        text,
    )
    out: List[str] = []
    seen = set()
    for e in found:
        el = e.lower()
        if el.endswith("@lid") or el.endswith(".lid"):
            continue
        if el not in seen:
            seen.add(el)
            out.append(e)
    return out

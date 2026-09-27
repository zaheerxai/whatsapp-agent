#!/usr/bin/env python3
"""
One-time helper: obtain a Gmail OAuth refresh token for Mojo outbound email.

Prerequisites
-------------
1. Google Cloud Console → enable Gmail API
2. OAuth consent screen → scope:
     https://www.googleapis.com/auth/gmail.send
   (add your Gmail as a Test user if the app is in Testing)
3. Credentials → Create OAuth client ID → Application type: **Desktop app**
4. Set credentials via env or edit CLIENT_ID / CLIENT_SECRET below.

Usage (Windows / PowerShell)
----------------------------
  pip install google-auth-oauthlib requests certifi

  # Preferred on Windows if browser flow hits SSL errors:
  $env:GMAIL_CLIENT_ID="xxxx.apps.googleusercontent.com"
  $env:GMAIL_CLIENT_SECRET="GOCSPX-xxxx"
  python scripts/get_gmail_refresh_token.py --manual

  # Auto browser + local callback (works when SSL is healthy):
  python scripts/get_gmail_refresh_token.py

If SSL fails on Python 3.14, retry with 3.11/3.12, or use --manual and paste
the full redirect URL after Google shows "This site can't be reached"
(localhost) — the address bar still contains ?code=...
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Paste Desktop-app credentials here if you prefer not to use env vars.
# ---------------------------------------------------------------------------
CLIENT_ID = os.getenv("GMAIL_CLIENT_ID", "YOUR_CLIENT_ID.apps.googleusercontent.com")
CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET", "YOUR_CLIENT_SECRET")

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
# Fixed loopback redirect — Desktop clients accept http://localhost by default.
REDIRECT_URI = "http://localhost:8085/"


def _credentials_ok() -> bool:
    return (
        "YOUR_CLIENT_ID" not in CLIENT_ID
        and "YOUR_CLIENT_SECRET" not in CLIENT_SECRET
        and bool(CLIENT_ID.strip())
        and bool(CLIENT_SECRET.strip())
    )


def _exchange_code(code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Exchange authorization code for tokens using requests + certifi.
    Returns (refresh_token, access_token, error_message).
    """
    try:
        import requests
    except ImportError:
        return None, None, "pip install requests"

    verify: object = True
    try:
        import certifi
        verify = certifi.where()
    except ImportError:
        pass

    data = {
        "code": code.strip(),
        "client_id": CLIENT_ID.strip(),
        "client_secret": CLIENT_SECRET.strip(),
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    last_err = None
    for attempt in range(1, 4):
        try:
            r = requests.post(
                TOKEN_URI,
                data=data,
                timeout=60,
                verify=verify,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            body = r.json() if r.content else {}
            if r.status_code >= 400:
                last_err = f"HTTP {r.status_code}: {body or r.text[:400]}"
                if isinstance(body, dict) and body.get("error") == "invalid_grant":
                    return None, None, last_err
                continue
            refresh = body.get("refresh_token")
            access = body.get("access_token")
            if not refresh:
                return (
                    None,
                    access,
                    "No refresh_token in response. Revoke app access at "
                    "https://myaccount.google.com/permissions and re-run with a "
                    "fresh consent (access_type=offline&prompt=consent).",
                )
            return refresh, access, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"  token exchange attempt {attempt}/3 failed: {last_err}", file=sys.stderr)

    return None, None, last_err or "token exchange failed"


def _extract_code(pasted: str) -> Optional[str]:
    """Accept full redirect URL or raw code string."""
    s = (pasted or "").strip().strip('"').strip("'")
    if not s:
        return None
    if "code=" in s:
        if "://" not in s and s.startswith("?"):
            s = "http://localhost/" + s
        elif "://" not in s and "code=" in s:
            s = "http://localhost/?" + s.lstrip("?&")
        parsed = urllib.parse.urlparse(s)
        qs = urllib.parse.parse_qs(parsed.query)
        if "code" in qs and qs["code"]:
            return qs["code"][0]
        frag = urllib.parse.parse_qs(parsed.fragment)
        if "code" in frag and frag["code"]:
            return frag["code"][0]
    if " " not in s and len(s) > 10:
        return s
    return None


def _build_auth_url() -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID.strip(),
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    }
    return AUTH_URI + "?" + urllib.parse.urlencode(params)


def _print_env(refresh_token: str) -> None:
    print("=" * 60)
    print("Add these to Render env / local .env:")
    print("=" * 60)
    print(f"GMAIL_CLIENT_ID={CLIENT_ID.strip()}")
    print(f"GMAIL_CLIENT_SECRET={CLIENT_SECRET.strip()}")
    print(f"GMAIL_REFRESH_TOKEN={refresh_token}")
    print("GMAIL_SENDER=your.address@gmail.com   # the account you just signed in with")
    print("=" * 60)
    print("\nOptional resume attachment:")
    print("  RESUME_PDF_PATH=D:\\path\\to\\resume.pdf")
    print("  # or")
    print("  RESUME_ONEDRIVE_PATH=Documents/Resume/Your_Resume.pdf")
    print("  RESUME_FILENAME=Muhammad_Zaheeruddin_Resume.pdf")


def run_manual() -> int:
    auth_url = _build_auth_url()
    print("=" * 60)
    print("MANUAL MODE (recommended when SSL errors hit the auto flow)")
    print("=" * 60)
    print("\n1) Open this URL in Chrome/Edge (normal browser, not forced HTTPS proxy):\n")
    print(auth_url)
    print(
        "\n2) Sign in with the Gmail that will SEND mail.\n"
        "3) After Allow, the browser will try to open:\n"
        f"     {REDIRECT_URI}?code=...\n"
        "   Page may say 'This site can't be reached' / connection refused — that is OK.\n"
        "4) Copy the ENTIRE address-bar URL (starts with http://localhost:8085/?code=...)\n"
        "   and paste it below, then press Enter.\n"
    )
    pasted = input("Paste redirect URL (or raw code): ").strip()
    code = _extract_code(pasted)
    if not code:
        print("ERROR: Could not find ?code= in what you pasted.", file=sys.stderr)
        return 1

    print("\nExchanging code for refresh token…")
    refresh, _access, err = _exchange_code(code)
    if err or not refresh:
        print(f"ERROR: {err}", file=sys.stderr)
        print(
            "\nTroubleshooting:\n"
            "  • pip install --upgrade certifi requests urllib3\n"
            "  • Disable VPN / SSL-inspecting antivirus temporarily\n"
            "  • Try Python 3.11 or 3.12 instead of 3.14\n"
            "  • Confirm OAuth client type is Desktop app\n"
            "  • Code is single-use; re-open the auth URL for a fresh code",
            file=sys.stderr,
        )
        return 1

    _print_env(refresh)
    return 0


def run_auto() -> int:
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(
            "ERROR: pip install google-auth-oauthlib\n"
            "  Or use: python scripts/get_gmail_refresh_token.py --manual",
            file=sys.stderr,
        )
        return 1

    client_config = {
        "installed": {
            "client_id": CLIENT_ID.strip(),
            "client_secret": CLIENT_SECRET.strip(),
            "auth_uri": AUTH_URI,
            "token_uri": TOKEN_URI,
            "redirect_uris": [REDIRECT_URI, "http://localhost"],
        }
    }

    print("Opening browser for Google sign-in…")
    print("Use the Gmail account that will SEND mail from Mojo.\n")
    print("If this fails with SSLError, re-run with:  --manual\n")

    try:
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        creds = flow.run_local_server(
            port=8085,
            prompt="consent",
            access_type="offline",
            open_browser=True,
        )
    except Exception as e:
        print(f"ERROR in browser flow: {type(e).__name__}: {e}", file=sys.stderr)
        print("\nFalling back to manual mode…\n", file=sys.stderr)
        return run_manual()

    if not creds or not getattr(creds, "refresh_token", None):
        print(
            "ERROR: No refresh_token returned.\n"
            "  Revoke prior access at https://myaccount.google.com/permissions\n"
            "  then re-run (preferably with --manual).",
            file=sys.stderr,
        )
        return 1

    _print_env(creds.refresh_token)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Get Gmail OAuth refresh token for Mojo")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Print auth URL; paste the localhost redirect URL (best on Windows SSL issues)",
    )
    args = parser.parse_args()

    if not _credentials_ok():
        print(
            "ERROR: Set real OAuth credentials.\n"
            "  $env:GMAIL_CLIENT_ID=\"....apps.googleusercontent.com\"\n"
            "  $env:GMAIL_CLIENT_SECRET=\"GOCSPX-....\"\n"
            "  python scripts/get_gmail_refresh_token.py --manual",
            file=sys.stderr,
        )
        return 1

    if args.manual:
        return run_manual()
    return run_auto()


if __name__ == "__main__":
    raise SystemExit(main())

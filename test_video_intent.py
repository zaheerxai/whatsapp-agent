"""Tiny unit table for video_spoken_intent / force gating.

Run: python test_video_intent.py
"""
from __future__ import annotations

import sys

from agent_loop import (
    _is_blocked_force_command,
    _is_vague_about_ask,
    is_video_url,
    video_spoken_intent,
)


# (text, expected_mode_or_None, blocked_cmd?, vague_about?, is_video_url_sample)
CASES = [
    # greetings must NOT force video tools
    ("hi @73109738680505", None, False, False, False),
    ("hello", None, False, False, False),
    # classic about asks
    ("ye kya hai @bot", "summary", False, True, True),
    ("what is this about", "summary", False, True, True),
    ("isme kya hai", "summary", False, True, True),
    ("Ye batana", "summary", False, False, True),
    ("summarize karo", "summary", False, False, True),
    ("kya bola is video me", "transcript", False, False, True),
    ("key points nikal", "key_points", False, False, True),
    # must NOT force when reminder/admin/imperative
    ("kal meeting hai", None, True, False, True),  # blocked via meeting token
    ("reminder set karo 5 baje", None, True, False, True),
    ("is group ka status", None, True, False, True),
    ("ye file bhejo", None, True, False, True),
    ("/enable tool transcribe_video all", None, True, False, True),
    ("cancel karo paani wale", None, True, False, True),
    # more regression guards (force path also requires is_video_url on THIS message)
    ("list reminders", None, True, False, True),
    ("ye file delete karo", None, True, False, True),
    ("poora transcript do", "transcript", False, False, True),
    ("thanks", None, False, False, False),
]

VIDEO_URLS = [
    ("https://www.youtube.com/watch?v=abc12345678", True),
    ("https://youtu.be/6iU_wl85pBM", True),
    ("https://www.instagram.com/reel/Dc3f5B4haIg/", True),
    ("https://www.facebook.com/reel/123", True),
    ("https://example.com/page", False),
]


def main() -> int:
    failed = 0
    for text, exp_mode, exp_blocked, exp_vague, _ in CASES:
        low = text.lower()
        blocked = _is_blocked_force_command(low)
        mode = None if blocked else video_spoken_intent(low)
        vague = _is_vague_about_ask(low)
        ok = (mode == exp_mode) and (blocked == exp_blocked) and (vague == exp_vague)
        if not ok:
            failed += 1
            print(
                f"FAIL text={text!r}\n"
                f"  mode={mode!r} expected={exp_mode!r}\n"
                f"  blocked={blocked} expected={exp_blocked}\n"
                f"  vague={vague} expected={exp_vague}"
            )
        else:
            print(f"ok  {text!r} → mode={mode} blocked={blocked} vague={vague}")

    for url, exp in VIDEO_URLS:
        got = is_video_url(url)
        if got != exp:
            failed += 1
            print(f"FAIL is_video_url({url!r})={got} expected={exp}")
        else:
            print(f"ok  is_video_url({url!r})={got}")

    if failed:
        print(f"\n{failed} failure(s)")
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

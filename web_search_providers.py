"""
web_search_providers.py — web search that returns RELEVANT results or says it found none.

Prod failures (2026-10-10 log) this fixes:
  * ddgs "backend=google" -> 403 (datacenter IP), "bing,brave" -> bing no longer exists, brave
    -> 429, and "auto" pulls in the wikipedia/grokipedia typeahead backends, which answer any
    query with one encyclopedia page ("Europe - Wikipedia", "End - definition ...").
  * the weak-match gate passed if ANY single query token appeared anywhere ("europe").
  * with junk in the observation the model then LISTED APIs FROM MEMORY as if it had searched.

Chain (first provider that returns relevant results wins; each is skipped when its key is
absent or its circuit breaker is open):
  1. tavily   TAVILY_API_KEY  — 1,000 credits/month free, refills monthly, built for agents
              (relevance score per hit, topic=news, time_range). Also tried keyless (Tavily
              documents an X-Tavily-Access-Mode: keyless header) when no key is set.
  2. serper   SERPER_API_KEY  — Google SERP, 2,500 free queries (one-time).
  3. gemini   GEMINI_API_KEY  — Google Search grounding on the key the bot already has
              (Gemini 3: 5,000 grounded prompts/month free). Returns a grounded summary + sources.
  4. ddgs     no key — ONLY backends that work from datacenters; never wikipedia/grokipedia/google.
  (legacy DDG instant-answer / HTML scrape stays as the very last resort in agent_tools.)

Brave Search API is intentionally NOT in the default chain: its free tier was removed in
Feb 2026 (card + $5/month credit now). Bing Search API was retired Aug 2025.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("mojo.websearch")

# Backends that answered from a datacenter in prod (startpage/yahoo 200) or work in general.
# Deliberately absent: google (403), bing (removed), wikipedia + grokipedia (typeahead junk).
DDGS_BACKENDS = os.getenv("DDGS_BACKENDS") or "startpage,yahoo,duckduckgo,brave,mojeek"
_TIMELIMIT = {"day": "d", "week": "w", "month": "m", "year": "y"}

_STOP = frozenset(
    "the a an of to in on for and or is are was were be been it this that with by from at as "
    "latest new news today current about what which who how why when where does do did can "
    "kya hai hain ka ki ke ko se me mein par aur ya ye wo koi kaun kaise kab kyun batao dekho "
    "search web karke kar find please".split()
)

_down_until: Dict[str, float] = {}
_cache: Dict[str, Tuple[str, float]] = {}
_lock = threading.Lock()
CACHE_TTL = 600


# Providers that are real search engines / agent-search APIs: their ranking IS the relevance
# signal. The prod regression (2026-10-12): Tavily AND Serper both returned good results for a
# 13-word LLM-hardware query and my gate rejected them ("needs 6 of 11 literal words in the
# snippets"), so the model answered from memory. The gate now only polices the keyless ddgs
# path, where one encyclopedia page answering everything ("Europe") was the real failure.
TRUSTED = frozenset({"tavily", "serper", "gemini"})

_WORD_RE = re.compile(r"[a-z0-9\+\#\.]{2,}")


def _tokens(q: str) -> List[str]:
    return [t for t in _WORD_RE.findall((q or "").lower()) if t not in _STOP]


def _stem(t: str) -> str:
    return t[:6] if len(t) >= 7 else t


def relevant(results: List[dict], query: str) -> bool:
    """Lenient topical check for UNTRUSTED results. A hit counts when it shares at least a
    third of the query's significant words (min 1, max 3) as WHOLE WORDS (so "eu" can no
    longer match "Europe", nor "ai" match "main"). Needs 2 such hits among the top 4
    (1 if fewer than 3 results). A provider score >= 0.5 short-circuits to relevant."""
    if not results:
        return False
    if any((r.get("score") or 0) >= 0.5 for r in results):
        return True
    toks = _tokens(query)
    if not toks:
        return True
    need = min(3, max(1, -(-len(toks) // 3)))
    good = 0
    for r in results[:4]:
        words = {_stem(w) for w in _WORD_RE.findall(f"{r.get('title', '')} {r.get('snippet', '')}".lower())}
        if sum(1 for t in toks if _stem(t) in words) >= need:
            good += 1
    return good >= (2 if len(results) >= 3 else 1)


def _open(name: str) -> bool:
    return _down_until.get(name, 0) <= time.time()


def _trip(name: str, secs: int, why: str) -> None:
    _down_until[name] = time.time() + secs
    log.warning("web_search provider %s benched %ss (%s)", name, secs, why)


# ----------------------------------------------------------------------------- providers
def _tavily(query: str, num: int, recency: str, topic: str) -> Optional[dict]:
    key = (os.getenv("TAVILY_API_KEY") or "").strip()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    elif (os.getenv("TAVILY_KEYLESS") or "1").strip().lower() in ("1", "true", "yes", "on"):
        headers["X-Tavily-Access-Mode"] = "keyless"
    else:
        return None
    body: Dict[str, Any] = {
        "query": query, "max_results": num, "search_depth": "basic",
        "topic": "news" if topic == "news" else "general", "include_answer": True,
    }
    if recency in _TIMELIMIT:
        body["time_range"] = recency
    r = requests.post("https://api.tavily.com/search", json=body, headers=headers, timeout=12)
    if r.status_code == 400 and "time_range" in body:       # older API: only valid with news
        body.pop("time_range")
        r = requests.post("https://api.tavily.com/search", json=body, headers=headers, timeout=12)
    if r.status_code in (401, 403):
        _trip("tavily", 3600, f"HTTP {r.status_code}")
        return None
    if r.status_code in (429, 432, 433):                    # rate / plan limit
        _trip("tavily", 6 * 3600 if r.status_code != 429 else 300, f"HTTP {r.status_code}")
        return None
    r.raise_for_status()
    d = r.json() or {}
    res = [
        {"title": x.get("title", ""), "url": x.get("url", ""),
         "snippet": (x.get("content") or "")[:320], "score": x.get("score")}
        for x in (d.get("results") or [])
    ]
    return {"results": res, "answer": (d.get("answer") or "").strip()}


def _serper(query: str, num: int, recency: str, topic: str) -> Optional[dict]:
    key = (os.getenv("SERPER_API_KEY") or "").strip()
    if not key:
        return None
    body: Dict[str, Any] = {"q": query, "num": num}
    if recency in _TIMELIMIT:
        body["tbs"] = f"qdr:{_TIMELIMIT[recency]}"
    ep = "news" if topic == "news" else "search"
    r = requests.post(f"https://google.serper.dev/{ep}", json=body,
                      headers={"X-API-KEY": key, "Content-Type": "application/json"}, timeout=12)
    if r.status_code in (401, 403):
        _trip("serper", 3600, f"HTTP {r.status_code}")
        return None
    if r.status_code == 429:
        _trip("serper", 600, "429")
        return None
    r.raise_for_status()
    d = r.json() or {}
    items = d.get("news") if ep == "news" else d.get("organic")
    res = [
        {"title": x.get("title", ""), "url": x.get("link", ""), "snippet": (x.get("snippet") or "")[:320],
         "score": None}
        for x in (items or [])[:num]
    ]
    ab = d.get("answerBox") or {}
    return {"results": res, "answer": str(ab.get("answer") or ab.get("snippet") or "").strip()}


def _gemini(query: str, num: int, recency: str, topic: str) -> Optional[dict]:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        return None
    model = (os.getenv("GEMINI_SEARCH_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.6-flash").strip()
    when = {"day": " from the last 24 hours", "week": " from the last week",
            "month": " from the last month", "year": " from the last year"}.get(recency, "")
    prompt = (
        f"Search the web{when} and answer concisely with concrete facts, names, numbers and dates. "
        f"If the search finds nothing credible, say so plainly.\n\nQuery: {query}"
    )
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key},
        json={"contents": [{"parts": [{"text": prompt}]}], "tools": [{"google_search": {}}],
              "generationConfig": {"temperature": 0.1, "maxOutputTokens": 900}},
        timeout=25,
    )
    if r.status_code in (400, 401, 403, 404):
        _trip("gemini", 3600, f"HTTP {r.status_code}: {r.text[:100]}")
        return None
    if r.status_code == 429:
        _trip("gemini", 900, "429")
        return None
    r.raise_for_status()
    cand = ((r.json() or {}).get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts") or []).strip()
    chunks = (cand.get("groundingMetadata") or {}).get("groundingChunks") or []
    res = [
        {"title": (c.get("web") or {}).get("title", ""), "url": (c.get("web") or {}).get("uri", ""),
         "snippet": "", "score": 1.0}
        for c in chunks[:num] if c.get("web")
    ]
    if not text:
        return None
    return {"results": res, "answer": text, "grounded": True}


def _ddgs(query: str, num: int, recency: str, topic: str) -> Optional[dict]:
    from ddgs import DDGS  # type: ignore
    kw: Dict[str, Any] = {"max_results": num, "backend": DDGS_BACKENDS}
    if recency in _TIMELIMIT:
        kw["timelimit"] = _TIMELIMIT[recency]
    try:
        raw = list(DDGS(timeout=8).text(query, **kw)) or []
    except TypeError:
        raw = list(DDGS().text(query, max_results=num)) or []
    res = [
        {"title": x.get("title", ""), "url": x.get("href") or x.get("url", ""),
         "snippet": (x.get("body") or "")[:320], "score": None}
        for x in raw
    ]
    return {"results": res, "answer": ""}


PROVIDERS = (("tavily", _tavily), ("serper", _serper), ("gemini", _gemini), ("ddgs", _ddgs))


# ----------------------------------------------------------------------------- public API
def format_results(name: str, query: str, data: dict, num: int) -> str:
    lines = [f"Source: web_search ({name}) | query={query!r}"]
    if data.get("answer"):
        lines.append(("Grounded summary: " if data.get("grounded") else "Answer: ") + data["answer"][:1400])
    shown = 0
    for r in data.get("results", [])[:num]:
        if not (r.get("title") or r.get("snippet")):
            continue
        head = f"• {r.get('title', '').strip() or r.get('url', '')}"
        if r.get("snippet"):
            head += f"\n  {r['snippet'].strip()}"
        if r.get("url"):
            head += f"\n  {r['url']}"
        lines.append(head)
        shown += 1
    return "\n\n".join(lines) if (shown or data.get("answer")) else ""


NO_RESULTS = (
    "NO RELIABLE RESULTS: the search providers returned nothing relevant to this query "
    "({why}). Tell the user plainly that the web search found nothing usable. Do NOT answer "
    "from memory as if these were search results; if you share background knowledge, label it "
    "clearly as 'from my own knowledge, not from the search'."
)


def current_year() -> int:
    return time.localtime().tm_year


def clean_query(query: str, recency: str = "") -> str:
    """Models write stale years into 'latest' queries ('...models 2024' in Oct 2026). A past
    year on a recency-style query poisons the results, so drop it (recency does the work)."""
    q = (query or "").strip()
    wants_latest = bool(recency) or bool(
        re.search(r"(?i)\b(latest|newest|current|recent|up[- ]to[- ]date|today|this (week|month|year)|now)\b", q)
    )
    if wants_latest:
        yr = current_year()
        q = re.sub(r"\b(?:19|20)\d{2}\b", lambda m: "" if int(m.group(0)) < yr else m.group(0), q)
    return re.sub(r"\s{2,}", " ", q).strip()


def search_data(query: str, num: int = 5, recency: str = "", topic: str = "general"):
    """-> (provider_name, data, why_log) ; data is None when nothing usable. Never raises."""
    query = clean_query(query, recency)
    if not query:
        return None, None, ["empty query"]
    ck = f"data|{query.lower()}|{num}|{recency}|{topic}"
    with _lock:
        hit = _cache.get(ck)
        if hit and hit[1] > time.time():
            return hit[0]
    why: List[str] = []
    for name, fn in PROVIDERS:
        if not _open(name):
            why.append(f"{name}:benched")
            continue
        try:
            data = fn(query, num, recency, topic)
        except ImportError:
            why.append(f"{name}:not installed")
            continue
        except Exception as e:
            if name != "ddgs":
                _trip(name, 120, str(e)[:80])
            why.append(f"{name}:{str(e)[:50]}")
            log.warning("web_search %s failed: %s", name, str(e)[:160])
            continue
        if data is None:
            why.append(f"{name}:skipped")
            continue
        results = data.get("results", [])
        if not (data.get("answer") or results):
            why.append(f"{name}:empty")
            continue
        if name not in TRUSTED and not relevant(results, query):
            why.append(f"{name}:off-topic")
            log.info("web_search %s returned off-topic results for %r", name, query)
            continue
        log.info("web_search OK via %s q=%r hits=%d", name, query, len(results))
        out = (name, data, why)
        with _lock:
            _cache[ck] = ((out), time.time() + CACHE_TTL)
        return out
    return None, None, why


def search(query: str, num: int = 5, recency: str = "", topic: str = "general") -> str:
    """Formatted observation, or the NO_RESULTS notice. Never raises."""
    query = (query or "").strip()
    if not query:
        return "Empty query."
    name, data, why = search_data(query, num, recency, topic)
    if data:
        out = format_results(name, clean_query(query, recency), data, num)
        if out:
            return out
    return NO_RESULTS.format(why="; ".join((why or [])[:6]) or "no provider configured")


def extract_pages(urls: List[str], max_chars: int = 3500) -> Dict[str, str]:
    """Clean page text for the given URLs via Tavily /extract (1 credit per 5 URLs, basic depth).
    Returns {url: text}; empty when no key or on failure (callers fall back to browse_url)."""
    key = (os.getenv("TAVILY_API_KEY") or "").strip()
    urls = [u for u in urls if u][:5]
    if not key or not urls or not _open("tavily_extract"):
        return {}
    try:
        r = requests.post(
            "https://api.tavily.com/extract",
            json={"urls": urls, "extract_depth": "basic"},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=25,
        )
        if r.status_code in (401, 403, 432, 433):
            _trip("tavily_extract", 3600, f"HTTP {r.status_code}")
            return {}
        r.raise_for_status()
        out = {}
        for x in (r.json() or {}).get("results", []):
            txt = re.sub(r"\s+", " ", (x.get("raw_content") or "")).strip()
            if txt:
                out[x.get("url", "")] = txt[:max_chars]
        return out
    except Exception as e:
        log.warning("tavily extract failed: %s", str(e)[:120])
        return {}

"""
github_ops.py — what a GitHub link actually CONTAINS, not just its star count.

Prod bug (2026-10-11): user forwarded a post about github.com/mattpocock/skills and said
"List all of them". The old tool returned only metadata (stars, forks, URL), so the model had
nothing to list and answered "The tool result contains: URL: ...".

Now for a repo URL we return, within one WhatsApp-sized observation (<= ~3.4k chars):
  1. header: name, description, stars, language, last push
  2. CATALOG: auto-detected collections — every dir holding SKILL.md (agent skills), plus
     monorepo packages — grouped by category, ALL names (that is what "list all of them" needs)
  3. top-level entries
  4. README excerpt with the remaining budget (descriptions usually live there)
Also handles /tree/<ref>/<path> (directory listing) and /blob/<ref>/<path> (file text).

GitHub REST API; unauthenticated = 60 req/h per IP (shared on Render!), so GITHUB_TOKEN is
used when set (5,000/h) and every answer is cached for 10 minutes.
"""
from __future__ import annotations

import base64
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import requests

API = "https://api.github.com"
_CACHE: Dict[str, Tuple[str, float]] = {}
_TTL = 600
_SKIP_DIRS = {"node_modules", ".git", ".github", "dist", "build", "vendor", "__pycache__"}


def _headers(raw: bool = False) -> dict:
    h = {
        "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json",
        "User-Agent": "MojoAgent/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    tok = (os.getenv("GITHUB_TOKEN") or "").strip()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def parse_github_url(url: str) -> Optional[dict]:
    """-> {owner, repo, kind: repo|tree|blob, ref, path} for repo-level URLs, else None."""
    m = re.match(
        r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?"
        r"(?:/(tree|blob)/([^/?#]+)(?:/([^?#]*))?)?(?:[/?#]|$)",
        (url or "").strip(),
    )
    if not m:
        return None
    owner, repo, kind, ref, path = m.groups()
    if repo.lower() in ("", "repositories", "stars", "followers", "following", "sponsors"):
        return None
    return {
        "owner": owner, "repo": repo, "kind": kind or "repo",
        "ref": ref or "", "path": (path or "").strip("/"),
    }


def _get(url: str, raw: bool = False, params: Optional[dict] = None):
    return requests.get(url, headers=_headers(raw), params=params, timeout=15)


def _rate_note(r) -> str:
    if r.status_code in (403, 429) and r.headers.get("x-ratelimit-remaining") == "0":
        return ("GitHub API rate limit hit (60/h unauthenticated). "
                "Set GITHUB_TOKEN to raise it to 5,000/h.")
    return ""


def _catalog(paths: List[str]) -> List[str]:
    """Group SKILL.md directories by category: skills/<category>/<name>/SKILL.md."""
    skills: Dict[str, str] = {}  # name -> category (shortest path wins, avoids copy dirs)
    depth: Dict[str, int] = {}
    for p in paths:
        parts = p.split("/")
        if parts[-1] != "SKILL.md" or any(x in _SKIP_DIRS or x.startswith(".") for x in parts[:-1]):
            continue
        if len(parts) < 2:
            continue
        name = parts[-2]
        cat = parts[-3] if len(parts) >= 3 else ""
        if name not in skills or len(parts) < depth[name]:
            skills[name], depth[name] = cat, len(parts)
    lines: List[str] = []
    if skills:
        groups: Dict[str, List[str]] = {}
        for n, c in sorted(skills.items()):
            groups.setdefault(c or "(root)", []).append(n)
        lines.append(f"SKILLS ({len(skills)}):")
        for c in sorted(groups):
            lines.append(f"• {c} ({len(groups[c])}): " + ", ".join(groups[c]))
    pk = sorted({
        p.split("/")[1] for p in paths
        if re.fullmatch(r"(packages|apps|plugins|libs)/[^/]+/(package\.json|plugin\.json|pyproject\.toml)", p)
    })
    if pk:
        lines.append(f"PACKAGES ({len(pk)}): " + ", ".join(pk))
    return lines


def _top_level(tree: List[dict]) -> str:
    dirs = sorted(t["path"] for t in tree if t.get("type") == "tree" and "/" not in t["path"]
                  and t["path"] not in _SKIP_DIRS)
    files = sorted(t["path"] for t in tree if t.get("type") == "blob" and "/" not in t["path"])
    out = []
    if dirs:
        out.append("DIRS: " + ", ".join(d + "/" for d in dirs[:30]))
    if files:
        out.append("FILES: " + ", ".join(files[:25]))
    return "\n".join(out)


def _clip(text: str, n: int) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    cut = text[:n]
    nl = cut.rfind("\n")
    return (cut[:nl] if nl > n * 0.6 else cut).rstrip() + " …"


def _clean_readme(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"<[^>]+>", "", md)                       # html badges/images
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)           # markdown images
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)       # links -> text
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def overview(owner: str, repo: str, kind: str = "repo", ref: str = "", path: str = "",
             max_chars: int = 3400) -> str:
    key = f"{owner}/{repo}|{kind}|{ref}|{path}"
    hit = _CACHE.get(key)
    if hit and hit[1] > time.time():
        return hit[0]
    try:
        out = _overview(owner, repo, kind, ref, path, max_chars)
    except Exception as e:
        return f"GitHub API error for {owner}/{repo}: {e}"
    if not out.startswith(("GitHub API error", "Repo ", "GitHub API rate")):
        _CACHE[key] = (out, time.time() + _TTL)
    return out


def _overview(owner, repo, kind, ref, path, max_chars) -> str:
    r = _get(f"{API}/repos/{owner}/{repo}")
    if r.status_code == 404:
        return f"Repo {owner}/{repo} not found (or private)."
    if r.status_code != 200:
        return _rate_note(r) or f"GitHub API error {r.status_code} for {owner}/{repo}."
    meta = r.json()
    branch = ref or meta.get("default_branch") or "main"
    head = (
        f"Repo: {meta.get('full_name')} — {meta.get('description') or '(no description)'}\n"
        f"★ {meta.get('stargazers_count', 0)} · forks {meta.get('forks_count', 0)} · "
        f"{meta.get('language') or '?'} · last push {(meta.get('pushed_at') or '')[:10]} · "
        f"branch {branch}"
    )

    # ---- file / directory deep links
    if kind == "blob" and path:
        fr = _get(f"{API}/repos/{owner}/{repo}/contents/{path}", raw=True, params={"ref": branch})
        if fr.status_code != 200:
            return head + f"\n\nCould not read file {path} (HTTP {fr.status_code})."
        return head + f"\n\nFILE {path}:\n" + _clip(fr.text, max_chars - len(head) - 40)
    if kind == "tree" and path:
        dr = _get(f"{API}/repos/{owner}/{repo}/contents/{path}", params={"ref": branch})
        if dr.status_code != 200 or not isinstance(dr.json(), list):
            return head + f"\n\nCould not list {path} (HTTP {dr.status_code})."
        items = dr.json()
        dirs = sorted(i["name"] + "/" for i in items if i.get("type") == "dir")
        files = sorted(i["name"] for i in items if i.get("type") == "file")
        body = f"\n\n{path}/ — {len(dirs)} folders, {len(files)} files:\n"
        body += ("FOLDERS: " + ", ".join(dirs) + "\n") if dirs else ""
        body += ("FILES: " + ", ".join(files)) if files else ""
        return head + _clip(body, max_chars - len(head))

    # ---- repo root: tree + README
    tr = _get(f"{API}/repos/{owner}/{repo}/git/trees/{branch}", params={"recursive": "1"})
    tree = (tr.json().get("tree") or []) if tr.status_code == 200 else []
    truncated = bool(tr.status_code == 200 and tr.json().get("truncated"))
    paths = [t["path"] for t in tree if t.get("type") == "blob"]

    parts = [head]
    cat = _catalog(paths)
    if cat:
        parts.append("\n".join(cat))
    top = _top_level(tree)
    if top:
        parts.append(top)
    if truncated:
        parts.append("(repo tree truncated by GitHub — catalog may be incomplete)")

    used = sum(len(p) for p in parts) + 2 * len(parts)
    budget = max(500, max_chars - used - 20)
    rr = _get(f"{API}/repos/{owner}/{repo}/readme", raw=True)
    if rr.status_code == 200 and rr.text.strip():
        parts.append("README:\n" + _clip(_clean_readme(rr.text), budget))
    elif tr.status_code != 200:
        parts.append(_rate_note(tr) or "(could not read file tree or README)")
    return "\n\n".join(parts)[: max_chars + 400]

"""
github_ops.py (v2) — what a GitHub link actually CONTAINS, robust to API rate limits.

History of failures this fixes (prod logs 2026-10-11/12):
  1. "List all of them" on github.com/<user>/skills returned only stars/forks: the API tree and
     README calls were rate-limited (60 req/h per IP, shared on Render) so only the first
     metadata call succeeded.
  2. After GITHUB_TOKEN was added, profile URLs still failed with "403 rate limit": the old
     profile-repos function built its own request and never sent the token.
  3. "Every detail from the README" was impossible: README excerpts only, no way to page.
  4. Raw exception text ("403 Client Error: rate limit exceeded for url: https://api...")
     was shown to the user.

Strategy (every path degrades gracefully; the token is used wherever the API is used):
  repo overview : API metadata (best effort) + the repo ZIP ARCHIVE from codeload (not API-limited):
                  tree, SKILL.md frontmatter (name + description of EVERY skill), packages,
                  top-level listing, README — all from one ~0.4 MB download.
                  Fallback: API tree + raw.githubusercontent README.
  profile repos : API (token) -> HTML scrape of github.com/<user>?tab=repositories (paginated).
  files         : raw.githubusercontent.com with offset paging ("chars 0-6000 of 21000").
Results cached 6 h (memory + Supabase kv) so repeat questions cost nothing.
"""
from __future__ import annotations

import html as _html
import io
import os
import re
import time
import zipfile
from typing import Dict, List, Optional, Tuple

import requests

import usage_budget

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"
_CACHE: Dict[str, Tuple[str, float]] = {}
_TTL = 6 * 3600
_SKIP_DIRS = {"node_modules", ".git", ".github", "dist", "build", "vendor", "__pycache__", ".venv", "venv"}
MAX_ZIP_BYTES = int(os.getenv("GITHUB_MAX_ZIP_MB") or "25") * 1024 * 1024
UA = "MojoAgent/2.0 (+https://github.com)"


# ----------------------------------------------------------------------------- http
def _headers(raw: bool = False, token: bool = True) -> dict:
    h = {
        "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json",
        "User-Agent": UA,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    tok = (os.getenv("GITHUB_TOKEN") or "").strip()
    if tok and token:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _http_get(url: str, raw: bool = False, params: Optional[dict] = None, timeout: int = 15):
    """Single choke point for GET requests (tests monkeypatch this). The token is attached
    ONLY to api.github.com, never to raw/html hosts."""
    return requests.get(url, headers=_headers(raw, token=url.startswith(API)), params=params, timeout=timeout)


def _rate_limited(r) -> bool:
    if r is None:
        return False
    if r.status_code == 429:
        return True
    return r.status_code == 403 and (
        r.headers.get("x-ratelimit-remaining") == "0" or "rate limit" in (getattr(r, "text", "") or "").lower()
    )


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
    return {"owner": owner, "repo": repo, "kind": kind or "repo", "ref": ref or "", "path": (path or "").strip("/")}


# ----------------------------------------------------------------------------- cache
def _cache_get(key: str) -> Optional[str]:
    hit = _CACHE.get(key)
    if hit and hit[1] > time.time():
        return hit[0]
    v = usage_budget.kv_get("gh2:" + key)
    if v:
        _CACHE[key] = (v, time.time() + 600)
    return v


def _cache_set(key: str, value: str) -> None:
    _CACHE[key] = (value, time.time() + _TTL)
    usage_budget.kv_set("gh2:" + key, value, _TTL)


def _clip(text: str, n: int) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    cut = text[:n]
    nl = cut.rfind("\n")
    return (cut[:nl] if nl > n * 0.6 else cut).rstrip() + " …"


def _clean_readme(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"<(?:details|summary)[^>]*>|</(?:details|summary)>", "", md)
    md = re.sub(r"<img[^>]*>|<br\s*/?>", "", md)
    md = re.sub(r"<[^>]+>", "", md)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)                    # images / badges
    md = re.sub(r"\[([^\]]+)\]\((?!http)[^)]*\)", r"\1", md)        # relative links -> text
    return re.sub(r"\n{3,}", "\n\n", md).strip()


# ----------------------------------------------------------------------------- archive parsing
def fetch_zip(owner: str, repo: str, ref: str = "") -> Optional[bytes]:
    """Repo archive from codeload (NOT subject to the REST API limit). None on failure/too big."""
    url = f"https://codeload.github.com/{owner}/{repo}/zip/{ref or 'HEAD'}"
    try:
        with requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=40, allow_redirects=True) as r:
            if r.status_code != 200:
                return None
            clen = int(r.headers.get("content-length") or 0)
            if clen and clen > MAX_ZIP_BYTES:
                return None
            buf = io.BytesIO()
            for chunk in r.iter_content(65536):
                buf.write(chunk)
                if buf.tell() > MAX_ZIP_BYTES:
                    return None
            return buf.getvalue()
    except Exception:
        return None


def _frontmatter(text: str) -> Dict[str, str]:
    m = re.match(r"\A\ufeff?---\s*\n(.*?)\n---", text, flags=re.S)
    if not m:
        return {}
    out: Dict[str, str] = {}
    lines = m.group(1).split("\n")
    i = 0
    while i < len(lines):
        km = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", lines[i])
        if km:
            key, val = km.group(1).lower(), km.group(2).strip()
            if val in (">", "|", ">-", "|-", ">+", "|+"):
                block = []
                while i + 1 < len(lines) and (lines[i + 1].startswith((" ", "\t")) or not lines[i + 1].strip()):
                    i += 1
                    block.append(lines[i].strip())
                val = " ".join(b for b in block if b)
            out[key] = val.strip().strip("\"'")
        i += 1
    return out


def read_archive(data: bytes) -> dict:
    """-> {paths, skills:{name:{cat,desc,path}}, readme, packages}. Pure function (tested)."""
    z = zipfile.ZipFile(io.BytesIO(data))
    all_names = z.namelist()
    root = all_names[0].split("/")[0] if all_names else ""
    pairs = [(n[len(root) + 1:], n) for n in all_names
             if not n.endswith("/") and "__MACOSX" not in n and n.startswith(root + "/")]
    paths = [p for p, _ in pairs]
    by_path = dict(pairs)

    skills: Dict[str, dict] = {}
    for p in paths:
        parts = p.split("/")
        if parts[-1] != "SKILL.md" or len(parts) < 2:
            continue
        if any(x in _SKIP_DIRS or x.startswith(".") for x in parts[:-1]):
            continue
        name, cat = parts[-2], (parts[-3] if len(parts) >= 3 else "")
        if name in skills and len(parts) >= len(skills[name]["path"].split("/")):
            continue  # keep the shallowest copy (ignores vendored duplicates)
        try:
            head = z.read(by_path[p])[:4000].decode("utf-8", "replace")
        except Exception:
            head = ""
        fm = _frontmatter(head)
        desc = fm.get("description") or ""
        if not desc:  # no frontmatter: first prose line after the heading
            for ln in head.split("\n"):
                ln = ln.strip()
                if ln and not ln.startswith(("#", "---", "name:")):
                    desc = ln
                    break
        skills[name] = {"cat": cat, "desc": re.sub(r"\s+", " ", desc).strip(), "path": p,
                        "title": fm.get("name") or name}

    readme = ""
    for p in sorted(paths, key=lambda x: (x.count("/"), x)):
        if "/" not in p and re.fullmatch(r"(?i)readme(\.(md|markdown|rst|txt))?", p):
            try:
                readme = z.read(by_path[p])[:80000].decode("utf-8", "replace")
            except Exception:
                readme = ""
            break

    packages = sorted({
        p.split("/")[1] for p in paths
        if re.fullmatch(r"(packages|apps|plugins|libs)/[^/]+/(package\.json|plugin\.json|pyproject\.toml)", p)
    })
    return {"paths": paths, "skills": skills, "readme": readme, "packages": packages}


def _top_level(paths: List[str]) -> Tuple[List[str], List[str]]:
    dirs = sorted({p.split("/")[0] for p in paths if "/" in p} - _SKIP_DIRS)
    files = sorted(p for p in paths if "/" not in p)
    return dirs, files


def _catalog_lines(skills: Dict[str, dict], budget: int) -> List[str]:
    """Every skill, grouped by category, with its description, shrinking descriptions to fit."""
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for n, v in sorted(skills.items()):
        groups.setdefault(v["cat"] or "(root)", []).append((n, v["desc"]))
    lines: List[str] = []
    for dlen in (110, 80, 55, 30, 0):
        lines = [f"SKILLS ({len(skills)}):"]
        for cat in sorted(groups):
            lines.append(f"{cat} ({len(groups[cat])}):")
            for n, d in groups[cat]:
                d = (d[: dlen - 1].rstrip() + "…") if dlen and len(d) > dlen else (d if dlen else "")
                lines.append(f"- {n}" + (f" — {d}" if d else ""))
        if sum(len(x) + 1 for x in lines) <= budget:
            return lines
    return lines


def _compose(meta: Optional[dict], branch: str, arch: dict, owner: str, repo: str, max_chars: int) -> str:
    if meta:
        head = (
            f"Repo: {meta.get('full_name')} — {meta.get('description') or '(no description)'}\n"
            f"★ {meta.get('stargazers_count', 0)} · forks {meta.get('forks_count', 0)} · "
            f"{meta.get('language') or '?'} · last push {(meta.get('pushed_at') or '')[:10]} · "
            f"branch {meta.get('default_branch') or branch}"
        )
    else:
        head = f"Repo: {owner}/{repo} (details from the repo archive)"
    parts = [head]
    paths = arch["paths"]
    if arch["skills"]:
        parts.append("\n".join(_catalog_lines(arch["skills"], int(max_chars * 0.62))))
    if arch["packages"]:
        parts.append(f"PACKAGES ({len(arch['packages'])}): " + ", ".join(arch["packages"]))
    dirs, files = _top_level(paths)
    top = []
    if dirs:
        top.append("DIRS: " + ", ".join(d + "/" for d in dirs[:40]) + (f" (+{len(dirs) - 40} more)" if len(dirs) > 40 else ""))
    if files:
        top.append(f"FILES ({len(files)}): " + ", ".join(files[:60]) + (f" (+{len(files) - 60} more)" if len(files) > 60 else ""))
    top.append(f"TOTAL: {len(paths)} files · README: {'yes' if arch['readme'] else 'NO'}")
    parts.append("\n".join(top))
    used = sum(len(p) for p in parts) + 2 * len(parts)
    budget = max(700, max_chars - used - 40)
    if arch["readme"]:
        readme = _clean_readme(arch["readme"])
        note = ""
        if len(readme) > budget:
            note = (f"\n[README is {len(readme)} chars; showing the first {budget}. Full text: browse_url on "
                    f"https://github.com/{owner}/{repo}/blob/HEAD/README.md with offset=0, then offset={budget}, …]")
        parts.append("README:\n" + _clip(readme, budget) + note)
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- repo overview
def overview(owner: str, repo: str, kind: str = "repo", ref: str = "", path: str = "",
             max_chars: int = 7000, offset: int = 0) -> str:
    key = f"{owner}/{repo}|{kind}|{ref}|{path}|{offset}|{max_chars}"
    cached = _cache_get(key)
    if cached:
        return cached
    try:
        out = _overview(owner, repo, kind, ref, path, max_chars, offset)
    except Exception as e:  # never leak a raw exception to the user
        return f"Couldn't read {owner}/{repo} from GitHub right now ({type(e).__name__}). Try again in a minute."
    if not out.startswith(("Couldn't", "Repo not found", "File not found", "Folder not found")):
        _cache_set(key, out)
    return out


def _meta(owner: str, repo: str) -> Tuple[Optional[dict], str]:
    try:
        r = _http_get(f"{API}/repos/{owner}/{repo}")
    except Exception:
        return None, ""
    if r.status_code == 404:
        return None, "404"
    if r.status_code == 200:
        return r.json(), ""
    return None, "rate" if _rate_limited(r) else str(r.status_code)


def read_file(owner: str, repo: str, ref: str, path: str, offset: int = 0, max_chars: int = 6000) -> str:
    """File text via raw.githubusercontent (no API limit) with offset paging."""
    r = None
    for candidate in ([ref] if ref else []) + ["HEAD", "main", "master"]:
        try:
            r = _http_get(f"{RAW}/{owner}/{repo}/{candidate}/{path}", timeout=20)
        except Exception:
            r = None
            continue
        if r.status_code == 200:
            break
    if r is None or r.status_code != 200:
        return f"File not found: {owner}/{repo}/{path} (is the path right, and is the repo public?)"
    text = r.text
    total = len(text)
    offset = max(0, int(offset or 0))
    chunk = text[offset: offset + max_chars]
    more = ""
    if offset + max_chars < total:
        more = (f"\n[chars {offset}-{offset + len(chunk)} of {total}. For the rest call browse_url again with "
                f"offset={offset + len(chunk)}]")
    return f"FILE {owner}/{repo}/{path} ({total} chars):\n{chunk}{more}"


def _overview(owner, repo, kind, ref, path, max_chars, offset) -> str:
    if kind == "blob" and path:
        return read_file(owner, repo, ref, path, offset, min(max_chars, 9000))

    meta, why = _meta(owner, repo)
    if why == "404":
        return f"Repo not found: {owner}/{repo} (does it exist, and is it public?)"
    branch = ref or (meta or {}).get("default_branch") or "HEAD"

    data = fetch_zip(owner, repo, ref)
    arch = None
    if data:
        try:
            arch = read_archive(data)
        except Exception:
            arch = None

    if arch is None:  # archive unavailable/too big -> API tree + raw README
        paths: List[str] = []
        try:
            tr = _http_get(f"{API}/repos/{owner}/{repo}/git/trees/{branch}", params={"recursive": "1"})
            if tr.status_code == 200:
                paths = [t["path"] for t in (tr.json().get("tree") or []) if t.get("type") == "blob"]
        except Exception:
            pass
        readme = ""
        for name in ("README.md", "readme.md", "README.rst", "README"):
            try:
                rr = _http_get(f"{RAW}/{owner}/{repo}/{branch}/{name}", timeout=20)
                if rr.status_code == 200:
                    readme = rr.text
                    break
            except Exception:
                continue
        if not paths and not readme and not meta:
            return (f"Couldn't read {owner}/{repo} from GitHub right now "
                    f"({'rate limit' if why == 'rate' else 'network error'}). Try again in a minute.")
        skills: Dict[str, dict] = {}
        for p in paths:
            parts = p.split("/")
            if parts[-1] == "SKILL.md" and len(parts) >= 2 and not any(x in _SKIP_DIRS or x.startswith(".") for x in parts[:-1]):
                skills.setdefault(parts[-2], {"cat": parts[-3] if len(parts) >= 3 else "", "desc": "",
                                              "path": p, "title": parts[-2]})
        arch = {"paths": paths, "skills": skills, "readme": readme, "packages": []}

    if kind == "tree" and path:
        sub = [p[len(path) + 1:] for p in arch["paths"] if p.startswith(path + "/")]
        if not sub:
            return f"Folder not found: {path} in {owner}/{repo}."
        dirs = sorted({p.split("/")[0] for p in sub if "/" in p})
        files = sorted(p for p in sub if "/" not in p)
        body = f"{owner}/{repo}/{path}/ — {len(dirs)} folders, {len(files)} files here, {len(sub)} files in total:\n"
        if dirs:
            body += "FOLDERS: " + ", ".join(d + "/" for d in dirs[:80]) + "\n"
        if files:
            body += "FILES: " + ", ".join(files[:80])
        return _clip(body, max_chars)

    return _compose(meta, branch, arch, owner, repo, max_chars)


# ----------------------------------------------------------------------------- profile repos
def _scrape_repos(user: str, pages: int = 4) -> Optional[List[dict]]:
    repos: List[dict] = []
    for page in range(1, pages + 1):
        try:
            r = _http_get(f"https://github.com/{user}", params={"tab": "repositories", "page": page}, timeout=20)
        except Exception:
            break
        if r.status_code == 404:
            return None if page == 1 else repos
        if r.status_code != 200:
            break
        h = r.text
        blocks = h.split('itemprop="owns"')[1:]
        if not blocks:
            break
        for b in blocks:
            m = re.search(r'itemprop="name codeRepository"[^>]*>\s*([^<]+?)\s*<', b)
            if not m:
                continue
            desc = re.search(r'itemprop="description"[^>]*>\s*(.*?)\s*</p>', b, re.S)
            lang = re.search(r'itemprop="programmingLanguage">([^<]+)<', b)
            stars = re.search(r'/stargazers"[^>]*>(.*?)</a>', b, re.S)
            when = re.search(r'<relative-time[^>]*datetime="([^"]+)"', b)
            repos.append({
                "name": _html.unescape(m.group(1)),
                "description": _html.unescape(re.sub(r"<[^>]+>|\s+", " ", desc.group(1))).strip() if desc else "",
                "language": lang.group(1) if lang else "",
                "stars": int(re.sub(r"\D", "", re.sub(r"<[^>]+>", "", stars.group(1))) or 0) if stars else 0,
                "pushed_at": when.group(1)[:10] if when else "",
                "fork": "Forked from" in b,
            })
        if 'rel="next"' not in h and 'class="next_page"' not in h:
            break
    return repos


def user_repos(user: str, limit: int = 100) -> str:
    """ALL public repos (up to `limit`) with description/language/stars/last push."""
    key = f"user|{user.lower()}|{limit}"
    cached = _cache_get(key)
    if cached:
        return cached
    repos: Optional[List[dict]] = None
    via = "API"
    try:
        collected: List[dict] = []
        for page in (1, 2):
            r = _http_get(f"{API}/users/{user}/repos",
                          params={"sort": "pushed", "direction": "desc", "per_page": 100, "page": page})
            if r.status_code == 404:
                return f"GitHub user '{user}' not found."
            if r.status_code != 200:
                raise RuntimeError("api")
            batch = r.json()
            collected += [{
                "name": x.get("name"), "description": (x.get("description") or "").strip(),
                "language": x.get("language") or "", "stars": x.get("stargazers_count", 0),
                "pushed_at": (x.get("pushed_at") or "")[:10], "fork": bool(x.get("fork")),
            } for x in batch]
            if len(batch) < 100:
                break
        repos = collected
    except Exception:
        repos = None
    if repos is None:  # rate-limited / blocked -> scrape the public HTML listing
        repos, via = _scrape_repos(user), "web"
        if repos is None:
            return f"GitHub user '{user}' not found."
    if not repos:
        return (f"Couldn't list repos for '{user}' from GitHub right now (rate limit/network). "
                "Try again in a minute, or send a specific repo link.")
    lines = [f"GitHub @{user} — {len(repos)} public repos (newest push first):"]
    for r in repos[:limit]:
        meta = " · ".join(x for x in (r["language"], f"★{r['stars']}" if r["stars"] else "", r["pushed_at"]) if x)
        d = (r["description"] or "no description")[:110]
        lines.append(f"- {r['name']}{' [fork]' if r['fork'] else ''} — {d}" + (f" ({meta})" if meta else ""))
    if len(repos) > limit:
        lines.append(f"(+{len(repos) - limit} more not shown)")
    out = "\n".join(lines)
    _cache_set(key, out)
    return out

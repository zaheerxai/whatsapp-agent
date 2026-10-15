"""Offline tests for github_ops v2. Run: python test_github_ops.py

Replays: (1) API rate-limited but the repo archive still yields all 38 skills; (2) profile URL
with the API blocked -> HTML fallback; (3) token only sent to api.github.com; (4) no raw
exception text reaches the user.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from types import SimpleNamespace as NS

import github_ops as g
import usage_budget

usage_budget._supabase = None
FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


class Resp:
    def __init__(self, status=200, j=None, text="", headers=None):
        self.status_code, self._j, self.text, self.headers = status, j, text, headers or {}

    def json(self):
        return self._j


def reset():
    g._CACHE.clear(); usage_budget._mem.clear()


# ---------------------------------------------------------------- url parsing
p = g.parse_github_url("https://github.com/mattpocock/skills")
check("parse repo", p == {"owner": "mattpocock", "repo": "skills", "kind": "repo", "ref": "", "path": ""}, p)
p = g.parse_github_url("github.com/o/r/tree/main/skills/eng")
check("parse tree", p["kind"] == "tree" and p["path"] == "skills/eng")
p = g.parse_github_url("https://github.com/o/r/blob/HEAD/README.md")
check("parse blob (HEAD ref)", p["kind"] == "blob" and p["ref"] == "HEAD" and p["path"] == "README.md")
check("profile -> None", g.parse_github_url("https://www.github.com/zaheerxai") is None)

# ---------------------------------------------------------------- archive parsing (synthetic)
def make_zip(files: dict) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for k, v in files.items():
            z.writestr("repo-main/" + k, v)
    return b.getvalue()

FILES = {
    "README.md": "# Repo\n\n![badge](x.png)\nHello <b>world</b> [link](docs/a.md)\n",
    "skills/eng/tdd/SKILL.md": "---\nname: tdd\ndescription: Test-driven development. Use when building features test-first.\n---\n# TDD\n",
    "skills/eng/review/SKILL.md": "---\nname: review\ndescription: >\n  Review a change along\n  two axes.\n---\n",
    "skills/misc/nofm/SKILL.md": "# Nofm\n\nA skill without frontmatter.\n",
    ".agents/skills/tdd/SKILL.md": "---\ndescription: VENDORED COPY\n---\n",
    "node_modules/x/SKILL.md": "---\ndescription: ignored\n---\n",
    "packages/cli/package.json": "{}", "packages/core/package.json": "{}",
    "LICENSE": "MIT",
}
arch = g.read_archive(make_zip(FILES))
check("archive: 3 real skills (vendored + node_modules ignored)", set(arch["skills"]) == {"tdd", "review", "nofm"}, arch["skills"].keys())
check("archive: frontmatter description", arch["skills"]["tdd"]["desc"].startswith("Test-driven development"))
check("archive: YAML folded (>) description joined", arch["skills"]["review"]["desc"] == "Review a change along two axes.", arch["skills"]["review"]["desc"])
check("archive: no frontmatter -> first prose line", arch["skills"]["nofm"]["desc"] == "A skill without frontmatter.")
check("archive: category from folder", arch["skills"]["tdd"]["cat"] == "eng" and arch["skills"]["nofm"]["cat"] == "misc")
check("archive: shallowest copy wins over vendored", arch["skills"]["tdd"]["path"] == "skills/eng/tdd/SKILL.md")
check("archive: monorepo packages", arch["packages"] == ["cli", "core"])
check("archive: README found", "Hello" in arch["readme"])

# ---------------------------------------------------------------- REAL repo (downloaded earlier), if present
if os.path.exists("/tmp/repo.zip"):
    real = g.read_archive(open("/tmp/repo.zip", "rb").read())
    check("REAL mattpocock/skills: all 38 skills found", len(real["skills"]) == 38, len(real["skills"]))
    check("REAL: every skill has a description", all(v["desc"] for v in real["skills"].values()))
    txt = g._compose(None, "main", real, "mattpocock", "skills", 7000)
    check("REAL: every one of the 38 names appears in the observation", all(f"- {n}" in txt for n in real["skills"]))
    check("REAL: observation fits the 7k budget", len(txt) <= 7600, len(txt))

# ---------------------------------------------------------------- overview: API rate-limited, archive works (THE prod bug)
META = {"full_name": "o/r", "description": "d", "stargazers_count": 5, "forks_count": 1, "language": "Python",
        "pushed_at": "2026-10-01T00:00:00Z", "default_branch": "main"}
reset()
calls = []
def api_limited(url, raw=False, params=None, timeout=15):
    calls.append(url)
    return Resp(403, text='{"message":"API rate limit exceeded"}', headers={"x-ratelimit-remaining": "0"})
g._http_get = api_limited
g.fetch_zip = lambda o, r, ref="": make_zip(FILES)
out = g.overview("o", "r")
check("API 403 rate limit: still returns the FULL catalog from the archive", "SKILLS (3)" in out and "- tdd" in out and "- review" in out, out[:200])
check("API 403: README included from archive", "README:" in out and "Hello world" in out)
check("API 403: no raw exception / URL leaked", "403" not in out and "api.github.com" not in out)
n = len(calls)
out2 = g.overview("o", "r")
check("second call served from cache (0 extra HTTP)", out2 == out and len(calls) == n)

# ---------------------------------------------------------------- overview: archive fails -> API tree + raw README
reset()
def api_ok(url, raw=False, params=None, timeout=15):
    if url.endswith("/repos/o/r"):
        return Resp(200, META)
    if "/git/trees/" in url:
        return Resp(200, {"tree": [{"path": "skills/eng/tdd/SKILL.md", "type": "blob"}, {"path": "README.md", "type": "blob"}]})
    if "raw.githubusercontent.com" in url and url.endswith("README.md"):
        return Resp(200, text="# raw readme")
    return Resp(404)
g._http_get = api_ok
g.fetch_zip = lambda o, r, ref="": None
out = g.overview("o", "r")
check("archive down: API tree + raw README fallback works", "tdd" in out and "raw readme" in out and "★ 5" in out, out)

# ---------------------------------------------------------------- everything down -> friendly message
reset()
def all_down(url, raw=False, params=None, timeout=15):
    raise ConnectionError("boom: https://api.github.com/x")
g._http_get = all_down
out = g.overview("o", "r")
check("everything down: friendly message, no raw exception text", out.startswith("Couldn't read o/r") and "boom" not in out and "http" not in out, out)
reset()
g._http_get = lambda url, raw=False, params=None, timeout=15: Resp(404)
check("404 repo -> clear not-found", g.overview("nope", "nope").startswith("Repo not found"))

# ---------------------------------------------------------------- tree / blob / paging
reset()
g._http_get = api_ok
g.fetch_zip = lambda o, r, ref="": make_zip(FILES)
out = g.overview("o", "r", kind="tree", ref="main", path="skills/eng")
check("tree: folder listing from archive", "tdd/" in out and "review/" in out, out)
check("tree: missing folder -> clear message", g.overview("o", "r", kind="tree", path="nope").startswith("Folder not found"))

reset()
BIG = "x" * 15000
g._http_get = lambda url, raw=False, params=None, timeout=15: Resp(200, text=BIG) if "raw.githubusercontent.com" in url else Resp(404)
out = g.overview("o", "r", kind="blob", ref="HEAD", path="README.md", max_chars=6000, offset=0)
check("blob: first page + next-offset hint", "FILE o/r/README.md (15000 chars)" in out and "offset=6000" in out, out[-120:])
out = g.overview("o", "r", kind="blob", ref="HEAD", path="README.md", max_chars=6000, offset=12000)
check("blob: last page has no 'more' hint", "offset=" not in out.split("]")[-1] and "[chars" not in out)
reset()
g._http_get = lambda url, raw=False, params=None, timeout=15: Resp(404)
check("blob: missing file -> clear message", g.overview("o", "r", kind="blob", path="nope.md").startswith("File not found"))

# ---------------------------------------------------------------- profile repos
HTML = """
<li class="col-12 d-flex" itemprop="owns" itemscope itemtype="http://schema.org/Code">
  <h3 class="wb-break-all"><a href="/u/alpha" itemprop="name codeRepository" >
        alpha-repo</a></h3>
  <p itemprop="description"> Does &amp; things </p>
  <span itemprop="programmingLanguage">Python</span>
  <a class="Link--muted" href="/u/alpha/stargazers"><svg></svg> 1,234 </a>
  <relative-time datetime="2026-09-30T10:00:00Z">Sep 30</relative-time>
</li>
<li class="col-12 d-flex" itemprop="owns" itemscope>
  <h3><a href="/u/beta" itemprop="name codeRepository" >beta</a></h3>
  <span>Forked from x/beta</span>
  <relative-time datetime="2026-01-02T10:00:00Z">Jan 2</relative-time>
</li>
<a rel="next" href="/u?page=2&amp;tab=repositories">Next</a>
"""
HTML2 = """<li itemprop="owns"><h3><a href="/u/gamma" itemprop="name codeRepository" >gamma</a></h3></li>"""
reset()
seen_urls = []
def api_blocked_html_ok(url, raw=False, params=None, timeout=15):
    seen_urls.append((url, (params or {}).get("page")))
    if url.startswith(g.API):
        return Resp(403, text="rate limit exceeded", headers={"x-ratelimit-remaining": "0"})
    return Resp(200, text=HTML if (params or {}).get("page") == 1 else HTML2)
g._http_get = api_blocked_html_ok
out = g.user_repos("u")
check("profile: API 403 -> HTML fallback lists ALL repos across pages", "alpha-repo" in out and "beta" in out and "gamma" in out and "3 public repos" in out, out)
check("profile: description entity decoded, lang + stars parsed", "Does & things" in out and "Python" in out and "★1234" in out, out)
check("profile: fork flagged", "beta [fork]" in out)
check("profile: pagination followed", any(pg == 2 for _, pg in seen_urls))

reset()
API_REPOS = [{"name": f"r{i}", "description": f"d{i}", "language": "Go", "stargazers_count": i, "pushed_at": "2026-01-01T00:00:00Z", "fork": False} for i in range(100)]
def api_two_pages(url, raw=False, params=None, timeout=15):
    pg = (params or {}).get("page")
    return Resp(200, API_REPOS) if pg == 1 else Resp(200, API_REPOS[:20])
g._http_get = api_two_pages
out = g.user_repos("u", limit=200)
check("profile: API path pages past 100 repos", "120 public repos" in out and out.count("\n- ") == 120, out.count("\n- "))
reset()
g._http_get = lambda url, raw=False, params=None, timeout=15: Resp(404)
check("profile: unknown user", g.user_repos("ghost") == "GitHub user 'ghost' not found.")
reset()
def everything_fails(url, raw=False, params=None, timeout=15):
    raise ConnectionError("net down https://github.com/x")
g._http_get = everything_fails
out = g.user_repos("u")
check("profile: total failure -> friendly, no raw exception", out.startswith("Couldn't list repos") and "net down" not in out, out)

# ---------------------------------------------------------------- token handling
import requests
os.environ["GITHUB_TOKEN"] = "ghp_secret"
captured = {}
def fake_requests_get(url, headers=None, params=None, timeout=None, **kw):
    captured[url] = dict(headers or {})
    return Resp(200, {})
real_get = requests.get
requests.get = fake_requests_get
try:
    import importlib
    h_api = g._headers(token=True)
    g.__dict__["_http_get"] = importlib.reload(g)._http_get  # restore the real choke point
    g._http_get("https://api.github.com/repos/o/r")
    g._http_get("https://raw.githubusercontent.com/o/r/HEAD/README.md")
    g._http_get("https://github.com/u", params={"tab": "repositories"})
finally:
    requests.get = real_get
    os.environ.pop("GITHUB_TOKEN", None)
check("token attached to api.github.com", captured["https://api.github.com/repos/o/r"].get("Authorization") == "Bearer ghp_secret")
check("token NEVER sent to raw/html hosts", "Authorization" not in captured["https://raw.githubusercontent.com/o/r/HEAD/README.md"] and "Authorization" not in captured["https://github.com/u"])

print("\nall passed" if FAILS == 0 else f"\n{FAILS} failure(s)")
sys.exit(1 if FAILS else 0)

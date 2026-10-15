"""
wa_format.py — turn model output (GitHub-flavoured Markdown) into clean WhatsApp text.

WhatsApp renders:   *bold*  _italic_  ~strike~  `mono`  ```block```  "> quote"  "• / 1." lists
WhatsApp does NOT:  **bold**, # headings, | pipe | tables |, [links](url), HTML, nested emphasis.

Why this replaces the regex stack in agent_loop._format_whatsapp_reply:
  prod screenshot (2026-10-12): a 7-column markdown table came out as one bullet per CELL plus
  rows of stray "|". The old code joined cells with " · " and a later pass turned every " · "
  into a new bullet. Patching regexes on top of regexes cannot fix that class of bug, so this is
  a small block parser:

    text -> [code fences | tables | headings | lists | quotes | paragraphs] -> rendered blocks

Guarantees (tested): code blocks and URLs/e-mails are never touched; no words are dropped;
the output is IDEMPOTENT (formatting formatted text changes nothing); tables become readable
per-row cards; numbering and bullets are normalised; weird model unicode (narrow no-break
spaces, non-breaking hyphens, zero-width characters) is cleaned up.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- unicode hygiene
_SPACE_CHARS = "\u202f\u00a0\u2009\u200a\u2007\u2002\u2003\u2004\u2005\u2006\u2008\u205f\u3000"
_HYPHEN_CHARS = "\u2010\u2011\u2012"
_ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff"
_UNI_TABLE = {ord(c): " " for c in _SPACE_CHARS}
_UNI_TABLE.update({ord(c): "-" for c in _HYPHEN_CHARS})
_UNI_TABLE.update({ord(c): None for c in _ZERO_WIDTH})

RAW = "\u2063"  # invisible separator: prefix a message with it to bypass reformatting entirely

_URL_RE = re.compile(r"https?://[^\s<>\"'\]\)]+[^\s<>\"'\]\).,;:!?]", re.I)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

_BULLET_RE = re.compile(r"^(\s*)([-*+•·●▪‣◦○])\s+(.*)$")
_NUM_RE = re.compile(r"^(\s*)(\d{1,3})([.)])\s+(.*)$")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_HR_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_FENCE_RE = re.compile(r"^\s*```")


def clean_unicode(t: str) -> str:
    return (t or "").translate(_UNI_TABLE)


# --------------------------------------------------------------------------- inline
def _protect(text: str) -> Tuple[str, List[str]]:
    """Swap URLs, e-mails and `code` spans for placeholders so emphasis rules can't touch them."""
    stash: List[str] = []

    def _hold(m: re.Match) -> str:
        stash.append(m.group(0))
        return f"\ue000{len(stash) - 1}\ue001"

    text = re.sub(r"`[^`\n]+`", _hold, text)
    text = _URL_RE.sub(_hold, text)
    text = _EMAIL_RE.sub(_hold, text)
    return text, stash


def _restore(text: str, stash: List[str]) -> str:
    for _ in range(2):  # placeholders may nest (url inside code)
        text = re.sub(r"\ue000(\d+)\ue001", lambda m: stash[int(m.group(1))], text)
    return text


def inline(text: str) -> str:
    """Markdown inline -> WhatsApp inline (single line of text)."""
    if not text:
        return ""
    # Links / images BEFORE protecting urls (they contain urls)
    text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)[^)]*\)", lambda m: (m.group(1) or m.group(2)), text)

    def _link(m: re.Match) -> str:
        label, url = m.group(1).strip(), m.group(2).strip()
        if not label or label == url or label.rstrip("/") == url.rstrip("/"):
            return url
        return f"{label} ({url})"

    text = re.sub(r"\[([^\]\n]*)\]\((https?://[^)\s]+)\)", _link, text)
    text = re.sub(r"</?(?:br|b|strong|i|em|u|s|strike|del|code|pre|p|div|span|sup|sub|small|center|font)[^>]*>",
                  lambda m: "\n" if m.group(0).lower().startswith("<br") else "", text, flags=re.I)
    text, stash = _protect(text)

    # strike
    text = re.sub(r"~~(?=\S)(.+?)(?<=\S)~~", r"~\1~", text)
    # bold: **x** / __x__  ->  *x*
    for _ in range(3):
        n = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"*\1*", text)
        n = re.sub(r"(?<![\w_])__(?=\S)(.+?)(?<=\S)__(?![\w_])", r"*\1*", n)
        if n == text:
            break
        text = n
    text = re.sub(r"\*{3,}", "*", text)
    # adjacent spans "*a**b*" -> "*a* *b*"
    text = re.sub(r"\*\*(?=\S)", "* *", text)
    # "* bold *" -> "*bold*" (same line)
    text = re.sub(r"(?<![\w*])\*\s+([^*\n]+?)\s+\*(?![\w*])", lambda m: f"*{m.group(1).strip()}*", text)
    # WhatsApp only bolds when the closing * is followed by a non-word char: "*Note:*keep" -> "*Note:* keep"
    text = re.sub(r"(?<![\w*])(\*[^*\n]+?\*)(?=[A-Za-z0-9])", r"\1 ", text)
    # remove lone, unmatched asterisks hugging a word (keep "5 * 3" maths)
    text = _drop_lone_markers(text, "*")
    text = _drop_lone_markers(text, "~")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return _restore(text, stash)


def _drop_lone_markers(line: str, ch: str) -> str:
    esc = re.escape(ch)
    spans = list(re.finditer(rf"(?<![\w{esc}]){esc}(?=\S)[^{esc}\n]*?(?<=\S){esc}(?![\w{esc}])", line))
    covered = set()
    for m in spans:
        covered.update({m.start(), m.end() - 1})
    out = []
    for i, c in enumerate(line):
        if c == ch and i not in covered:
            prev_c = line[i - 1] if i else " "
            next_c = line[i + 1] if i + 1 < len(line) else " "
            if (prev_c.isalnum() and next_c in " \n.,;:!?)") or (next_c.isalnum() and prev_c in " \n(") or i == 0 or i == len(line) - 1:
                if not (c == "*" and (prev_c == " " and next_c == " ")):
                    continue  # lone marker: drop
        out.append(c)
    return "".join(out)


def _plain(cell: str) -> str:
    """Cell text without emphasis markers (used for titles/labels)."""
    c = inline(cell.strip())
    c = re.sub(r"^\*(.+)\*$", r"\1", c)
    return c.strip()


# --------------------------------------------------------------------------- tables
def _split_row(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells = re.split(r"(?<!\\)\|", s)
    return [c.replace("\\|", "|").strip() for c in cells]


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.count("|") >= 2 or (s.startswith("|") and s.count("|") >= 1 and len(s) > 2)


def _render_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    ncol = max(len(r) for r in rows)
    header = header + [""] * (ncol - len(header))
    body = [r + [""] * (ncol - len(r)) for r in body]
    if not body:  # header only -> treat as a list
        return "\n".join(f"• {inline(c)}" for c in header if c.strip())

    out: List[str] = []
    if ncol == 1:
        return "\n".join(f"• {inline(r[0])}" for r in body if r[0].strip())
    if ncol == 2:
        for r in body:
            k, v = _plain(r[0]), inline(r[1])
            out.append(f"• *{k}* — {v}" if k and v else f"• {k or v}")
        return "\n".join(out)

    ordinal_first = all(re.fullmatch(r"#?\s*\d{1,3}[.)]?|[ivxIVX]{1,4}", (r[0] or "").strip()) for r in body if r[0].strip())
    for r in body:
        cells = list(r)
        if ordinal_first and cells[1].strip():
            title = f"{_plain(cells[0]).lstrip('#').strip().rstrip('.)')}. {_plain(cells[1])}"
            start = 2
        else:
            title = _plain(cells[0]) or _plain(cells[1])
            start = 1
        block = [f"*{title}*"] if title else []
        for j in range(start, ncol):
            val = inline(cells[j])
            lab = _plain(header[j])
            if not val.strip() or val.strip() in {"-", "—", "–"}:
                continue
            block.append(f"• {lab}: {val}" if lab else f"• {val}")
        out.append("\n".join(block))
    return "\n\n".join(out)


# --------------------------------------------------------------------------- block parser
_INLINE_NUM_RE = re.compile(r"(?:(?<=\s)|^)(\d{1,2})[.)]\s+(?=\S)")


def _split_inline_numbered(line: str) -> List[str]:
    """'Steps: 1. Install 2. Configure 3. Run' -> real list lines (models do this a lot)."""
    if len(line) < 20 or _NUM_RE.match(line) and len(_INLINE_NUM_RE.findall(line)) < 3:
        return [line]
    marks = list(_INLINE_NUM_RE.finditer(line))
    if len(marks) < 3:
        return [line]
    nums = [int(m.group(1)) for m in marks]
    ascending = all(b == a + 1 for a, b in zip(nums, nums[1:]))
    if not (ascending or len(set(nums)) == 1):
        return [line]
    out: List[str] = []
    head = line[: marks[0].start()].rstrip()
    if head:
        out.append(head)
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(line)
        out.append(f"{m.group(1)}. {line[m.end():end].strip()}")
    return out


def _convert_blocks(text: str) -> str:
    raw_lines = [ln for ln in text.split("\n") if not re.fullmatch(r"[\s|]*\|[\s|]*", ln)]
    lines: List[str] = []
    for ln in raw_lines:
        lines.extend(_split_inline_numbered(ln) if not ln.lstrip().startswith("|") else [ln])
    out: List[str] = []
    i, n = 0, len(lines)

    def push_block(b: str, attach: bool = False) -> None:
        if out and out[-1] != "" and not attach:
            out.append("")
        out.extend(b.split("\n"))
        out.append("")

    while i < n:
        line = lines[i]
        s = line.strip()

        if not s:
            if out and out[-1] != "":
                out.append("")
            i += 1
            continue

        # ---- table: header row + separator row (works with or without outer pipes),
        #      or a run of 3+ rows that each have 2+ pipes
        nxt = lines[i + 1] if i + 1 < n else ""
        sep_next = bool(nxt.strip()) and bool(_TABLE_SEP_RE.match(nxt)) and "-" in nxt and ("|" in nxt or "|" in line)
        if ("|" in line and sep_next) or (_is_table_row(line) and s.count("|") >= 2):
            j = i
            while j < n and lines[j].strip() and "|" in lines[j]:
                j += 1
            block = lines[i:j]
            has_sep = any(_TABLE_SEP_RE.match(b) and "-" in b and "|" in b for b in block[:3])
            if has_sep or len(block) >= 3:
                rows = [_split_row(b) for b in block if not (_TABLE_SEP_RE.match(b) and "-" in b)]
                rows = [r for r in rows if any(c.strip() for c in r)]
                if len(rows) >= 2 or (rows and has_sep):
                    push_block(_render_table(rows))
                    i = j
                    continue

        # ---- stray pipe-only debris lines ("|", "| |") left by broken tables
        if re.fullmatch(r"[|\s:\-]*\|[|\s:\-]*", s):
            i += 1
            continue

        # ---- horizontal rule
        if _HR_RE.match(line):
            if out and out[-1] != "":
                out.append("")
            i += 1
            continue

        # ---- heading
        m = _HEADING_RE.match(line)
        if m:
            title = _plain(re.sub(r"[*_]{1,3}", "", m.group(2)))
            push_block(f"*{title}*")
            i += 1
            continue

        # ---- blockquote
        if s.startswith(">"):
            q = []
            while i < n and lines[i].strip().startswith(">"):
                q.append("> " + inline(re.sub(r"^\s*>\s?", "", lines[i])))
                i += 1
            push_block("\n".join(q))
            continue

        # ---- list (bullets + numbers, with wrapped continuation lines)
        if _BULLET_RE.match(line) or _NUM_RE.match(line):
            start_i = i
            items: List[Tuple[int, Optional[int], str]] = []  # (indent, number|None, text)
            while i < n:
                ln = lines[i]
                mb, mn = _BULLET_RE.match(ln), _NUM_RE.match(ln)
                if mb or mn:
                    indent = len((mb or mn).group(1).expandtabs(2))
                    if mb:
                        items.append((indent, None, mb.group(3)))
                    else:
                        items.append((indent, int(mn.group(2)), mn.group(4)))
                    i += 1
                elif ln.strip() and items and ln.startswith((" ", "\t")) and not _is_table_row(ln):
                    ind, num, txt = items[-1]
                    items[-1] = (ind, num, txt + " " + ln.strip())
                    i += 1
                else:
                    break
            # attached (no blank line) when the source put the list right under a line of text
            attached = bool(out) and out[-1] != "" and start_i > 0 and lines[start_i - 1].strip() != ""
            push_block(_render_list(items), attach=attached)
            continue

        # ---- paragraph (consecutive plain lines keep their breaks)
        para = []
        while i < n and lines[i].strip() and not (
            _HEADING_RE.match(lines[i]) or _HR_RE.match(lines[i]) or _BULLET_RE.match(lines[i])
            or _NUM_RE.match(lines[i]) or lines[i].strip().startswith(">")
            or ("|" in lines[i] and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1] or "") and "-" in lines[i + 1] and "|" in lines[i + 1])
        ):
            para.append(inline(lines[i].rstrip()))
            i += 1
        if para:
            if out and out[-1] != "":
                out.append("")
            out.extend(para)
        else:
            out.append(inline(line))
            i += 1

    return "\n".join(out)


def _render_list(items: List[Tuple[int, Optional[int], str]]) -> str:
    """Indent stack (works for 2- and 4-space nesting alike, and re-parses its own output)."""
    out: List[str] = []
    stack: List[int] = []          # indent of each open level
    counters: dict = {}            # depth -> last number
    for indent, num, txt in items:
        while stack and indent < stack[-1]:
            counters.pop(len(stack) - 1, None)
            stack.pop()
        if not stack or indent > stack[-1]:
            stack.append(indent)
        depth = min(len(stack) - 1, 2)
        for d in [k for k in counters if k > depth]:
            counters.pop(d)
        pad = "    " * depth
        body = inline(txt.strip())
        if num is None:
            out.append(f"{pad}{'•' if depth == 0 else '◦'} {body}")
        else:
            counters[depth] = counters.get(depth, num - 1) + 1
            out.append(f"{pad}{counters[depth]}. {body}")
    return "\n".join(out)


# --------------------------------------------------------------------------- public API
def to_whatsapp(text: Optional[str]) -> str:
    """Model markdown -> clean WhatsApp text. Safe on already-formatted text (idempotent)."""
    if not text or not isinstance(text, str):
        return text or ""
    if text.startswith(RAW):  # pre-formatted card: only hygiene, never restructure
        body = clean_unicode(text[len(RAW):].replace("\r\n", "\n"))
        return "\n".join(ln.rstrip() for ln in body.split("\n")).strip()
    t = clean_unicode(text.replace("\r\n", "\n").replace("\r", "\n"))

    # split out fenced code: pass through untouched (drop the language tag only)
    parts: List[Tuple[bool, str]] = []
    buf: List[str] = []
    in_code = False
    for ln in t.split("\n"):
        if _FENCE_RE.match(ln) and ln.count("```") % 2 == 1:
            if in_code:
                buf.append("```")
                parts.append((True, "\n".join(buf)))
                buf, in_code = [], False
            else:
                if buf:
                    parts.append((False, "\n".join(buf)))
                buf, in_code = ["```"], True
        else:
            buf.append(ln)
    if buf:
        parts.append((in_code, "\n".join(buf) + ("\n```" if in_code else "")))

    rendered: List[str] = []
    for is_code, chunk in parts:
        rendered.append(chunk if is_code else _convert_blocks(chunk))
    out = "\n\n".join(r.strip("\n") for r in rendered if r.strip())

    out = "\n".join(ln.rstrip() for ln in out.split("\n"))
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()

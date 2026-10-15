"""Offline tests for wa_format.to_whatsapp. Run: python test_wa_format.py

Includes replicas of the two ugly production outputs:
  * 2026-10-12 07:00  7-column markdown table  -> one bullet per CELL + stray "|" lines
  * 2026-10-12 04:35  GitHub answer             -> "* *Repository:*value* *Description:*..." glued on one line
"""
from __future__ import annotations

import re
import sys

from wa_format import to_whatsapp as F

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


def words(t):
    return set(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]{2,}", t))


# ------------------------------------------------------------------ the screenshot table
TABLE = """**Top 3 LLMs (reasoning\u202f+\u202fcoding) & their minimum local\u2011run specs**

| Rank | Model | Reasoning / Coding Strength | Minimum GPU (VRAM) | System RAM | CPU | Storage | Notes |
|------|-------|-----------------------------|--------------------|------------|-----|---------|-------|
| 1 | DeepSeek\u2011Coder\u2011V2 16B | Best overall reasoning (math, logic) + solid coding | RTX 3090 24 GB (or A100 40 GB) \u2013 20 GB min | 64 GB (\u2265 32 GB for Q4) | 8\u2011core CPU | 40 GB NVMe | Runs fully in VRAM |
| 2 | Qwen2.5\u2011Coder 32B | Top coding benchmark | 24 GB VRAM | 64 GB | 8 cores | 60 GB | Q4 fits one 3090 |
| 3 | Llama 3.1 70B | Strong general reasoning | 2x 24 GB | 128 GB | 16 cores | 140 GB | Needs offload |"""
out = F(TABLE)
check("table: no pipe characters survive", "|" not in out, out[:200])
check("table: each model becomes its own titled card",
      all(f"*{n}*" in out for n in ("1. DeepSeek-Coder-V2 16B", "2. Qwen2.5-Coder 32B", "3. Llama 3.1 70B")))
check("table: column labels kept as 'Label: value' lines", "• Minimum GPU (VRAM): 24 GB VRAM" in out and "• System RAM: 128 GB" in out)
check("table: not one bullet per cell (bullets = rows x 6 details)", out.count("• ") == 18, out.count("• "))
lost = words(TABLE.replace("\u2011", "-")) - words(out) - {"Rank", "Model"}  # header cells of the ordinal/title columns become the card title
check("table: no data lost", not lost, lost)
check("table: weird unicode cleaned", not any(c in out for c in "\u202f\u2011"))
check("table: heading preserved as bold line", out.startswith("*Top 3 LLMs"))

# two-column + no outer pipes + degenerate pipe debris
out = F("Name | Value\n--- | ---\nRAM | 64 GB\nGPU | 3090")
check("2-col table without outer pipes -> 'key — value'", "• *RAM* — 64 GB" in out and "|" not in out, out)
out = F("Intro line\n|\n\n|\n\n| \nafter")
check("stray pipe-only debris lines are dropped", "|" not in out and "Intro line" in out and "after" in out, out)

# ------------------------------------------------------------------ the GitHub answer
GH = """Here are the repository details found for **mattpocock/skills**:

* **Repository:** mattpocock/skills (https://github.com/mattpocock/skills)
* **Description:** Skills for Real Engineers. Straight from my `.agents` directory.
* **Stars:** 286,537
* **Forks:** 24,155"""
out = F(GH)
check("github: bullets on separate lines", out.count("• ") == 4 and all(l.startswith("• ") for l in out.splitlines() if "Stars" in l or "Forks" in l), out)
check("github: labels bold, values kept", "• *Repository:* mattpocock/skills (https://github.com/mattpocock/skills)" in out, out)
check("github: inline code untouched", "`.agents`" in out)
check("github: **bold** -> *bold*", "**" not in out and "*mattpocock/skills*" in out)

# ------------------------------------------------------------------ lists
out = F("1. First\n1. Second\n1. Third")
check("numbering repaired (1. 1. 1. -> 1. 2. 3.)", out.splitlines() == ["1. First", "2. Second", "3. Third"], out)
out = F("- a\n  - nested b\n    - deeper c\n- d")
check("nested bullets indent + glyphs", out.splitlines()[0] == "• a" and out.splitlines()[1].startswith("    ◦ nested b"), out)
out = F("- item that wraps\n  onto a second line\n- next")
check("wrapped list item joined", "• item that wraps onto a second line" in out and "• next" in out, out)
check("mixed markers all become bullets", F("* a\n+ b\n- c\n· d\n• e") == "• a\n• b\n• c\n• d\n• e")

# ------------------------------------------------------------------ headings / rules / quotes / html
out = F("# Title\ntext\n## Sub **bold** heading\n---\nmore")
check("headings -> bold lines with spacing", out.splitlines()[0] == "*Title*" and "*Sub bold heading*" in out, out)
check("horizontal rule removed", "---" not in out)
check("blockquote kept as WhatsApp quote", F("> quoted line") == "> quoted line")
check("html tags stripped, <br> -> newline", F("a<br>b <b>x</b>") == "a\nb x")
out = F("See [the docs](https://x.dev/a) and ![logo](https://x.dev/l.png) and [https://y.dev](https://y.dev)")
check("links rendered readable", "the docs (https://x.dev/a)" in out and "logo" in out and "https://y.dev" in out and "](" not in out, out)

# ------------------------------------------------------------------ inline safety
out = F("Email john_doe_99@my_site.com and https://x.com/a_b_c?q=*1*&r=__2__ now")
check("emails / URLs never touched by emphasis rules", "john_doe_99@my_site.com" in out and "https://x.com/a_b_c?q=*1*&r=__2__" in out, out)
check("maths asterisk kept", F("5 * 3 = 15") == "5 * 3 = 15")
check("glued spans split", F("**One****Two**") == "*One* *Two*", F("**One****Two**"))
check("lone asterisk removed", F("this is *broken text") == "this is broken text", F("this is *broken text"))
check("strike", F("~~old~~ new") == "~old~ new")
check("bold glued to a word gets a space (WhatsApp wouldn't bold it)", F("**Note:**keep") == "*Note:* keep", F("**Note:**keep"))
# /help style: inline ```cmd``` lines must NOT open a code fence, and quotes stay quotes
HELP = "*`AVAILABLE ADMIN COMMANDS:`*\n\n```/enable``` ¬\n\n> Usage: /enable agent | /enable <feature|all> <chat|all>\n\n```/status``` ¬\n\n> Usage: /status [phone]"
check("inline ```cmd``` lines are not fences", F(HELP).count("```/") == 2 and "> Usage: /status [phone]" in F(HELP), F(HELP))
from wa_format import RAW
check("RAW marker: card passes through untouched", F(RAW + "*To:* a@b.co\n- keep **this** | as is") == "*To:* a@b.co\n- keep **this** | as is")

# ------------------------------------------------------------------ code
code = "```python\nx = [1, 2]  # **not bold** | not | a | table\n- not a bullet\n```"
out = F("before\n" + code + "\nafter")
check("code block content untouched (language tag dropped)", "x = [1, 2]  # **not bold** | not | a | table" in out and "- not a bullet" in out, out)
check("code fence normalised", out.count("```") == 2 and "python" not in out)

# ------------------------------------------------------------------ idempotence on a varied corpus
CORPUS = [TABLE, GH, "# H\n1. a\n1. b\n- x\n  - y\n> q\n---\n**bold** and *it* and ~~s~~",
          "Plain paragraph.\nSecond line.\n\nNew para with https://a.com/x_y and a@b.co",
          "Name | Value\n--- | ---\nRAM | 64", "*Already* formatted\n• one\n• two\n\n1. a\n2. b",
          "Rank\n- Model\n- CPU\n|\n\n|\n1"]
for k, c in enumerate(CORPUS):
    once = F(c)
    check(f"idempotent #{k}", F(once) == once, repr(F(once))[:160])

# ------------------------------------------------------------------ never crashes / empty
check("None/empty safe", F("") == "" and F(None) == "")
check("huge input fast & intact", len(F("word " * 20000)) > 90000)
long_table = "| a | b | c |\n|---|---|---|\n" + "\n".join(f"| {i} | row{i} | v{i} |" for i in range(200))
check("200-row table handled", F(long_table).count("*") >= 400)

out = F("Steps to set up: 1. Install the CLI 2. Configure your key 3. Run the agent")
check("inline '1. a 2. b 3. c' becomes a real list", out.splitlines() == ["Steps to set up:", "1. Install the CLI", "2. Configure your key", "3. Run the agent"], out)
check("a lone '1.' in a sentence is NOT split", F("In version 1. we shipped it and later 2. fixed it") == "In version 1. we shipped it and later 2. fixed it")
check("price/number prose untouched", F("It costs 3.5 dollars, about 2) maybe") == "It costs 3.5 dollars, about 2) maybe")

print("\nall passed" if FAILS == 0 else f"\n{FAILS} failure(s)")
sys.exit(1 if FAILS else 0)

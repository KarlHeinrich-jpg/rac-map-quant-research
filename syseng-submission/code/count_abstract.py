# -*- coding: utf-8 -*-
"""统计中文摘要与英文摘要的篇幅。"""
import re
from pathlib import Path

MAIN = Path(__file__).resolve().parent / "paper" / "main.tex"
s = MAIN.read_text(encoding="utf-8")

cn = re.search(r"摘\\quad 要\}：(.*?)\\vspace", s, re.S).group(1)
cn_clean = re.sub(r"\$[^$]*\$", "", cn)
cn_clean = re.sub(r"\\[a-zA-Z]+", "", cn_clean)
print("中文摘要汉字数：", len(re.findall(r"[\u4e00-\u9fff]", cn_clean)))

en = re.search(r"\\textbf\{Abstract\}:(.*?)\\vspace", s, re.S).group(1)
print("英文摘要词数：", len(re.findall(r"[A-Za-z][A-Za-z\-']*", en)))

kw = re.search(r"关键词\}：(.*?)\n", s).group(1)
print("中文关键词条数：", len([x for x in re.split(r"[；;]", re.sub(r"\\[a-zA-Z]+", "", kw)) if x.strip()]))

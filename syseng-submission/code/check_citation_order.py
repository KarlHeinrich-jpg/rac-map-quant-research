# -*- coding: utf-8 -*-
"""核验顺序编码制：正文中文献编号的首次出现顺序应为 1,2,3,…"""
import re
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parent / "paper"
txt = PAPER / "main.txt"
subprocess.run(["D:/texlive/2025/bin/windows/pdftotext.exe", "-layout",
                str(PAPER / "main.pdf"), str(txt)], check=True)

s = txt.read_text(encoding="utf-8")
# 只保留正文（去掉文末参考文献表）
body = s.split("参考文献")[0]
group = re.compile(r"\[(\d{1,2}(?:\s*[,\-–]\s*\d{1,2})*)\]")
marks = []
for m in group.finditer(body):
    for n in re.split(r"\s*[,\-–]\s*", m.group(1)):
        if n.isdigit():
            marks.append(int(n))

seen, order = set(), []
for n in marks:
    if n not in seen:
        seen.add(n)
        order.append(n)

bad = [(i, n) for i, n in enumerate(order, 1) if i != n]
print("正文引用的文献编号总数为", len(order))
if bad:
    print("首次出现顺序与编号不一致处（位置, 编号）：", bad[:10])
else:
    print("✓ 编号首次出现顺序严格递增：1,2,3,…")
print("前 12 个首次出现的编号：", order[:12])

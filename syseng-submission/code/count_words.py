# -*- coding: utf-8 -*-
"""统计正文汉字数（剔除表格、图注与文献表）。"""
import glob
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent / "paper"

ENV = r"table|figure|tabular|threeparttable|tablenotes|thebibliography|equation|align"


def clean(s: str) -> str:
    s = re.sub(r"\\begin\{(" + ENV + r")\}.*?\\end\{\1\}", "", s, flags=re.S)
    s = re.sub(r"\\(cite|ref|label)\{[^}]*\}", "", s)
    s = re.sub(r"\$[^$]*\$", "X", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", "", s)
    return s


def main() -> None:
    total = 0
    files = sorted(glob.glob(str(PAPER / "sec0*.tex"))) + [str(PAPER / "appendix.tex")]
    for f in files:
        s = Path(f).read_text(encoding="utf-8")
        n = len(re.findall(r"[\u4e00-\u9fff]", clean(s)))
        total += n
        print(f"{Path(f).name}: {n} 字")
    main_tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    abs_cn = re.search(r"摘\\quad 要\}：(.*?)\\vspace", main_tex, re.S)
    if abs_cn:
        print("中文摘要：", len(re.findall(r"[\u4e00-\u9fff]", abs_cn.group(1))), "字")
    print("正文章节合计（含公式符号占位）：", total, "汉字")


if __name__ == "__main__":
    main()

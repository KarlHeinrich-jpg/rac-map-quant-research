"""把论文各部分拼接为完整稿。

用法::

    python src/assemble_paper.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

PUBLIC_INFO = """% =====================================================================
% 本文件由 src/assemble_paper.py 自动拼接生成，请勿直接编辑。
% 各部分源文件：
%   paper/preamble.tex              —— 导言区（宏包、定理环境、标题）
%   paper/paper_theory_part.tex     —— 摘要与第 1--3 节（引言、文献、理论）
%   paper/paper_empirics_part.tex   —— 第 4--5 节（数据设计、主结果）
%   paper/paper_mechanism_part.tex  —— 第 6--7 节（机制检验、稳健性）
%   paper/paper_conclusion_part.tex —— 第 8 节与附录、参考文献
% 编译：xelatex rac_map_paper_full.tex（连续两次以解决交叉引用）
% =====================================================================
"""

# 标题之后的正文起始标记：摘要
BODY_START = r"\begin{abstract}"


def extract_body(text: str, start_marker: str = BODY_START) -> str:
    """从部分文件中提取正文（去掉其自身的导言区与 \\begin{document}）。"""
    idx = text.find(start_marker)
    if idx < 0:
        # 没有摘要的部分：从第一个 \section 开始
        idx = text.find(r"\section{")
        if idx < 0:
            raise ValueError("无法定位正文起点")
    body = text[idx:]
    # 去掉部分文件自带的 \end{document}，否则会把整篇论文提前截断
    return body.replace(r"\end{document}", "").rstrip() + "\n"


def main() -> None:
    preamble = (PAPER / "preamble.tex").read_text(encoding="utf-8")
    parts = [
        extract_body((PAPER / "paper_theory_part.tex").read_text(encoding="utf-8")),
        extract_body((PAPER / "paper_empirics_part.tex").read_text(encoding="utf-8")),
        extract_body((PAPER / "paper_mechanism_part.tex").read_text(encoding="utf-8")),
        extract_body((PAPER / "paper_conclusion_part.tex").read_text(encoding="utf-8")),
    ]
    full = PUBLIC_INFO + preamble + "\n" + "\n".join(parts) + "\n\\end{document}\n"
    out = PAPER / "rac_map_paper_full.tex"
    out.write_text(full, encoding="utf-8")

    n_sections = full.count(r"\section{")
    n_props = sum(full.count(m) for m in (r"\begin{proposition}", r"\begin{theorem}",
                                          r"\begin{corollary}", r"\begin{lemma}"))
    n_tables = full.count(r"\begin{table}")
    n_figs = full.count(r"\begin{figure}")
    print(f"已生成 {out}")
    print(f"  文件大小   : {len(full)/1024:.1f} KB")
    print(f"  章         : {n_sections}")
    print(f"  定理类命题 : {n_props}")
    print(f"  表         : {n_tables}")
    print(f"  图         : {n_figs}")
    print("\n下一步：")
    print("  cd paper")
    print("  xelatex -interaction=nonstopmode rac_map_paper_full.tex")
    print("  xelatex -interaction=nonstopmode rac_map_paper_full.tex")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""按正文首次引用顺序生成 refs.tex（GB/T 7714 顺序编码制）。

文献来源：
1. 《系统工程理论与实践》专刊项目的已验证文献库（setp-review/refs_data.py，151 条）；
2. 原 RAC-MAP 论文参考文献表中已核实条目的补录（本文件 MANUAL）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # rac-map-theory
PAPER = ROOT / "syseng" / "paper"
SRC_REFS = Path(r"F:\work-environment-for-dsh\setp-review\refs_data.py")

# 原论文参考文献表中已核实条目（不在文献库中的部分）
MANUAL = {
    "hamilton1989":
        "Hamilton J D. A new approach to the economic analysis of nonstationary time "
        "series and the business cycle[J]. Econometrica, 1989, 57(2): 357--384.",
    "guidolin2007":
        "Guidolin M, Timmermann A. Asset allocation under multivariate regime "
        "switching[J]. Journal of Economic Dynamics and Control, 2007, 31(11): 3503--3544.",
    "green2017":
        "Green J, Hand J R M, Zhang X F. The characteristics that provide independent "
        "information about average U.S. monthly stock returns[J]. Review of Financial "
        "Studies, 2017, 30(12): 4389--4436.",
    "timmermann2006":
        "Timmermann A. Forecast combinations[M]//Elliott G, Granger C W J, "
        "Timmermann A. Handbook of Economic Forecasting, Vol. 1. Amsterdam: Elsevier, "
        "2006: 135--196.",
    "jiang2017":
        "Jiang Z, Xu D, Liang J. A deep reinforcement learning framework for the "
        "financial portfolio management problem[EB/OL]. arXiv:1706.10059, 2017.",
    "liu2020":
        "Liu X Y, Yang H, Chen Q, et al. FinRL: A deep reinforcement learning library "
        "for automated stock trading in quantitative finance[EB/OL]. arXiv:2011.09607, 2020.",
    "li2024":
        "Li Z, Tam V, Yeung K. Developing a multi-agent and self-adaptive framework with "
        "deep reinforcement learning for dynamic portfolio risk management[EB/OL]. "
        "arXiv:2402.00515, 2024.",
    # ---- 行为金融与信息经济学（新增，均已逐条核验出版信息）----
    "grossman1980":
        "Grossman S J, Stiglitz J E. On the impossibility of informationally efficient "
        "markets[J]. American Economic Review, 1980, 70(3): 393--408.",
    "shleifer1997":
        "Shleifer A, Vishny R W. The limits of arbitrage[J]. Journal of Finance, 1997, "
        "52(1): 35--55.",
    "sims2003":
        "Sims C A. Implications of rational inattention[J]. Journal of Monetary "
        "Economics, 2003, 50(3): 665--690.",
    "hongstein1999":
        "Hong H, Stein J C. A unified theory of underreaction, momentum trading, and "
        "overreaction in asset markets[J]. Journal of Finance, 1999, 54(6): 2143--2184.",
    "odean1998":
        "Odean T. Are investors reluctant to realize their losses?[J]. Journal of "
        "Finance, 1998, 53(5): 1775--1798.",
    "barberodean2001":
        "Barber B M, Odean T. Boys will be boys: Gender, overconfidence, and common "
        "stock investment[J]. Quarterly Journal of Economics, 2001, 116(1): 261--292.",
    "dellavigna2009":
        "DellaVigna S, Pollet J M. Investor inattention and Friday earnings "
        "announcements[J]. Journal of Finance, 2009, 64(2): 709--749.",
    "brunnermeier2009":
        "Brunnermeier M K, Pedersen L H. Market liquidity and funding liquidity[J]. "
        "Review of Financial Studies, 2009, 22(6): 2201--2238.",
}


def load_library() -> dict[str, str]:
    ns: dict = {}
    exec(SRC_REFS.read_text(encoding="utf-8"), ns)  # noqa: S102 受信任的本地文献数据
    out = {}
    for key, zh, _en in ns["REFS"]:
        out[key] = zh
    return out


def citation_order() -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    files = ["main.tex"] + [f"sec0{i}_{n}.tex" for i, n in
                            enumerate(["intro", "lit", "framework", "design", "factors",
                                       "marginal", "robust", "policy", "conclusion"], start=1)] \
            + ["appendix.tex"]
    for f in files:
        p = PAPER / f
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"\\cite\{([^}]*)\}", text):
            for k in (x.strip() for x in m.group(1).split(",")):
                if k and k not in seen:
                    seen.add(k)
                    order.append(k)
    return order


def main() -> None:
    lib = load_library()
    order = citation_order()
    missing = [k for k in order if k not in lib and k not in MANUAL]
    if missing:
        print("警告：以下 key 无对应文献：", missing)
        sys.exit(1)

    lines = ["\\begin{thebibliography}{99}", "\\footnotesize",
             "\\setlength{\\itemsep}{0.1em}", ""]
    for i, k in enumerate(order, 1):
        entry = MANUAL.get(k) or lib[k]
        lines.append(f"\\bibitem{{{k}}} {entry}")
    lines += ["", "\\end{thebibliography}"]
    (PAPER / "refs.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"引用 {len(order)} 条文献，已写入 refs.tex")
    for i, k in enumerate(order, 1):
        print(f"  [{i}] {k}")


if __name__ == "__main__":
    main()

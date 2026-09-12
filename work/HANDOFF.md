# RAC-MAP 研究交接记录

更新时间：2026-09-08（Asia/Shanghai）

交接对象：Claude。以下内容是当前工作区的唯一交接摘要；所有结果均需以文件中的 manifest、CSV 和 JSON 为准。

## 当前研究定位

论文题目暂定为《基于多智能体的量化投资研究》。核心模型为制度自适应、置信度加权并带尾部风险约束的多智能体组合模型 RAC-MAP。论文主比较对象应为非智能体方法，而不是把 MASA/FinRL 作为唯一基线。

## 已完成

- 公开 ETF 日频面板：SPY、QQQ、IWM、EFA、EEM、TLT、GLD、VNQ、LQD、DBC。
- FRED 宏观状态变量：VIX、联邦基金利率、10 年期国债收益率、10Y--2Y 利差、BAA--10Y 信用利差。
- 宏观变量按公开发布日期代理规则滞后一个交易日；当前不是完整 vintage 数据。
- RAC-MAP 联合门控：专家动作、专家置信度、专家风险和近期表现进入门控网络；训练目标包含净收益、换手、CVaR 和门控熵。
- 严格年度滚动实验：五年训练、一年验证、一年测试；测试年份 2018--2025；五个 seed 为 7、19、42、73、101。
- 统计推断已改为先在测试年份内平均 seed，再以测试年份作为市场层面的配对单位，避免把同一市场年份伪重复。
- 规则与架构对照已完成：buy-and-hold、equal-weight、risk parity、static multi-agent、MASA-compatible、确定性 RAC-MAP、联合 RAC-MAP。
- 已新增 `work/rac_map/ml_baselines.py`，接入 OLS、Ridge、Lasso、Elastic Net、PLS、SVR、GBDT 的滚动预测基线；尚未完成正式运行和论文整合。
- ML 五 seed 滚动结果已完成：`work/rac_map/results/ml_rolling_v2`。当前平均 Sharpe（测试年份内先平均 seed）：GBDT 0.961、Lasso 0.846、Elastic Net 0.718、Ridge 0.612、OLS 0.599、PLS 0.562、SVR 0.308。
- GARCH(1,1)+60日动量滚动基线已完成：`work/rac_map/results/garch_rolling`，平均 Sharpe 0.641；使用 scipy 高斯准最大似然估计，不生成模拟数据。
- 统一结果包已生成：`work/rac_map/results/full_v1`，包含规则、RAC-MAP、ML 和 GARCH；全模型 Friedman 检验当前 p=0.0001，但不代表联合 RAC-MAP 优于 GBDT。
- RAC-MAP 消融已完成：`work/rac_map/results/ablations`。平均 Sharpe：去 regime 0.929、去 confidence 0.935、去 risk score 0.931、joint no-CVaR 0.941；联合模型完整版本为 0.940。当前结果不支持“每个模块都单调提升收益”的强断言，门控模块的主要差异需要结合换手、CVaR 和状态分组解释。
- 论文工作稿已开始按《中国工业经济》式高信息密度结构重写：研究问题、模型对象、比较对象和经济量级前置；不预先承诺RAC-MAP优越性；主结果同时报告梯度提升树的高收益点估计与高换手成本。
- 论文主稿当前文件：`outputs/rac_map_paper_zh.tex`；PDF需在最终表格更新后重新编译。
- 当前 PDF 已重新编译为 8 页；`pytest` 为 5 passed，新增基线脚本均已通过 `py_compile`。
- Claude 交接压缩包：`outputs/rac_map_claude_handoff.zip`。
- 中文经济学写作已按 `econ-writing-workflow`、`cn-top-econ-writing`、`econ-table-figure-design` 路由；正文采用问题—机制—比较—量级—含义顺序，表格采用三线表。

## 当前证据边界

`work/rac_map/results/rolling_v3` 只支持规则组合与 RAC-MAP 族的公开 ETF 结果。它不支持“RAC-MAP 优于机器学习或 GARCH”的论文主张。完整非智能体基线完成前，主摘要和结论不得写成全面优越性。

## 当前正式运行顺序

1. 检查并冻结 ML、GARCH 和统一结果包。
2. 完成并审计 ablation 输出。
3. 如需最终主表，基于全模型包重算相对等权的年份层面 bootstrap/Wilcoxon/Friedman。
4. 更新论文主表、基线表、消融表、摘要和结论；重新编译 XeLaTeX 并进行 PDF 渲染检查。

## 关键结果（当前正式候选版本）

测试年份为 2018--2025，五年训练、一年验证、一年测试，交易成本为单边 0.10% 乘以半换手率。测试年份内先平均 seed：

| 模型 | CAGR | Sharpe | Sortino | 最大回撤 | CVaR | 换手率 |
|---|---:|---:|---:|---:|---:|---:|
| 梯度提升树 | 0.119 | 0.961 | 1.498 | -0.124 | -0.0198 | 0.061 |
| 等权 | 0.096 | 0.941 | 1.442 | -0.117 | -0.0178 | 0.003 |
| 联合 RAC-MAP | 0.095 | 0.940 | 1.438 | -0.116 | -0.0175 | 0.003 |
| GARCH+动量 | 0.070 | 0.641 | 0.994 | -0.131 | -0.0205 | 0.121 |

全模型相对等权的 Friedman 检验统计量为 42.60，p=0.0001；该检验只说明模型排序存在差异，不说明联合 RAC-MAP 优于 GBDT。联合 RAC-MAP 相对等权 Sharpe 差值为 -0.0013，95% 区间 [-0.0070, 0.0037]，Holm 校正 p=1.000。

## 可恢复命令

```powershell
D:\Codex\PythonEnv\Scripts\python.exe -m py_compile work\rac_map\ml_baselines.py
D:\Codex\PythonEnv\Scripts\python.exe work\rac_map\ml_baselines.py --data work\rac_map\data\features_macro.parquet --output-dir work\rac_map\results\ml_rolling --seed 42
D:\Codex\PythonEnv\Scripts\python.exe work\rac_map\merge_results.py --rule work\rac_map\results\rolling_v3\rolling_metrics.csv --ml work\rac_map\results\ml_rolling_v2\ml_rolling_metrics.csv --garch work\rac_map\results\garch_rolling\garch_rolling_metrics.csv --output-dir work\rac_map\results\full_v1
```

## 不应做的事情

- 不把 Yahoo/ETF 结果写成 CRSP/WRDS 或完整 SEC point-in-time 结果。
- 不把公开发布日期滞后代理写成完整 FRED vintage。
- 不把 MASA-compatible 写成 byte-identical 上游复现。
- 不把验证期调参结果称为测试期结果。
- 不在机器学习/GARCH 基线尚未完成前宣称 RAC-MAP 全面优于传统方法。
- 当前 ML 基线使用 ETF 技术、市场和滞后 FRED 特征；没有价值或情绪变量，因此不能把它们称为完整的基本面/情绪预测实验。
- 当前 GARCH 实现为 scipy 高斯准最大似然的单变量 GARCH(1,1)+60日动量倾斜；不是 DCC-GARCH，也不是完整的条件均值—方差联合模型。
- 当前五个智能体仍是动量、趋势、低波动、防御、均值回复的可解释代理，不应在没有估值和文本数据时写成严格的价值、情绪、波动率、宏观五智能体。

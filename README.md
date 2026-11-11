# RAC-MAP：基于多智能体的制度自适应风险感知量化投资研究

本仓库保存 RAC-MAP 研究项目的可复现实验材料：公开 ETF 与 FRED 数据、特征工程、滚动窗口实验、非智能体机器学习与 GARCH 基线、多智能体模型、统计检验、结果文件、中文 LaTeX 论文和交接文档。

## 目录

- `work/rac_map/`：RAC-MAP 模型、基线、数据下载/处理脚本、滚动实验与结果。
- `work/rac_map/data/`：十个 ETF（日频）和 FRED 宏观变量及数据元数据。
- `work/rac_map/results/`：规则组合、RAC-MAP、机器学习、GARCH 和消融实验的原始输出。
- `work/MASA/`、`work/FinRL/`：经授权保存的参考实现与复现材料；其中含第三方凭据的示例文件不纳入版本库。
- `outputs/`：汇总 CSV/JSON、中文论文、文献综述 PDF/LaTeX、研究交接压缩包。
- `work/HANDOFF.md`、`work/current_plan.md`：研究状态、证据边界和后续计划。
- `syseng-submission/`：《系统工程》专刊投稿稿的完整交付目录（A 股 868 只证券样本，
  含论文源文件与 PDF、图件、实验与二次分析结果、数据与代码；详见该目录 README）。

## 实验协议

- 测试年份：2018--2025；五年训练、一年验证、一年测试的严格年度滚动窗口。
- 随机种子：7、19、42、73、101。
- 资产池：SPY、QQQ、IWM、EFA、EEM、TLT、GLD、VNQ、LQD、DBC。
- 交易成本：单边 0.10%，按半换手计入组合净收益。
- 主要基线：买入持有、等权、风险平价、OLS/Ridge/Lasso/Elastic Net/PLS/SVR/GBDT、GARCH(1,1)+动量，以及静态和动态多智能体聚合。

## 复现入口

```powershell
D:\Codex\PythonEnv\Scripts\python.exe work\rac_map\rolling_experiment.py --help
D:\Codex\PythonEnv\Scripts\python.exe work\rac_map\ml_baselines.py --help
D:\Codex\PythonEnv\Scripts\python.exe work\rac_map\garch_baseline.py --help
D:\Codex\PythonEnv\Scripts\python.exe work\rac_map\stat_tests.py --help
```

论文主稿为 `outputs/rac_map_paper_zh.tex`，已编译 PDF 为 `outputs/rac_map_paper_zh.pdf`。当前公开数据版本使用 ETF—FRED 样本和公开发布日期滞后代理，不应表述为完整 CRSP/WRDS 或 FRED vintage 数据。

## 凭据与数据边界

仓库不保存 API key、token、`.env` 或第三方私有凭据。被 `.gitignore` 排除的少数参考示例仅因其中包含硬编码第三方凭据；RAC-MAP 实验所需数据文件和元数据均保留。数据的来源、抓取时间、字段和哈希值见 `work/rac_map/data/*metadata.json`。

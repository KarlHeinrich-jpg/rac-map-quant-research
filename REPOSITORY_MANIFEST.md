# 版本库备份清单

本清单对应首次 GitHub 备份。纳入范围为当前工作区中的研究数据、实验源码、模型权重/压缩包、结果 CSV/JSON、论文 LaTeX/PDF、文献综述和交接文档。

纳入：

- `work/rac_map/data/`、`work/rac_map/models/`、`work/rac_map/results/`
- `work/rac_map/*.py` 与 `work/rac_map/README.md`
- `work/MASA/` 和 `work/FinRL/` 中不含凭据的源代码与复现材料
- `outputs/` 中的论文、文献综述、结果汇总与 handoff 压缩包
- `work/HANDOFF.md`、`work/current_plan.md`、`work/Paper_attachment.txt`

排除：

- Python/pytest 缓存、LaTeX 编译中间文件和临时 PDF 审阅渲染图；
- `.env`、私钥、token、`config_private.py`；
- 五个包含硬编码第三方 API key 的参考示例文件。

排除项不影响 RAC-MAP 主实验的重跑；若需使用对应参考示例，应在本地自行配置凭据。

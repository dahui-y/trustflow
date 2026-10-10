# TrustFlow — 给 Coding Agent 的入口说明

本仓库是 FlowMM 官方代码的 fork，用于 TrustFlow 研究项目。开始任何工作前先读：

1. `PROJECT_BRIEF.md` — 项目定义、三个 Research Questions、Milestone 0 范围与硬停止线。
2. `REFERENCE_CODE_MAP.md` — FlowMM 与 `help_code/` 各库的代码事实、数据流、已确认决策、环境方案。

## 不可违反的规则

- **Milestone 0 只做 trajectory reliability analysis。不实现任何 guidance。** 停止线：`STOP. Do not implement reliability-aware guidance until the preliminary results have been reviewed.`
- `help_code/` 下的仓库是只读参考，不修改，不在其中实现项目代码。
- `src/flowmm/` 是 backbone，Milestone 0 不修改其采样代码；TrustFlow 自己的代码放在 `src/trustflow/` 与 `scripts_analysis/`。
- 永远不要把 Cartesian MLIP 力直接加到分数坐标速度场上。坐标转换与切向投影见 `REFERENCE_CODE_MAP.md` §1.3。
- 按"复现 → 测量 → 分析 → 判断假设 → 再实现"推进，每次只做一个阶段，产出结果后停下来总结。
- 实验必须可复现：config、seed、输出元数据一起保存。
- 新研究代码要有单元测试。

## 仓库注意事项

- `.gitignore` 会忽略 `*.png` 与 `**/analysis`。图表需 `git add -f`，不要给目录起名 `analysis`。
- `data/*.csv` 是 git-lfs 指针，需要 `git lfs pull`。
- `remote/` 子模块未 checkout 时 FlowMM 无法 import。

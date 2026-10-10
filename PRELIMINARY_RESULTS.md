# TrustFlow — Milestone 0 Preliminary Results

> 申请阶段的初步证据。所有数字来自一个未训完的 FlowMM checkpoint（MP-20，epoch 1300 / 2000）、64 条生成轨迹、5905 个原子。细节见 `EXPERIMENT_LOG.md`，图在 `figures/m0/`。

## 问题

晶体生成模型（FlowMM）的采样轨迹上，大原子模型（CHGNet、DPA）是否在所有时间、所有原子上都可靠？如果不是，可靠性如何随时间和局部环境变化？

## 做了什么

1. 在 MP-20 上自训 FlowMM，导出 64 条 1000 步生成轨迹的全部中间结构。
2. 对 11 个时间步的中间结构用 CHGNet 计算力、site energy 和 64 维原子特征；用 UQ-MLIP 的方法（quantile GBM，在 MP-20 训练集上拟合）得到 per-atom 不确定性，以 MP-20 验证集作为分布内参考。
3. 用独立的 DPA-3.1-3M（MPtrj head）对同一批结构计算力，得到逐原子跨模型分歧 D_F = |F_CHGNet − F_DPA|。

## 三个发现

**1. 早期中间态在 MLIP 分布之外，可靠性沿轨迹系统性恢复，但只在后 40% 的流中发生。**

| | t ≤ 0.4 | t = 1.0 | MP-20 分布内参考 |
|---|---|---|---|
| per-atom UQ 中位 (eV) | 1.7 | 0.96 | 0.42 |
| 超过参考 p95 的原子比例 | 35% | 9.6% | 5% |
| CHGNet 力中位 (eV/Å) | 10–12 | 1.4 | — |

t ≤ 0.5 是平台，t ∈ [0.6, 1] 快速下降（fig1、fig2、fig3）。即使 t = 1 的生成结构，UQ 仍是分布内的 2.3 倍。

**2. 两个独立的大原子模型在中间态上严重分歧。**

相对分歧 D_F / (|F_C| + |F_D|) 的中位数从 t = 0 的 0.67 降到 t = 1 的 0.18（fig5）。早期两模型给出的力方向接近无关，无条件使用任一模型的力做 guidance 都不可靠。这条证据不依赖任何 UQ 方法。

**3. CHGNet 的 per-atom UQ 能部分预测跨模型分歧，且携带了力大小之外的信息，但在后期失效。**

- 绝对分歧：ρ(UQ, D_F) = 0.37–0.50，top-decile AUROC 0.64–0.82（fig6，分歧随 UQ 十分位单调上升约 15 倍）。但 |F| 本身是更强的预测器（ρ 0.57–0.73）。
- 去掉量级效应后：早中期 ρ(UQ, D_F_rel) ≈ 0.2 而 ρ(|F|, D_F_rel) ≈ 0；固定 |F| 后 UQ 的条件 Spearman 0.14–0.31。
- t ≥ 0.8 时 UQ 对相对分歧的 AUROC 降到 0.5 左右。

## 对假设的判断

- **RQ1（when）成立**：可靠性有明确的时间结构，门控的有效区间集中在 t ∈ [0.6, 1]。
- **RQ2（where）弱成立**：同一时间步内 UQ 与跨模型分歧有独立于几何的相关，但强度有限。
- **估计器需要改进**：UQ-MLIP 的默认目标（site energy 分位宽度）在接近流形的后期失去判别力，而这正是 guidance 起作用的区间。按 PROJECT_BRIEF 的转向条件，下一步优先研究 reliability estimator（例如直接以跨模型分歧为目标训练、embedding 距离、DPA 多模型 deviation），再进入 selective guidance。

## 局限

- 单个未训完的 checkpoint、64 条轨迹；训练结束后需用最终 checkpoint 复跑。
- 早期有 16% 的结构因原子重叠被预过滤，早期可信度被低估。
- 跨模型分歧是真实误差的代理，不是 DFT。
- UQ 的 GBM 在验证集覆盖率 0.78（名义 0.90），只作排序用。
- 中间态的原子类型用最终步回填，只测量几何 OOD。

## 下一步（申请后、实习期间）

1. 比较 reliability estimator：site-energy 分位宽度 vs 以 D_F 为目标的 GBM vs embedding 距离 vs DPA 模型 deviation。
2. 固定 guidance baseline，测量 negative transfer rate。
3. 基于最优 estimator 的 atom-wise selective guidance（TrustFlow），paired experiments。

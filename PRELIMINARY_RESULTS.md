# TrustFlow — Milestone 0 Preliminary Results

> 申请阶段的初步证据。所有数字来自在 MP-20 上训完的 FlowMM（2000 epoch）、256 条生成轨迹、24688 个原子。细节见 `EXPERIMENT_LOG.md`，图在 `figures/m0_final/`（`figures/m0/` 是 epoch 1300、64 条轨迹的早期版本，结论一致）。

## 问题

晶体生成模型（FlowMM）的采样轨迹上，大原子模型（CHGNet、DPA）是否在所有时间、所有原子上都可靠？如果不是，可靠性如何随时间和局部环境变化？

## 做了什么

1. 在 MP-20 上自训 FlowMM，导出 256 条 1000 步生成轨迹的全部中间结构。
2. 对 11 个时间步的中间结构用 CHGNet 计算力、site energy 和 64 维原子特征；用 UQ-MLIP 的方法（quantile GBM，在 MP-20 训练集上拟合）得到 per-atom 不确定性，以 MP-20 验证集作为分布内参考。
3. 用独立的 DPA-3.1-3M（MPtrj head）对同一批结构计算力，得到逐原子跨模型分歧 D_F = |F_CHGNet − F_DPA|。

## 三个发现

**1. 早期中间态在 MLIP 分布之外，可靠性沿轨迹系统性恢复，但只在后 40% 的流中发生。**

| | t ≤ 0.4 | t = 1.0 | MP-20 分布内参考 |
|---|---|---|---|
| per-atom UQ 中位 (eV) | 1.7–1.8 | 1.03 | 0.42 |
| 超过参考 p95 的原子比例 | 33–36% | 8% | 5% |
| CHGNet 力中位 (eV/Å) | 13–14 | 1.3 | — |

t ≤ 0.5 是平台，t ∈ [0.6, 1] 快速下降（fig1、fig2、fig3）。即使 t = 1 的生成结构，UQ 仍是分布内的 2.5 倍。

**2. 两个独立的大原子模型在中间态上严重分歧。**

相对分歧 D_F / (|F_C| + |F_D|) 的中位数从 t = 0 的 0.56 降到 t = 1 的 0.19（fig5）。早期两模型给出的力方向接近无关，无条件使用任一模型的力做 guidance 都不可靠。这条证据不依赖任何 UQ 方法。

**3. CHGNet 的 per-atom UQ 能部分预测跨模型分歧，且携带了力大小之外的信息，但在后期失效。**

- 绝对分歧：ρ(UQ, D_F) = 0.44–0.53，全体 0.57（fig6，分歧随 UQ 十分位单调上升约 20 倍）。但 |F| 本身是更强的预测器（ρ 0.61–0.70）。
- 去掉量级效应后：早中期 ρ(UQ, D_F_rel) ≈ 0.15–0.21 而 ρ(|F|, D_F_rel) ≈ 0；固定 |F| 后 UQ 的条件 Spearman 0.13–0.30。
- t ≥ 0.8 时 UQ 对相对分歧的 AUROC 降到 0.47–0.59。

## 对假设的判断

- **RQ1（when）成立**：可靠性有明确的时间结构，门控的有效区间集中在 t ∈ [0.6, 1]。
- **RQ2（where）弱成立**：同一时间步内 UQ 与跨模型分歧有独立于几何的相关，但强度有限。
- **估计器需要改进**：UQ-MLIP 的默认目标（site energy 分位宽度）在接近流形的后期失去判别力，而这正是 guidance 起作用的区间。按 PROJECT_BRIEF 的转向条件，下一步优先研究 reliability estimator（例如直接以跨模型分歧为目标训练、embedding 距离、DPA 多模型 deviation），再进入 selective guidance。

## 与已有工作的关系

- **uMLIP 在分布外环境的系统性误差**：Deng 等（npj Comput. Mater. 2025，[Systematic softening in universal MLIPs](https://www.nature.com/articles/s41524-024-01500-6)）证明 M3GNet、CHGNet、MACE-MP-0 在高能态与分布外原子环境下系统性低估力。本项目把这一现象放到生成轨迹的时间维与原子维上刻画。
- **uMLIP 的不确定性估计**：[Heterogeneous Ensemble 一文](https://arxiv.org/pdf/2507.21297)报告 MACE 的 quantile regression 与 Orb 的置信头在 OOD 检测上效果有限，特征空间距离与误差相关更强。与本项目第三个发现一致，并支持阶段 1 把 embedding 距离列为候选 estimator。
- **力场梯度引导生成**：[Generative structure search](https://arxiv.org/pdf/2604.27636) 在每一步扩散中加入外部力场的能量梯度；OMatG-IRL 用 MACE 能量作 RL 奖励；DiffCSP 用学习到的能量网络做梯度引导。这些工作默认力场在轨迹各处可靠，未对该前提做检验。
- **时间相关的引导调度**：图像领域已有工作指出不精确的物理引导会淹没扩散模型自身形成的结构，需随时间衰减引导（如 [generative prior for ptychography](https://arxiv.org/pdf/2507.17800)）。本项目的发现为晶体生成中这类调度提供了可测量的依据，并进一步细化到原子级。

截至 2026-10-16 的检索未发现对大原子模型可靠性沿晶体生成轨迹做逐原子刻画的工作，但未做系统综述，不以"首个"自称。

## 局限

- 单个训练 seed、256 条轨迹；epoch 1300 的 64 条轨迹复跑给出一致结论，说明结果对 checkpoint 和样本量不敏感。
- 早期有 20% 的结构因原子重叠被预过滤，早期可信度被低估。
- 跨模型分歧是真实误差的代理，不是 DFT。
- UQ 的 GBM 在验证集覆盖率 0.78（名义 0.90），只作排序用。
- 中间态的原子类型用最终步回填，只测量几何 OOD。

## 下一步（申请后、实习期间）

1. 比较 reliability estimator：site-energy 分位宽度 vs 以 D_F 为目标的 GBM vs embedding 距离 vs DPA 模型 deviation。
2. 固定 guidance baseline，测量 negative transfer rate。
3. 基于最优 estimator 的 atom-wise selective guidance（TrustFlow），paired experiments。

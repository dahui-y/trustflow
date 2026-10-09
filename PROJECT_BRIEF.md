# TrustFlow 项目说明

我们正在准备一个面向**深势科技（DP Technology）AI for Science 实习申请**的研究项目。这个项目不仅需要能够运行代码，还需要体现明确的 AI4Science 研究问题、材料科学应用价值、实验可验证性，以及在较短周期内形成 preliminary results 的能力。

深势科技的研究重点主要集中在材料、化学、分子模拟、Large Atomic Models、科学基础模型与 AI for Science。申请要求提交一份约 500 字的项目提议，核心包含：具体目标、核心价值和可行性简析。因此，本项目必须同时满足两个要求：一是具有真实研究价值，而不是简单复现已有工作；二是能够在约 2–4 周内形成可展示的代码、实验和结果。

项目暂定名称为：

## TrustFlow
### Knowing When and Where to Trust Large Atomic Models in Crystal Generative Flows
### 面向晶体生成的可信大原子模型物理引导

当前晶体生成模型，例如 FlowMM、CrystalFlow、MatterGen 等，可以学习真实材料的结构分布；DPA、MACE、CHGNet 等机器学习原子势模型（MLIP / Large Atomic Model）可以预测结构的能量、力和其他物理信息。因此，一个自然思路是在生成过程中利用 MLIP 提供物理梯度，引导生成模型得到更加稳定的材料。

然而，已有研究已经表明，**“使用能量或物理梯度引导生成”本身并不是新的问题**。DiffCSP、DAO、OMatG-IRL、CrystalREPA 等工作已经从不同角度研究了 energy guidance、inference-time optimization 或从 MLIP 中注入物理知识。

因此，本项目不能简单实现：

`FlowMM + MLIP force guidance`

并将其作为创新。

TrustFlow 真正关注的问题是：

> **生成过程中，Large Atomic Model 并不是在所有时间、所有局部原子环境中都可靠。能否先判断“什么时候、什么位置值得相信它”，再选择性地使用物理引导？**

晶体生成早期的 intermediate structures 可能具有异常原子间距、不合理配位、异常晶格或者处于 MLIP 训练分布之外。此时 MLIP 输出的 force / energy gradient 可能存在较大误差。如果仍然无条件使用这些梯度，可能造成 negative transfer，即所谓的“physics guidance”反而让最终生成结构变差。

因此，本项目研究：

$$
(t,i)\rightarrow r_{i,t},
$$

其中 $t$ 表示生成时间步，$i$ 表示具体原子，$r_{i,t}\in[0,1]$ 表示 MLIP 在当前时间和当前局部原子环境下的可信程度。

最终物理引导不再是：

$$
g_{i,t}=F_{i,t},
$$

而是：

$$
g_{i,t}=r_{i,t}F_{i,t}.
$$

即：

**可靠位置 → 强物理引导；
不可靠位置 → 减弱或关闭物理引导。**

项目核心不是提出新的晶体 foundation model，也不是重新训练大型材料模型，而是研究：

> **Generative Foundation Model 与 Large Atomic Model 在 inference time 如何可靠协同。**

---

## 项目必须回答的三个 Research Questions

**RQ1：When?**
Large Atomic Model 沿 crystal generative trajectory 的可靠性是否随 sampling timestep 系统变化？

**RQ2：Where?**
在同一个 intermediate crystal 内，不同局部原子环境的 MLIP reliability 是否存在显著差异？

**RQ3：Can selective trust improve generation?**
基于 atom-wise reliability 的 selective physics guidance，是否能够相比 unconditional physics guidance 降低 negative transfer，同时维持或提高最终结构的稳定性？

---

## 项目的核心研究假设

本项目的核心假设为：

> MLIP reliability 在晶体生成轨迹中具有显著的 temporal 和 spatial heterogeneity；无条件使用 MLIP force guidance 会在低可信区域产生错误 steering，而 reliability-aware atom-wise gating 可以减少这种 failure。

首先需要通过实验验证该假设，而不能直接假定其成立。

如果实验显示 MLIP uncertainty 与真实 error 没有明显关系，则项目应暂停当前 guidance 方法，转而研究更加可靠的 reliability estimator。

如果 unconditional guidance 本身几乎没有 negative transfer，则应重新检查实验设置、OOD 程度和 guidance strength，而不能强行声称 TrustFlow 有优势。

---

## MVP 技术范围

MVP 阶段优先使用：

**FlowMM**：作为 crystal generative flow backbone。

**MP-20**：作为主要晶体数据集。

**CHGNet**：作为第一阶段 MLIP，用于快速得到 energy / force。

**UQ-MLIP 或 ensemble / embedding-based estimator**：用于得到 per-atom uncertainty。

**DPA / DeepMD**：作为第二阶段验证，用于提高与深势科技技术体系的相关性。

第一阶段不要求训练新的 foundation model。

第一阶段不做：

property optimization、LLM agent、reinforcement learning、multi-objective generation、智能实验室、完整 DFT 高通量筛选等。

这些内容只有在核心假设得到实验支持后才考虑扩展。

---

## 期望最终产出

项目完成后至少应形成：

- 一个可复现的 crystal generative trajectory reliability benchmark；
- 一个模块化的 TrustFlow implementation；
- baseline / fixed guidance / time-based guidance / reliability-aware guidance 的 paired experiments；
- MLIP uncertainty 随 timestep 和 atom/local environment 变化的可视化；
- Negative Transfer Rate、结构稳定性、validity、diversity 等结果；
- 一个结构清晰、可以直接展示给深势科技研究人员的 GitHub repository；
- 以及用于申请的约 500 字中文项目提议。

Claude Code 在后续开发过程中必须遵循一个原则：

> **不要一次实现整个 TrustFlow。应按照“复现 → 测量 → 分析 → 判断假设 → 再实现下一阶段”的研究流程逐步推进。**

任何方法设计都必须由前一步实验结果支持。

---

# 深势科技申请用项目 Proposal

## TrustFlow：面向晶体生成的可信大原子模型物理引导

**具体目标：**
晶体生成模型可结合 DPA、CHGNet 等大原子模型提供的能量和力信息，提高生成材料的物理稳定性。但生成过程中的中间结构可能偏离真实材料分布，此时原子势模型的预测未必可靠，无条件使用物理梯度可能反而破坏生成结果。本项目拟系统研究大原子模型在晶体生成轨迹中的可信度变化，并构建 TrustFlow：依据时间步和局部原子环境的可信度，对原子级物理力进行选择性加权，仅在可靠区域实施物理引导。预期产出包括生成轨迹可信度 benchmark、可插拔 TrustFlow 模块及晶体生成实验结果。

**核心价值：**
现有材料生成方法通常将物理模型视为始终可靠的 oracle，而 TrustFlow 将这一过程从“调用物理模型”升级为“调用—验证—选择性采用”。该机制有望减少错误物理梯度造成的 negative transfer，使材料生成基础模型与 DPA 等领域基础模型能够更可靠地协同，并作为可复用的科学验证模块接入后续材料生成与筛选工作流。

**可行性简析：**
项目基于开源 FlowMM 和 MP-20 构建晶体生成轨迹，利用 CHGNet/DPA 计算能量与力，并通过 ensemble disagreement、per-atom uncertainty 或局部环境 OOD 指标估计模型可信度。首先验证可信度与 MLIP 预测误差之间的相关性，再比较无引导、固定物理引导和 TrustFlow 在稳定性、结构有效性、多样性及 negative-transfer rate 上的差异。主要风险是不确定性指标无法准确预测真实误差；届时将比较多种 reliability estimator，并使用少量高质量参考计算进行校准。

---

## 仓库现状（供后续 session 参考）

- 本仓库是 FlowMM 官方代码的 fork，backbone 位于 `src/flowmm`，数据位于 `data/mp_20` 等。
- `help_code/` 下已放置参考代码：DiffCSP、OMatG、UQ-MLIP、chgnet、deepmd-kit。
- 下一步计划：阅读 `help_code` 与 `src/flowmm`，生成 `REFERENCE_CODE_MAP.md`，确认可复用模块、sampling hook 位置和数据格式衔接。该步骤不实现 TrustFlow 代码。

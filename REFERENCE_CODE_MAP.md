# REFERENCE_CODE_MAP.md

> 目的：在实现 Milestone 0 之前，确认 `src/flowmm` 与 `help_code/` 中哪些模块可以直接复用、生成轨迹的 hook 在哪里、各库之间的数据格式如何衔接。本文件只记录代码事实与由此得出的决策，不实现任何 TrustFlow 代码。
>
> 阅读范围：`src/flowmm`、`scripts_model`、`scripts_analysis`、`help_code/{UQ-MLIP, chgnet, deepmd-kit, DiffCSP, OMatG}`。所有行号基于当前分支 `claude/intelligent-davinci-6o2z1v`。

---

## 0. 一页结论

| 问题 | 结论 |
|---|---|
| FlowMM 能否导出完整生成轨迹 | **能，原生支持，无需改采样代码。** `scripts_model/evaluate.py:524` 的 `gen_trajectory` 命令通过 `entire_traj=True` 返回每个 Euler step 的 `(atom_types, frac_coords, lattices, lengths, angles)`。 |
| Milestone 0 的 MLIP 接入点 | 轨迹 `.pt` → `flowmm.pymatgen_.diffcsp_to_structure` → pymatgen `Structure` → `CHGNet.predict_graph(..., return_site_energies=True, return_atom_feas=True)`。全部现成。 |
| per-atom uncertainty 怎么来 | UQ-MLIP 提供 quantile-GBM on CHGNet 64-d `atom_fea`，**但不附带预训练 UQ 模型，且目标是 CHGNet 自己的 site energy，不是力误差**。需要自己训练，并把它当作 proxy。 |
| Experiment B 的第二个 MLIP | DPA-3 via `deepmd.infer.DeepPot("DPA-3.1-3M", head=...)`，权重自动下载。多任务模型，**必须指定 head**（候选 `MP_traj_v024_alldata_mixu` 或 `OMat24`）。 |
| 仓库里现在缺什么 | ① `remote/` 三个子模块（cdvae, DiffCSP-official, riemannian-fm）未 checkout，FlowMM **无法 import**；② `data/*.csv` 是 git-lfs 指针，不是数据；③ 没有任何 FlowMM checkpoint；④ 本容器无 GPU、无 torch。 |
| 最大时间风险 | FlowMM 没有官方发布 checkpoint（README 无下载链接，`.gitignore` 里有 `**/official_ckpts/` 暗示作者本地有）。若拿不到，需在 MP-20 上自训，这一步可能超出 3–7 天预算。 |

---

## 1. FlowMM（`src/flowmm`，backbone）

### 1.1 依赖与运行环境

- `environment.yml`：Python 3.9，torch 2.1.0 + cu118，pyg 2.4.0，pymatgen 2023.10.11，pytorch-lightning 1.8.5，**chgnet==0.3.1**，hydra 1.2，torchdiffeq，geoopt。
- 三个外部包以 editable 方式安装：`remote/cdvae`、`remote/DiffCSP-official`（bkmi fork）、`remote/riemannian-fm`（提供 `manifm`）。
- `src/flowmm` 对外部包的 import 统计：

| 被 import 的模块 | 用途 |
|---|---|
| `diffcsp.common.data_utils` | `lattices_to_params_shape`, `lattice_params_to_matrix_torch`, `radius_graph_pbc` 等 |
| `diffcsp.pl_modules.cspnet` | `CSPNet`/`CSPLayer` 基类（`src/flowmm/model/arch.py`） |
| `diffcsp.script_utils.GenDataset` | 生成时按训练集分布采样 `num_atoms`（`scripts_model/evaluate.py:17`） |
| `diffcsp.eval_utils`, `diffcsp.pl_data.*` | 数据模块、评估 |
| `manifm.solvers.projx_integrator` / `projx_integrator_return_last` | **采样积分器**（`model_pl.py:23`） |
| `manifm.model_pl.ManifoldFMLitModule`, `manifm.ema.EMA` | Lightning 基类与 EMA |

- **`help_code/DiffCSP` 是上游 jiaor17 版本，不是 bkmi fork。** 它没有 `diffcsp/script_utils.py`，等价物是 `help_code/DiffCSP/scripts/generation.py:104` 的 `SampleDataset`（按 `train_dist[dataset]` 采样原子数）。因此 `help_code/DiffCSP` **不能**直接替代 `remote/DiffCSP-official`。
- 本容器：Python 3.13，无 torch，无 GPU（`nvidia-smi` 不存在）。本仓库当前状态下 **任何 FlowMM 代码都跑不起来**，Milestone 0 必须在有 GPU 的机器上执行。

### 1.2 采样循环与轨迹导出（Milestone 0 核心）

调用链：

```
scripts_model/evaluate.py:574 gen_trajectory()
  └─ cfg.integrate.entire_traj = True  (evaluate.py:605)
  └─ GenDataset(dataset, total_num)  → DataLoader → pl.Trainer.predict
       └─ model_pl.py:1042 predict_step()
            └─ model_pl.py:919 compute_gen_trajectory()
                 └─ model_pl.py:221 gen_sample(entire_traj=True)
                      └─ model_pl.py:306 finish_sampling()
                           └─ model_pl.py:380 manifm.solvers.projx_integrator(...)  → xs, vs
```

关键事实：

- `finish_sampling` 在 `entire_traj=True` 时走 `projx_integrator`，返回 `xs` 形状 `(num_steps+1, B, D_flat)`（`model_pl.py:380-388`）。时间网格 `torch.linspace(0, 1, num_steps+1)`，默认 `num_steps=1000`（`scripts_model/conf/default.yaml` 的 `integrate.num_steps`）。
- `compute_gen_trajectory`（`model_pl.py:919-965`）对每个 step 调用 `manifold_getter.flatrep_to_crystal`，输出 dict：

| key | shape | 说明 |
|---|---|---|
| `atom_types` | `(T+1, N_total)` 或 `(T+1, N_total, 7)` | analog bits 解码后为整数原子序数；解码在 `analog_bits_to_int` 按符号取整，**中间步也会给出一个"当前猜测"的元素** |
| `frac_coords` | `(T+1, N_total, 3)` | torus 上，`[0,1)` |
| `lattices` | `(T+1, B, 3, 3)` | 由 lattice params 转矩阵 |
| `lengths`, `angles` | `(T+1, B, 3)` | Å 与度 |
| `num_atoms` | `(B,)` | |
| `input_data_batch` | PyG Batch | |

- `TorchPredictionWriter`（`evaluate.py:43`）把每个 rank 的预测写到 `<ckpt_dir>/<subdir>/gen_trajectory_00/predictions_??.pt`，并写 `num_steps.txt`。
- `evaluate.py consolidate`（`evaluate.py:786 _consolidate`）对 trajectory 任务沿 **dim=1** 拼接各 batch，输出 `consolidated_gen_trajectory.pt`。
- 默认 `--num_samples 256 --batch_size 256`，正好匹配 Milestone 0 的"几十到几百条轨迹"。
- `--compute_traj_velo_norms` 可额外输出每步在 atom/coord/lattice 三个流形上的速度范数（`model_pl.py:402-430`），可作为"生成进度"的一个免费附加指标。
- self-conditioning 模型（`cfg.model.self_cond=True`）走 `projx_cond_integrator_return_last`（`src/flowmm/model/solvers.py:28`），**不支持 entire_traj**。Milestone 0 必须用 `self_cond=false` 的 checkpoint（默认配置 `abits_params.yaml` 即是）。

**结论：Milestone 0 的轨迹导出零改动。**

### 1.3 未来 guidance hook 位置（仅记录，Milestone 0 不实现）

- 向量场包装函数 `scheduled_fn_to_integrate`（`model_pl.py:348-366`）：接收 `(t, x_flat, cond)`，调用 `self.vecfield`，然后按 `inference_anneal_*` 对 `out[:, dims.a]`、`out[:, dims.a:-dims.l]`、`out[:, -dims.l:]` 三段分别乘 anneal factor。**这就是将来加 `r_{i,t} F_{i,t}` 的唯一位置**：它已经按 (atom types | frac coords | lattice) 切片，加项只需填入 coords 段。
- 需要处理的坐标转换（现有代码均无）：MLIP 给 Cartesian 力 `F` (eV/Å)，flow 在分数坐标 torus 上。`frac_force = F @ inv(L)^T` 之类的转换要自己写；`help_code/DiffCSP/diffcsp/common/data_utils.py` 的 `frac_to_cart_coords` / `cart_to_frac_coords` 可参考。
- 切向投影：`ProjectedConjugatedCSPNet.forward`（`arch.py:604-630`）对输出做 `manifold.proju`，`MaskedNoDriftFlatTorus01.proju`（`flat_torus.py:140-158`）会**减去每个晶体的平均速度**。任何注入的力也必须去掉质心漂移，否则会被投影吃掉或引入不一致。
- 现有的 anneal 机制本身就是一个 "time-based guidance" 的 baseline 雏形（`_annealing_schedule`，`model_pl.py:162`）。

### 1.4 几何表示（影响中间结构能否喂给 MLIP）

- 分数坐标：`MaskedNoDriftFlatTorus01`（`flat_torus.py:85`），`projx` 对 1 取模，所有中间步坐标都在 `[0,1)`，**可直接构造 pymatgen Structure**。
- 晶格：`LatticeParams`（`lattice_params.py:139`），长度经 log-space 正欧氏，角度经 `UnconstrainedCompact` 映射到 `(59.9°, 120.1°)`。因此**任何中间步的晶格都是合法的**（正长度、合理角度），不会出现奇异矩阵。但早期长度可能很小或很大 → 原子重叠或孤立。
- 原子类型：`MultiAtomAnalogBits`（`analog_bits.py:122`），7 bits 编码 0–127，`NUM_ATOMIC_TYPES=100`。中间步解码可能得到 >94 或 0 的无效原子序数。`diffcsp_to_structure(..., clip_atom_types=True)` 会 clip 到 `[0,118]`，但 **CHGNet 只支持 Z=1–94**（`AtomEmbedding(max_num_elements=94)`），需要额外过滤或把早期步的原子类型固定为最终类型（见 §6 决策）。
- 向量场图：`CSPNet.gen_edges`（`arch.py:332`）默认 `edge_style=fc` 全连接 + torus logmap，不依赖 cutoff，所以模型本身对异常距离不敏感；敏感的是下游 MLIP。

### 1.5 轨迹 → pymatgen Structure

- `src/flowmm/pymatgen_.py:94 diffcsp_to_structure(gen, i, clip_atom_types)`：输入 dict（`num_atoms`, `lengths`, `angles`, `atom_types`, `frac_coords`，按晶体 flat 排布，用 `cumsum(num_atoms)` 切片），输出 `Structure`。对 trajectory 文件，先取某个 step 的切片 `gen_t = {k: v[t] for k in [...]}` 再调用即可。
- `pymatgen_.py:179 torch_geometric_to_structures`：批量版本，joblib 并行，异常返回 `None`。
- `scripts_analysis/prerelax.py:44-80` 展示了"加载 `.pt` → squeeze → 逐个 `get_structure` → CHGNet"的完整样例，可直接改成"逐 step"。

### 1.6 FlowMM 已有的 CHGNet 集成

- `src/flowmm/chgnet_.py`：`prerelax_with_chgnet`（`CHGNet.load()` + `predict_structure` + `StructOptimizer.relax`），`structures_to_chgnet_graphs`（joblib 并行建图）。只用于生成后 relax，**不碰 per-atom 特征，不做 batch**。
- 仅调用方：`scripts_analysis/prerelax.py`。输出 `RelaxationData`（`e_gen`, `e_relax`, `n_to_relax`, `rms_dist`, `matched`, ...）。
- `environment.yml` 固定 `chgnet==0.3.1`，而 `help_code/chgnet` 是 0.4.1（torch>=2.4.1）。两者 API 一致，但 **0.4.1 要求的 torch 与 FlowMM 的 torch 2.1 冲突**，见 §7。

### 1.7 评估指标

- `src/flowmm/old_eval/generation_metrics.py:26 GenEval`：`comp_valid`, `struct_valid`（最小原子间距 ≥0.5 Å 且体积 ≥0.1，见 `help_code/DiffCSP/scripts/eval_utils.py:214`）, `wdist_density`, `wdist_num_elems`, `wdist_prop`, `cov_recall`, `cov_precision`。需要 ≥1000 个 valid 样本才算分布指标。
- `old_eval/lattice_metrics.py`：晶格参数分布对比。
- `scripts_analysis/ehull.py`：需要 Matbench Discovery 的 MP hull 文件（`mp_02072023/`，未包含）。
- **Milestone 0 只需要 `struct_valid` 的判据作为"是否喂给 MLIP"的预过滤**，不需要跑完整 GenEval。

### 1.8 数据

- `data/mp_20/{train,val,test}.csv` 是 LFS 指针（`.gitattributes: *.csv filter=lfs`），`git lfs ls-files` 显示 10 个文件，`git lfs pull` 后才有内容。真实列：`material_id, formation_energy_per_atom, band_gap, pretty_formula, e_above_hull, elements, cif, spacegroup.number, ...`，FlowMM 只读 `cif` 与 `prop`。
- `scripts_model/conf/data/mp_20.yaml`：`max_atoms: 20`, `niggli: true`, `graph_method: crystalnn`，首次加载会预处理成 `train_ori.pt` 等。

### 1.9 仓库卫生问题（会直接影响 Milestone 0 产出）

| `.gitignore` 规则 | 后果 |
|---|---|
| `*.png` | **preliminary figures 会被忽略**，需要改成白名单或用 `git add -f` |
| `**/analysis` | 已经吞掉了 `help_code/OMatG/omg/analysis`（OMatG 因此 import 失败）；TrustFlow 自己的任何 `analysis/` 目录也会被忽略 |
| `**/results/`, `**/official_ckpts/`, `**/our_ckpts/` | checkpoint 与结果目录默认不入库，符合预期，但 `PRELIMINARY_RESULTS.md` 不能放在这些目录下 |

---

## 2. CHGNet（`help_code/chgnet`，v0.4.1）

### 2.1 推理 API（`chgnet/model/model.py`）

```python
CHGNet.load(model_name="0.3.0" | "0.2.0" | "r2scan", use_device=None)
model.predict_structure(structure | [structures], task="efsm",
    return_site_energies=False, return_atom_feas=False,
    return_crystal_feas=False, batch_size=16)
model.predict_graph(graph | [graphs], ...)   # 同上，跳过建图
```

输出（每个结构一个 dict，numpy，无梯度）：

| key | shape | 单位 / 含义 |
|---|---|---|
| `e` | scalar | **eV/atom**（intensive） |
| `f` | (N,3) | eV/Å，autograd 对 Cartesian 坐标求导 |
| `s` | (3,3) | GPa |
| `m` | (N,) | μB |
| `site_energies` | (N,) | per-atom energy，含 composition-model 元素参考能 |
| `atom_fea` | (N,64) | **倒数第二层 conv 之后**的原子特征 |
| `crystal_fea` | (64,) | mean-pool |

- 单个输入返回裸 dict，不是长度 1 的 list。
- `batch_size` 只作用于前向，**建图是串行 pymatgen**。用 `flowmm.chgnet_.structures_to_chgnet_graphs` 并行建图再 `predict_graph`。
- 需要可微分力/梯度时直接调 `model.forward(graphs, task="ef", ...)`，`CrystalGraph.atom_frac_coord` 和 `lattice` 都是 `requires_grad=True`。

### 2.2 建图与异常结构行为（`chgnet/graph/converter.py`）

- `CrystalGraphConverter(atom_graph_cutoff=6, bond_graph_cutoff=3, on_isolated_atoms="error")`。
- **`CHGNet.load()` 得到的 converter 默认 `on_isolated_atoms="error"`**：任何原子在 6 Å 内无邻居 → `ValueError`。早期轨迹的大晶格会触发。修复：`model.graph_converter.set_isolated_atom_response("warn" | "ignore")`。
- bond graph 构建失败会把 `bond_graph_error.cif` 写到 **当前工作目录** 并抛 `RuntimeError`。批处理脚本要在 scratch 目录下运行并 try/except。
- **没有最小距离检查**。原子重叠时 `BondEncoder` 的 `bond_vectors / bond_lengths` 与 Bessel basis 的 `sin(freq*d)/d` 会产生 inf/NaN，不抛异常。必须自己预过滤（建议沿用 DiffCSP `structure_validity` 的 0.5 Å 阈值，并记录过滤比例作为轨迹早期的一个指标）。
- 元素范围 Z=1–94。
- Cython 快速建图 `cygraph.pyx` 需要编译，否则退回 legacy Python 实现（慢但可用）。

### 2.3 预训练权重

- 随包自带，无需下载：`chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar`（4.9 MB，MPtrj，力 MAE 68 meV/Å）。

---

## 3. UQ-MLIP（`help_code/UQ-MLIP`）

### 3.1 方法本质

- **只实现一种方法**：对 MLIP 的 per-atom embedding 训练 XGBoost quantile regression（`objective: reg:quantileerror`，默认 α=0.05/0.95），uncertainty = `|q_upper − q_lower| / 2`。
- **训练目标是 MLIP 自己预测的 per-atom energy**（CHGNet `site_energies` / MACE `node_energy`），**不是 DFT 残差，也不是力误差**。代码中没有任何读取参考力/能量的地方。
- 自己的 SKILL.md 承认：只量化 aleatoric 区间不确定性，不做 OOD 检测，分布偏移下 "GBM doesn't know it's extrapolating"。**这正是 FlowMM 早期中间结构的情形**，所以它作为 reliability estimator 的有效性本身是 RQ1/RQ2 要检验的东西，而非前提。
- 无 ensemble、无 latent-distance 方法。

### 3.2 可复用代码

| 组件 | 位置 | 说明 |
|---|---|---|
| `CHGNetExtractor` | `src/uq_mlip/backends/chgnet.py:21` | 输入 **ASE Atoms** 列表；内部 `predict_graph(graphs, task="e", return_site_energies=True, return_atom_feas=True)`；默认 `on_isolated_atoms="warn"` |
| `EmbeddingData` | `src/uq_mlip/data.py:12` | flat 数组：`node_feats (N_total,64)`, `node_energies (N_total,)`, `node_type`, `num_atoms (B,)`；npz 读写 |
| `UQModel.fit / predict_embeddings / from_dir` | `src/uq_mlip/model.py:46,116` | 训练 / 预测 / 加载；输出 `{"lower","upper","uncertainty"}` 每个 `(N_total,)` |
| `uq_evaluate_uncertainty` | `src/uq_mlip/integrations/scilink.py:186` | 一次调用：结构列表 → per-structure per-atom uncertainty |
| `run_embeddings_chgnet.py` | 仓库根 | 同逻辑的 CLI |

- **核心逻辑约 150 行**，若环境冲突可直接复制而不安装包。
- `run-gbm.py:67-71` 有 upper/lower 标签对调的 bug，不要用旧脚本路径。
- **不附带任何预训练 GBM**。需要先在 "CHGNet 训练分布内" 的结构上提取 embedding 并训练。可用数据源：MP-20 训练集本身（这些结构都在 MPtrj 里）。
- `EmbeddingData.node_energies` 只是一个数组，**可以塞入任意 per-atom 目标**（例如 CHGNet-vs-DPA 的 per-atom 力差）而不改库。这给了一个比默认方法更贴近 RQ 的 estimator 选项。

### 3.3 依赖

- `requires-python >=3.11,<3.14`；core 依赖 `ase, numpy, pandas, xgboost`（未 pin，`reg:quantileerror` 需要 xgboost ≥2.0）。
- 与 FlowMM Python 3.9 **不兼容安装**，但代码本身只依赖 numpy/xgboost，复制即可。

---

## 4. DeePMD-kit / DPA（`help_code/deepmd-kit`）

### 4.1 推理 API

```python
from deepmd.infer import DeepPot
dp = DeepPot("DPA-3.1-3M", head="MP_traj_v024_alldata_mixu")  # 自动下载到 ~/.cache/deepmd/pretrained/models
e, f, v = dp.eval(coords, cells, atom_types)                   # numpy
e, f, v, ae, av = dp.eval(coords, cells, atom_types, atomic=True)
desc, atom_feat, struct_feat = dp.eval_embedding(coords, cells, atom_types)
```

- `coords` 为 **Cartesian** `(nf, natoms, 3)` Å；`cells` `(nf, 9)` 行主序；`atom_types` 为 `dp.get_type_map()` 的索引。
- **一次调用内所有帧 natoms 必须相同**。MP-20 轨迹需按原子数分组，或逐结构调用。
- 多任务模型 **必须指定 `head`**，否则断言失败并列出可用 head（`deepmd/pt/infer/deep_eval.py:175-214`）。候选：`MP_traj_v024_alldata_mixu`（MPtrj，与 CHGNet 训练数据同源，适合做"独立模型、同分布"的对照）、`OMat24`。用 `dp --pt show <model.pt> model-branch type-map` 确认。
- ASE calculator：`deepmd.calculator.DP(model, head=...)`，逐帧，不暴露 atomic energy。
- 无最小距离检查；邻居超出 `sel` 会被静默丢弃。

### 4.2 预训练权重

- **不随仓库附带**，`deepmd/pretrained/registry.py` 列出 `DPA-3.1-3M`, `DPA-3.2-5M`, `DPA-3.3-1M`, `DPA-2.4-7M`, `DPA3-Omol-Large`, `DPA4-*-OMat24`，从 HuggingFace / hf-mirror / modelscope 下载并校验 sha256。
- `dpa_adapt/` 是属性预测的微调工具包，与力推理无关。

### 4.3 不确定性

- `deepmd/infer/model_devi.py`：`calc_model_devi_f(fs, atomic=True)` 给出 **per-atom** 跨模型力标准差。需要多个模型；可用不同 DPA 版本做 ad-hoc ensemble，但 `_check_tmaps` 要求 type_map 一致。
- `eval_embedding` 给 per-atom descriptor，可喂给 UQ-MLIP 的 GBM。

### 4.4 安装

- `pip install deepmd-kit[torch]`，torch ≥2.1；纯 Python PyTorch 路径可用，不需要 C++ 编译或 TensorFlow。
- `skills/deepmd-python-inference/SKILL.md` 是最相关的文档，并警告不要在循环中重复加载模型。

---

## 5. DiffCSP 与 OMatG（对照/参考，不进入 Milestone 0 代码路径）

### 5.1 DiffCSP（`help_code/DiffCSP`，上游 jiaor17 NeurIPS 2023 版）

- 采样循环 `diffcsp/pl_modules/diffusion.py CSPDiffusion.sample`：predictor-corrector，T=1000，`traj_stack` 已保存 `all_frac_coords (T+1,N,3)` 与 `all_lattices (T+1,B,3,3)`，但所有调用方都丢弃了它。
- **已有的 energy guidance**：`diffcsp/pl_modules/energy_model.py CSPEnergy.sample`，用一个学习到的 time-dependent 属性网络，对 **分数坐标和晶格矩阵直接求梯度**，`x -= std_x**2 * aug * grad_x`，默认 `aug=50`，无 clipping，无 Cartesian→fractional 转换。这是文献中 "classifier guidance" 的参考实现，也是 TrustFlow 将来 "fixed guidance" baseline 的形式参考。
- 可复用工具：`diffcsp/common/data_utils.py` 的 `frac_to_cart_coords`, `cart_to_frac_coords`, `lattice_params_to_matrix_torch`, `radius_graph_pbc`；`energy_model.py` 的 `RequiresGradContext`。
- 评估：`scripts/compute_metrics.py`，`eval_gen.pt` 格式 `{frac_coords (N_total,3), num_atoms (B,), atom_types (N_total,), lengths (B,3), angles (B,3)}`，与 FlowMM `consolidate` 输出兼容。
- 依赖栈 torch 1.9 / PL 1.3.8，**不要装进 FlowMM 环境**，只读代码。

### 5.2 OMatG（`help_code/OMatG`，v2.0.0，stochastic interpolants）

- **没有 force guidance，也没有 Cartesian→fractional 力转换。** MLIP（MACE-MPA-0）只出现在 OMatG-IRL 的 reward（`omg/omg_irl/rewards/energy_reward.py`）、能量指标和 relax 中。项目说明里提到的 "OMatG-IRL" 是 **policy-gradient 微调**（GRPO/PPO），不是 inference-time guidance。
- 与 TrustFlow 最相关的结构：`OMGIRLScale` 学习 `velocity = (1 + s_θ(t)) * base_b`（`omg_irl_lightning/omg_irl_scale.py:344`），即一个 per-atom、time-dependent 的速度缩放，是 "learned time-based gating" 的最近邻。
- `StochasticInterpolants.integrate(save_intermediate=True)` 可返回每步中间态；IRL 模块中的手写 Euler-Maruyama 循环（`omg_irl_velocity.py:213-346`）是将来加 per-atom gated 力项的最干净模板。
- 可复用小件：`omg/si/corrector.py PeriodicBoundaryConditionsCorrector.correct/unwrap`（最小镜像），`EnergyReward._is_valid`（体积 ≥0.1、最小距离 ≥0.5 Å、polar sine ≥1e-3 的预过滤）。
- 本仓库副本 **缺 `omg/analysis` 包**（被 `.gitignore` 的 `**/analysis` 吞掉），import 会失败。
- 依赖 torch 2.8 / pymatgen 2025 / mace-torch，与 FlowMM 环境不兼容，只读代码。

---

## 6. Milestone 0 数据流与决策

```
[GPU 机器, flowmm env]
  FlowMM ckpt (abits_params, mp_20, self_cond=false)
    │  python scripts_model/evaluate.py gen_trajectory <ckpt> --num_samples 256 --num_steps 1000 --single_gpu
    │  python scripts_model/evaluate.py consolidate <ckpt> --subdir ...
    ▼
  consolidated_gen_trajectory.pt     # atom_types (T+1,N), frac_coords (T+1,N,3), lengths/angles (T+1,B,3), num_atoms (B,)
    │  选取 K 个 timestep（如 t ∈ {0, .1, ..., 1} × T）
    │  每个 step: diffcsp_to_structure(gen_t, i, clip_atom_types=True) → Structure
    │  预过滤: Z∈[1,94]; min_dist ≥ 0.5 Å; volume ≥ 0.1; 记录各 step 的过滤比例
    ▼
[MLIP env, 可与 flowmm env 分离]
  CHGNet 0.3.0: predict_graph(..., return_site_energies=True, return_atom_feas=True)
    → e, f (N,3), site_energies (N,), atom_fea (N,64)
  UQ-MLIP GBM (自训于 MP-20 train 的 CHGNet embedding) → per-atom U_{i,t}
  [Exp B] DPA-3.1-3M (head=MP_traj) eval(atomic=True) → F_DPA
    → D_F,i = ||F_CHGNet,i − F_DPA,i||
    ▼
  per-atom table: (traj_id, t, atom_idx, Z, U, |F_CHGNet|, D_F, site_e, ...)
    → 图 1: U vs t (median + IQR, 按轨迹配对)
    → 图 2: high-U atom fraction vs t
    → 图 3 (Exp B): D_F vs U 散点 / 分箱, 按 t 着色
    → PRELIMINARY_RESULTS.md
```

**已确认的决策（2026-10-09）**

| # | 决策 | 结论 |
|---|---|---|
| 1 | FlowMM checkpoint 来源 | **在 MP-20 上自训**，按 FlowMM README 的 Unconditional Training 流程：`python scripts_model/run.py data=mp_20 model=abits_params` |
| 2 | 中间步原子类型 | 主图用最终步类型回填（只测几何 OOD）；每步解码的"当前猜测"作为补充 |
| 3 | reliability estimator 主选 | UQ-MLIP 默认 GBM（目标 = site energy）作为 Exp A；CHGNet-vs-DPA 力差 D_F 作为 Exp B 验证 |
| 4 | DPA head | 主用 `MP_traj_v024_alldata_mixu`，`OMat24` 作为 robustness 检查 |

**自训 FlowMM 的要点（来自 `scripts_model/run.py` 与 `conf/default.yaml`）**

- 默认 `train_max_epochs: 2000`（`conf/data/mp_20.yaml`），单卡、`precision: 32`、`gradient_clip_val: 0.5`。**先跑 1 个 epoch 测时间**再决定总 epoch 数。
- `every_n_epochs_checkpoint.every_n_epochs: 100, save_top_k: -1`：每 100 epoch 保存一个 checkpoint 到 `<run_dir>/every_n_epochs/`。这意味着 **训练到几百 epoch 时就可以用早期 checkpoint 启动轨迹审计**，不必等 2000 epoch 结束。另有 `monitor_metric: val/loss` 的 top-1 checkpoint。
- 输出目录由 `conf/hydra/trash.yaml` 决定：`./runs/trash/<date>/<time>/<model>-<vectorfield>-<id>/`，其中包含 `.hydra/config.yaml`。`load_cfg`（`eval_utils.py:140`）靠这个文件定位，**不要移动 checkpoint 离开其 run 目录**。`runs/` 已在 `.gitignore`。
- wandb 默认 `mode: online`。无账号时设置环境变量 `WANDB_MODE=offline` 或 `disabled`（`run.py:136` 读取 `WANDB_MODE`）。注意 `load_id_from_wandb` 等辅助函数依赖 run 目录下的 `wandb/` 子目录，但 `load_model` 不依赖。
- `preprocess_workers: 30`、`num_workers: 40` 按机器核数调低。首次加载会把 CSV 预处理为 `data/mp_20/{train,val,test}_ori.pt`（`.gitignore` 的 `*.pt` 规则会忽略它们）。
- `val_check_interval: 5` 是 epoch 为单位的验证频率，验证只算 loss（`val.compute_nll: false`），开销可接受。
- 模型配置 `abits_params`：`self_cond: false`，满足 §1.2 对 `entire_traj` 的要求。

---

## 7. 环境方案

两个环境，避免 torch 版本冲突：

| 环境 | 内容 | 用途 |
|---|---|---|
| `flowmm` | `environment.yml` 原样：py3.9, torch 2.1, pyg 2.4, chgnet 0.3.1, 三个 `remote/` 子模块 | 训练/采样/导出轨迹 |
| `mlip` | py3.11, torch ≥2.4, chgnet 0.4.1 (`help_code/chgnet`), deepmd-kit[torch], xgboost ≥2, 复制的 UQ-MLIP 核心 | 读 `.pt` 轨迹做 MLIP/UQ 评估 |

两者通过 `consolidated_gen_trajectory.pt`（纯 torch tensor dict）衔接，不共享任何 Python 包。`flowmm` 环境自带的 chgnet 0.3.1 也可以直接做 Exp A，但 DPA 需要新 torch，所以 Exp B 必然要第二个环境。

**实际建环境时踩到的坑（2026-10-09，RTX 4090 / 驱动 580 / conda 24）**

1. fork 丢失了子模块 gitlink，已用 `git submodule add` 重新注册并改为 https 地址。
2. `data/*.csv` 的 LFS 对象不在 fork 的 LFS 服务器上。`help_code/DiffCSP/data/mp_20/*.csv` 与指针 sha256 完全一致，直接 `cp` 过来，然后 `git update-index --assume-unchanged data/mp_20/*.csv` 防止误提交。
3. `remote/riemannian-fm/manifm/` 没有 `__init__.py`，新版 pip 的 editable 安装注册为空包。修法：把该目录写进 site-packages 的 `manifm.pth`。
4. conda 装的 Pillow 链接 `libtiff.so.5` 失败，`pip install --force-reinstall --no-deps pillow` 解决。
5. 直接 `from manifm.solvers import ...` 会触发 manifm 内部循环 import，FlowMM 自身的 import 顺序不会触发，验证环境时用 `from flowmm.model.model_pl import MaterialsRFMLitModule`。
6. pip 阶段把 pymatgen 升到 2024.8.9（matbench-discovery 拉的），暂未见问题。

**在当前仓库跑通 FlowMM 之前必须做的事**（这些不是 TrustFlow 代码）：

1. `git submodule update --init`（`.gitmodules` 用的是 `git@github.com:` SSH 地址，需改为 https 或配置 SSH）。
2. `git lfs pull` 取回 `data/mp_20/*.csv`。
3. `bash create_env_file.sh` 生成 `.env`（`PROJECT_ROOT` 等）。
4. 取得或训练 checkpoint，且其目录下必须有 `.hydra/config.yaml`（`load_cfg` 靠它定位，`eval_utils.py:111-150`）。

---

## 8. 下一步

按 PROJECT_BRIEF 的 Milestone 0：

1. 用户确认 §6 的四个决策（尤其 checkpoint 来源）。
2. 在 GPU 机器上完成 §7 的环境准备并跑通 `gen_trajectory` 一次（哪怕只有 8 条轨迹），确认 `.pt` 格式与本文件描述一致。
3. 写一个独立脚本 `scripts_analysis/trajectory_mlip_eval.py`：读轨迹 → 选步 → 构造 Structure → 预过滤 → CHGNet per-atom 输出 → 保存 parquet/npz。**不修改 `src/flowmm` 采样代码。**
4. Exp A 作图；若趋势存在，再加 DPA 做 Exp B。
5. `PRELIMINARY_RESULTS.md`。然后 **STOP**，不实现 guidance。

# mlip 环境（Experiment B：DPA 参考力）

独立于 `flowmm` 环境，只装 DPA 推理所需。两个环境通过 `outputs/m0/traj.npz` 和 parquet 表衔接。

```bash
conda create -y -n mlip python=3.11
conda activate mlip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "deepmd-kit[torch]" pymatgen pandas pyarrow matplotlib scipy
pip install -e . --no-deps          # 让 trustflow 可 import（flowmm 本身不会被用到）
python -c "import deepmd, trustflow.dpa_eval, torch; print(deepmd.__version__, torch.cuda.is_available())"
```

若 `download.pytorch.org` 走不通，去掉 `--index-url` 用默认源，pip 上的 torch 自带 CUDA 12.x 运行库，驱动 580 可用。

首次运行 `trajectory_dpa_eval.py` 会自动从 HuggingFace / hf-mirror / modelscope 下载 `DPA-3.1-3M.pt`（约 GB 级）到 `~/.cache/deepmd/pretrained/models`。

```bash
# 在 mlip 环境
python scripts_analysis/trajectory_dpa_eval.py outputs/m0/traj.npz outputs/m0/dpa_ep1300 --n_steps 11
# 任一环境（只需 pandas/matplotlib）
python scripts_analysis/cross_mlip_analysis.py outputs/m0/chgnet_ep1300 outputs/m0/dpa_ep1300 --out figures/m0
```

head 名称若报错，脚本会列出模型里可用的 head，用 `--head` 指定。

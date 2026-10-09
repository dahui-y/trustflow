# uq-mlip: Per-atom Uncertainty Quantification for MLIPs

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://github.com/pnnl/UQ-MLIP/blob/master/pyproject.toml)
[![Research](https://img.shields.io/badge/papers-npj%20Computational%20Materials-4b32c3)](#research-background)
[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue)](https://github.com/pnnl/UQ-MLIP/blob/master/LICENSE)

![uq-mlip banner: uncertainty quantification for neural network potentials in chemistry](https://raw.githubusercontent.com/pnnl/UQ-MLIP/master/docs/assets/banner-uq-mlip.png)

This repository contains code for training and evaluating per-atom uncertainty
quantification for machine learning interatomic potentials (MLIPs). The
recommended interface is the installable `uq-mlip` package, which provides a
unified command-line and Python API for the same extract, train, and predict
workflow available in the standalone scripts.

## New to UQ for NNPs?

> New to uncertainty quantification for neural network potentials? Start with
> the [UQ for NNPs primer](https://github.com/pnnl/UQ-MLIP/blob/master/docs/UQ_for_NNPs_Primer.pdf).

The primer gives a beginner-friendly overview of what uncertainty
quantification means for neural network potentials, why per-atom uncertainty is
useful, and how the workflow in this repository fits into simulation practice.

## Research background

`uq-mlip` implements uncertainty quantification workflows developed through two
peer-reviewed studies in
![npj Computational Materials](https://img.shields.io/badge/npj%20Computational%20Materials-4b32c3):

- [Uncertainty quantification for neural network potential foundation models](https://doi.org/10.1038/s41524-025-01572-y)
  introduces the UQ approach for neural network potential foundation models.
- [Assessing universal MLIP robustness with per-atom uncertainty for simulations of solid-liquid interfaces](https://doi.org/10.1038/s41524-026-02051-8)
  demonstrates per-atom uncertainty as a practical diagnostic for MLIP
  robustness in challenging interfacial simulations.

## Dataset

The simulation data used in these studies is available at
[this Google Drive folder](https://drive.google.com/drive/folders/1KcdqAS8SG_ybEOy0oI5X_bywJN2JSYUv?usp=sharing).

## Why per-atom UQ?

MLIP predictions can look stable even when local atomic environments are
outside the model's reliable domain. `uq-mlip` adds per-atom uncertainty
estimates so simulations can be inspected, filtered, or monitored at the level
where failures often begin.

## Quick start

`uq-mlip` makes per-atom uncertainty quantification easy to add to MLIP
workflows. The UQ model is specific to the MLIP and dataset, so users should
train a UQ model on validation or representative configurations before using it
in simulations.

MACE, UMA, and CHGNet are supported out of the box as default model backends:

```
git clone https://github.com/pnnl/UQ-MLIP.git
cd UQ-MLIP
git checkout master
pip install -e .
```

To include a backend in the same environment:

```
pip install -e ".[mace]"    # for MACE extraction
pip install -e ".[uma]"     # for UMA extraction
pip install -e ".[chgnet]"  # for CHGNet extraction
```

MACE and UMA currently depend on incompatible `e3nn` versions, so use separate
environments if you need to exercise both dependency stacks. CHGNet does not
depend on `e3nn` and can share an environment with either. After the first
PyPI release, this section will be updated to use `pip install uq-mlip`.

To automate backend-specific setup:

```
scripts/create_backend_env.sh mace
scripts/create_backend_env.sh uma
scripts/create_backend_env.sh chgnet
```

On macOS, XGBoost may also require the OpenMP runtime:

```
brew install libomp
```

For UMA on systems where `~/.cache` is not writable:

```
export FAIRCHEM_CACHE_DIR=/path/to/writable/fairchem-cache
```

Train a UQ model:

```
uq-mlip extract \
  --backend mace \
  --sample validation.xyz \
  --savedir embeddings/

uq-mlip train \
  --embeddings embeddings/embedding_info_validation.npz \
  --savedir uq-model/
```

Run the local hello world example to train a small UQ model, predict a synthetic
trajectory, and generate a UQ profile visualization:

```
python examples/hello_world/train_run_visualize.py
```

To make a tiny complete-frame fixture from a large XYZ/extXYZ trajectory:

```
python scripts/slice_xyz.py /path/to/large.xyz examples/test_data/my_slice.xyz --frames 8
```

To smoke-test a real backend in its own environment:

```
scripts/run_hello_world.sh mace
scripts/run_hello_world.sh uma
scripts/run_hello_world.sh chgnet
```

You can pass custom train/run slices to the backend smoke test without copying a
large trajectory into the repository:

```
scripts/run_hello_world.sh mace .venv-mace outputs/mace-small \
  examples/test_data/aimd_pbe_train.xyz examples/test_data/aimd_pbe_run.xyz
```

Use it in existing code with the decorator-style calculator:

```python
from uq_mlip import UQCalculator

atoms.calc = UQCalculator(
    base_calculator=mace_calc,
    uq_model="uq-model/",
    backend="mace",
    model="medium-0b",
)
```

Or use the convenience helper:

```python
from uq_mlip import with_uq

atoms.calc = with_uq(
    mace_calc,
    uq_model="uq-model/",
    backend="mace",
    model="medium-0b",
)
```

Energy and force calls continue to behave like the original calculator. After a
calculation, per-atom uncertainty is available as `atoms.arrays["uq"]`,
`atoms.arrays["uq_lower"]`, and `atoms.arrays["uq_upper"]`.

To verify a local checkout:

```
pip install -e ".[dev]"
python -m pytest
uq-mlip --help
python examples/hello_world/train_run_visualize.py
```

## Use with SciLink and other agent orchestrators

`uq-mlip` ships a tool plug-in that exposes the extract → train → predict
workflow to agent frameworks. It lives in
`uq_mlip.integrations.scilink` and provides three tools:

| Tool | Purpose |
|------|---------|
| `uq_extract_embeddings` | Extract per-atom embeddings + energies into a `.npz` bundle. |
| `uq_train_model` | Train the quantile-GBM UQ model on a validation-set bundle. |
| `uq_evaluate_uncertainty` | Score calibrated per-atom uncertainty for a trajectory/structures. |

The plug-in adds no dependency on SciLink. When SciLink is installed it
reuses SciLink's own `ToolSpec` dataclass; otherwise it falls back to a bundled
equivalent so the specs and their OpenAI/JSON schemas are available with
`uq-mlip` installed alone.

### SciLink

SciLink discovers a tool by importing a module, reading its `TOOL_SPEC` /
`TOOL_SPECS` attribute, and pairing each spec with a module-level callable of
the same `name` (see `scilink.skills._shared._registry`). The plug-in follows
that contract exactly, so the tools can be registered without editing SciLink.

`uq_evaluate_uncertainty` is a calibrated, per-atom drop-in for SciLink's
force-variance heuristic in `mlip_tools.evaluate_uncertainty`: it returns the
same result keys (`per_structure`, `mean_energy_uncertainty`,
`n_extrapolating`, `extrapolation_indices`) plus per-atom prediction intervals.

Register it inside a SciLink `simulation` skill bundle by re-exporting the
plug-in from a `.py` module in the skill folder — the registry pairs each spec
with the callable of the same name:

```python
# scilink/skills/<domain>/<skill>/uq_mlip_tools.py
from uq_mlip.integrations.scilink import (
    TOOL_SPECS,
    uq_extract_embeddings,
    uq_train_model,
    uq_evaluate_uncertainty,
)
```

Then resolve/inspect the tools through SciLink's registry:

```python
from scilink.skills._shared._registry import get_tools_for, get_tool_function

specs = get_tools_for("simulation", active_skills=["<skill>"])
report = get_tool_function("uq_evaluate_uncertainty", active_skills=["<skill>"])(
    backend="mace",
    uq_model="uq-model/",
    structures="md_run.xyz",
    model_size="medium-0b",
)
bad_frames = report["extrapolation_indices"]
```

### Claude Code, Codex, Cline, and other tool-calling agents

For orchestrators that consume OpenAI-style function schemas, the plug-in emits
ready-made tool schemas and a dispatcher:

```python
from uq_mlip.integrations.scilink import openai_tool_schemas, get_tool_function

tools = openai_tool_schemas()          # pass to the model's `tools=[...]`

# When the model calls a tool, dispatch by name:
result = get_tool_function(tool_name)(**tool_args)
```

`tool_prompt()` returns the same tools rendered as markdown for prompt
injection when a framework prefers documented tools over a schema list.

Because every tool is also a plain function, the simplest integration for a CLI
agent (Claude Code, Codex, Cline, ...) is to drive the `uq-mlip` command
directly — the agent runs shell commands, and the CLI mirrors the tools:

```
# 1. extract embeddings from validation configurations
uq-mlip extract --backend mace --sample validation.xyz \
  --savedir embeddings/ --model-size medium-0b

# 2. train the per-atom UQ model
uq-mlip train --embeddings embeddings/embedding_info_validation.npz \
  --savedir uq-model/

# 3a. extract embeddings for the MD trajectory to score
uq-mlip extract --backend mace --sample md_run.xyz \
  --savedir embeddings/ --model-size medium-0b

# 3b. predict per-atom uncertainty (writes UQ_<stem>.csv.gz)
uq-mlip predict --embeddings embeddings/embedding_info_md_run.npz \
  --model-dir uq-model/ --savedir results/
```

The `uq_evaluate_uncertainty` tool collapses steps 3a+3b into one call and
returns per-frame / per-atom results in memory (with `extrapolation_indices`),
which is the more convenient form when an agent needs to branch on the result;
the CLI path is preferred when the agent should persist a CSV artifact.

## Standalone script interface

The commands below expose the same workflow through the original script-level
entry points. They remain useful for direct inspection, debugging, and
reproducing the paper-era workflow.

### Extract embeddings and per-atom energies

To train the GBM model, first extract per-atom embeddings and per-atom energies
from a trained MLIP. The following commands provide methods to extract this
information for MACE, UMA, and CHGNet. If using a finetuned checkpoint, the
`--checkpoint` flag can be used to specify the path to the checkpoint file. The
sample should be in a format readable by ASE and contain configurations *from
the validation set* used to train or finetune the MLIP.

```
python run_embeddings_mace.py \
  --sample data/example.xyz \
  --savedir data/embeddings_mace \
  --model-size medium-0b \
  --index ":"
```

```
python run_embeddings_uma.py \
  --sample data/example.xyz \
  --savedir data/embeddings_uma \
  --model-size uma-s-1p1 \
  --head 'omat' \
  --index ":"
```

```
python run_embeddings_chgnet.py \
  --sample data/example.xyz \
  --savedir data/embeddings_chgnet \
  --model-size 0.3.0 \
  --index ":"
```

CHGNet is a crystal model and expects periodic inputs. Non-periodic
configurations (e.g. isolated molecules) are automatically wrapped in a vacuum
box before graph construction.


### Train GBM on per-atom embeddings and energies

Once the embeddings and energies have been extracted, the GBM model can be
trained using the following command.

```
python train-gbm.py --embeddings data/embeddings_mace/embedding_info_example.npz \
  --savedir data/gbm_mace \
  --upper-alpha 0.95 \
  --lower-alpha 0.05 \
  --estimators 1000
```


### Compute per-atom uncertainties using the trained GBM model

To compute per-atom uncertainties for a trajectory produced using the MLIP, the
per-atom embeddings must be extracted in the same way as described above.

```
python run_embeddings_mace.py \
  --sample data/md_run.xyz \
  --savedir data/embeddings_mace \
  --model-size medium-0b \
  --index ":"
```

Then, the following command can be used to compute per-atom uncertainties using
the trained GBM model.

```
python run-gbm.py --embeddings 'data/embeddings/embedding_info_md_run.npz' --savedir 'results/gbm_mace'
```

## Citation
If you use this model or code in your research, please cite the following papers:

```bibtex
@article{Bilbrey2025,
  author = {Bilbrey, Jenna A. and Firoz, Jesun S. and Lee, Mal-Soon and Choudhury, Sutanay},
  title = {Uncertainty quantification for neural network potential foundation models},
  journal = {npj Computational Materials},
  year = {2025},
  volume = {11},
  number = {1},
  pages = {109},
  doi = {10.1038/s41524-025-01572-y},
  url = {https://doi.org/10.1038/s41524-025-01572-y}
}

@article{Bilbrey2026,
  author = {Bilbrey, Jenna A. and Firoz, Jesun S. and Allec, Sarah I. and Sprueill, Henry W. and von Rueden, Alexander D. and Jackson, Benjamin A. and Raugei, Simone and Lee, Mal-Soon and Choudhury, Sutanay},
  title = {Assessing universal MLIP robustness with per-atom uncertainty for simulations of solid-liquid interfaces},
  journal = {npj Computational Materials},
  year = {2026},
  volume = {12},
  pages = {291},
  doi = {10.1038/s41524-026-02051-8},
  url = {https://doi.org/10.1038/s41524-026-02051-8}
}
```
## Acknowledgements
This work was supported by the "Transferring exascale computational chemistry to cloud computing environment and emerging hardware technologies (TEC4)" project, which is funded by the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences, the Division of Chemical Sciences, Geosciences, and Biosciences (under FWP 82037). The simulation data used in this study was supported by the U.S. Department of Energy (DOE), Office of Science, Office of Basic Energy Sciences, Division of Chemical Sciences, Geosciences & Biosciences (under FWP 47319). Computational research was partly supported by a DOE Office of Science Graduate Student Research (SCGSR) award. The SCGSR program is administered by the Oak Ridge Institute for Science and Education (ORISE) for the DOE. ORISE is managed by ORAU under Contract No. DE-SC001-4664. This work used resources of the National Energy Research Scientific Computing Center (NERSC), a DOE Office of Science User Facility (Contract No. DE-AC02-05CH11231), using NERSC awards BES-ERCAP0032414 and BES-ERCAP0032416. Pacific Northwest National Laboratory (PNNL) is a multiprogram national laboratory operated for the U.S. Department of Energy (DOE) by Battelle Memorial Institute under Contract No. DE-AC05-76RL0-1830.
## Visual assets

The README banner image was generated with ChatGPT.

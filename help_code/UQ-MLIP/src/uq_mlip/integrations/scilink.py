"""SciLink tool plug-in for the uq-mlip workflow.

This module adapts uq-mlip's extract → train → predict workflow to the tool
contract SciLink's registry expects, and to any other tool-calling orchestrator
(Claude Code, Codex, Cline, ...) via OpenAI function-call schemas.

SciLink discovers tools by importing a module and reading its ``TOOL_SPEC`` /
``TOOL_SPECS`` attribute, then pairing each spec with a module-level callable of
the same ``name`` (see ``scilink.skills._shared._registry``). The three public
functions here — :func:`uq_extract_embeddings`, :func:`uq_train_model`,
:func:`uq_evaluate_uncertainty` — are named to match the specs in
``TOOL_SPECS`` so the registry can resolve them directly.

uq-mlip does not depend on SciLink. When SciLink is importable this module
reuses its real ``ToolSpec`` dataclass so the specs are byte-for-byte what the
registry produces; otherwise it falls back to a bundled shim exposing the same
``to_prompt`` / ``to_openai_schema`` methods, so the specs (and their
OpenAI/JSON-schema renderings) are available with uq-mlip installed alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from uq_mlip.backends import get_extractor
from uq_mlip.data import EmbeddingData
from uq_mlip.model import UQModel

# ── ToolSpec: SciLink's if available, else a compatible local shim ──────────
try:  # pragma: no cover - exercised only inside a SciLink environment
    from scilink.skills._shared._spec import ToolSpec  # type: ignore

    _HAVE_SCILINK_SPEC = True
except Exception:  # SciLink not installed — provide a drop-in equivalent.
    from dataclasses import dataclass, field

    _HAVE_SCILINK_SPEC = False

    @dataclass
    class ToolSpec:  # type: ignore[no-redef]
        """Local mirror of ``scilink.skills._shared._spec.ToolSpec``.

        Keeps the same fields and the ``to_prompt`` / ``to_openai_schema``
        renderings so specs work standalone. When SciLink is installed its own
        ToolSpec is used instead and this class is never defined.
        """

        name: str
        description: str
        parameters: dict = field(default_factory=dict)
        required: list = field(default_factory=list)
        import_line: str = ""
        signature: str = ""
        agents: list = field(default_factory=list)
        when_to_use: str = ""
        returns: str = ""
        example: str = ""

        def to_prompt(self) -> str:
            lines = [f"### `{self.name}`"]
            if self.description:
                lines.append(self.description)
            if self.when_to_use:
                lines.append(f"**When to use:** {self.when_to_use}")
            if self.import_line:
                lines.append(f"**Import:** `{self.import_line}`")
            if self.signature:
                lines.append(f"**Signature:** `{self.signature}`")
            if self.parameters:
                lines.append("**Parameters:**")
                for pname, pinfo in self.parameters.items():
                    pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else str(pinfo)
                    ptype = pinfo.get("type", "") if isinstance(pinfo, dict) else ""
                    req_marker = " *(required)*" if pname in self.required else ""
                    type_str = f" ({ptype})" if ptype else ""
                    lines.append(f"- `{pname}`{type_str}{req_marker}: {pdesc}")
            if self.returns:
                lines.append(f"**Returns:** {self.returns}")
            if self.example:
                lines.append(f"**Example:**\n```python\n{self.example}\n```")
            return "\n".join(lines)

        def to_openai_schema(self) -> dict:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": self.parameters,
                        "required": self.required,
                    },
                },
            }


# ── helpers ─────────────────────────────────────────────────────────────────

def _as_atoms_list(structures: Union[str, Path, Sequence[Any]], index: str = ":") -> List[Any]:
    """Normalize the ``structures`` argument to a list of ``ase.Atoms``.

    Accepts a trajectory/extXYZ path (anything ASE can read) or an in-memory
    sequence of Atoms. A path is the natural form when an orchestrator passes a
    filename by reference; the in-memory form is what SciLink's MLIP agent
    already holds after ``ase.io.read``.
    """
    if isinstance(structures, (str, Path)):
        from ase.io import read

        frames = read(str(structures), index=index)
        return frames if isinstance(frames, list) else [frames]
    return list(structures)


def _split_per_structure(values: np.ndarray, num_atoms: Sequence[int]) -> List[np.ndarray]:
    """Slice a flat per-atom array back into one array per structure."""
    out, start = [], 0
    for n in num_atoms:
        out.append(np.asarray(values[start : start + n]))
        start += int(n)
    return out


# ── tools ─────────────────────────────────────────────────────────────────

def uq_extract_embeddings(
    backend: str,
    sample: str,
    savedir: str,
    model_size: Optional[str] = None,
    checkpoint: Optional[str] = None,
    index: str = ":",
    device: str = "cpu",
    head: str = "omat",
    batch_size: int = 4,
) -> str:
    """Extract per-atom embeddings + per-atom energies from an MLIP.

    Thin wrapper over the backend extractors. Returns the path to the written
    ``embedding_info_<stem>.npz`` bundle, ready to feed :func:`uq_train_model`
    (validation set) or :func:`uq_evaluate_uncertainty` (a trajectory).
    """
    kwargs: Dict[str, Any] = {"checkpoint": checkpoint, "device": device}
    if model_size is not None:
        kwargs["model"] = model_size
    if backend == "uma":
        kwargs["head"] = head
        kwargs["batch_size"] = batch_size
    elif backend == "chgnet":
        kwargs["batch_size"] = batch_size
    extractor = get_extractor(backend, **kwargs)
    output = extractor.extract_file(sample, savedir, index=index)
    return str(output)


def uq_train_model(
    embeddings: str,
    savedir: str,
    lower_alpha: float = 0.05,
    upper_alpha: float = 0.95,
    estimators: int = 100,
    device: str = "cpu",
) -> str:
    """Train a quantile-GBM per-atom UQ model on an embeddings bundle.

    ``embeddings`` is the ``.npz`` produced by :func:`uq_extract_embeddings` on
    *validation-set* configurations. Returns the path to the saved model
    pickle (pass its parent directory as ``uq_model`` to
    :func:`uq_evaluate_uncertainty`).
    """
    model = UQModel.train_from_file(
        embeddings,
        savedir,
        lower_alpha=lower_alpha,
        upper_alpha=upper_alpha,
        n_estimators=estimators,
        device=device,
    )
    return str(model.model_path)


def uq_evaluate_uncertainty(
    backend: str,
    uq_model: str,
    structures: Union[str, Sequence[Any]],
    model_size: Optional[str] = None,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    index: str = ":",
    lower_alpha: float = 0.05,
    upper_alpha: float = 0.95,
    extrapolation_percentile: float = 95.0,
    head: str = "omat",
    batch_size: int = 4,
) -> Dict[str, Any]:
    """Score per-atom uncertainty for structures with a trained UQ model.

    This is a calibrated, per-atom drop-in replacement for the force-variance
    heuristic in SciLink's ``mlip_tools.evaluate_uncertainty``: it extracts the
    same MLIP embeddings used at training time and predicts per-atom prediction
    intervals with the trained quantile-GBM model.

    Args:
        backend:      MLIP backend the UQ model was trained against ("mace" |
                      "uma" | "chgnet"). Must match the training backend/model.
        uq_model:     Directory holding the trained UQ model pickle (output of
                      :func:`uq_train_model`), or the pickle path's directory.
        structures:   Trajectory/extXYZ path or list of ``ase.Atoms``.
        model_size:   Backend model identifier used for extraction (must match
                      the model the UQ model was trained on).
        extrapolation_percentile: A structure is flagged as extrapolating when
                      its mean per-atom uncertainty exceeds this percentile of
                      the per-structure distribution over the given frames.

    Returns:
        A dict shaped like SciLink's ``evaluate_uncertainty`` result, enriched
        with calibrated per-atom values::

            {
              "method": "uq-mlip-quantile-gbm",
              "backend": str,
              "per_structure": [
                 {"index": int, "n_atoms": int,
                  "mean_uncertainty": float, "max_uncertainty": float,
                  "energy_uncertainty": float,    # alias of mean_uncertainty
                  "max_atom_index": int, "is_extrapolation": bool,
                  "per_atom_uncertainty": [float, ...],
                  "elements": [int, ...]},
                 ...
              ],
              "mean_energy_uncertainty": float,
              "max_energy_uncertainty": float,
              "n_extrapolating": int,
              "extrapolation_indices": [int, ...],
              "extrapolation_threshold": float,
            }
    """
    atoms_list = _as_atoms_list(structures, index=index)

    ext_kwargs: Dict[str, Any] = {"checkpoint": checkpoint, "device": device}
    if model_size is not None:
        ext_kwargs["model"] = model_size
    if backend == "uma":
        ext_kwargs["head"] = head
        ext_kwargs["batch_size"] = batch_size
    elif backend == "chgnet":
        ext_kwargs["batch_size"] = batch_size
    extractor = get_extractor(backend, **ext_kwargs)

    model = UQModel.from_dir(
        Path(uq_model),
        lower_alpha=lower_alpha,
        upper_alpha=upper_alpha,
        device=device,
    )

    # Extract every frame into a single bundle, then predict once.
    bundle: EmbeddingData = extractor.extract(atoms_list)
    predictions = model.predict_embeddings(bundle)
    per_atom = np.asarray(predictions["uncertainty"], dtype=float)

    num_atoms = [int(n) for n in bundle.num_atoms]
    unc_by_structure = _split_per_structure(per_atom, num_atoms)
    type_by_structure = _split_per_structure(np.asarray(bundle.node_type), num_atoms)

    structure_means = np.array([float(np.mean(u)) if u.size else 0.0 for u in unc_by_structure])
    finite = structure_means[np.isfinite(structure_means)]
    threshold = float(np.percentile(finite, extrapolation_percentile)) if finite.size else float("inf")

    per_structure: List[Dict[str, Any]] = []
    extrapolation_indices: List[int] = []
    for i, unc in enumerate(unc_by_structure):
        mean_u = float(np.mean(unc)) if unc.size else 0.0
        max_u = float(np.max(unc)) if unc.size else 0.0
        is_extrap = bool(mean_u > threshold) if finite.size else False
        if is_extrap:
            extrapolation_indices.append(i)
        per_structure.append(
            {
                "index": i,
                "n_atoms": int(num_atoms[i]),
                "mean_uncertainty": mean_u,
                "max_uncertainty": max_u,
                # alias so callers keyed on SciLink's field keep working
                "energy_uncertainty": mean_u,
                "max_atom_index": int(np.argmax(unc)) if unc.size else -1,
                "is_extrapolation": is_extrap,
                "per_atom_uncertainty": [float(x) for x in unc],
                "elements": [int(z) for z in type_by_structure[i]],
            }
        )

    return {
        "method": "uq-mlip-quantile-gbm",
        "backend": backend,
        "per_structure": per_structure,
        "mean_energy_uncertainty": float(np.mean(finite)) if finite.size else 0.0,
        "max_energy_uncertainty": float(np.max(finite)) if finite.size else 0.0,
        "n_extrapolating": len(extrapolation_indices),
        "extrapolation_indices": extrapolation_indices,
        "extrapolation_threshold": threshold,
    }


# ── SciLink tool specs ──────────────────────────────────────────────────────

_IMPORT = "from uq_mlip.integrations.scilink import {name}"

TOOL_SPECS = [
    ToolSpec(
        name="uq_evaluate_uncertainty",
        description=(
            "Calibrated per-atom uncertainty for MLIP structures/trajectories "
            "using a trained uq-mlip quantile-GBM model. Drop-in, better-grounded "
            "replacement for force-variance heuristics: it reuses the MLIP's own "
            "per-atom embeddings and predicts per-atom prediction intervals."
        ),
        import_line=_IMPORT.format(name="uq_evaluate_uncertainty"),
        signature=(
            "uq_evaluate_uncertainty(backend, uq_model, structures, model_size=None, "
            "device='cpu', extrapolation_percentile=95.0) -> dict"
        ),
        agents=["simulation"],
        when_to_use=(
            "Use to flag which frames / which atoms of an MLIP trajectory are "
            "outside the model's reliable domain, once a uq-mlip UQ model has "
            "been trained (uq_train_model) on validation configurations for the "
            "same backend + base model. Prefer this over energy/force heuristics "
            "when per-atom, calibrated uncertainty is available."
        ),
        parameters={
            "backend": {"type": "string", "description": "MLIP backend: 'mace' | 'uma' | 'chgnet'. Must match the UQ model's training backend."},
            "uq_model": {"type": "string", "description": "Directory holding the trained UQ model pickle (output of uq_train_model)."},
            "structures": {"type": "string", "description": "Trajectory/extXYZ path (or an in-memory list of ase.Atoms when called from Python)."},
            "model_size": {"type": "string", "description": "Base-model identifier used for embedding extraction; must match training (e.g. 'medium-0b' for MACE)."},
            "device": {"type": "string", "description": "'cpu' or 'cuda'."},
            "extrapolation_percentile": {"type": "number", "description": "Per-structure mean-uncertainty percentile above which a frame is flagged as extrapolating (default 95)."},
        },
        required=["backend", "uq_model", "structures"],
        returns=(
            "dict with per_structure (per-atom uncertainty, mean/max, "
            "is_extrapolation), mean/max_energy_uncertainty, n_extrapolating, "
            "extrapolation_indices, extrapolation_threshold."
        ),
        example=(
            "report = uq_evaluate_uncertainty(\n"
            "    backend='mace',\n"
            "    uq_model='uq-model/',\n"
            "    structures='md_run.xyz',\n"
            "    model_size='medium-0b',\n"
            ")\n"
            "bad_frames = report['extrapolation_indices']"
        ),
    ),
    ToolSpec(
        name="uq_extract_embeddings",
        description=(
            "Extract per-atom embeddings and per-atom energies from a MACE/UMA/"
            "CHGNet model into a .npz bundle — the input to uq_train_model "
            "(validation set) and, indirectly, uq_evaluate_uncertainty."
        ),
        import_line=_IMPORT.format(name="uq_extract_embeddings"),
        signature=(
            "uq_extract_embeddings(backend, sample, savedir, model_size=None, "
            "checkpoint=None, index=':', device='cpu') -> str"
        ),
        agents=["simulation"],
        when_to_use=(
            "Use as the first step of the uq-mlip workflow, or to (re)generate "
            "embeddings for a trajectory when calling the model directly. For the "
            "common evaluate-a-trajectory case, uq_evaluate_uncertainty extracts "
            "embeddings internally — call this only when you need the .npz."
        ),
        parameters={
            "backend": {"type": "string", "description": "'mace' | 'uma' | 'chgnet'."},
            "sample": {"type": "string", "description": "Path to an ASE-readable structure/trajectory file."},
            "savedir": {"type": "string", "description": "Output directory for the embedding .npz."},
            "model_size": {"type": "string", "description": "Backend model identifier; backend default if omitted."},
            "checkpoint": {"type": "string", "description": "Optional path to a fine-tuned checkpoint."},
            "index": {"type": "string", "description": "ASE slice of frames to read (default ':')."},
            "device": {"type": "string", "description": "'cpu' or 'cuda'."},
        },
        required=["backend", "sample", "savedir"],
        returns="str path to the written embedding_info_<stem>.npz bundle.",
        example=(
            "npz = uq_extract_embeddings(\n"
            "    backend='mace', sample='validation.xyz',\n"
            "    savedir='embeddings/', model_size='medium-0b')"
        ),
    ),
    ToolSpec(
        name="uq_train_model",
        description=(
            "Train a quantile-GBM per-atom UQ model on an embeddings bundle "
            "produced by uq_extract_embeddings from validation configurations."
        ),
        import_line=_IMPORT.format(name="uq_train_model"),
        signature=(
            "uq_train_model(embeddings, savedir, lower_alpha=0.05, "
            "upper_alpha=0.95, estimators=100, device='cpu') -> str"
        ),
        agents=["simulation"],
        when_to_use=(
            "Use once per (backend, base model, dataset) to produce the UQ model "
            "that uq_evaluate_uncertainty consumes. The embeddings MUST come from "
            "validation-set configurations of the same MLIP."
        ),
        parameters={
            "embeddings": {"type": "string", "description": "Path to the .npz bundle from uq_extract_embeddings (validation set)."},
            "savedir": {"type": "string", "description": "Directory to write the trained UQ model into."},
            "lower_alpha": {"type": "number", "description": "Lower quantile (default 0.05)."},
            "upper_alpha": {"type": "number", "description": "Upper quantile (default 0.95)."},
            "estimators": {"type": "integer", "description": "Number of boosting rounds (default 100)."},
            "device": {"type": "string", "description": "'cpu' or 'cuda'."},
        },
        required=["embeddings", "savedir"],
        returns="str path to the saved UQ model pickle (pass its parent dir as uq_model).",
        example=(
            "model = uq_train_model(\n"
            "    embeddings='embeddings/embedding_info_validation.npz',\n"
            "    savedir='uq-model/')"
        ),
    ),
]

# Single-tool convenience alias — some registries look for TOOL_SPEC first.
TOOL_SPEC = TOOL_SPECS[0]

_CALLABLES = {
    "uq_extract_embeddings": uq_extract_embeddings,
    "uq_train_model": uq_train_model,
    "uq_evaluate_uncertainty": uq_evaluate_uncertainty,
}


def openai_tool_schemas() -> List[dict]:
    """Return OpenAI function-call schemas for every uq-mlip tool.

    For orchestrators that consume the OpenAI/JSON tool-schema format
    (Claude Code custom tools, Codex, Cline, ...). Pair each returned schema's
    ``function.name`` with :func:`get_tool_function` to dispatch a call.
    """
    return [spec.to_openai_schema() for spec in TOOL_SPECS]


def tool_prompt() -> str:
    """Markdown description of all uq-mlip tools for prompt injection."""
    return "\n\n".join(spec.to_prompt() for spec in TOOL_SPECS)


def get_tool_function(name: str):
    """Resolve a tool name to its callable (for manual dispatch loops)."""
    try:
        return _CALLABLES[name]
    except KeyError:
        raise LookupError(
            f"Unknown uq-mlip tool {name!r}. Available: {sorted(_CALLABLES)}."
        )


__all__ = [
    "ToolSpec",
    "TOOL_SPEC",
    "TOOL_SPECS",
    "uq_extract_embeddings",
    "uq_train_model",
    "uq_evaluate_uncertainty",
    "openai_tool_schemas",
    "tool_prompt",
    "get_tool_function",
]

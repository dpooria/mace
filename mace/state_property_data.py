"""Data plumbing for the invariant state-property model."""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from ase import Atoms
from ase.io import read

from mace import data
from mace.tools import AtomicNumberTable


def read_property_data(
    path: str,
    backbone: torch.nn.Module,
    names: Sequence[str],
    require_targets: bool = True,
) -> Tuple[List[data.AtomicData], Dict[str, torch.Tensor], List[Atoms]]:
    """Graphs plus a ``(num_configs, len(names))`` target table.

    Target ``name`` is read from the scalar ``atoms.info["REF_<name>"]``, which
    ``db2xyz.py --props`` writes straight from the NEMO ensemble columns.
    """
    atoms_list = read(path, index=":", format="extxyz")
    z_table = AtomicNumberTable([int(z) for z in backbone.atomic_numbers])
    dtype = torch.get_default_dtype()
    keys = [f"REF_{name}" for name in names]
    graphs: List[data.AtomicData] = []
    rows = []
    for index, atoms in enumerate(atoms_list):
        missing = [key for key in keys if key not in atoms.info]
        if missing and require_targets:
            raise KeyError(
                f"configuration {index} is missing {missing}; "
                "rebuild the dataset with db2xyz.py --props"
            )
        row = np.array(
            [float(np.asarray(atoms.info.get(key, np.nan)).reshape(-1)[0]) for key in keys]
        )
        configuration = data.Configuration(
            atomic_numbers=atoms.numbers,
            positions=atoms.positions,
            properties={},
            property_weights={},
            cell=atoms.cell.array,
            pbc=tuple(atoms.pbc),
        )
        graph = data.AtomicData.from_config(
            configuration, z_table=z_table, cutoff=float(backbone.r_max)
        )
        graph.props = torch.tensor(row, dtype=dtype).reshape(1, -1)
        graphs.append(graph)
        rows.append(row)
    targets = {"props": torch.tensor(np.asarray(rows), dtype=dtype)}
    return graphs, targets, atoms_list


def batch_targets(batch) -> Dict[str, torch.Tensor]:
    num_graphs = int(batch.ptr.numel() - 1)
    return {"props": batch.props.view(num_graphs, -1)}


def property_report(
    names: Sequence[str],
    log_mask: Sequence[bool],
    target: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """MAE / RMSE / R^2 per target, plus R^2 of log10 for log targets."""

    def r2(reference, predicted):
        total = float(((reference - reference.mean()) ** 2).sum())
        residual = float(((reference - predicted) ** 2).sum())
        return 1.0 - residual / total if total > 0 else float("nan")

    report = {}
    for column, (name, logged) in enumerate(zip(names, log_mask)):
        reference, predicted = target[:, column], prediction[:, column]
        keep = np.isfinite(reference)
        reference, predicted = reference[keep], predicted[keep]
        errors = predicted - reference
        entry = {
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt((errors**2).mean())),
            "r2": r2(reference, predicted),
        }
        if logged:
            floor = 1.0e-30
            entry["r2_log10"] = r2(
                np.log10(np.clip(reference, floor, None)),
                np.log10(np.clip(predicted, floor, None)),
            )
            # the ensemble mean is what a spectrum or a rate integrates
            entry["mean_ratio"] = float(predicted.mean() / reference.mean())
        report[name] = entry
    return report

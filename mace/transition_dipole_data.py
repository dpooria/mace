"""Data plumbing for the excited-state (transition-dipole) model.

Per configuration the extxyz ``atoms.info`` may carry

    REF_energy_ex   singlet excitation energies, eV, (n_S,)          required
    REF_tdm         <S0|mu|S_n>, a.u., flattened (n_S, 3)            required
    REF_osce        singlet oscillator strengths, (n_S,)             required
    REF_energy_t    triplet excitation energies, eV, (n_T,)          optional
    REF_soc         |<S_m|H_SO|T_k>|, eV, flattened (n_S + 1, n_T)  optional
    REF_soc_shape   the two dimensions of REF_soc                    with REF_soc
    REF_soc_vec     the couplings as Cartesian axial vectors, (n_S + 1, n_T, 3)  optional
    REF_osce_t      T_k -> S0 strengths, (n_T,)                      optional
    REF_d_s, REF_d_t, REF_gp   PCM susceptibilities and ground-state polarization, eV, optional

Optional labels that are absent are NaN, and every loss and metric masks
NaN, so the same model trains on databases with different label sets.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from ase import Atoms
from ase.io import read

from mace import data
from mace.tools import AtomicNumberTable

TARGET_KEYS = ("REF_energy_ex", "REF_tdm", "REF_osce")


def _take(values: np.ndarray, count: int, index: int, name: str) -> np.ndarray:
    if values.shape[0] < count:
        raise ValueError(
            f"configuration {index} has {values.shape[0]} {name} entries, "
            f"fewer than the {count} requested"
        )
    return values[:count]


def _optional(atoms: Atoms, key: str, count: int) -> np.ndarray:
    """First ``count`` entries of an optional per-state label, NaN-padded."""
    values = np.full(count, np.nan)
    if key in atoms.info and count > 0:
        found = np.asarray(atoms.info[key], float).reshape(-1)
        take = min(found.size, count)
        values[:take] = found[:take]
    return values


def _soc_matrix(
    atoms: Atoms, num_states: int, num_triplets: int, index: int,
    key: str = "REF_soc", depth: int = 0,
) -> np.ndarray:
    """(num_states + 1, num_triplets[, depth]) block of couplings, NaN where not labelled.

    ``REF_soc`` holds |SOC| (depth 0) and ``REF_soc_vec`` the Cartesian axial
    vectors (depth 3); both are flattened with the leading shape in ``REF_soc_shape``.
    """
    shape = (num_states + 1, num_triplets) + ((depth,) if depth else ())
    block = np.full(shape, np.nan)
    if key not in atoms.info:
        return block
    flat = np.asarray(atoms.info[key], float).reshape(-1)
    if "REF_soc_shape" in atoms.info:
        rows, cols = (int(x) for x in np.asarray(atoms.info["REF_soc_shape"]).reshape(-1))
    else:
        raise KeyError(f"configuration {index} has {key} but no REF_soc_shape")
    matrix = flat.reshape((rows, cols) + ((depth,) if depth else ()))
    take_rows, take_cols = min(rows, num_states + 1), min(cols, num_triplets)
    block[:take_rows, :take_cols] = matrix[:take_rows, :take_cols]
    return block


def read_tdm_data(
    path: str,
    backbone: torch.nn.Module,
    num_states: int,
    num_triplets: int = 0,
    require_targets: bool = True,
) -> Tuple[List[data.AtomicData], Dict[str, torch.Tensor], List[Atoms]]:
    atoms_list = read(path, index=":", format="extxyz")
    z_table = AtomicNumberTable([int(z) for z in backbone.atomic_numbers])
    dtype = torch.get_default_dtype()
    graphs: List[data.AtomicData] = []
    collected: Dict[str, list] = {
        key: [] for key in
        ("energy_ex", "tdm", "osce", "energy_t", "soc", "soc_vec", "osce_t", "d_s", "d_t", "gp")
    }
    for index, atoms in enumerate(atoms_list):
        if require_targets:
            missing = [key for key in TARGET_KEYS if key not in atoms.info]
            if missing:
                raise KeyError(
                    f"configuration {index} is missing {missing}; "
                    "rebuild the dataset with db2xyz.py"
                )
        if all(key in atoms.info for key in TARGET_KEYS):
            energy = _take(
                np.asarray(atoms.info["REF_energy_ex"], float).reshape(-1),
                num_states, index, "singlet energy",
            )
            dipole = _take(
                np.asarray(atoms.info["REF_tdm"], float).reshape(-1, 3),
                num_states, index, "transition dipole",
            )
            strength = _take(
                np.asarray(atoms.info["REF_osce"], float).reshape(-1),
                num_states, index, "oscillator strength",
            )
        else:
            energy = np.full(num_states, np.nan)
            dipole = np.full((num_states, 3), np.nan)
            strength = np.full(num_states, np.nan)
        energy_t = np.full(max(num_triplets, 0), np.nan)
        osce_t = np.full(max(num_triplets, 0), np.nan)
        soc = np.full((num_states + 1, max(num_triplets, 0)), np.nan)
        soc_vec = np.full((num_states + 1, max(num_triplets, 0), 3), np.nan)
        d_s = _optional(atoms, "REF_d_s", num_states)
        d_t = _optional(atoms, "REF_d_t", max(num_triplets, 0))
        gp = float(np.asarray(atoms.info.get("REF_gp", np.nan), float).reshape(-1)[0])
        if num_triplets > 0:
            if "REF_energy_t" in atoms.info:
                energy_t = _take(
                    np.asarray(atoms.info["REF_energy_t"], float).reshape(-1),
                    num_triplets, index, "triplet energy",
                )
            osce_t = _optional(atoms, "REF_osce_t", num_triplets)
            soc = _soc_matrix(atoms, num_states, num_triplets, index)
            soc_vec = _soc_matrix(atoms, num_states, num_triplets, index, key="REF_soc_vec", depth=3)

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
        graph.energy_ex = torch.tensor(energy, dtype=dtype).reshape(1, -1)
        graph.tdm = torch.tensor(dipole, dtype=dtype).reshape(1, num_states, 3)
        graph.osce = torch.tensor(strength, dtype=dtype).reshape(1, -1)
        graph.d_s = torch.tensor(d_s, dtype=dtype).reshape(1, -1)
        graph.gp = torch.tensor([gp], dtype=dtype)
        if num_triplets > 0:
            graph.energy_t = torch.tensor(energy_t, dtype=dtype).reshape(1, -1)
            graph.soc = torch.tensor(soc, dtype=dtype).reshape(1, num_states + 1, num_triplets)
            graph.soc_vec = torch.tensor(soc_vec, dtype=dtype).reshape(1, num_states + 1, num_triplets, 3)
            graph.osce_t = torch.tensor(osce_t, dtype=dtype).reshape(1, -1)
            graph.d_t = torch.tensor(d_t, dtype=dtype).reshape(1, -1)
        graphs.append(graph)
        for key, value in (
            ("energy_ex", energy), ("tdm", dipole), ("osce", strength),
            ("energy_t", energy_t), ("soc", soc), ("soc_vec", soc_vec), ("osce_t", osce_t),
            ("d_s", d_s), ("d_t", d_t), ("gp", gp),
        ):
            collected[key].append(value)
    targets = {
        key: torch.tensor(np.asarray(value), dtype=dtype) for key, value in collected.items()
    }
    if num_triplets <= 0:
        for key in ("energy_t", "soc", "soc_vec", "osce_t", "d_t"):
            targets.pop(key)
    return graphs, targets, atoms_list


def batch_targets(batch) -> Dict[str, torch.Tensor]:
    num_graphs = int(batch.ptr.numel() - 1)
    targets = {
        "energy_ex": batch.energy_ex.view(num_graphs, -1),
        "tdm": batch.tdm.view(num_graphs, -1, 3),
        "osce": batch.osce.view(num_graphs, -1),
    }
    targets["d_s"] = batch.d_s.view(num_graphs, -1)
    targets["gp"] = batch.gp.view(num_graphs)
    if "energy_t" in batch:
        num_triplets = batch.energy_t.numel() // num_graphs
        targets["energy_t"] = batch.energy_t.view(num_graphs, num_triplets)
        targets["soc"] = batch.soc.view(num_graphs, -1, num_triplets)
        targets["soc_vec"] = batch.soc_vec.view(num_graphs, -1, num_triplets, 3)
        targets["osce_t"] = batch.osce_t.view(num_graphs, num_triplets)
        targets["d_t"] = batch.d_t.view(num_graphs, num_triplets)
    return targets


class CachedFeatures(torch.utils.data.Dataset):
    """Per-graph frozen-backbone features, kept as variable-length atom blocks."""

    def __init__(
        self,
        scalars: Sequence[torch.Tensor],
        vectors: Sequence[torch.Tensor],
        positions: Sequence[torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ):
        self.scalars = list(scalars)
        self.vectors = list(vectors)
        self.positions = list(positions)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.scalars)

    def __getitem__(self, index: int):
        return (
            self.scalars[index],
            self.vectors[index],
            self.positions[index],
            {key: value[index] for key, value in self.targets.items()},
        )


def collate_features(items):
    scalars = torch.cat([item[0] for item in items])
    vectors = torch.cat([item[1] for item in items])
    positions = torch.cat([item[2] for item in items])
    device = scalars.device
    batch = torch.cat(
        [
            torch.full((len(item[0]),), index, dtype=torch.long, device=device)
            for index, item in enumerate(items)
        ]
    )
    targets = {
        key: torch.stack([item[3][key] for item in items]) for key in items[0][3]
    }
    return scalars, vectors, positions, batch, targets


@torch.no_grad()
def cache_features(model, loader, device, keep_on: Optional[str] = None):
    """Frozen-backbone features per graph.

    ``keep_on`` is the device the cache lives on (default: the compute
    device).  Keeping it there makes the per-batch collate a device-side
    ``cat`` instead of a CPU one; with many OpenMP threads the CPU version is
    hundreds of times slower than the head step it feeds.
    """
    keep_on = device if keep_on is None else keep_on
    model.eval()
    scalars, vectors, positions = [], [], []
    for batch in loader:
        batch = batch.to(device)
        node_scalars, node_vectors = model.encode(batch.to_dict())
        pointer = batch.ptr.cpu()
        for index in range(len(pointer) - 1):
            start, stop = int(pointer[index]), int(pointer[index + 1])
            scalars.append(node_scalars[start:stop].to(keep_on).clone())
            vectors.append(node_vectors[start:stop].to(keep_on).clone())
            positions.append(batch.positions[start:stop].to(keep_on).clone())
    return scalars, vectors, positions


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    keep = np.isfinite(target) & np.isfinite(prediction)
    if keep.sum() < 2:
        return float("nan")
    target, prediction = target[keep], prediction[keep]
    residual = float(((target - prediction) ** 2).sum())
    total = float(((target - target.mean()) ** 2).sum())
    return 1.0 - residual / total if total > 0 else float("nan")


def _column_metrics(target: np.ndarray, prediction: np.ndarray, log: bool = False):
    """Per-column MAE / RMSE / R^2 with NaN labels dropped, optional log metrics."""
    report: Dict[str, List[float]] = {"mae": [], "rmse": [], "r2": []}
    if log:
        report.update({"r2_log10": [], "mean_ratio": []})
    for column in range(target.shape[1]):
        reference, predicted = target[:, column], prediction[:, column]
        keep = np.isfinite(reference)
        reference, predicted = reference[keep], predicted[keep]
        if reference.size == 0:
            for key in report:
                report[key].append(float("nan"))
            continue
        errors = predicted - reference
        report["mae"].append(float(np.abs(errors).mean()))
        report["rmse"].append(float(np.sqrt((errors**2).mean())))
        report["r2"].append(r2_score(reference, predicted))
        if log:
            floor = 1.0e-30
            report["r2_log10"].append(
                r2_score(
                    np.log10(np.clip(reference, floor, None)),
                    np.log10(np.clip(predicted, floor, None)),
                )
            )
            report["mean_ratio"].append(float(predicted.mean() / max(reference.mean(), floor)))
    return report


def soc_pair_labels(num_states: int, num_triplets: int) -> List[str]:
    return [f"S{m}-T{k}" for m in range(num_states + 1) for k in range(1, num_triplets + 1)]


def regression_report(
    target: Dict[str, np.ndarray], prediction: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, List[float]]]:
    """Per-state metrics.

    Transition dipoles are scored phase free: on ``|mu|^2`` (the quantity the
    oscillator strength uses) and on ``|cos|`` between the predicted and
    reference directions, never on the signed components.  Spin-orbit
    couplings are scored linearly and in ``log10``; ``mean_ratio`` is what a
    rate, which integrates the ensemble mean, sees.
    """
    report: Dict[str, Dict[str, List[float]]] = {}
    for key in ("energy_ex", "osce", "energy_t", "d_s", "d_t", "gp"):
        if key in target and key in prediction:
            ref, pred = target[key], prediction[key]
            if ref.ndim == 1:
                ref, pred = ref[:, None], pred[:, None]
            if np.isfinite(ref).any():
                report[key] = _column_metrics(ref, pred)
    if "osce_t" in target and "osce_t" in prediction and np.isfinite(target["osce_t"]).any():
        report["osce_t"] = _column_metrics(target["osce_t"], prediction["osce_t"], log=True)

    target_square = (target["tdm"] ** 2).sum(axis=-1)
    prediction_square = (prediction["tdm"] ** 2).sum(axis=-1)
    overlap = np.abs((target["tdm"] * prediction["tdm"]).sum(axis=-1))
    norms = np.sqrt(np.clip(target_square * prediction_square, 1.0e-24, None))
    report["tdm_square"] = _column_metrics(target_square, prediction_square)
    report["tdm_abs_cosine"] = {"mean": np.nanmean(overlap / norms, axis=0).tolist()}

    if "osce" in target and "osce" in prediction:
        total_target = target["osce"].sum(axis=1)
        total_prediction = prediction["osce"].sum(axis=1)
        report["osce_total"] = {
            "mae": [float(np.nanmean(np.abs(total_prediction - total_target)))],
            "r2": [r2_score(total_target, total_prediction)],
        }
    if "soc" in target and "soc" in prediction:
        num_states = target["soc"].shape[1] - 1
        num_triplets = target["soc"].shape[2]
        flat_target = target["soc"].reshape(len(target["soc"]), -1)
        flat_prediction = prediction["soc"].reshape(len(prediction["soc"]), -1)
        report["soc"] = _column_metrics(flat_target, flat_prediction, log=True)
        report["soc"]["pairs"] = soc_pair_labels(num_states, num_triplets)
        if "soc_vec" in target and "soc_vectors" in prediction:
            ref = target["soc_vec"].reshape(len(target["soc_vec"]), -1, 3)
            pred = prediction["soc_vectors"].reshape(len(prediction["soc_vectors"]), -1, 3)
            overlap = np.abs((ref * pred).sum(-1))
            norms = np.sqrt(np.clip((ref**2).sum(-1) * (pred**2).sum(-1), 1.0e-60, None))
            with np.errstate(invalid="ignore"):
                report["soc"]["abs_cosine"] = np.nanmean(overlap / norms, axis=0).tolist()
    return report


def save_json(path: str, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

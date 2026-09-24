"""Data plumbing for the transition-dipole model."""

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from ase import Atoms
from ase.io import read

from mace import data
from mace.tools import AtomicNumberTable

TARGET_KEYS = ("REF_energy_ex", "REF_tdm", "REF_osce")


def read_tdm_data(
    path: str,
    backbone: torch.nn.Module,
    num_states: int,
    require_targets: bool = True,
) -> Tuple[List[data.AtomicData], Dict[str, torch.Tensor], List[Atoms]]:
    atoms_list = read(path, index=":", format="extxyz")
    z_table = AtomicNumberTable([int(z) for z in backbone.atomic_numbers])
    dtype = torch.get_default_dtype()
    graphs: List[data.AtomicData] = []
    energies, dipoles, strengths = [], [], []
    for index, atoms in enumerate(atoms_list):
        if require_targets:
            missing = [key for key in TARGET_KEYS if key not in atoms.info]
            if missing:
                raise KeyError(
                    f"configuration {index} is missing {missing}; "
                    "rebuild the dataset with logs2xyz.py"
                )
        if all(key in atoms.info for key in TARGET_KEYS):
            energy = np.asarray(atoms.info["REF_energy_ex"], float).reshape(-1)
            dipole = np.asarray(atoms.info["REF_tdm"], float).reshape(-1, 3)
            strength = np.asarray(atoms.info["REF_osce"], float).reshape(-1)
            if min(energy.size, dipole.shape[0], strength.size) < num_states:
                raise ValueError(
                    f"configuration {index} has fewer than {num_states} states"
                )
            energy, dipole, strength = (
                energy[:num_states],
                dipole[:num_states],
                strength[:num_states],
            )
        else:
            energy = np.full(num_states, np.nan)
            dipole = np.full((num_states, 3), np.nan)
            strength = np.full(num_states, np.nan)
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
        graphs.append(graph)
        energies.append(energy)
        dipoles.append(dipole)
        strengths.append(strength)
    targets = {
        "energy_ex": torch.tensor(np.asarray(energies), dtype=dtype),
        "tdm": torch.tensor(np.asarray(dipoles), dtype=dtype),
        "osce": torch.tensor(np.asarray(strengths), dtype=dtype),
    }
    return graphs, targets, atoms_list


def batch_targets(batch) -> Dict[str, torch.Tensor]:
    num_graphs = int(batch.ptr.numel() - 1)
    return {
        "energy_ex": batch.energy_ex.view(num_graphs, -1),
        "tdm": batch.tdm.view(num_graphs, -1, 3),
        "osce": batch.osce.view(num_graphs, -1),
    }


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
    batch = torch.cat(
        [
            torch.full((len(item[0]),), index, dtype=torch.long)
            for index, item in enumerate(items)
        ]
    )
    targets = {
        key: torch.stack([item[3][key] for item in items]) for key in items[0][3]
    }
    return scalars, vectors, positions, batch, targets


@torch.no_grad()
def cache_features(model, loader, device):
    model.eval()
    scalars, vectors, positions = [], [], []
    for batch in loader:
        batch = batch.to(device)
        node_scalars, node_vectors = model.encode(batch.to_dict())
        pointer = batch.ptr.cpu()
        for index in range(len(pointer) - 1):
            start, stop = int(pointer[index]), int(pointer[index + 1])
            scalars.append(node_scalars[start:stop].cpu())
            vectors.append(node_vectors[start:stop].cpu())
            positions.append(batch.positions[start:stop].cpu())
    return scalars, vectors, positions


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    residual = float(((target - prediction) ** 2).sum())
    total = float(((target - target.mean()) ** 2).sum())
    return 1.0 - residual / total if total > 0 else float("nan")


def regression_report(
    target: Dict[str, np.ndarray], prediction: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, List[float]]]:
    """Per-state metrics.

    Transition dipoles are scored phase free: on ``|mu|^2`` (the quantity the
    oscillator strength uses) and on ``|cos|`` between the predicted and
    reference directions, never on the signed components.
    """
    report: Dict[str, Dict[str, List[float]]] = {}
    for key in ("energy_ex", "osce"):
        if key not in target:
            continue
        errors = prediction[key] - target[key]
        report[key] = {
            "mae": np.abs(errors).mean(axis=0).tolist(),
            "rmse": np.sqrt((errors**2).mean(axis=0)).tolist(),
            "r2": [
                r2_score(target[key][:, i], prediction[key][:, i])
                for i in range(target[key].shape[1])
            ],
        }

    target_square = (target["tdm"] ** 2).sum(axis=-1)
    prediction_square = (prediction["tdm"] ** 2).sum(axis=-1)
    overlap = np.abs((target["tdm"] * prediction["tdm"]).sum(axis=-1))
    norms = np.sqrt(np.clip(target_square * prediction_square, 1.0e-24, None))
    report["tdm_square"] = {
        "mae": np.abs(prediction_square - target_square).mean(axis=0).tolist(),
        "r2": [
            r2_score(target_square[:, i], prediction_square[:, i])
            for i in range(target_square.shape[1])
        ],
    }
    report["tdm_abs_cosine"] = {
        "mean": (overlap / norms).mean(axis=0).tolist()
    }

    if "osce" in target:
        total_target = target["osce"].sum(axis=1)
        total_prediction = prediction["osce"].sum(axis=1)
        report["osce_total"] = {
            "mae": [float(np.abs(total_prediction - total_target).mean())],
            "r2": [r2_score(total_target, total_prediction)],
        }
    return report


def save_json(path: str, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from ase import Atoms
from ase.io import read, write

from mace import data
from mace.calculators import mace_off
from mace.modules.oscillator import OscillatorStrengthMACE
from mace.tools import AtomicNumberTable, torch_geometric


def load_foundation_model(
    model: str,
    device: str,
    dtype: torch.dtype,
) -> torch.nn.Module:
    model_name = model.removesuffix("_off")
    if model_name in {"small", "medium", "large"}:
        backbone = mace_off(
            model=model_name,
            device=device,
            return_raw_model=True,
        )
    else:
        backbone = torch.load(model, map_location=device, weights_only=False)
    return backbone.to(device=device, dtype=dtype)


def read_oscillator_data(
    path: str,
    backbone: torch.nn.Module,
    target_key: str = "REF_osce",
    require_targets: bool = True,
    num_transitions: Optional[int] = None,
) -> Tuple[List[data.AtomicData], torch.Tensor, List[Atoms]]:
    atoms_list = read(path, index=":", format="extxyz")
    z_table = AtomicNumberTable([int(z) for z in backbone.atomic_numbers])
    graphs = []
    targets = []
    for index, atoms in enumerate(atoms_list):
        if target_key not in atoms.info and require_targets:
            raise KeyError(f"{target_key!r} is missing from configuration {index}")
        if target_key in atoms.info:
            target = np.asarray(atoms.info[target_key], dtype=float).reshape(-1)
        elif num_transitions is not None:
            target = np.full(num_transitions, np.nan)
        else:
            raise ValueError("num_transitions is required for unlabeled data")
        if require_targets and (
            not np.all(np.isfinite(target)) or np.any(target < 0)
        ):
            raise ValueError(
                f"Configuration {index} has negative or non-finite targets"
            )
        if targets and target.shape != targets[0].shape:
            raise ValueError(
                f"Configuration {index} has {target.size} targets; "
                f"expected {targets[0].size}"
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
            configuration,
            z_table=z_table,
            cutoff=float(backbone.r_max),
        )
        graph.oscillator_strength = torch.tensor(
            target, dtype=torch.get_default_dtype()
        ).reshape(1, -1)
        graph.sample_id = torch.tensor([index], dtype=torch.long)
        graphs.append(graph)
        targets.append(target)
    return (
        graphs,
        torch.tensor(np.asarray(targets), dtype=torch.get_default_dtype()),
        atoms_list,
    )


def make_loader(
    graphs: Sequence[data.AtomicData],
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset=graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def split_indices(
    size: int,
    valid_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")
    generator = np.random.default_rng(seed)
    indices = generator.permutation(size)
    valid_size = max(1, round(size * valid_fraction))
    return indices[valid_size:], indices[:valid_size]


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, List[float]]:
    error = prediction - target
    denominator = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    r2 = np.full(target.shape[1], np.nan)
    valid = denominator > 0
    r2[valid] = 1.0 - np.sum(error[:, valid] ** 2, axis=0) / denominator[valid]
    return {
        "mae": np.mean(np.abs(error), axis=0).tolist(),
        "rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "r2": r2.tolist(),
        "log_mae": np.mean(
            np.abs(
                np.log(np.clip(prediction, 1.0e-12, None))
                - np.log(np.clip(target, 1.0e-12, None))
            ),
            axis=0,
        ).tolist(),
    }


@torch.no_grad()
def predict(
    model: OscillatorStrengthMACE,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    indices = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.to_dict())
        predictions.append(output["oscillator_strength"].cpu())
        targets.append(batch.oscillator_strength.cpu())
        indices.append(batch.sample_id.cpu())
    return (
        torch.cat(predictions).numpy(),
        torch.cat(targets).numpy(),
        torch.cat(indices).numpy(),
    )


def save_predictions(
    atoms_list: Sequence[Atoms],
    predictions: np.ndarray,
    indices: np.ndarray,
    output: str,
    prediction_key: str = "MACE_osce",
) -> None:
    output_atoms = []
    for prediction, index in zip(predictions, indices):
        atoms = atoms_list[int(index)].copy()
        atoms.info[prediction_key] = prediction
        output_atoms.append(atoms)
    write(output, output_atoms, format="extxyz")


def save_json(path: str, values: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2)


def load_trained_model(
    checkpoint_path: str,
    device: str,
    foundation_model: Optional[str] = None,
) -> Tuple[OscillatorStrengthMACE, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != 2:
        raise ValueError(
            "This checkpoint uses the old oscillator model. Retrain it with "
            "the current mace_run_train_oscillator command."
        )
    dtype = getattr(torch, checkpoint["dtype"])
    torch.set_default_dtype(dtype)
    source = foundation_model or checkpoint["foundation_model"]
    backbone = load_foundation_model(source, device, dtype)
    model = OscillatorStrengthMACE(
        backbone=backbone,
        num_transitions=checkpoint["num_transitions"],
        latent_mean=checkpoint["latent_mean"].to(dtype),
        latent_std=checkpoint["latent_std"].to(dtype),
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
        num_distance_basis=checkpoint.get("num_distance_basis", 0),
    ).to(device)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state"], strict=False
    )
    if unexpected or any(name != "distance_centers" for name in missing):
        raise ValueError(
            f"Checkpoint does not match the oscillator model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model, checkpoint

"""Evaluate a transition-dipole model and assemble oscillator strengths.

    f_n = 2/3 * dE_n[Hartree] * |mu_n|^2

The excitation energies used for ``f`` are chosen with ``--energy-source``:

``model``     the eigenvalues of the model's own mixing matrix
``info``      a per-configuration key in ``atoms.info`` -- use this to feed in
              energies predicted by X-MACE
``reference`` the TDDFT values in ``REF_energy_ex``, which isolates the error
              contributed by the dipole alone
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import write

from mace.modules.transition_dipole import (
    TransitionDipoleMACE,
    oscillator_strength,
)
from mace.oscillator_strength import load_foundation_model, make_loader
from mace.transition_dipole_data import (
    batch_targets,
    read_tdm_data,
    regression_report,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--energy-source",
        choices=["model", "info", "reference"],
        default="model",
    )
    parser.add_argument(
        "--energy-key",
        default="XMACE_energy_ex",
        help="atoms.info key read when --energy-source info",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-targets",
        action="store_true",
        help="the configurations carry no TDDFT labels, so skip scoring",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
    dtype = getattr(torch, checkpoint.get("dtype", "float32"))
    torch.set_default_dtype(dtype)

    backbone = load_foundation_model(
        checkpoint["foundation_model"], args.device, dtype
    )
    model = TransitionDipoleMACE(
        backbone=backbone,
        num_states=checkpoint["num_states"],
        energy_reference=checkpoint["energy_reference"].to(args.device),
        basis=checkpoint.get("basis", "diabatic"),
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
        coupling_scale=checkpoint["coupling_scale"],
        charge_scale=checkpoint["charge_scale"],
        vector_scale=checkpoint["vector_scale"],
        eigh_eps=checkpoint["eigh_eps"],
        factorised=checkpoint.get("factorised", True),
        solvation=checkpoint.get("solvation", False),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    num_states = checkpoint["num_states"]
    solvation = checkpoint.get("solvation", False)
    graphs, targets, atoms_list = read_tdm_data(
        args.configs,
        backbone,
        num_states,
        require_targets=not args.no_targets,
        solvation=solvation,
    )
    loader = make_loader(graphs, args.batch_size, False)

    dipoles, energies, susceptibilities, polarizations = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            prediction = model(batch.to_dict())
            dipoles.append(prediction["tdm"].cpu().numpy())
            energies.append(prediction["energy_ex"].cpu().numpy())
            if solvation:
                susceptibilities.append(prediction["susceptibility"].cpu().numpy())
                polarizations.append(prediction["ground_polarization"].cpu().numpy())
    dipoles = np.concatenate(dipoles)
    model_energies = np.concatenate(energies)
    if solvation:
        susceptibilities = np.concatenate(susceptibilities)
        polarizations = np.concatenate(polarizations)

    if args.energy_source == "model":
        used_energies = model_energies
    elif args.energy_source == "reference":
        if args.no_targets:
            raise ValueError("--energy-source reference needs REF_energy_ex")
        used_energies = targets["energy_ex"].numpy()
    else:
        missing = [
            index
            for index, atoms in enumerate(atoms_list)
            if args.energy_key not in atoms.info
        ]
        if missing:
            raise KeyError(
                f"{args.energy_key!r} is missing from {len(missing)} configurations; "
                "run the energy model over them first"
            )
        used_energies = np.array(
            [
                np.asarray(atoms.info[args.energy_key], float).reshape(-1)[:num_states]
                for atoms in atoms_list
            ]
        )

    strengths = (
        oscillator_strength(
            torch.tensor(dipoles), torch.tensor(used_energies)
        )
        .numpy()
    )

    for index, atoms in enumerate(atoms_list):
        atoms.info["MACE_tdm"] = dipoles[index].reshape(-1)
        atoms.info["MACE_energy_ex"] = model_energies[index]
        atoms.info["MACE_osce"] = strengths[index]
        if solvation:
            atoms.info["MACE_susceptibility"] = susceptibilities[index]
            atoms.info["MACE_gp"] = float(polarizations[index])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write(args.output, atoms_list, format="extxyz")

    if args.no_targets:
        print(f"wrote {len(atoms_list)} predictions to {args.output}")
        return

    reference = {key: value.numpy() for key, value in targets.items()}
    prediction = {
        "tdm": dipoles,
        "energy_ex": model_energies,
        "osce": strengths,
    }
    if solvation:
        prediction["susceptibility"] = susceptibilities
        prediction["ground_polarization"] = polarizations
    metrics = regression_report(reference, prediction)
    metrics["energy_source"] = args.energy_source
    print(json.dumps(metrics, indent=2))
    save_json(str(Path(args.output).with_suffix(".metrics.json")), metrics)


if __name__ == "__main__":
    main()

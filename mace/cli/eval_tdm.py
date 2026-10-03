"""Evaluate an excited-state model and assemble oscillator strengths.

    f_n = 2/3 * dE_n[Hartree] * |mu_n|^2

The excitation energies used for ``f`` are chosen with ``--energy-source``:

``model``     the eigenvalues of the model's own mixing matrix
``info``      a per-configuration key in ``atoms.info`` -- use this to feed in
              energies predicted by another model
``reference`` the TDDFT values in ``REF_energy_ex``, which isolates the error
              contributed by the dipole alone

Written into ``atoms.info``: ``MACE_energy_ex``, ``MACE_tdm``, ``MACE_osce``
and, depending on the model, ``MACE_energy_t``, ``MACE_soc`` (flattened, with
``MACE_soc_shape``), ``MACE_soc_vec``, ``MACE_osce_t``, ``MACE_d_s``, ``MACE_d_t``,
``MACE_gp``.
"""

import os

# e3nn 0.4.4 loads its constants with torch.load at import time; torch >= 2.6
# defaults to weights_only=True and refuses them
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import write

from mace.modules.transition_dipole import TransitionDipoleMACE, oscillator_strength
from mace.oscillator_strength import load_foundation_model, make_loader
from mace.transition_dipole_data import (
    batch_targets,
    read_tdm_data,
    regression_report,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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


def load_model(path: str, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version", 1) < 2:
        raise ValueError(
            f"{path} was written by an older transition-dipole model; retrain it"
        )
    dtype = getattr(torch, checkpoint.get("dtype", "float32"))
    torch.set_default_dtype(dtype)
    backbone = load_foundation_model(checkpoint["foundation_model"], device, dtype)
    model = TransitionDipoleMACE(backbone=backbone, **checkpoint["options"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def main() -> None:
    args = parse_args()
    model, checkpoint = load_model(args.model, args.device)
    options = checkpoint["options"]
    num_states, num_triplets = options["num_states"], options["num_triplets"]

    graphs, targets, atoms_list = read_tdm_data(
        args.configs,
        model.backbone,
        num_states,
        num_triplets,
        require_targets=not args.no_targets,
    )
    loader = make_loader(graphs, args.batch_size, False)

    collected = {}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            prediction = model(batch.to_dict())
            for key in ("tdm", "energy_ex", "energy_t", "soc", "soc_vectors", "tdm_t", "osce_t",
                        "susceptibility_s", "susceptibility_t", "ground_polarization"):
                if key in prediction:
                    collected.setdefault(key, []).append(prediction[key].cpu().numpy())
    collected = {key: np.concatenate(value) for key, value in collected.items()}
    dipoles, model_energies = collected["tdm"], collected["energy_ex"]

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

    strengths = oscillator_strength(
        torch.tensor(dipoles), torch.tensor(used_energies)
    ).numpy()

    for index, atoms in enumerate(atoms_list):
        atoms.info["MACE_tdm"] = dipoles[index].reshape(-1)
        atoms.info["MACE_energy_ex"] = model_energies[index]
        atoms.info["MACE_osce"] = strengths[index]
        if "energy_t" in collected:
            atoms.info["MACE_energy_t"] = collected["energy_t"][index]
        if "soc" in collected:
            atoms.info["MACE_soc"] = collected["soc"][index].reshape(-1)
            atoms.info["MACE_soc_shape"] = np.array(collected["soc"][index].shape)
        if "soc_vectors" in collected:
            atoms.info["MACE_soc_vec"] = collected["soc_vectors"][index].reshape(-1)
        if "tdm_t" in collected:
            atoms.info["MACE_tdm_t"] = collected["tdm_t"][index].reshape(-1)
        for key, name in (("osce_t", "MACE_osce_t"), ("susceptibility_s", "MACE_d_s"),
                          ("susceptibility_t", "MACE_d_t")):
            if key in collected:
                atoms.info[name] = collected[key][index]
        if "ground_polarization" in collected:
            atoms.info["MACE_gp"] = float(collected["ground_polarization"][index])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write(args.output, atoms_list, format="extxyz")

    if args.no_targets:
        print(f"wrote {len(atoms_list)} predictions to {args.output}")
        return

    reference = {key: value.numpy() for key, value in targets.items()}
    prediction = dict(collected)
    prediction["osce"] = strengths
    # score the auxiliary outputs under their label names
    for key, name in (("susceptibility_s", "d_s"), ("susceptibility_t", "d_t"),
                      ("ground_polarization", "gp")):
        if key in prediction:
            prediction[name] = prediction.pop(key)
    metrics = regression_report(reference, prediction)
    metrics["energy_source"] = args.energy_source
    print(json.dumps(metrics, indent=2))
    save_json(str(Path(args.output).with_suffix(".metrics.json")), metrics)


if __name__ == "__main__":
    main()

"""Evaluate a state-property model and write ``MACE_<name>`` into extxyz.

Every ``atoms.info`` entry already in ``--configs`` is kept, so this can be
chained after ``eval_tdm`` on the same file to collect the energies,
susceptibilities and phosphorescence strengths one spectrum needs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import write

from mace.modules.state_properties import StatePropertyMACE
from mace.oscillator_strength import load_foundation_model, make_loader
from mace.state_property_data import property_report, read_property_data
from mace.transition_dipole_data import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-targets",
        action="store_true",
        help="the configurations carry no reference labels, so skip scoring",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
    if checkpoint.get("kind") != "state_properties":
        raise ValueError(f"{args.model} is not a state-property model")
    dtype = getattr(torch, checkpoint.get("dtype", "float32"))
    torch.set_default_dtype(dtype)

    backbone = load_foundation_model(
        checkpoint["foundation_model"], args.device, dtype
    )
    specs = checkpoint["targets"]
    model = StatePropertyMACE(
        backbone,
        specs,
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint["dropout"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    names = [str(item["name"]) for item in specs]
    log_mask = [bool(item["log"]) for item in specs]
    graphs, targets, atoms_list = read_property_data(
        args.configs, backbone, names, require_targets=not args.no_targets
    )
    predicted = []
    with torch.no_grad():
        for batch in make_loader(graphs, args.batch_size, False):
            batch = batch.to(args.device)
            predicted.append(model(batch.to_dict())["props"].cpu().numpy())
    predicted = np.concatenate(predicted)

    for index, atoms in enumerate(atoms_list):
        for column, name in enumerate(names):
            atoms.info[f"MACE_{name}"] = float(predicted[index, column])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write(args.output, atoms_list, format="extxyz")

    if args.no_targets:
        print(f"wrote {len(atoms_list)} predictions to {args.output}")
        return
    metrics = property_report(names, log_mask, targets["props"].numpy(), predicted)
    print(json.dumps(metrics, indent=2))
    # not ".metrics.json": eval_tdm writes that name for the same eval file
    save_json(str(Path(args.output).with_suffix(".props.metrics.json")), metrics)


if __name__ == "__main__":
    main()

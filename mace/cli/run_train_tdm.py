"""Train the transition-dipole model on top of a MACE backbone.

Stage 1 trains the head against cached frozen-backbone features, which is cheap
enough to run for many epochs.  Stage 2 optionally unfreezes part of the
backbone and continues end to end.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from mace.modules.transition_dipole import (
    TransitionDipoleLoss,
    TransitionDipoleMACE,
    oscillator_strength,
    target_statistics,
)
from mace.oscillator_strength import load_foundation_model, make_loader, split_indices
from mace.transition_dipole_data import (
    CachedFeatures,
    batch_targets,
    cache_features,
    collate_features,
    read_tdm_data,
    regression_report,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file")
    parser.add_argument("--output", default="tdm.model")
    parser.add_argument("--foundation-model", default="medium_off")
    parser.add_argument("--num-states", type=int, default=3)
    parser.add_argument(
        "--basis",
        choices=["diabatic", "adiabatic"],
        default="diabatic",
        help="diabatic predicts smooth dipoles plus a mixing matrix and "
        "diagonalises; adiabatic predicts the adiabatic dipoles directly",
    )
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--head-lr", type=float, default=1.0e-3)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dipole-weight", type=float, default=1.0)
    parser.add_argument("--tensor-weight", type=float, default=0.3)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--strength-weight", type=float, default=0.3)
    parser.add_argument(
        "--total-weight",
        type=float,
        default=0.0,
        help="weight on the summed-intensity term; measured to hurt both "
        "per-state f and the sum itself, so it is off by default",
    )
    parser.add_argument(
        "--solvation",
        action="store_true",
        help="also predict the NEMO susceptibilities REF_susceptibility and "
        "the ground-state polarization REF_gp used for solvent interpolation",
    )
    parser.add_argument("--susceptibility-weight", type=float, default=1.0)
    parser.add_argument("--polarization-weight", type=float, default=1.0)
    parser.add_argument("--coupling-scale", type=float, default=0.2)
    parser.add_argument("--charge-scale", type=float, default=1.0)
    parser.add_argument("--vector-scale", type=float, default=1.0)
    parser.add_argument("--eigh-eps", type=float, default=1.0e-3)
    parser.add_argument(
        "--direct-dipole",
        action="store_true",
        help="read the dipole straight off the vector head instead of "
        "factorising it into an invariant magnitude and a unit direction",
    )
    parser.add_argument("--fine-tune", choices=["head", "last", "full"], default="last")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    return parser.parse_args()


def evaluate_graphs(model, loader, loss_fn, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            size = int(batch.ptr.numel() - 1)
            loss = loss_fn(model(batch.to_dict()), batch_targets(batch))
            total += loss.item() * size
            count += size
    return total / count


def evaluate_cached(model, loader, loss_fn, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for scalars, vectors, positions, batch, targets in loader:
            size = len(targets["tdm"])
            prediction = model.head(
                scalars.to(device),
                vectors.to(device),
                positions.to(device),
                batch.to(device),
                size,
            )
            loss = loss_fn(prediction, {k: v.to(device) for k, v in targets.items()})
            total += loss.item() * size
            count += size
    return total / count


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    keys = ["energy_ex", "tdm"]
    if model.head.solvation:
        keys += ["susceptibility", "ground_polarization"]
    collected = {key: [] for key in keys}
    reference = {key: [] for key in keys}
    for batch in loader:
        batch = batch.to(device)
        prediction = model(batch.to_dict())
        targets = batch_targets(batch)
        for key in keys:
            collected[key].append(prediction[key].cpu().numpy())
            reference[key].append(targets[key].cpu().numpy())
        collected.setdefault("osce", []).append(
            oscillator_strength(prediction["tdm"], prediction["energy_ex"])
            .cpu()
            .numpy()
        )
        reference.setdefault("osce", []).append(targets["osce"].cpu().numpy())
    return (
        {key: np.concatenate(value) for key, value in collected.items()},
        {key: np.concatenate(value) for key, value in reference.items()},
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(getattr(torch, args.dtype))

    backbone = load_foundation_model(
        args.foundation_model, args.device, getattr(torch, args.dtype)
    )
    graphs, targets, _ = read_tdm_data(
        args.train_file, backbone, args.num_states, solvation=args.solvation
    )
    if args.valid_file:
        train_graphs, train_targets = graphs, targets
        valid_graphs, valid_targets, _ = read_tdm_data(
            args.valid_file, backbone, args.num_states, solvation=args.solvation
        )
    else:
        train_indices, valid_indices = split_indices(
            len(graphs), args.valid_fraction, args.seed
        )
        train_graphs = [graphs[index] for index in train_indices]
        valid_graphs = [graphs[index] for index in valid_indices]
        train_targets = {key: value[train_indices] for key, value in targets.items()}
        valid_targets = {key: value[valid_indices] for key, value in targets.items()}

    statistics = target_statistics(train_targets)
    model = TransitionDipoleMACE(
        backbone=backbone,
        num_states=args.num_states,
        energy_reference=statistics["energy_reference"],
        basis=args.basis,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        coupling_scale=args.coupling_scale,
        charge_scale=args.charge_scale,
        vector_scale=args.vector_scale,
        eigh_eps=args.eigh_eps,
        factorised=not args.direct_dipole,
        solvation=args.solvation,
    ).to(args.device)
    model.set_magnitude_reference(train_targets["tdm"].to(args.device))
    if args.solvation:
        model.set_solvation_statistics(
            train_targets["susceptibility"].to(args.device),
            train_targets["ground_polarization"].to(args.device),
        )
    energy_weight = 0.0 if args.basis == "adiabatic" else args.energy_weight
    loss_fn = TransitionDipoleLoss(
        dipole_std=statistics["dipole_std"],
        tensor_std=statistics["tensor_std"],
        energy_std=statistics["energy_std"],
        strength_std=statistics["strength_std"],
        dipole_weight=args.dipole_weight,
        tensor_weight=args.tensor_weight,
        energy_weight=energy_weight,
        strength_weight=args.strength_weight,
        total_weight=args.total_weight,
        susceptibility_std=statistics.get("susceptibility_std"),
        polarization_std=statistics.get("polarization_std"),
        susceptibility_weight=args.susceptibility_weight if args.solvation else 0.0,
        polarization_weight=args.polarization_weight if args.solvation else 0.0,
    ).to(args.device)

    model.set_backbone_trainable("head")
    train_loader = make_loader(train_graphs, args.batch_size, False)
    valid_loader = make_loader(valid_graphs, args.batch_size, False)
    train_cache = cache_features(model, train_loader, args.device)
    valid_cache = cache_features(model, valid_loader, args.device)
    model.set_feature_statistics(torch.cat(train_cache[0]).to(args.device))

    head_train = torch.utils.data.DataLoader(
        CachedFeatures(*train_cache, train_targets),
        batch_size=args.head_batch_size,
        shuffle=True,
        collate_fn=collate_features,
    )
    head_valid = torch.utils.data.DataLoader(
        CachedFeatures(*valid_cache, valid_targets),
        batch_size=args.head_batch_size,
        shuffle=False,
        collate_fn=collate_features,
    )

    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        head_parameters, lr=args.head_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=25, min_lr=1.0e-6
    )
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = []
    for epoch in range(1, args.head_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for scalars, vectors, positions, batch, batch_target in head_train:
            size = len(batch_target["tdm"])
            optimizer.zero_grad(set_to_none=True)
            prediction = model.head(
                scalars.to(args.device),
                vectors.to(args.device),
                positions.to(args.device),
                batch.to(args.device),
                size,
            )
            loss = loss_fn(
                prediction,
                {key: value.to(args.device) for key, value in batch_target.items()},
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_parameters, 10.0)
            optimizer.step()
            total += loss.item() * size
            count += size
        valid_loss = evaluate_cached(model, head_valid, loss_fn, args.device)
        scheduler.step(valid_loss)
        record = {
            "stage": "head",
            "epoch": epoch,
            "train_loss": total / count,
            "valid_loss": valid_loss,
        }
        history.append(record)
        if epoch == 1 or epoch % 25 == 0:
            print(json.dumps(record), flush=True)
        if valid_loss < best_loss:
            best_loss, stale = valid_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)

    if args.fine_tune != "head":
        model.set_backbone_trainable(args.fine_tune)
        optimizer = torch.optim.AdamW(
            [
                {"params": head_parameters, "lr": args.head_lr * 0.1},
                {
                    "params": [
                        parameter
                        for parameter in model.backbone.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": args.backbone_lr,
                },
            ],
            weight_decay=args.weight_decay,
        )
        shuffled = make_loader(train_graphs, args.batch_size, True)
        stale = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            total, count = 0.0, 0
            for batch in shuffled:
                batch = batch.to(args.device)
                size = int(batch.ptr.numel() - 1)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch.to_dict()), batch_targets(batch))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                total += loss.item() * size
                count += size
            valid_loss = evaluate_graphs(model, valid_loader, loss_fn, args.device)
            record = {
                "stage": args.fine_tune,
                "epoch": epoch,
                "train_loss": total / count,
                "valid_loss": valid_loss,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if valid_loss < best_loss:
                best_loss, stale = valid_loss, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= args.patience:
                break
        model.load_state_dict(best_state)

    prediction, reference = predict(model, valid_loader, args.device)
    metrics = regression_report(reference, prediction)
    print(json.dumps({"validation": metrics}, indent=2))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": 1,
            "model_state": model.state_dict(),
            "foundation_model": args.foundation_model,
            "num_states": args.num_states,
            "basis": args.basis,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "coupling_scale": args.coupling_scale,
            "charge_scale": args.charge_scale,
            "vector_scale": args.vector_scale,
            "eigh_eps": args.eigh_eps,
            "factorised": not args.direct_dipole,
            "solvation": args.solvation,
            "energy_reference": statistics["energy_reference"].cpu(),
            "dtype": args.dtype,
            "metrics": metrics,
        },
        output,
    )
    save_json(str(output.with_suffix(".history.json")), {"history": history})
    save_json(str(output.with_suffix(".metrics.json")), metrics)


if __name__ == "__main__":
    main()

"""Train the excited-state (transition-dipole) model on top of a MACE backbone.

Stage 1 trains the head against cached frozen-backbone features, which is cheap
enough to run for many epochs.  Stage 2 optionally unfreezes part of the
backbone and continues end to end.

The model predicts singlet excitation energies and transition dipoles (hence
oscillator strengths) and, with ``--num-triplets N``, triplet excitation
energies and the spin-orbit coupling magnitudes |<S_m|H_SO|T_k>| for
m = 0..num_states, k = 1..N.  Labels a database does not have are NaN in the
extxyz file and drop out of the loss, so the same command serves every
ensemble.
"""

import os

# e3nn 0.4.4 loads its constants with torch.load at import time; torch >= 2.6
# defaults to weights_only=True and refuses them
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from mace.modules.transition_dipole import (
    SOC_MODES,
    TransitionDipoleLoss,
    TransitionDipoleMACE,
    oscillator_strength,
    target_statistics,
)
from mace.oscillator_strength import (
    FOUNDATION_MODELS,
    load_foundation_model,
    make_loader,
    split_indices,
)
from mace.transition_dipole_data import (
    CachedFeatures,
    batch_targets,
    cache_features,
    collate_features,
    read_tdm_data,
    regression_report,
    save_json,
)

CHECKPOINT_VERSION = 2
PREDICTION_KEYS = ("energy_ex", "tdm", "energy_t", "soc", "soc_vectors", "osce_t",
                   "susceptibility_s", "susceptibility_t", "ground_polarization")
TARGET_OF = {"soc_vectors": "soc_vec", "susceptibility_s": "d_s", "susceptibility_t": "d_t",
             "ground_polarization": "gp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file")
    parser.add_argument("--output", default="tdm.model")
    parser.add_argument(
        "--foundation-model",
        default="mh-1",
        help=f"one of {sorted(FOUNDATION_MODELS)}, a MACE-MP release name or a "
        ".model file (default mh-1)",
    )
    parser.add_argument("--num-states", type=int, default=5, help="singlet states")
    parser.add_argument(
        "--num-triplets",
        type=int,
        default=0,
        help="triplet states; 0 disables the triplet manifold and the SOC head",
    )
    parser.add_argument(
        "--soc-mode",
        choices=SOC_MODES,
        default="diabatic",
        help="diabatic: axial SOC vectors rotated by both manifolds; direct: "
        "invariant log-magnitude readout per pair; none: no SOC head",
    )
    parser.add_argument(
        "--basis",
        choices=["diabatic", "adiabatic"],
        default="diabatic",
        help="diabatic predicts smooth diabatic properties plus a mixing matrix "
        "per manifold and diagonalises; adiabatic predicts the adiabatic "
        "properties directly",
    )
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--head-lr", type=float, default=1.0e-3)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--vector-dim",
        type=int,
        default=32,
        help="1o channels kept after an equivariant channel mixing",
    )
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
    parser.add_argument("--triplet-energy-weight", type=float, default=1.0)
    parser.add_argument("--soc-weight", type=float, default=1.0)
    parser.add_argument(
        "--soc-vector-weight",
        type=float,
        default=1.0,
        help="sign-matched loss on the Cartesian SOC vectors (REF_soc_vec); only "
        "acts with --soc-mode diabatic",
    )
    parser.add_argument(
        "--solvation",
        action="store_true",
        help="predict PCM susceptibilities d_n (REF_d_s, REF_d_t) and the ground-state "
        "polarization gp (REF_gp) on a detached auxiliary branch",
    )
    parser.add_argument("--solvation-weight", type=float, default=1.0)
    parser.add_argument(
        "--phosphorescence",
        action="store_true",
        help="predict the T_n -> S0 strengths REF_osce_t (log fit, mean-calibrated) on "
        "the detached auxiliary branch",
    )
    parser.add_argument("--phosphorescence-weight", type=float, default=1.0)
    parser.add_argument(
        "--calibrate",
        choices=["none", "mean"],
        default="mean",
        help="rescale log-fitted magnitudes (SOC, osce_t) to the training-set arithmetic mean",
    )
    parser.add_argument(
        "--coupling-scale",
        type=float,
        default=1.0,
        help="dimensionless multiplier on the off-diagonal mixing, in units "
        "of the training-set energy spread",
    )
    parser.add_argument("--eigh-eps", type=float, default=1.0e-3, help="eV")
    parser.add_argument(
        "--direct-dipole",
        action="store_true",
        help="read the dipole straight off the vector head instead of "
        "factorising it into an invariant magnitude and a unit direction",
    )
    parser.add_argument("--fine-tune", choices=["head", "last", "full"], default="head")
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=1,
        help="with --fine-tune last: how many interaction/product blocks from the top train",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="torch CPU threads; the small CPU ops in the loop get slower, not "
        "faster, with many OpenMP threads",
    )
    parser.add_argument(
        "--cache-device",
        default=None,
        help="where the cached backbone features live (default: --device)",
    )
    return parser.parse_args()


def to_device(targets, device):
    return {key: value.to(device) for key, value in targets.items()}


def evaluate_graphs(model, loader, loss_fn, device):
    """Mean validation loss: (total, main terms only)."""
    model.eval()
    total, main, count = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            size = int(batch.ptr.numel() - 1)
            prediction, targets = model(batch.to_dict()), batch_targets(batch)
            total += loss_fn(prediction, targets).item() * size
            main += loss_fn(prediction, targets, main_only=True).item() * size
            count += size
    return total / count, main / count


def evaluate_cached(model, loader, loss_fn, device):
    """Mean validation loss on cached features: (total, main terms only)."""
    model.eval()
    total, main, count = 0.0, 0.0, 0
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
            targets = to_device(targets, device)
            total += loss_fn(prediction, targets).item() * size
            main += loss_fn(prediction, targets, main_only=True).item() * size
            count += size
    return total / count, main / count


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    collected, reference = {}, {}
    for batch in loader:
        batch = batch.to(device)
        prediction = model(batch.to_dict())
        targets = batch_targets(batch)
        prediction["osce"] = oscillator_strength(prediction["tdm"], prediction["energy_ex"])
        for key in PREDICTION_KEYS + ("osce",):
            target_key = TARGET_OF.get(key, key)
            if key in prediction and target_key in targets:
                stored = key if key == "soc_vectors" else target_key
                collected.setdefault(stored, []).append(prediction[key].cpu().numpy())
                reference.setdefault(target_key, []).append(targets[target_key].cpu().numpy())
    return (
        {key: np.concatenate(value) for key, value in collected.items()},
        {key: np.concatenate(value) for key, value in reference.items()},
    )


def model_options(args) -> dict:
    return {
        "num_states": args.num_states,
        "num_triplets": args.num_triplets,
        "basis": args.basis,
        "hidden_dim": args.hidden_dim,
        "vector_dim": args.vector_dim,
        "dropout": args.dropout,
        "coupling_scale": args.coupling_scale,
        "eigh_eps": args.eigh_eps,
        "factorised": not args.direct_dipole,
        "soc_mode": args.soc_mode,
        "solvation": args.solvation,
        "phosphorescence": args.phosphorescence,
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.num_threads))
    torch.set_default_dtype(getattr(torch, args.dtype))

    backbone = load_foundation_model(
        args.foundation_model, args.device, getattr(torch, args.dtype)
    )
    graphs, targets, _ = read_tdm_data(
        args.train_file, backbone, args.num_states, args.num_triplets
    )
    if args.valid_file:
        train_graphs, train_targets = graphs, targets
        valid_graphs, valid_targets, _ = read_tdm_data(
            args.valid_file, backbone, args.num_states, args.num_triplets
        )
    else:
        train_indices, valid_indices = split_indices(
            len(graphs), args.valid_fraction, args.seed
        )
        train_graphs = [graphs[index] for index in train_indices]
        valid_graphs = [graphs[index] for index in valid_indices]
        train_targets = {key: value[train_indices] for key, value in targets.items()}
        valid_targets = {key: value[valid_indices] for key, value in targets.items()}

    for key, value in train_targets.items():
        labelled = int(torch.isfinite(value.reshape(len(value), -1)).all(-1).sum())
        print(f"{key}: {labelled}/{len(value)} training configurations labelled")

    statistics = target_statistics(train_targets)
    options = model_options(args)
    model = TransitionDipoleMACE(backbone=backbone, **options).to(args.device)
    model.set_target_statistics(to_device(train_targets, args.device))
    energy_weight = args.energy_weight
    loss_fn = TransitionDipoleLoss(
        statistics,
        dipole_weight=args.dipole_weight,
        tensor_weight=args.tensor_weight,
        energy_weight=energy_weight,
        strength_weight=args.strength_weight,
        total_weight=args.total_weight,
        triplet_energy_weight=args.triplet_energy_weight,
        soc_weight=args.soc_weight,
        soc_vector_weight=args.soc_vector_weight if args.soc_mode == "diabatic" else 0.0,
        phosphorescence_weight=args.phosphorescence_weight if args.phosphorescence else 0.0,
        solvation_weight=args.solvation_weight if args.solvation else 0.0,
    ).to(args.device)

    model.set_backbone_trainable("head")
    train_loader = make_loader(train_graphs, args.batch_size, False)
    valid_loader = make_loader(valid_graphs, args.batch_size, False)
    cache_device = args.cache_device or args.device
    train_cache = cache_features(model, train_loader, args.device, cache_device)
    valid_cache = cache_features(model, valid_loader, args.device, cache_device)
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
            loss = loss_fn(prediction, to_device(batch_target, args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_parameters, 10.0)
            optimizer.step()
            total += loss.item() * size
            count += size
        valid_loss, valid_main = evaluate_cached(model, head_valid, loss_fn, args.device)
        # the schedule and the selection see the main objective only, so the
        # auxiliary branch cannot change how the main head is trained
        scheduler.step(valid_main)
        record = {
            "stage": "head",
            "epoch": epoch,
            "train_loss": total / count,
            "valid_loss": valid_loss,
            "valid_main_loss": valid_main,
        }
        history.append(record)
        if epoch == 1 or epoch % 25 == 0:
            print(json.dumps(record), flush=True)
        if valid_main < best_loss:
            best_loss, stale = valid_main, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)

    if args.fine_tune != "head":
        model.set_backbone_trainable(args.fine_tune, args.fine_tune_layers)
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
            valid_loss, valid_main = evaluate_graphs(model, valid_loader, loss_fn, args.device)
            record = {
                "stage": args.fine_tune,
                "epoch": epoch,
                "train_loss": total / count,
                "valid_loss": valid_loss,
                "valid_main_loss": valid_main,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if valid_main < best_loss:
                best_loss, stale = valid_main, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= args.patience:
                break
        model.load_state_dict(best_state)

    if args.calibrate == "mean":
        # log fits estimate geometric means; rates and spectra integrate arithmetic ones
        train_prediction, train_reference = predict(model, train_loader, args.device)
        model.set_calibration(
            {k: torch.tensor(v, device=args.device) for k, v in train_prediction.items()},
            {k: torch.tensor(v, device=args.device) for k, v in train_reference.items()},
        )
        print(json.dumps({
            "calibration": {
                "soc": model.head.soc_calibration.flatten().cpu().tolist(),
                "osce_t": model.head.osce_t_calibration.cpu().tolist(),
            }
        }), flush=True)

    prediction, reference = predict(model, valid_loader, args.device)
    metrics = regression_report(reference, prediction)
    print(json.dumps({"validation": metrics}, indent=2))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_state": model.state_dict(),
            "foundation_model": args.foundation_model,
            "options": options,
            "dtype": args.dtype,
            "metrics": metrics,
            "args": vars(args),
        },
        output,
    )
    save_json(str(output.with_suffix(".history.json")), {"history": history})
    save_json(str(output.with_suffix(".metrics.json")), metrics)


if __name__ == "__main__":
    main()

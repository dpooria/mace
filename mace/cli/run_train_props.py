"""Train the invariant state-property model on top of a MACE backbone.

Targets are NEMO ensemble columns written by ``db2xyz.py --props`` as
``REF_<name>``; pass them as ``--targets name[:log] ...``, e.g.

    --targets e_t1 d_t1 osce_t1:log soc_t1_s0:log soc_t1_s1:log

Stage 1 trains the head against cached frozen-backbone features; stage 2
optionally unfreezes part of the backbone, exactly as ``run_train_tdm``.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from mace.modules.state_properties import (
    StatePropertyLoss,
    StatePropertyMACE,
    parse_target_specs,
)
from mace.oscillator_strength import load_foundation_model, make_loader
from mace.state_property_data import batch_targets, property_report, read_property_data
from mace.transition_dipole_data import (
    CachedFeatures,
    cache_features,
    collate_features,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file", required=True)
    parser.add_argument("--output", default="props.model")
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="one loss weight per target (default: all 1)",
    )
    parser.add_argument("--foundation-model", default="medium_off")
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
    parser.add_argument("--fine-tune", choices=["head", "last", "full"], default="last")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    return parser.parse_args()


def run_head(model, scalars, batch, size, device):
    return model.head(scalars.to(device), batch.to(device), size)


def evaluate_cached(model, loader, loss_fn, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for scalars, _, _, batch, targets in loader:
            size = len(targets["props"])
            prediction = run_head(model, scalars, batch, size, device)
            loss = loss_fn(prediction, {k: v.to(device) for k, v in targets.items()})
            total += loss.item() * size
            count += size
    return total / count


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


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predicted, reference = [], []
    for batch in loader:
        batch = batch.to(device)
        predicted.append(model(batch.to_dict())["props"].cpu().numpy())
        reference.append(batch_targets(batch)["props"].cpu().numpy())
    return np.concatenate(predicted), np.concatenate(reference)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(getattr(torch, args.dtype))

    specs = parse_target_specs(args.targets)
    names = [str(item["name"]) for item in specs]
    log_mask = [bool(item["log"]) for item in specs]
    weights = args.weights or [1.0] * len(specs)
    if len(weights) != len(specs):
        raise ValueError("--weights needs one value per target")

    backbone = load_foundation_model(
        args.foundation_model, args.device, getattr(torch, args.dtype)
    )
    train_graphs, train_targets, _ = read_property_data(args.train_file, backbone, names)
    valid_graphs, valid_targets, _ = read_property_data(args.valid_file, backbone, names)
    for label, table in (("train", train_targets), ("valid", valid_targets)):
        bad = [
            name
            for name, logged, column in zip(names, log_mask, table["props"].T)
            if logged and bool((column <= 0).any())
        ]
        if bad:
            raise ValueError(f"{label}: log targets {bad} have non-positive values")

    model = StatePropertyMACE(
        backbone, specs, hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(args.device)
    model.set_target_statistics(train_targets["props"].to(args.device))
    loss_fn = StatePropertyLoss(model.head, weights).to(args.device)

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
        for scalars, _, _, batch, batch_target in head_train:
            size = len(batch_target["props"])
            optimizer.zero_grad(set_to_none=True)
            prediction = run_head(model, scalars, batch, size, args.device)
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

    # log targets: undo the geometric-mean bias so ensemble means are right
    train_prediction, train_reference = predict(model, train_loader, args.device)
    model.head.set_calibration(
        torch.tensor(train_prediction, device=args.device),
        torch.tensor(train_reference, device=args.device),
    )
    print(
        json.dumps(
            {"calibration": dict(zip(names, model.head.calibration.cpu().tolist()))}
        ),
        flush=True,
    )

    prediction, reference = predict(model, valid_loader, args.device)
    metrics = property_report(names, log_mask, reference, prediction)
    print(json.dumps({"validation": metrics}, indent=2))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": 1,
            "kind": "state_properties",
            "model_state": model.state_dict(),
            "foundation_model": args.foundation_model,
            "targets": specs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "dtype": args.dtype,
            "metrics": metrics,
        },
        output,
    )
    save_json(str(output.with_suffix(".history.json")), {"history": history})
    save_json(str(output.with_suffix(".metrics.json")), metrics)


if __name__ == "__main__":
    main()

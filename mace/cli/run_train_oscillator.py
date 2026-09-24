import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from mace.modules.oscillator import (
    OscillatorStrengthLoss,
    OscillatorStrengthMACE,
    target_statistics,
)
from mace.oscillator_strength import (
    load_foundation_model,
    make_loader,
    predict,
    read_oscillator_data,
    regression_metrics,
    save_json,
    split_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an invariant MACE oscillator-strength model"
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file")
    parser.add_argument("--output", default="oscillator_strength.model")
    parser.add_argument("--foundation-model", default="medium_off")
    parser.add_argument("--target-key", default="REF_osce")
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--head-epochs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--head-lr", type=float, default=1.0e-3)
    parser.add_argument("--backbone-lr", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--raw-loss-weight", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-distance-basis", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--fine-tune", choices=["head", "last", "full"], default="head"
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    return parser.parse_args()


@torch.no_grad()
def cache_features(model, loader, device):
    model.eval()
    features = []
    targets = []
    for batch in loader:
        batch = batch.to(device)
        features.append(model.encode(batch.to_dict()).cpu())
        targets.append(batch.oscillator_strength.cpu())
    return torch.cat(features), torch.cat(targets)


def cached_loader(features, targets, batch_size, train):
    dataset = torch.utils.data.TensorDataset(features, targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train
    )
    return loader


def evaluate_graphs(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            loss = loss_fn(model(batch.to_dict()), batch.oscillator_strength)
            total += loss.item() * batch.num_graphs
            count += batch.num_graphs
    return total / count


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)

    backbone = load_foundation_model(args.foundation_model, args.device, dtype)
    graphs, targets, _ = read_oscillator_data(
        args.train_file, backbone, args.target_key
    )
    if args.valid_file:
        valid_graphs, _, _ = read_oscillator_data(
            args.valid_file, backbone, args.target_key
        )
        train_graphs = graphs
        train_targets = targets
    else:
        train_indices, valid_indices = split_indices(
            len(graphs), args.valid_fraction, args.seed
        )
        train_graphs = [graphs[index] for index in train_indices]
        valid_graphs = [graphs[index] for index in valid_indices]
        train_targets = targets[train_indices]

    statistics = target_statistics(train_targets)
    model = OscillatorStrengthMACE(
        backbone=backbone,
        num_transitions=train_targets.shape[1],
        latent_mean=statistics["latent_mean"],
        latent_std=statistics["latent_std"],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_distance_basis=args.num_distance_basis,
    ).to(args.device)
    model.set_backbone_trainable("head")
    loss_fn = OscillatorStrengthLoss(
        **statistics, raw_weight=args.raw_loss_weight
    ).to(args.device)
    train_loader = make_loader(train_graphs, args.batch_size, False)
    valid_loader = make_loader(valid_graphs, args.batch_size, False)

    train_features, cached_train_targets = cache_features(
        model, train_loader, args.device
    )
    valid_features, cached_valid_targets = cache_features(
        model, valid_loader, args.device
    )
    model.set_feature_statistics(train_features.to(args.device))
    train_features = train_features.to(args.device)
    cached_train_targets = cached_train_targets.to(args.device)
    valid_features = valid_features.to(args.device)
    cached_valid_targets = cached_valid_targets.to(args.device)

    feature_variation = (
        (torch.std(train_features, dim=0) > 1.0e-6).float().mean().item()
    )
    print(
        json.dumps(
            {
                "cached_graphs": len(train_features) + len(valid_features),
                "feature_dimension": train_features.shape[1],
                "varying_feature_fraction": feature_variation,
            }
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.readout.parameters(), lr=args.head_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20, min_lr=1.0e-6
    )
    head_train_loader = cached_loader(
        train_features,
        cached_train_targets,
        args.head_batch_size,
        True,
    )
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.head_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for features, target in head_train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model.predict_from_features(features), target)
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(target)
            count += len(target)
        model.eval()
        with torch.no_grad():
            valid_loss = loss_fn(
                model.predict_from_features(valid_features), cached_valid_targets
            ).item()
        train_loss = total / count
        scheduler.step(valid_loss)
        record = {
            "stage": "head",
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }
        history.append(record)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(record), flush=True)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break

    model.load_state_dict(best_state)
    if args.fine_tune != "head":
        model.set_backbone_trainable(args.fine_tune)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.readout.parameters(), "lr": args.head_lr * 0.1},
                {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            ],
            weight_decay=args.weight_decay,
        )
        train_loader = make_loader(train_graphs, args.batch_size, True)
        epochs_without_improvement = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            count = 0
            for batch in train_loader:
                batch = batch.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch.to_dict()), batch.oscillator_strength)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                total += loss.detach().item() * batch.num_graphs
                count += batch.num_graphs
            valid_loss = evaluate_graphs(
                model, valid_loader, loss_fn, args.device
            )
            record = {
                "stage": args.fine_tune,
                "epoch": epoch,
                "train_loss": total / count,
                "valid_loss": valid_loss,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
        model.load_state_dict(best_state)

    predictions, valid_targets_array, _ = predict(model, valid_loader, args.device)
    metrics = regression_metrics(valid_targets_array, predictions)
    print(json.dumps({"validation_metrics": metrics}, indent=2))

    checkpoint = {
        "checkpoint_version": 2,
        "model_state": model.state_dict(),
        "foundation_model": args.foundation_model,
        "num_transitions": train_targets.shape[1],
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "num_distance_basis": args.num_distance_basis,
        "latent_mean": statistics["latent_mean"].cpu(),
        "latent_std": statistics["latent_std"].cpu(),
        "raw_std": statistics["raw_std"].cpu(),
        "dtype": args.dtype,
        "target_key": args.target_key,
        "fine_tune": args.fine_tune,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    save_json(str(output.with_suffix(".history.json")), {"history": history})
    save_json(str(output.with_suffix(".metrics.json")), metrics)


if __name__ == "__main__":
    main()

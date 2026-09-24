import argparse
import json

import torch

from mace.oscillator_strength import (
    load_trained_model,
    make_loader,
    predict,
    read_oscillator_data,
    regression_metrics,
    save_json,
    save_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a MACE oscillator-strength model"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--foundation-model")
    parser.add_argument("--target-key")
    parser.add_argument("--no-targets", action="store_true")
    parser.add_argument("--prediction-key", default="MACE_osce")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model, checkpoint = load_trained_model(
        args.model,
        args.device,
        args.foundation_model,
    )
    target_key = args.target_key or checkpoint["target_key"]
    graphs, _, atoms_list = read_oscillator_data(
        args.configs,
        model.backbone,
        target_key,
        require_targets=not args.no_targets,
        num_transitions=checkpoint["num_transitions"],
    )
    loader = make_loader(graphs, args.batch_size, False)
    predictions, targets, indices = predict(model, loader, args.device)
    save_predictions(
        atoms_list,
        predictions,
        indices,
        args.output,
        args.prediction_key,
    )
    if not args.no_targets:
        metrics = regression_metrics(targets, predictions)
        metrics_path = args.metrics or f"{args.output}.metrics.json"
        save_json(metrics_path, metrics)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

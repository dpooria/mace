"""Invariant per-geometry state properties on a MACE backbone.

Some quantities NEMO needs are rotational invariants of one geometry with no
phase to worry about, so they need no vector head:

    e_t1       T1 emission energy, eV
    d_t1       T1 susceptibility, eV (solvent interpolation)
    osce_t1    SOC-borrowed T1 -> S0 oscillator strength (phosphorescence)
    soc_*      total spin-orbit coupling magnitudes, eV (ISC / RISC rates)

Each target is regressed in a standardised space, optionally after ``log10``.
Oscillator strengths and SOC magnitudes span two to three decades, so a
linear loss would be dominated by the few brightest geometries; the log
space puts every geometry on the same footing and keeps the prediction
positive.  It also means the fit estimates the *geometric* mean, while a
spectrum or a rate is an arithmetic ensemble mean; ``calibration`` rescales
each log target once after training so that <prediction> = <reference> on the
training set (a smearing correction).  Without it the DiKTA T1 phosphorescence
rate came out ~20 % low.

``osce_t1`` predicted directly is the baseline.  It is built in NEMO from an
interfering second-order sum over singlet and triplet intermediates, so it is
expected to be rougher in the geometry than the singlet ``f``: in the DiKTA
T1 ensemble ``log osce_t1`` correlates with no single SOC magnitude above
r = 0.06.  The equivariant effective-dipole head is the planned replacement.
"""

from typing import Dict, List, Sequence

import torch

from mace.modules.transition_dipole import (
    NodeFeatureSplit,
    TransitionDipoleMACE,
    _mlp,
)
from mace.tools.scatter import scatter_mean

LOG_FLOOR = 1.0e-30


def parse_target_specs(specs: Sequence[str]) -> List[Dict[str, object]]:
    """``["osce_t1:log", "e_t1"]`` -> ``[{"name": ..., "log": ...}, ...]``."""
    parsed = []
    for spec in specs:
        name, _, transform = spec.partition(":")
        if transform not in {"", "linear", "log"}:
            raise ValueError(f"unknown transform {transform!r} in {spec!r}")
        parsed.append({"name": name, "log": transform == "log"})
    names = [item["name"] for item in parsed]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate targets in {specs}")
    return parsed


class StatePropertyHead(torch.nn.Module):
    """Pooled invariant readout; one output column per target."""

    def __init__(
        self,
        scalar_dim: int,
        num_targets: int,
        log_mask: Sequence[bool],
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(log_mask) != num_targets:
            raise ValueError("log_mask must have one entry per target")
        self.atom_mlp = _mlp([scalar_dim, hidden_dim, hidden_dim], dropout)
        self.output_mlp = _mlp([2 * hidden_dim, hidden_dim, num_targets], dropout)
        torch.nn.init.zeros_(self.output_mlp[-1].bias)
        self.register_buffer("log_mask", torch.tensor(list(log_mask), dtype=torch.bool))
        self.register_buffer("feature_mean", torch.zeros(scalar_dim))
        self.register_buffer("feature_std", torch.ones(scalar_dim))
        self.register_buffer("target_mean", torch.zeros(num_targets))
        self.register_buffer("target_std", torch.ones(num_targets))
        self.register_buffer("calibration", torch.ones(num_targets))

    def set_feature_statistics(self, scalars: torch.Tensor) -> None:
        self.feature_mean.copy_(scalars.mean(dim=0))
        self.feature_std.copy_(scalars.std(dim=0).clamp_min(1.0e-6))

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Physical units -> the space the loss is taken in (before scaling)."""
        logged = torch.log10(values.clamp_min(LOG_FLOOR))
        return torch.where(self.log_mask, logged, values)

    def untransform(self, values: torch.Tensor) -> torch.Tensor:
        return torch.where(self.log_mask, torch.pow(10.0, values), values)

    def set_target_statistics(self, targets: torch.Tensor) -> None:
        transformed = self.transform(targets)
        self.target_mean.copy_(torch.nanmean(transformed, dim=0))
        centred = transformed - self.target_mean
        variance = torch.nanmean(centred.square(), dim=0)
        self.target_std.copy_(variance.sqrt().clamp_min(1.0e-6))

    def set_calibration(self, prediction: torch.Tensor, targets: torch.Tensor) -> None:
        """Ratio of means on the training set, for log targets only.

        ``prediction`` must come from this head with the calibration at one.
        """
        keep = torch.isfinite(targets)
        ratio = torch.where(keep, targets, 0).sum(0) / torch.where(
            keep, prediction, 0
        ).sum(0).clamp_min(LOG_FLOOR)
        self.calibration.copy_(torch.where(self.log_mask, ratio, torch.ones_like(ratio)))

    def standardise(self, targets: torch.Tensor) -> torch.Tensor:
        return (self.transform(targets) - self.target_mean) / self.target_std

    def forward(
        self,
        scalars: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.atom_mlp((scalars - self.feature_mean) / self.feature_std)
        mean = scatter_mean(hidden, batch, dim=0, dim_size=num_graphs)
        mean_square = scatter_mean(hidden.square(), batch, dim=0, dim_size=num_graphs)
        spread = (mean_square - mean.square()).clamp_min(1.0e-12).sqrt()
        standardised = self.output_mlp(torch.cat([mean, spread], dim=-1))
        values = self.untransform(standardised * self.target_std + self.target_mean)
        values = values * self.calibration
        return {"standardised": standardised, "props": values}


class StatePropertyMACE(torch.nn.Module):
    """A MACE backbone with a :class:`StatePropertyHead` on top."""

    # backbone handling is identical to the transition-dipole model
    set_backbone_trainable = TransitionDipoleMACE.set_backbone_trainable
    encode = TransitionDipoleMACE.encode

    def __init__(
        self,
        backbone: torch.nn.Module,
        targets: Sequence[Dict[str, object]],
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.targets = [dict(item) for item in targets]
        self.names = [str(item["name"]) for item in targets]
        layer_irreps = [product.linear.irreps_out for product in backbone.products]
        self.split = NodeFeatureSplit(layer_irreps)
        self.head = StatePropertyHead(
            scalar_dim=self.split.scalar_dim,
            num_targets=len(self.names),
            log_mask=[bool(item["log"]) for item in targets],
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def set_feature_statistics(self, scalars: torch.Tensor) -> None:
        self.head.set_feature_statistics(scalars)

    def set_target_statistics(self, targets: torch.Tensor) -> None:
        self.head.set_target_statistics(targets)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        scalars, _ = self.encode(data)
        num_graphs = int(data["ptr"].numel() - 1)
        return self.head(scalars, data["batch"], num_graphs)


class StatePropertyLoss(torch.nn.Module):
    """Standardised MSE in the (log-)transformed space, NaN targets skipped."""

    def __init__(self, head: StatePropertyHead, weights: Sequence[float]):
        super().__init__()
        self.head = head
        self.register_buffer("weights", torch.tensor(list(weights)))

    def forward(
        self, prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        reference = self.head.standardise(target["props"])
        mask = torch.isfinite(reference)
        error = torch.where(
            mask, prediction["standardised"] - reference, torch.zeros_like(reference)
        )
        weights = self.weights.to(error.dtype) * mask.to(error.dtype)
        return (weights * error.square()).sum() / weights.sum().clamp_min(1.0)

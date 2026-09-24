from typing import Dict, List

import torch
from e3nn import o3

from mace.tools.scatter import scatter_mean


class InvariantFeatures(torch.nn.Module):
    def __init__(self, irreps: List[o3.Irreps]):
        super().__init__()
        self.irreps = irreps
        self.input_dim = sum(item.dim for layer in irreps for item in layer)
        self.output_dim = sum(item.mul for layer in irreps for item in layer)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        if node_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} node features, got "
                f"{node_features.shape[-1]}"
            )

        invariants = []
        offset = 0
        for layer in self.irreps:
            for mul, irrep in layer:
                width = mul * irrep.dim
                values = node_features[:, offset : offset + width]
                values = values.reshape(values.shape[0], mul, irrep.dim)
                if irrep.l == 0 and irrep.p == 1:
                    invariant = values.squeeze(-1)
                else:
                    invariant = torch.linalg.vector_norm(values, dim=-1)
                invariants.append(invariant)
                offset += width
        return torch.cat(invariants, dim=-1)


def target_to_latent(target: torch.Tensor) -> torch.Tensor:
    if target.shape[-1] < 2:
        return torch.log(target.clamp_min(1.0e-12))
    total = (target[:, 0] + target[:, 1]).clamp_min(1.0e-12)
    fraction = (target[:, 0] / total).clamp(1.0e-6, 1.0 - 1.0e-6)
    exchange = torch.log(fraction) - torch.log1p(-fraction)
    values = [torch.log(total), exchange]
    if target.shape[-1] > 2:
        values.append(torch.log(target[:, 2:].clamp_min(1.0e-12)))
    return torch.cat(
        [value.unsqueeze(-1) if value.ndim == 1 else value for value in values],
        dim=-1,
    )


def latent_to_target(latent: torch.Tensor) -> torch.Tensor:
    if latent.shape[-1] < 2:
        return torch.exp(latent)
    total = torch.exp(latent[:, 0])
    fraction = torch.sigmoid(latent[:, 1])
    values = [total * fraction, total * (1.0 - fraction)]
    if latent.shape[-1] > 2:
        values.append(torch.exp(latent[:, 2:]))
    return torch.cat(
        [value.unsqueeze(-1) if value.ndim == 1 else value for value in values],
        dim=-1,
    )


class OscillatorStrengthMACE(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        num_transitions: int,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        num_distance_basis: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_elements = len(backbone.atomic_numbers)
        self.num_distance_basis = num_distance_basis
        self.register_buffer(
            "distance_centers",
            torch.linspace(0.5, 12.0, num_distance_basis)
            if num_distance_basis > 0
            else torch.empty(0),
        )
        layer_irreps = [product.linear.irreps_out for product in backbone.products]
        self.invariant_features = InvariantFeatures(layer_irreps)
        local_dim = 2 * self.invariant_features.output_dim
        global_dim = self.num_elements * self.num_elements * num_distance_basis
        pooled_dim = local_dim + global_dim
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(pooled_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, num_transitions),
        )
        self.register_buffer("latent_mean", latent_mean.clone())
        self.register_buffer("latent_std", latent_std.clone())
        self.register_buffer("feature_mean", torch.zeros(pooled_dim))
        self.register_buffer("feature_std", torch.ones(pooled_dim))

    def set_feature_statistics(self, features: torch.Tensor) -> None:
        self.feature_mean.copy_(features.mean(dim=0))
        self.feature_std.copy_(features.std(dim=0).clamp_min(1.0e-6))

    def set_backbone_trainable(self, mode: str) -> None:
        if mode not in {"head", "last", "full"}:
            raise ValueError("mode must be 'head', 'last', or 'full'")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = mode == "full"
        if mode == "last":
            for parameter in self.backbone.interactions[-1].parameters():
                parameter.requires_grad = True
            for parameter in self.backbone.products[-1].parameters():
                parameter.requires_grad = True

    def pool(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        mean = scatter_mean(node_features, batch, dim=0, dim_size=num_graphs)
        mean_square = scatter_mean(
            node_features.square(), batch, dim=0, dim_size=num_graphs
        )
        std = (mean_square - mean.square()).clamp_min(1.0e-12).sqrt()
        return torch.cat([mean, std], dim=-1)

    def global_distance_features(
        self,
        data: Dict[str, torch.Tensor],
        num_graphs: int,
    ) -> torch.Tensor:
        if self.num_distance_basis == 0:
            return data["positions"].new_zeros((num_graphs, 0))
        outputs = []
        for graph_index in range(num_graphs):
            mask = data["batch"] == graph_index
            positions = data["positions"][mask]
            species = torch.argmax(data["node_attrs"][mask], dim=-1)
            distances = torch.cdist(positions, positions)
            pair_types = (
                species[:, None] * self.num_elements + species[None, :]
            )
            pair_mask = ~torch.eye(
                len(positions), dtype=torch.bool, device=positions.device
            )
            distances = distances[pair_mask]
            pair_types = pair_types[pair_mask]
            width = self.distance_centers[1] - self.distance_centers[0]
            radial = torch.exp(
                -0.5
                * (
                    (distances[:, None] - self.distance_centers[None, :])
                    / width
                ).square()
            )
            histogram = radial.new_zeros(
                self.num_elements * self.num_elements,
                self.num_distance_basis,
            )
            histogram.index_add_(0, pair_types, radial)
            outputs.append(histogram.flatten() / len(positions))
        return torch.stack(outputs)

    def encode(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.backbone(
            data,
            training=self.training,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_hessian=False,
        )
        node_features = self.invariant_features(output["node_feats"])
        num_graphs = int(data["ptr"].numel() - 1)
        local_features = self.pool(node_features, data["batch"], num_graphs)
        global_features = self.global_distance_features(data, num_graphs)
        return torch.cat([local_features, global_features], dim=-1)

    def predict_from_features(
        self, graph_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        features = (graph_features - self.feature_mean) / self.feature_std
        normalized_latent = self.readout(features)
        latent = normalized_latent * self.latent_std + self.latent_mean
        return {
            "oscillator_strength": latent_to_target(latent),
            "latent": latent,
            "normalized_latent": normalized_latent,
        }

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.predict_from_features(self.encode(data))


class OscillatorStrengthLoss(torch.nn.Module):
    def __init__(
        self,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        raw_std: torch.Tensor,
        raw_weight: float = 0.2,
    ):
        super().__init__()
        self.register_buffer("latent_mean", latent_mean.clone())
        self.register_buffer("latent_std", latent_std.clone())
        self.register_buffer("raw_std", raw_std.clone())
        self.raw_weight = raw_weight
        self.latent_loss = torch.nn.SmoothL1Loss()

    def forward(
        self,
        prediction: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        target_latent = target_to_latent(target)
        target_normalized = (target_latent - self.latent_mean) / self.latent_std
        loss_latent = self.latent_loss(
            prediction["normalized_latent"], target_normalized
        )
        scaled_error = (
            prediction["oscillator_strength"] - target
        ) / self.raw_std
        return loss_latent + self.raw_weight * torch.mean(scaled_error.square())


def target_statistics(targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    if torch.any(targets < 0):
        raise ValueError("Oscillator strengths must be non-negative")
    latent = target_to_latent(targets)
    return {
        "latent_mean": latent.mean(dim=0),
        "latent_std": latent.std(dim=0).clamp_min(1.0e-8),
        "raw_std": targets.std(dim=0).clamp_min(1.0e-8),
    }

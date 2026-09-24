"""Equivariant transition-dipole-moment head for a MACE backbone.

The oscillator strength of an electronic transition is not an independent
quantity:

    f_n = 2/3 * dE_n[Hartree] * |mu_n|^2

so a model that predicts the transition dipole moment ``mu_n`` can be combined
with excitation energies from any source (X-MACE, or the model's own mixing
matrix) to give ``f``.  Predicting ``mu`` rather than ``f`` is worth doing for
three reasons: ``mu`` is a vector, so the equivariant part of the backbone can
be used instead of discarded; it carries three numbers per state instead of
one; and the state mixing that makes ``f`` discontinuous acts on ``mu``
linearly, so it can be represented exactly.

Two bases are available.

``adiabatic``
    One vector per adiabatic state, predicted directly.  Simple, but adiabatic
    transition dipoles exchange between states wherever two states approach
    each other, so the target is not a smooth function of geometry.

``diabatic`` (default)
    The network predicts ``num_states`` *diabatic* transition dipoles together
    with a symmetric mixing matrix ``H`` -- both smooth in the nuclear
    coordinates.  Diagonalising ``H`` supplies the adiabatic rotation:

        E, U   = eigh(H)
        mu_i   = sum_k U[k, i] mu_d[k]

    The eigenvalues double as excitation energies, so the same model can also
    predict ``E`` if it is trained on it; if energies come from elsewhere only
    the eigenvectors are used.

Equivariance
    The dipole is the sum of a transition-charge term ``sum_i q_i r_i`` with
    ``sum_i q_i = 0`` -- which is what makes it translation invariant -- and a
    learned contraction of the backbone's ``1o`` node features, which supplies
    the local correction a point-charge picture misses.  Both terms rotate with
    the molecule and are invariant to atom permutation.

Phase
    The sign of ``mu_n`` is an arbitrary wavefunction phase and differs between
    single-point calculations, so nothing here -- loss or metric -- may depend
    on it.  Supervision goes through ``mu mu^T``, which is phase free and still
    carries the direction.
"""

from typing import Dict, List, Optional, Tuple

import torch
from e3nn import o3

from mace.tools.scatter import scatter_mean, scatter_sum

HARTREE_PER_EV = 1.0 / 27.211386245988


class _SafeEigh(torch.autograd.Function):
    """``torch.linalg.eigh`` with the eigenvector gradient clamped.

    The eigenvector derivative carries ``1 / (lambda_j - lambda_i)``, which
    diverges exactly at the near-degeneracies this model exists to describe.
    The denominator is clamped to ``eps`` so training stays finite there.
    """

    @staticmethod
    def forward(ctx, matrix: torch.Tensor, eps: float):
        values, vectors = torch.linalg.eigh(matrix)
        ctx.save_for_backward(values, vectors)
        ctx.eps = eps
        return values, vectors

    @staticmethod
    def backward(ctx, grad_values: torch.Tensor, grad_vectors: torch.Tensor):
        values, vectors = ctx.saved_tensors
        gaps = values.unsqueeze(-2) - values.unsqueeze(-1)
        sign = torch.where(gaps < 0, -torch.ones_like(gaps), torch.ones_like(gaps))
        clamped = torch.where(gaps.abs() < ctx.eps, sign * ctx.eps, gaps)
        diagonal = torch.eye(
            values.shape[-1], dtype=torch.bool, device=values.device
        ).expand_as(gaps)
        factor = torch.where(diagonal, torch.zeros_like(clamped), 1.0 / clamped)
        inner = factor * (vectors.transpose(-1, -2) @ grad_vectors)
        grad = (
            vectors
            @ (torch.diag_embed(grad_values) + inner)
            @ vectors.transpose(-1, -2)
        )
        return 0.5 * (grad + grad.transpose(-1, -2)), None


def safe_eigh(matrix: torch.Tensor, eps: float = 1.0e-3):
    return _SafeEigh.apply(matrix, eps)


class NodeFeatureSplit(torch.nn.Module):
    """Split concatenated MACE ``node_feats`` into invariants and ``1o`` vectors.

    Scalars are the ``0e`` channels plus the norms of every ``l > 0`` channel.
    Vectors are the ``1o`` channels: odd parity, ``l = 1``, which is how a
    polar vector such as a transition dipole transforms.
    """

    def __init__(self, layer_irreps: List[o3.Irreps]):
        super().__init__()
        scalar_slices: List[Tuple[int, int]] = []
        norm_slices: List[Tuple[int, int, int, int]] = []
        vector_slices: List[Tuple[int, int]] = []
        offset = 0
        for irreps in layer_irreps:
            for mul, irrep in irreps:
                width = mul * irrep.dim
                if irrep.l == 0 and irrep.p == 1:
                    scalar_slices.append((offset, width))
                else:
                    norm_slices.append((offset, width, mul, irrep.dim))
                    if irrep.l == 1 and irrep.p == -1:
                        vector_slices.append((offset, mul))
                offset += width
        self.input_dim = offset
        self.scalar_slices = scalar_slices
        self.norm_slices = norm_slices
        self.vector_slices = vector_slices
        self.scalar_dim = sum(width for _, width in scalar_slices) + sum(
            mul for _, _, mul, _ in norm_slices
        )
        self.vector_channels = sum(mul for _, mul in vector_slices)
        if self.vector_channels == 0:
            raise ValueError(
                "the backbone exposes no 1o node features, so it cannot "
                "predict a transition dipole"
            )

    def forward(self, node_feats: torch.Tensor):
        if node_feats.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} node features, got {node_feats.shape[-1]}"
            )
        scalars = [
            node_feats[:, offset : offset + width] for offset, width in self.scalar_slices
        ]
        for offset, width, mul, dim in self.norm_slices:
            block = node_feats[:, offset : offset + width].reshape(-1, mul, dim)
            scalars.append(torch.linalg.vector_norm(block, dim=-1))
        vectors = [
            node_feats[:, offset : offset + 3 * mul].reshape(-1, mul, 3)
            for offset, mul in self.vector_slices
        ]
        return torch.cat(scalars, dim=-1), torch.cat(vectors, dim=-2)


def _mlp(sizes: List[int], dropout: float = 0.0) -> torch.nn.Sequential:
    layers: List[torch.nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(torch.nn.SiLU())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
    return torch.nn.Sequential(*layers)


class TransitionDipoleHead(torch.nn.Module):
    """Readout from per-atom features to transition dipoles.

    Kept separate from the backbone so it can be trained against cached
    features, which is what makes hyper-parameter selection affordable on a
    dataset this size.
    """

    def __init__(
        self,
        scalar_dim: int,
        vector_channels: int,
        num_states: int,
        energy_reference: torch.Tensor,
        basis: str = "diabatic",
        hidden_dim: int = 64,
        dropout: float = 0.0,
        coupling_scale: float = 0.2,
        charge_scale: float = 0.1,
        vector_scale: float = 0.1,
        eigh_eps: float = 1.0e-3,
        factorised: bool = True,
    ):
        super().__init__()
        if basis not in {"adiabatic", "diabatic"}:
            raise ValueError("basis must be 'adiabatic' or 'diabatic'")
        self.num_states = num_states
        self.basis = basis
        self.factorised = factorised
        self.coupling_scale = coupling_scale
        self.charge_scale = charge_scale
        self.vector_scale = vector_scale
        self.eigh_eps = eigh_eps

        self.atom_mlp = _mlp([scalar_dim, hidden_dim, hidden_dim], dropout)
        self.charge_head = torch.nn.Linear(hidden_dim, num_states)
        self.vector_head = torch.nn.Linear(hidden_dim, num_states * vector_channels)
        self.mixing_mlp = _mlp(
            [2 * hidden_dim, hidden_dim, num_states * (num_states + 1) // 2],
            dropout,
        )
        # |mu| is a rotational invariant, so it can be read out from the
        # invariant features directly instead of emerging from the vector
        # components.  Predicting the two separately stops direction error from
        # leaking into the magnitude -- and |mu|^2 is what f actually uses.
        self.magnitude_mlp = _mlp([2 * hidden_dim, hidden_dim, num_states], dropout)
        self.register_buffer("magnitude_reference", torch.ones(num_states))
        # the mixing matrix starts at the mean adiabatic energies; the dipole
        # path deliberately does NOT start near zero (see phase_matched_error)
        torch.nn.init.zeros_(self.mixing_mlp[-1].bias)
        self.mixing_mlp[-1].weight.data.mul_(0.1)

        indices = torch.triu_indices(num_states, num_states)
        self.register_buffer("triu_rows", indices[0])
        self.register_buffer("triu_cols", indices[1])
        self.register_buffer("energy_reference", energy_reference.clone())
        self.register_buffer("feature_mean", torch.zeros(scalar_dim))
        self.register_buffer("feature_std", torch.ones(scalar_dim))

    def set_feature_statistics(self, scalars: torch.Tensor) -> None:
        self.feature_mean.copy_(scalars.mean(dim=0))
        self.feature_std.copy_(scalars.std(dim=0).clamp_min(1.0e-6))

    def set_magnitude_reference(self, dipole: torch.Tensor) -> None:
        self.magnitude_reference.copy_(
            dipole.square().sum(-1).mean(dim=0).sqrt().clamp_min(1.0e-6)
        )

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        energy_ex: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.atom_mlp((scalars - self.feature_mean) / self.feature_std)
        mean = scatter_mean(hidden, batch, dim=0, dim_size=num_graphs)
        mean_square = scatter_mean(hidden.square(), batch, dim=0, dim_size=num_graphs)
        spread = (mean_square - mean.square()).clamp_min(1.0e-12).sqrt()
        pooled = torch.cat([mean, spread], dim=-1)

        # transition charges; zero net charge makes the term translation invariant
        charges = self.charge_head(hidden)
        charges = (
            charges - scatter_mean(charges, batch, dim=0, dim_size=num_graphs)[batch]
        )
        centroid = scatter_mean(positions, batch, dim=0, dim_size=num_graphs)
        relative = positions - centroid[batch]
        dipole_charge = scatter_sum(
            charges.unsqueeze(-1) * relative.unsqueeze(-2),
            batch,
            dim=0,
            dim_size=num_graphs,
        )

        # equivariant local correction from the backbone 1o channels
        dipole_local = torch.zeros_like(dipole_charge)
        if self.vector_scale != 0.0:
            weights = self.vector_head(hidden).view(
                -1, self.num_states, vectors.shape[-2]
            )
            dipole_local = scatter_sum(
                torch.einsum("nkc,nca->nka", weights, vectors),
                batch,
                dim=0,
                dim_size=num_graphs,
            )
        diabatic = self.charge_scale * dipole_charge + self.vector_scale * dipole_local
        if self.factorised:
            direction = diabatic / diabatic.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            magnitude = torch.nn.functional.softplus(
                self.magnitude_mlp(pooled) + 0.5413
            ) * self.magnitude_reference
            diabatic = magnitude.unsqueeze(-1) * direction

        if self.basis == "adiabatic":
            eye = torch.eye(self.num_states, device=scalars.device).expand(
                num_graphs, -1, -1
            )
            return {
                "tdm": diabatic,
                "tdm_diabatic": diabatic,
                "rotation": eye,
                "mixing": torch.diag_embed(
                    self.energy_reference.expand(num_graphs, -1)
                ),
                "energy_ex": self.energy_reference.expand(num_graphs, -1)
                if energy_ex is None
                else energy_ex,
            }

        # smooth symmetric mixing matrix from graph-level invariants
        entries = self.mixing_mlp(pooled) * self.coupling_scale
        matrix = entries.new_zeros(num_graphs, self.num_states, self.num_states)
        matrix[:, self.triu_rows, self.triu_cols] = entries
        matrix = matrix + matrix.transpose(-1, -2)
        matrix = matrix - 0.5 * torch.diag_embed(
            torch.diagonal(matrix, dim1=-2, dim2=-1)
        )
        mixing = matrix + torch.diag_embed(self.energy_reference.expand(num_graphs, -1))

        values, rotation = safe_eigh(mixing, self.eigh_eps)
        adiabatic = torch.einsum("nki,nka->nia", rotation, diabatic)
        return {
            "tdm": adiabatic,
            "tdm_diabatic": diabatic,
            "rotation": rotation,
            "mixing": mixing,
            "energy_ex": values if energy_ex is None else energy_ex,
        }


class TransitionDipoleMACE(torch.nn.Module):
    """A MACE backbone with a :class:`TransitionDipoleHead` on top."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        num_states: int,
        energy_reference: torch.Tensor,
        basis: str = "diabatic",
        hidden_dim: int = 64,
        dropout: float = 0.0,
        coupling_scale: float = 0.2,
        charge_scale: float = 0.1,
        vector_scale: float = 0.1,
        eigh_eps: float = 1.0e-3,
        factorised: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_states = num_states
        layer_irreps = [product.linear.irreps_out for product in backbone.products]
        self.split = NodeFeatureSplit(layer_irreps)
        self.head = TransitionDipoleHead(
            scalar_dim=self.split.scalar_dim,
            vector_channels=self.split.vector_channels,
            num_states=num_states,
            energy_reference=energy_reference,
            basis=basis,
            hidden_dim=hidden_dim,
            dropout=dropout,
            coupling_scale=coupling_scale,
            charge_scale=charge_scale,
            vector_scale=vector_scale,
            eigh_eps=eigh_eps,
            factorised=factorised,
        )

    @property
    def basis(self) -> str:
        return self.head.basis

    def set_backbone_trainable(self, mode: str) -> None:
        if mode not in {"head", "last", "full"}:
            raise ValueError("mode must be 'head', 'last' or 'full'")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = mode == "full"
        if mode == "last":
            for parameter in self.backbone.interactions[-1].parameters():
                parameter.requires_grad = True
            for parameter in self.backbone.products[-1].parameters():
                parameter.requires_grad = True

    def encode(self, data: Dict[str, torch.Tensor]):
        output = self.backbone(
            data,
            training=self.training,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_hessian=False,
        )
        return self.split(output["node_feats"])

    def set_feature_statistics(self, scalars: torch.Tensor) -> None:
        self.head.set_feature_statistics(scalars)

    def set_magnitude_reference(self, dipole: torch.Tensor) -> None:
        self.head.set_magnitude_reference(dipole)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        energy_ex: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        scalars, vectors = self.encode(data)
        num_graphs = int(data["ptr"].numel() - 1)
        return self.head(
            scalars, vectors, data["positions"], data["batch"], num_graphs, energy_ex
        )


def intensity_tensor(dipole: torch.Tensor) -> torch.Tensor:
    """``mu mu^T`` per state -- phase free, unlike the dipole itself."""
    return torch.einsum("nia,nib->niab", dipole, dipole)


def phase_matched_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """``|mu_pred - s mu_ref|`` with ``s`` the sign that makes them agree.

    The sign of a transition dipole is an arbitrary wavefunction phase and is
    effectively random between single-point calculations, so the loss must not
    see it.  Matching the sign is the phase-free choice that stays *quadratic*
    in the prediction.  ``mu mu^T`` is also phase free but is quartic, and
    quartic means ``mu = 0`` is a critical point with vanishing gradient -- a
    model initialised near zero never leaves it.  That trap is why this is the
    default.
    """
    sign = torch.sign((prediction * target).sum(-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return prediction - sign * target


def oscillator_strength(
    dipole: torch.Tensor, energy_ex_ev: torch.Tensor
) -> torch.Tensor:
    """``f = 2/3 dE[Hartree] |mu|^2`` from a dipole and an excitation energy in eV."""
    return (2.0 / 3.0) * (energy_ex_ev * HARTREE_PER_EV) * dipole.square().sum(-1)


class TransitionDipoleLoss(torch.nn.Module):
    """Phase-free loss on the dipole, with optional energy and ``f`` terms.

    ``dipole_weight`` drives the sign-matched quadratic term, which is what
    actually trains the model.  ``tensor_weight`` adds the quartic ``mu mu^T``
    term; it is a sharper match to how ``f`` uses the dipole but is useless on
    its own, so it defaults to a supporting role.
    """

    def __init__(
        self,
        dipole_std: torch.Tensor,
        tensor_std: torch.Tensor,
        energy_std: torch.Tensor,
        strength_std: torch.Tensor,
        dipole_weight: float = 1.0,
        tensor_weight: float = 0.3,
        energy_weight: float = 1.0,
        strength_weight: float = 0.3,
        total_weight: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("dipole_std", dipole_std.clamp_min(1.0e-8))
        self.register_buffer("tensor_std", tensor_std.clamp_min(1.0e-8))
        self.register_buffer("energy_std", energy_std.clamp_min(1.0e-8))
        self.register_buffer("strength_std", strength_std.clamp_min(1.0e-8))
        self.dipole_weight = dipole_weight
        self.tensor_weight = tensor_weight
        self.energy_weight = energy_weight
        self.strength_weight = strength_weight
        self.total_weight = total_weight

    def forward(
        self,
        prediction: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        error = phase_matched_error(prediction["tdm"], target["tdm"])
        total = (
            self.dipole_weight
            * (error / self.dipole_std.view(1, -1, 1)).square().mean()
        )
        if self.tensor_weight > 0.0:
            tensor_error = (
                intensity_tensor(prediction["tdm"]) - intensity_tensor(target["tdm"])
            ) / self.tensor_std.view(1, -1, 1, 1)
            total = total + self.tensor_weight * tensor_error.square().mean()
        if self.energy_weight > 0.0:
            energy_error = (
                prediction["energy_ex"] - target["energy_ex"]
            ) / self.energy_std
            total = total + self.energy_weight * energy_error.square().mean()
        if self.strength_weight > 0.0:
            strength = oscillator_strength(prediction["tdm"], prediction["energy_ex"])
            strength_error = (strength - target["osce"]) / self.strength_std
            total = total + self.strength_weight * strength_error.square().mean()
        if self.total_weight > 0.0:
            # Off by default: measured, and it hurts.  The summed intensity is
            # better conditioned than any single state, but it is a function of
            # predictions the per-state terms already constrain, so it adds no
            # information and only reweights the objective towards a coarser
            # target.  At weight 0.3 it cost 0.06 on per-state f AND 0.15 on
            # the sum itself.  Kept only so the experiment is repeatable.
            strength = oscillator_strength(prediction["tdm"], prediction["energy_ex"])
            total_error = (
                strength.sum(-1) - target["osce"].sum(-1)
            ) / self.strength_std.sum()
            total = total + self.total_weight * total_error.square().mean()
        return total


def target_statistics(targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    dipole = targets["tdm"]
    return {
        "energy_reference": targets["energy_ex"].mean(dim=0),
        "energy_std": targets["energy_ex"].std(dim=0),
        # RMS rather than std: the sign is arbitrary so the mean is meaningless
        "dipole_std": dipole.square().sum(-1).mean(dim=0).sqrt(),
        "tensor_std": intensity_tensor(dipole).flatten(2).std(dim=(0, 2)),
        "strength_std": targets["osce"].std(dim=0),
    }

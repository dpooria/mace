"""Equivariant excited-state head for a MACE backbone.

One head predicts, for any molecule,

    singlets  S_1..S_nS   excitation energies E, transition dipoles <S0|mu|S_n>
                          and hence the oscillator strengths
                              f_n = 2/3 * dE_n[Hartree] * |mu_n|^2
    triplets  T_1..T_nT   excitation energies
    couplings             spin-orbit coupling magnitudes |<S_m|H_SO|T_k>|,
                          m = 0..nS, k = 1..nT (sublevel root-sum-square)

Predicting ``mu`` rather than ``f`` is worth doing for three reasons: ``mu``
is a vector, so the equivariant part of the backbone can be used instead of
discarded; it carries three numbers per state instead of one; and the state
mixing that makes ``f`` discontinuous acts on ``mu`` linearly, so it can be
represented exactly.

Bases
    ``diabatic`` (default): every manifold is described by a smooth symmetric
    mixing matrix ``H``.  Diagonalising it gives the adiabatic energies and the
    rotation that turns smooth *diabatic* state properties into the
    discontinuous adiabatic ones::

        E, U = eigh(H)          mu_i = sum_k U[k, i] mu_d[k]

    ``adiabatic``: energies are regressed directly and state properties are
    predicted per adiabatic state.  Simpler, and a useful control, but the
    targets are not smooth functions of geometry.

Standardisation
    Nothing in the head carries a unit.  Energies are ``reference + scale *
    output`` with the reference and scale taken from the training labels,
    vector magnitudes are multiples of the training RMS, and spin-orbit
    couplings are log-normal around the training log-mean.  The only physical
    constant is the eigenvector-gradient clamp ``eigh_eps`` (eV).

Equivariance
    Polar vectors (transition dipoles) are a transition-charge term
    ``sum_i q_i r_i`` with ``sum_i q_i = 0`` plus a learned contraction of the
    backbone's ``1o`` node features.  Axial vectors (spin-orbit couplings:
    ``<S|H_SO|T_a>`` is ``i`` times a real axial vector in the Cartesian
    sublevel basis) are cross products of such contractions.  Every vector is
    factorised into an invariant magnitude and a unit direction, so direction
    error never reaches ``|mu|`` or ``|SOC|``, which is what the observables use.

Phase
    The sign of ``mu_n`` is an arbitrary wavefunction phase and differs between
    single-point calculations, so nothing here -- loss or metric -- may depend
    on it.  Supervision uses the sign-matched quadratic error (see
    :func:`phase_matched_error`); ``mu mu^T`` is a supporting term only.
"""

from typing import Dict, List, Optional, Tuple

import torch
from e3nn import o3

from mace.tools.scatter import scatter_mean, scatter_sum

HARTREE_PER_EV = 1.0 / 27.211386245988
UNIT_MAGNITUDE = 0.5413  # softplus(0.5413) = 1: magnitudes start at the reference
LOG_FLOOR = 1.0e-12
SOC_MODES = ("none", "direct", "diabatic")


# --------------------------------------------------------------------- eigh
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


# ------------------------------------------------------------ small helpers
def nanmean(values: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.nanmean(values, dim=dim)


def nanstd(values: torch.Tensor, dim: int = 0) -> torch.Tensor:
    centred = values - torch.nanmean(values, dim=dim, keepdim=True)
    return torch.sqrt(torch.nanmean(centred.square(), dim=dim))


def _fill(values: torch.Tensor, default: float) -> torch.Tensor:
    """Replace NaN statistics (a label nobody supplied) by ``default``."""
    return torch.nan_to_num(values, nan=default, posinf=default, neginf=default)


def _symmetric(entries: torch.Tensor, size: int) -> torch.Tensor:
    """Upper-triangle entries (..., size(size+1)/2) -> symmetric (..., size, size)."""
    rows, cols = torch.triu_indices(size, size, device=entries.device)
    matrix = entries.new_zeros(*entries.shape[:-1], size, size)
    matrix[..., rows, cols] = entries
    matrix = matrix + matrix.transpose(-1, -2)
    return matrix - 0.5 * torch.diag_embed(torch.diagonal(matrix, dim1=-2, dim2=-1))


def _mlp(sizes: List[int], dropout: float = 0.0) -> torch.nn.Sequential:
    layers: List[torch.nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(torch.nn.SiLU())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
    return torch.nn.Sequential(*layers)


def _small_output(mlp: torch.nn.Sequential, factor: float = 0.1) -> None:
    torch.nn.init.zeros_(mlp[-1].bias)
    mlp[-1].weight.data.mul_(factor)


# ---------------------------------------------------------- feature split
class NodeFeatureSplit(torch.nn.Module):
    """Split concatenated MACE ``node_feats`` into invariants and ``1o`` vectors.

    Scalars are the ``0e`` channels plus the norms of every ``l > 0`` channel.
    Vectors are the ``1o`` channels: odd parity, ``l = 1``, which is how a
    polar vector such as a transition dipole transforms.  Works for any MACE
    backbone whose products expose ``linear.irreps_out`` (MACE-OFF, MACE-MP,
    MACE-MH, MACE-OMOL).
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


# ---------------------------------------------------------------- blocks
class Manifold(torch.nn.Module):
    """Excitation energies of one spin manifold and the adiabatic rotation.

    ``diabatic``: a smooth symmetric mixing matrix whose eigenvalues are the
    energies and whose eigenvectors rotate every state property.
    ``adiabatic``: energies regressed directly, identity rotation.

    The matrix is built in standardised units -- diagonal entries are
    ``reference + scale * output`` per state, off-diagonal couplings are in
    units of the mean energy scale -- so nothing here depends on the dataset
    beyond the statistics set from its training labels.
    """

    def __init__(
        self,
        hidden_dim: int,
        states: int,
        basis: str = "diabatic",
        dropout: float = 0.0,
        coupling_scale: float = 1.0,
        eigh_eps: float = 1.0e-3,
    ):
        super().__init__()
        if basis not in {"adiabatic", "diabatic"}:
            raise ValueError("basis must be 'adiabatic' or 'diabatic'")
        self.states = states
        self.basis = basis
        self.coupling_scale = coupling_scale
        self.eigh_eps = eigh_eps
        self.mixing = _mlp([2 * hidden_dim, hidden_dim, states * (states + 1) // 2], dropout)
        _small_output(self.mixing)
        self.register_buffer("energy_reference", torch.zeros(states))
        self.register_buffer("energy_scale", torch.ones(states))

    def set_statistics(self, energies: torch.Tensor) -> None:
        """``energies``: training labels (num_configs, states), NaN allowed."""
        self.energy_reference.copy_(_fill(nanmean(energies), 0.0))
        self.energy_scale.copy_(_fill(nanstd(energies), 1.0).clamp_min(1.0e-3))

    def forward(self, pooled: torch.Tensor):
        num_graphs = pooled.shape[0]
        raw = _symmetric(self.mixing(pooled), self.states)
        diagonal = torch.diagonal(raw, dim1=-2, dim2=-1)
        off_diagonal = raw - torch.diag_embed(diagonal)
        mixing = torch.diag_embed(
            self.energy_reference + self.energy_scale * diagonal
        ) + self.coupling_scale * self.energy_scale.mean() * off_diagonal
        if self.basis == "adiabatic":
            energies = torch.diagonal(mixing, dim1=-2, dim2=-1)
            rotation = torch.eye(
                self.states, device=pooled.device, dtype=pooled.dtype
            ).expand(num_graphs, -1, -1)
        else:
            energies, rotation = safe_eigh(mixing, self.eigh_eps)
        return energies, rotation, mixing


class VectorReadout(torch.nn.Module):
    """``count`` equivariant vectors per graph, factorised as magnitude x direction.

    ``kind="polar"``: transition charges (zero net charge, so translation
    invariant) times centroid-relative positions, plus a contraction of the
    ``1o`` node features.  Odd under inversion, like a dipole.

    ``kind="axial"``: cross products of two such contractions plus
    ``r x (contraction)``.  Even under inversion, like angular momentum and
    hence like a spin-orbit coupling vector.

    The magnitude is an invariant readout of the pooled features.  It is
    either a multiple of a reference RMS (``scale="linear"``, softplus) or
    log-normal around a reference (``scale="log"``), for quantities that span
    decades.  With ``factorised=False`` the raw summed vector is returned,
    scaled by the reference -- the ablation control.
    """

    def __init__(
        self,
        hidden_dim: int,
        channels: int,
        count: int,
        kind: str = "polar",
        dropout: float = 0.0,
        scale: str = "linear",
        factorised: bool = True,
    ):
        super().__init__()
        if kind not in {"polar", "axial"}:
            raise ValueError("kind must be 'polar' or 'axial'")
        if scale not in {"linear", "log"}:
            raise ValueError("scale must be 'linear' or 'log'")
        self.kind, self.scale, self.count, self.channels = kind, scale, count, channels
        self.factorised = factorised
        self.width = 1 if kind == "polar" else 3
        self.charge = torch.nn.Linear(hidden_dim, count)
        self.weights = torch.nn.Linear(hidden_dim, self.width * count * channels)
        self.magnitude = _mlp([2 * hidden_dim, hidden_dim, count], dropout)
        self.register_buffer("reference", torch.ones(count))
        self.register_buffer("log_reference", torch.zeros(count))
        self.register_buffer("log_scale", torch.ones(count))

    def set_reference(self, rms: torch.Tensor) -> None:
        """Linear scale: the training RMS of each vector (NaN -> 1)."""
        self.reference.copy_(_fill(rms, 1.0).clamp_min(LOG_FLOOR))

    def set_log_statistics(self, magnitudes: torch.Tensor) -> None:
        """Log scale: mean and std of ``log`` of the training magnitudes.

        ``magnitudes``: (num_configs, count), NaN where a label is missing; a
        column nobody labelled takes the statistics of the labelled ones.
        """
        logged = torch.log(magnitudes.clamp_min(LOG_FLOOR))
        logged = torch.where(torch.isfinite(magnitudes), logged, torch.nan)
        mean, std = nanmean(logged), nanstd(logged)
        overall_mean = _fill(torch.nanmean(mean), 0.0)
        overall_std = _fill(torch.nanmean(std), 1.0)
        self.log_reference.copy_(_fill(mean, float(overall_mean)))
        self.log_scale.copy_(_fill(std, float(overall_std)).clamp_min(1.0e-3))

    def forward(
        self,
        hidden: torch.Tensor,
        pooled: torch.Tensor,
        vectors: torch.Tensor,
        relative: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        weights = self.weights(hidden).view(
            hidden.shape[0], self.width, self.count, self.channels
        )
        contraction = torch.einsum("nwkc,nca->nwka", weights, vectors)
        lever = relative.unsqueeze(-2).expand(-1, self.count, -1)
        if self.kind == "polar":
            charges = self.charge(hidden)
            charges = (
                charges - scatter_mean(charges, batch, dim=0, dim_size=num_graphs)[batch]
            )
            atomic = contraction[:, 0] + charges.unsqueeze(-1) * lever
        else:
            atomic = torch.cross(contraction[:, 0], contraction[:, 1], dim=-1)
            atomic = atomic + torch.cross(lever, contraction[:, 2], dim=-1)
        raw = scatter_sum(atomic, batch, dim=0, dim_size=num_graphs)
        if not self.factorised:
            return raw * self.reference.unsqueeze(-1)
        direction = raw / raw.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
        output = self.magnitude(pooled)
        if self.scale == "linear":
            size = torch.nn.functional.softplus(output + UNIT_MAGNITUDE) * self.reference
        else:
            size = torch.exp(self.log_reference + self.log_scale * output)
        return size.unsqueeze(-1) * direction


# ------------------------------------------------------------------ head
class TransitionDipoleHead(torch.nn.Module):
    """Readout from per-atom features to energies, dipoles and couplings.

    Kept separate from the backbone so it can be trained against cached
    features, which is what makes hyper-parameter selection affordable.

    Outputs (``G`` graphs, ``nS`` singlets, ``nT`` triplets):

        energy_ex      (G, nS)        singlet excitation energies, eV
        tdm            (G, nS, 3)     adiabatic <S0|mu|S_n>, a.u.
        tdm_diabatic   (G, nS, 3)
        rotation       (G, nS, nS)    U_S
        mixing         (G, nS, nS)    H_S
        energy_t       (G, nT)        triplet excitation energies, eV
        rotation_t     (G, nT, nT)    U_T
        mixing_t       (G, nT, nT)    H_T
        soc            (G, nS+1, nT)  |<S_m|H_SO|T_k>|, m = 0..nS, eV
        soc_vectors    (G, nS+1, nT, 3)  adiabatic axial vectors (diabatic mode)
    """

    def __init__(
        self,
        scalar_dim: int,
        vector_channels: int,
        num_states: int,
        num_triplets: int = 0,
        basis: str = "diabatic",
        hidden_dim: int = 64,
        vector_dim: int = 32,
        dropout: float = 0.0,
        coupling_scale: float = 1.0,
        eigh_eps: float = 1.0e-3,
        factorised: bool = True,
        soc_mode: str = "diabatic",
        solvation: bool = False,
        phosphorescence: bool = False,
    ):
        super().__init__()
        if soc_mode not in SOC_MODES:
            raise ValueError(f"soc_mode must be one of {SOC_MODES}")
        if num_states < 1:
            raise ValueError("at least one singlet state is needed")
        self.num_states = num_states
        self.num_triplets = num_triplets
        self.basis = basis
        self.factorised = factorised
        self.soc_mode = soc_mode if num_triplets > 0 else "none"
        self.eigh_eps = eigh_eps

        self.atom_mlp = _mlp([scalar_dim, hidden_dim, hidden_dim], dropout)
        self.register_buffer("feature_mean", torch.zeros(scalar_dim))
        self.register_buffer("feature_std", torch.ones(scalar_dim))
        # channel mixing of the 1o features is equivariant; it keeps the
        # vector readouts small whatever the backbone width
        vector_dim = min(vector_dim, vector_channels)
        self.vector_mix = torch.nn.Linear(vector_channels, vector_dim, bias=False)

        self.singlet = Manifold(hidden_dim, num_states, basis, dropout, coupling_scale, eigh_eps)
        self.dipole = VectorReadout(
            hidden_dim, vector_dim, num_states, "polar", dropout, "linear", factorised
        )
        if num_triplets > 0:
            self.triplet = Manifold(
                hidden_dim, num_triplets, basis, dropout, coupling_scale, eigh_eps
            )
        pairs = (num_states + 1) * num_triplets
        if self.soc_mode == "diabatic":
            self.soc = VectorReadout(hidden_dim, vector_dim, pairs, "axial", dropout, "log")
        elif self.soc_mode == "direct":
            self.soc = _mlp([2 * hidden_dim, hidden_dim, pairs], dropout)
            self.register_buffer("soc_log_reference", torch.zeros(pairs))
            self.register_buffer("soc_log_scale", torch.ones(pairs))
        # ensemble-mean calibration of log-fitted magnitudes (set after training)
        self.register_buffer("soc_calibration", torch.ones(max(num_states + 1, 1), max(num_triplets, 1)))
        self.register_buffer("osce_t_calibration", torch.ones(max(num_triplets, 1)))

        # Auxiliary branch: solvation (PCM susceptibilities d_n, ground-state
        # polarization gp) and the T_n -> S0 phosphorescence strength.  It has
        # its own atom trunk and receives the manifold rotations *detached*, so
        # its losses cannot move any parameter that energies, dipoles or SOCs
        # depend on: the main outputs are identical with or without it.
        self.solvation = solvation
        self.phosphorescence = phosphorescence and num_triplets > 0
        if self.solvation or self.phosphorescence:
            self.aux_mlp = _mlp([scalar_dim, hidden_dim, hidden_dim], dropout)
        if self.solvation:
            # PCM susceptibilities are solvent stabilisations and never negative
            # (NEMO's rate widths take sqrt(2 lambda kT + kT^2) of them), so the
            # diabatic matrix is positive semi-definite, D = B B^T, and every
            # rotated d_i = (U^T D U)_ii >= 0 follows
            self.d_s_mlp = _mlp([2 * hidden_dim, hidden_dim, num_states * num_states], dropout)
            _small_output(self.d_s_mlp)
            self.register_buffer("d_s_mean", torch.ones(num_states))
            self.register_buffer("d_s_scale", torch.ones(num_states))
            if num_triplets > 0:
                self.d_t_mlp = _mlp([2 * hidden_dim, hidden_dim, num_triplets * num_triplets], dropout)
                _small_output(self.d_t_mlp)
                self.register_buffer("d_t_mean", torch.ones(num_triplets))
                self.register_buffer("d_t_scale", torch.ones(num_triplets))
            self.gp_mlp = _mlp([2 * hidden_dim, hidden_dim, 1], dropout)
            _small_output(self.gp_mlp)
            self.register_buffer("gp_mean", torch.zeros(()))
            self.register_buffer("gp_scale", torch.ones(()))
        if self.phosphorescence:
            self.osce_t_mlp = _mlp([2 * hidden_dim, hidden_dim, num_triplets], dropout)
            self.register_buffer("osce_t_log_reference", torch.zeros(num_triplets))
            self.register_buffer("osce_t_log_scale", torch.ones(num_triplets))

    # ------------------------------------------------------------ statistics
    def set_feature_statistics(self, scalars: torch.Tensor) -> None:
        self.feature_mean.copy_(scalars.mean(dim=0))
        self.feature_std.copy_(scalars.std(dim=0).clamp_min(1.0e-6))

    @staticmethod
    def _log_statistics(values: torch.Tensor):
        logged = torch.where(torch.isfinite(values), torch.log(values.clamp_min(LOG_FLOOR)), torch.nan)
        mean, std = nanmean(logged), nanstd(logged)
        mean = _fill(mean, float(_fill(torch.nanmean(mean), 0.0)))
        std = _fill(std, float(_fill(torch.nanmean(std), 1.0))).clamp_min(1.0e-3)
        return mean, std

    def set_calibration(self, prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> None:
        """Rescale log-fitted magnitudes so their training-set arithmetic means match.

        A log fit estimates the geometric mean; a rate or a spectrum integrates the
        arithmetic one.  Geometries whose log residual is more than 3 robust sigma
        (1.4826 MAD) from the median are left out so a few resonant outliers cannot
        set the factor.  ``prediction`` must come from this head with unit calibration.
        """

        def factor(pred, ref):
            valid = torch.isfinite(ref) & (ref > 0)
            residual = torch.where(valid, torch.log(pred.clamp_min(LOG_FLOOR)) - torch.log(ref.clamp_min(LOG_FLOOR)), torch.nan)
            median = torch.nanmedian(residual, dim=0).values
            mad = torch.nanmedian((residual - median).abs(), dim=0).values * 1.4826
            keep = valid & ((residual - median).abs() <= 3.0 * mad.clamp_min(1.0e-6))
            ratio = torch.where(keep, ref, 0.0).sum(0) / torch.where(keep, pred, 0.0).sum(0).clamp_min(LOG_FLOOR)
            return torch.where(keep.any(0), ratio, torch.ones_like(ratio))

        if self.soc_mode != "none" and "soc" in target:
            self.soc_calibration.copy_(factor(prediction["soc"], target["soc"]))
        if self.phosphorescence and "osce_t" in target:
            self.osce_t_calibration.copy_(factor(prediction["osce_t"], target["osce_t"]))

    def set_target_statistics(self, targets: Dict[str, torch.Tensor]) -> None:
        """Reference levels from the training labels (NaN = missing label)."""
        if self.solvation:
            self.d_s_mean.copy_(_fill(nanmean(targets["d_s"]), 1.0).clamp_min(1.0e-6))
            self.d_s_scale.copy_(_fill(nanstd(targets["d_s"]), 1.0).clamp_min(1.0e-6))
            if self.num_triplets > 0:
                self.d_t_mean.copy_(_fill(nanmean(targets["d_t"]), 1.0).clamp_min(1.0e-6))
                self.d_t_scale.copy_(_fill(nanstd(targets["d_t"]), 1.0).clamp_min(1.0e-6))
            self.gp_mean.copy_(_fill(nanmean(targets["gp"]), 0.0))
            self.gp_scale.copy_(_fill(nanstd(targets["gp"]), 1.0).clamp_min(1.0e-6))
        if self.phosphorescence and "osce_t" in targets:
            mean, std = self._log_statistics(targets["osce_t"])
            self.osce_t_log_reference.copy_(mean)
            self.osce_t_log_scale.copy_(std)
        self.singlet.set_statistics(targets["energy_ex"])
        dipole = targets["tdm"]
        self.dipole.set_reference(torch.sqrt(nanmean(dipole.square().sum(-1))))
        if self.num_triplets > 0 and "energy_t" in targets:
            self.triplet.set_statistics(targets["energy_t"])
        if self.soc_mode != "none" and "soc" in targets:
            soc = targets["soc"].reshape(targets["soc"].shape[0], -1)
            if self.soc_mode == "diabatic":
                self.soc.set_log_statistics(soc)
            else:
                mean, std = self._log_statistics(soc)
                self.soc_log_reference.copy_(mean)
                self.soc_log_scale.copy_(std)

    # --------------------------------------------------------------- forward
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
        centroid = scatter_mean(positions, batch, dim=0, dim_size=num_graphs)
        relative = positions - centroid[batch]
        mixed = torch.einsum("nca,cm->nma", vectors, self.vector_mix.weight.t())
        args = (hidden, pooled, mixed, relative, batch, num_graphs)

        energies, rotation, mixing = self.singlet(pooled)
        diabatic = self.dipole(*args)
        output = {
            "tdm": torch.einsum("gki,gka->gia", rotation, diabatic),
            "tdm_diabatic": diabatic,
            "rotation": rotation,
            "mixing": mixing,
            "energy_ex": energies if energy_ex is None else energy_ex,
        }
        rotation_t = None
        if self.num_triplets > 0:
            energies_t, rotation_t, mixing_t = self.triplet(pooled)
            output.update({"energy_t": energies_t, "rotation_t": rotation_t, "mixing_t": mixing_t})
        output.update(self._auxiliary(scalars, batch, num_graphs, rotation, rotation_t))
        if self.num_triplets == 0:
            return output

        rows, cols = self.num_states + 1, self.num_triplets
        if self.soc_mode == "diabatic":
            # S0 is not rotated; S_1..S_nS rotate with U_S, triplets with U_T
            block = torch.zeros(
                num_graphs, rows, rows, device=scalars.device, dtype=scalars.dtype
            )
            block[:, 0, 0] = 1.0
            block[:, 1:, 1:] = rotation
            coupling = self.soc(*args).view(num_graphs, rows, cols, 3)
            coupling = torch.einsum("gkm,gkla,gln->gmna", block, coupling, rotation_t)
            coupling = coupling * self.soc_calibration.unsqueeze(-1)
            output["soc_vectors"] = coupling
            output["soc"] = coupling.norm(dim=-1)
        elif self.soc_mode == "direct":
            logged = self.soc_log_reference + self.soc_log_scale * self.soc(pooled)
            output["soc"] = torch.exp(logged).view(num_graphs, rows, cols) * self.soc_calibration
        return output

    def _auxiliary(self, scalars, batch, num_graphs, rotation, rotation_t) -> Dict[str, torch.Tensor]:
        """Solvation and phosphorescence outputs from the detached auxiliary branch."""
        if not (self.solvation or self.phosphorescence):
            return {}
        hidden = self.aux_mlp((scalars - self.feature_mean) / self.feature_std)
        mean = scatter_mean(hidden, batch, dim=0, dim_size=num_graphs)
        mean_square = scatter_mean(hidden.square(), batch, dim=0, dim_size=num_graphs)
        pooled = torch.cat([mean, (mean_square - mean.square()).clamp_min(1.0e-12).sqrt()], -1)
        out: Dict[str, torch.Tensor] = {}
        if self.solvation:
            # d_n is a state property exchanged at crossings like mu: a smooth
            # diabatic symmetric matrix rotated with the (detached) eigenvectors,
            # d_i = (U^T D U)_ii; trace preserving, so no susceptibility is created
            def rotated(mlp, states, mean_, scale_, rot):
                # B = sqrt(mean) (I + A), A from the network; D = B B^T starts on
                # the training means and stays positive semi-definite
                factor = mlp(pooled).view(num_graphs, states, states)
                factor = torch.eye(states, device=pooled.device, dtype=pooled.dtype) + factor
                factor = mean_.sqrt().unsqueeze(-1) * factor
                matrix = factor @ factor.transpose(-1, -2)
                rot = rot.detach()
                return torch.einsum("gki,gkl,gli->gi", rot, matrix, rot)

            out["susceptibility_s"] = rotated(self.d_s_mlp, self.num_states, self.d_s_mean, self.d_s_scale, rotation)
            if self.num_triplets > 0:
                out["susceptibility_t"] = rotated(self.d_t_mlp, self.num_triplets, self.d_t_mean, self.d_t_scale, rotation_t)
            out["ground_polarization"] = self.gp_mlp(pooled).squeeze(-1) * self.gp_scale + self.gp_mean
        if self.phosphorescence:
            logged = self.osce_t_log_reference + self.osce_t_log_scale * self.osce_t_mlp(pooled)
            out["osce_t"] = torch.exp(logged) * self.osce_t_calibration
        return out


class TransitionDipoleMACE(torch.nn.Module):
    """A MACE backbone with a :class:`TransitionDipoleHead` on top."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        num_states: int,
        num_triplets: int = 0,
        energy_reference: Optional[torch.Tensor] = None,
        basis: str = "diabatic",
        hidden_dim: int = 64,
        vector_dim: int = 32,
        dropout: float = 0.0,
        coupling_scale: float = 1.0,
        eigh_eps: float = 1.0e-3,
        factorised: bool = True,
        soc_mode: str = "diabatic",
        solvation: bool = False,
        phosphorescence: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_states = num_states
        self.num_triplets = num_triplets
        layer_irreps = [product.linear.irreps_out for product in backbone.products]
        self.split = NodeFeatureSplit(layer_irreps)
        self.head = TransitionDipoleHead(
            scalar_dim=self.split.scalar_dim,
            vector_channels=self.split.vector_channels,
            num_states=num_states,
            num_triplets=num_triplets,
            basis=basis,
            hidden_dim=hidden_dim,
            vector_dim=vector_dim,
            dropout=dropout,
            coupling_scale=coupling_scale,
            eigh_eps=eigh_eps,
            factorised=factorised,
            soc_mode=soc_mode,
            solvation=solvation,
            phosphorescence=phosphorescence,
        )
        if energy_reference is not None:
            self.head.singlet.energy_reference.copy_(energy_reference)

    @property
    def basis(self) -> str:
        return self.head.basis

    def set_backbone_trainable(self, mode: str, layers: int = 1) -> None:
        """``head``: frozen backbone; ``last``: the top ``layers`` interaction and
        product blocks train; ``full``: everything trains."""
        if mode not in {"head", "last", "full"}:
            raise ValueError("mode must be 'head', 'last' or 'full'")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = mode == "full"
        if mode == "last":
            layers = max(1, min(layers, len(self.backbone.interactions)))
            for block in list(self.backbone.interactions[-layers:]) + list(self.backbone.products[-layers:]):
                for parameter in block.parameters():
                    parameter.requires_grad = True

    def set_calibration(self, prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> None:
        self.head.set_calibration(prediction, target)

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

    def set_target_statistics(self, targets: Dict[str, torch.Tensor]) -> None:
        self.head.set_target_statistics(targets)

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


# ------------------------------------------------------------------ loss
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


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over the entries whose label exists; zero when none does."""
    weight = valid.to(values.dtype)
    return (torch.nan_to_num(values) * weight).sum() / weight.sum().clamp_min(1.0)


def target_statistics(targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Scales the loss standardises with; every entry NaN-aware."""
    dipole = targets["tdm"]
    statistics = {
        "energy_reference": _fill(nanmean(targets["energy_ex"]), 0.0),
        "energy_std": _fill(nanstd(targets["energy_ex"]), 1.0),
        # RMS rather than std: the sign is arbitrary so the mean is meaningless
        "dipole_std": _fill(torch.sqrt(nanmean(dipole.square().sum(-1))), 1.0),
        "tensor_std": _fill(nanstd(intensity_tensor(dipole).flatten(2), dim=(0, 2)), 1.0),
        "strength_std": _fill(nanstd(targets["osce"]), 1.0),
    }
    if "energy_t" in targets:
        statistics["energy_t_std"] = _fill(nanstd(targets["energy_t"]), 1.0)
    def log_std(values):
        logged = torch.where(torch.isfinite(values), torch.log(values.clamp_min(LOG_FLOOR)), torch.nan)
        std = nanstd(logged)
        return _fill(std, float(_fill(torch.nanmean(std), 1.0)))

    if "soc" in targets:
        statistics["soc_log_std"] = log_std(targets["soc"])
    if "soc_vec" in targets:
        rms = torch.sqrt(nanmean(targets["soc_vec"].square().sum(-1)))
        statistics["soc_vec_rms"] = _fill(rms, float(_fill(torch.nanmean(rms), 1.0)))
    if "osce_t" in targets:
        statistics["osce_t_log_std"] = log_std(targets["osce_t"])
    for key in ("d_s", "d_t", "gp"):
        if key in targets:
            statistics[f"{key}_std"] = _fill(nanstd(targets[key]), 1.0)
    return statistics


class TransitionDipoleLoss(torch.nn.Module):
    """Standardised, NaN-masked loss on every predicted quantity.

    ``dipole_weight`` drives the sign-matched quadratic term, which is what
    actually trains the dipoles.  ``tensor_weight`` adds the quartic
    ``mu mu^T`` term; it is a sharper match to how ``f`` uses the dipole but is
    useless on its own, so it defaults to a supporting role.  Spin-orbit
    couplings span decades and are compared in log space.  A label that is
    NaN for a configuration simply drops out of the mean, so one loss serves
    databases with different label sets.
    """

    def __init__(
        self,
        statistics: Dict[str, torch.Tensor],
        dipole_weight: float = 1.0,
        tensor_weight: float = 0.3,
        energy_weight: float = 1.0,
        strength_weight: float = 0.3,
        total_weight: float = 0.0,
        triplet_energy_weight: float = 1.0,
        soc_weight: float = 1.0,
        soc_vector_weight: float = 0.0,
        phosphorescence_weight: float = 0.0,
        solvation_weight: float = 0.0,
    ):
        super().__init__()
        for key in ("dipole_std", "tensor_std", "energy_std", "strength_std"):
            self.register_buffer(key, statistics[key].clamp_min(1.0e-8))
        self.register_buffer(
            "energy_t_std", statistics.get("energy_t_std", torch.ones(1)).clamp_min(1.0e-8)
        )
        self.register_buffer(
            "soc_log_std", statistics.get("soc_log_std", torch.ones(1)).clamp_min(1.0e-3)
        )
        self.register_buffer("soc_vec_rms", statistics.get("soc_vec_rms", torch.ones(1)).clamp_min(LOG_FLOOR))
        self.register_buffer("osce_t_log_std", statistics.get("osce_t_log_std", torch.ones(1)).clamp_min(1.0e-3))
        for key in ("d_s", "d_t", "gp"):
            self.register_buffer(f"{key}_std", statistics.get(f"{key}_std", torch.ones(1)).clamp_min(1.0e-8))
        self.dipole_weight = dipole_weight
        self.tensor_weight = tensor_weight
        self.energy_weight = energy_weight
        self.strength_weight = strength_weight
        self.total_weight = total_weight
        self.triplet_energy_weight = triplet_energy_weight
        self.soc_weight = soc_weight
        self.soc_vector_weight = soc_vector_weight
        self.phosphorescence_weight = phosphorescence_weight
        self.solvation_weight = solvation_weight

    def terms(
        self, prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        reference = target["tdm"]
        valid = torch.isfinite(reference).all(-1)
        error = phase_matched_error(prediction["tdm"], torch.nan_to_num(reference))
        out["dipole"] = masked_mean(
            (error / self.dipole_std.view(1, -1, 1)).square().mean(-1), valid
        )
        if self.tensor_weight > 0.0:
            tensor_error = (
                intensity_tensor(prediction["tdm"])
                - intensity_tensor(torch.nan_to_num(reference))
            ) / self.tensor_std.view(1, -1, 1, 1)
            out["tensor"] = masked_mean(tensor_error.square().mean((-1, -2)), valid)
        if self.energy_weight > 0.0:
            reference = target["energy_ex"]
            error = (prediction["energy_ex"] - torch.nan_to_num(reference)) / self.energy_std
            out["energy"] = masked_mean(error.square(), torch.isfinite(reference))
        if self.strength_weight > 0.0 or self.total_weight > 0.0:
            strength = oscillator_strength(prediction["tdm"], prediction["energy_ex"])
            reference = target["osce"]
            valid = torch.isfinite(reference)
            if self.strength_weight > 0.0:
                error = (strength - torch.nan_to_num(reference)) / self.strength_std
                out["strength"] = masked_mean(error.square(), valid)
            if self.total_weight > 0.0:
                # Off by default: measured, and it hurts.  The summed intensity
                # is a function of predictions the per-state terms already
                # constrain, so it adds no information and only reweights the
                # objective towards a coarser target.
                error = (strength.sum(-1) - torch.nan_to_num(reference).sum(-1)) / (
                    self.strength_std.sum()
                )
                out["total"] = masked_mean(error.square(), valid.all(-1))
        if self.triplet_energy_weight > 0.0 and "energy_t" in prediction and "energy_t" in target:
            reference = target["energy_t"]
            error = (prediction["energy_t"] - torch.nan_to_num(reference)) / self.energy_t_std
            out["energy_t"] = masked_mean(error.square(), torch.isfinite(reference))
        if self.soc_weight > 0.0 and "soc" in prediction and "soc" in target:
            reference = target["soc"]
            valid = torch.isfinite(reference) & (reference > 0)
            error = (
                torch.log(prediction["soc"].clamp_min(LOG_FLOOR))
                - torch.log(torch.nan_to_num(reference, nan=1.0).clamp_min(LOG_FLOOR))
            ) / self.soc_log_std.view(1, *reference.shape[1:])
            out["soc"] = masked_mean(error.square(), valid)
        if self.soc_vector_weight > 0.0 and "soc_vectors" in prediction and "soc_vec" in target:
            # the coupling vector carries an arbitrary sign per pair: sign-matched
            # quadratic error, scaled by the training RMS of the pair
            reference = target["soc_vec"]
            valid = torch.isfinite(reference).all(-1)
            error = phase_matched_error(prediction["soc_vectors"], torch.nan_to_num(reference))
            scale = self.soc_vec_rms.view(1, *reference.shape[1:-1], 1)
            out["soc_vec"] = masked_mean((error / scale).square().mean(-1), valid)
        if self.phosphorescence_weight > 0.0 and "osce_t" in prediction and "osce_t" in target:
            reference = target["osce_t"]
            valid = torch.isfinite(reference) & (reference > 0)
            error = (
                torch.log(prediction["osce_t"].clamp_min(LOG_FLOOR))
                - torch.log(torch.nan_to_num(reference, nan=1.0).clamp_min(LOG_FLOOR))
            ) / self.osce_t_log_std
            out["osce_t"] = masked_mean(error.square(), valid)
        if self.solvation_weight > 0.0:
            for key, name in (("susceptibility_s", "d_s"), ("susceptibility_t", "d_t"), ("ground_polarization", "gp")):
                if key in prediction and name in target:
                    reference = target[name]
                    error = (prediction[key] - torch.nan_to_num(reference)) / getattr(self, f"{name}_std")
                    out[name] = masked_mean(error.square(), torch.isfinite(reference))
        return out

    AUXILIARY = ("osce_t", "d_s", "d_t", "gp")

    def weights(self) -> Dict[str, float]:
        return {
            "dipole": self.dipole_weight,
            "tensor": self.tensor_weight,
            "energy": self.energy_weight,
            "strength": self.strength_weight,
            "total": self.total_weight,
            "energy_t": self.triplet_energy_weight,
            "soc": self.soc_weight,
            "soc_vec": self.soc_vector_weight,
            "osce_t": self.phosphorescence_weight,
            "d_s": self.solvation_weight,
            "d_t": self.solvation_weight,
            "gp": self.solvation_weight,
        }

    def forward(
        self,
        prediction: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        main_only: bool = False,
    ) -> torch.Tensor:
        """Weighted sum of the terms; ``main_only`` drops the auxiliary-branch
        terms so that scheduling and model selection see the same objective
        with or without the branch."""
        weights = self.weights()
        terms = self.terms(prediction, target)
        total = sum(
            weights[key] * value
            for key, value in terms.items()
            if not (main_only and key in self.AUXILIARY)
        )
        return total if isinstance(total, torch.Tensor) else prediction["tdm"].new_zeros(())

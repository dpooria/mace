"""Symmetry and consistency tests for the transition-dipole head."""

import numpy as np
import pytest
import torch
from ase import Atoms
from scipy.spatial.transform import Rotation

from mace import data
from mace.modules.transition_dipole import (
    HARTREE_PER_EV,
    TransitionDipoleMACE,
    oscillator_strength,
    safe_eigh,
)
from mace.oscillator_strength import load_foundation_model
from mace.tools import AtomicNumberTable, torch_geometric

NUM_STATES = 3


@pytest.fixture(scope="module")
def backbone():
    torch.set_default_dtype(torch.float64)
    return load_foundation_model("medium_off", "cpu", torch.float64)


@pytest.fixture(scope="module")
def model(backbone):
    torch.manual_seed(0)
    built = TransitionDipoleMACE(
        backbone=backbone,
        num_states=NUM_STATES,
        energy_reference=torch.tensor([3.6, 3.7, 3.9], dtype=torch.float64),
    ).double()
    built.eval()
    return built


def make_atoms(seed: int = 0) -> Atoms:
    generator = np.random.default_rng(seed)
    numbers = [6, 6, 7, 8, 1, 1, 1, 1]
    positions = generator.normal(scale=1.6, size=(len(numbers), 3))
    return Atoms(numbers=numbers, positions=positions)


def to_batch(atoms_list, backbone):
    z_table = AtomicNumberTable([int(z) for z in backbone.atomic_numbers])
    graphs = [
        data.AtomicData.from_config(
            data.Configuration(
                atomic_numbers=atoms.numbers,
                positions=atoms.positions,
                properties={},
                property_weights={},
                cell=atoms.cell.array,
                pbc=tuple(atoms.pbc),
            ),
            z_table=z_table,
            cutoff=float(backbone.r_max),
        )
        for atoms in atoms_list
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=graphs, batch_size=len(graphs), shuffle=False
    )
    return next(iter(loader))


def run(model, backbone, atoms_list):
    with torch.no_grad():
        return model(to_batch(atoms_list, backbone).to_dict())


def test_rotation_equivariance(model, backbone):
    atoms = make_atoms()
    rotation = Rotation.random(random_state=3).as_matrix()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T

    reference = run(model, backbone, [atoms])
    turned = run(model, backbone, [rotated])

    expected = reference["tdm"].numpy() @ rotation.T
    assert np.allclose(turned["tdm"].numpy(), expected, atol=1e-8)
    # the mixing matrix is a scalar object and must not move
    assert np.allclose(
        turned["mixing"].numpy(), reference["mixing"].numpy(), atol=1e-8
    )


def test_translation_invariance(model, backbone):
    atoms = make_atoms()
    shifted = atoms.copy()
    shifted.positions = atoms.positions + np.array([3.1, -2.4, 5.0])

    reference = run(model, backbone, [atoms])
    moved = run(model, backbone, [shifted])
    assert np.allclose(moved["tdm"].numpy(), reference["tdm"].numpy(), atol=1e-8)


def test_permutation_invariance(model, backbone):
    atoms = make_atoms()
    order = np.array([3, 0, 2, 1, 6, 4, 7, 5])
    permuted = Atoms(numbers=atoms.numbers[order], positions=atoms.positions[order])

    reference = run(model, backbone, [atoms])
    swapped = run(model, backbone, [permuted])
    assert np.allclose(swapped["tdm"].numpy(), reference["tdm"].numpy(), atol=1e-8)


def test_reflection_is_odd(model, backbone):
    """A transition dipole is a polar vector: it flips under inversion."""
    atoms = make_atoms()
    mirrored = atoms.copy()
    mirrored.positions = -atoms.positions

    reference = run(model, backbone, [atoms])
    flipped = run(model, backbone, [mirrored])
    assert np.allclose(flipped["tdm"].numpy(), -reference["tdm"].numpy(), atol=1e-8)


def test_batching_matches_single(model, backbone):
    first, second = make_atoms(0), make_atoms(1)
    together = run(model, backbone, [first, second])
    apart = [run(model, backbone, [atoms]) for atoms in (first, second)]
    for index in range(2):
        assert np.allclose(
            together["tdm"][index].numpy(), apart[index]["tdm"][0].numpy(), atol=1e-8
        )


def test_adiabatic_basis_has_no_mixing(backbone):
    torch.manual_seed(0)
    model = TransitionDipoleMACE(
        backbone=backbone,
        num_states=NUM_STATES,
        energy_reference=torch.tensor([3.6, 3.7, 3.9], dtype=torch.float64),
        basis="adiabatic",
    ).double()
    model.eval()
    output = run(model, backbone, [make_atoms()])
    assert np.allclose(output["rotation"].numpy(), np.eye(NUM_STATES), atol=1e-12)
    assert np.allclose(
        output["tdm"].numpy(), output["tdm_diabatic"].numpy(), atol=1e-12
    )


def test_oscillator_strength_matches_definition():
    dipole = torch.tensor([[[-1.4374, 0.0061, 0.2550]]], dtype=torch.float64)
    energy = torch.tensor([[3.5808]], dtype=torch.float64)
    expected = (2.0 / 3.0) * (3.5808 * HARTREE_PER_EV) * float((dipole**2).sum())
    assert oscillator_strength(dipole, energy).item() == pytest.approx(
        expected, rel=1e-12
    )
    # and against the value QChem printed for this transition
    assert oscillator_strength(dipole, energy).item() == pytest.approx(
        0.1869656307, abs=2e-5
    )


def test_safe_eigh_gradient_is_finite_at_degeneracy():
    """The whole point of the clamp: a degenerate matrix must still train."""
    matrix = torch.eye(3, dtype=torch.float64).unsqueeze(0).requires_grad_(True)
    values, vectors = safe_eigh(matrix, 1.0e-3)
    (values.sum() + vectors.sum()).backward()
    assert torch.isfinite(matrix.grad).all()

    plain = torch.eye(3, dtype=torch.float64).unsqueeze(0).requires_grad_(True)
    values, vectors = torch.linalg.eigh(plain)
    (values.sum() + vectors.sum()).backward()
    assert not torch.isfinite(plain.grad).all()


def test_safe_eigh_matches_autograd_away_from_degeneracy():
    torch.manual_seed(0)
    base = torch.randn(2, 3, 3, dtype=torch.float64)
    base = 0.5 * (base + base.transpose(-1, -2)) + torch.diag_embed(
        torch.tensor([[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]], dtype=torch.float64)
    )
    weights = torch.randn(2, 3, 3, dtype=torch.float64)

    safe_input = base.clone().requires_grad_(True)
    values, vectors = safe_eigh(safe_input, 1.0e-6)
    (values.sum() + (vectors * weights).sum()).backward()

    plain_input = base.clone().requires_grad_(True)
    values, vectors = torch.linalg.eigh(plain_input)
    (values.sum() + (vectors * weights).sum()).backward()

    assert torch.allclose(safe_input.grad, plain_input.grad, atol=1e-10)


def test_factorised_magnitude_is_rotationally_invariant(backbone):
    """|mu| must come from the invariant head, not from the vector components.

    This is what stops direction error leaking into the intensity: |mu|^2 is
    the quantity f uses, and in the factorised head it is produced by a purely
    invariant readout.
    """
    torch.manual_seed(0)
    model = TransitionDipoleMACE(
        backbone=backbone,
        num_states=NUM_STATES,
        energy_reference=torch.tensor([3.6, 3.7, 3.9], dtype=torch.float64),
        basis="adiabatic",
        factorised=True,
    ).double()
    model.eval()

    atoms = make_atoms()
    rotation = Rotation.random(random_state=11).as_matrix()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T

    reference = run(model, backbone, [atoms])["tdm"].numpy()
    turned = run(model, backbone, [rotated])["tdm"].numpy()
    assert np.allclose(
        (reference**2).sum(-1), (turned**2).sum(-1), atol=1e-10
    )


def test_factorisation_changes_the_dipole(backbone):
    """Guard against the flag silently doing nothing."""
    outputs = {}
    for factorised in (True, False):
        torch.manual_seed(0)
        model = TransitionDipoleMACE(
            backbone=backbone,
            num_states=NUM_STATES,
            energy_reference=torch.tensor([3.6, 3.7, 3.9], dtype=torch.float64),
            basis="adiabatic",
            factorised=factorised,
        ).double()
        model.eval()
        outputs[factorised] = run(model, backbone, [make_atoms()])["tdm"].numpy()
    assert not np.allclose(outputs[True], outputs[False])
    # the direction is shared; only the length is reassigned
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    assert np.allclose(unit(outputs[True]), unit(outputs[False]), atol=1e-10)

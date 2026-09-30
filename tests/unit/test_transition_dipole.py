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


# ------------------------------------------------------- triplets and SOC
NUM_TRIPLETS = 2


def _labelled_targets(num_configs: int = 6, seed: int = 5):
    """Synthetic training labels, with the S0 row of the SOC block missing."""
    generator = torch.Generator().manual_seed(seed)
    targets = {
        "energy_ex": 3.6 + 0.1 * torch.randn(num_configs, NUM_STATES, generator=generator, dtype=torch.float64),
        "tdm": torch.randn(num_configs, NUM_STATES, 3, generator=generator, dtype=torch.float64),
        "osce": 0.1 * torch.rand(num_configs, NUM_STATES, generator=generator, dtype=torch.float64),
        "energy_t": 2.8 + 0.1 * torch.randn(num_configs, NUM_TRIPLETS, generator=generator, dtype=torch.float64),
        "soc": 1.0e-4 * torch.exp(torch.randn(num_configs, NUM_STATES + 1, NUM_TRIPLETS, generator=generator, dtype=torch.float64)),
    }
    targets["soc"][:, 0, :] = float("nan")
    return targets


@pytest.fixture(scope="module")
def soc_model(backbone):
    torch.manual_seed(2)
    built = TransitionDipoleMACE(
        backbone=backbone, num_states=NUM_STATES, num_triplets=NUM_TRIPLETS
    ).double()
    built.set_target_statistics(_labelled_targets())
    with torch.no_grad():  # move the axial directions off their tiny initial values
        built.head.soc.weights.weight.normal_(std=0.5)
        built.head.soc.magnitude[-1].weight.normal_(std=0.5)
    built.eval()
    return built


def test_triplet_manifold_shapes_and_ordering(soc_model, backbone):
    output = run(soc_model, backbone, [make_atoms(), make_atoms(3)])
    assert output["energy_t"].shape == (2, NUM_TRIPLETS)
    assert output["soc"].shape == (2, NUM_STATES + 1, NUM_TRIPLETS)
    assert output["soc_vectors"].shape == (2, NUM_STATES + 1, NUM_TRIPLETS, 3)
    # eigenvalues come out ascending: the adiabatic ordering
    assert bool((output["energy_t"].diff(dim=-1) >= 0).all())
    assert bool((output["energy_ex"].diff(dim=-1) >= 0).all())
    assert bool((output["soc"] > 0).all())


def test_soc_vectors_are_axial(soc_model, backbone):
    """|SOC| is invariant; the coupling vector rotates but does NOT flip under inversion."""
    atoms = make_atoms()
    rotation = Rotation.random(random_state=8).as_matrix()
    rotated, mirrored = atoms.copy(), atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    mirrored.positions = -atoms.positions

    reference = run(soc_model, backbone, [atoms])
    turned = run(soc_model, backbone, [rotated])
    flipped = run(soc_model, backbone, [mirrored])
    assert np.allclose(turned["soc"].numpy(), reference["soc"].numpy(), atol=1e-8)
    assert np.allclose(
        turned["soc_vectors"].numpy(), reference["soc_vectors"].numpy() @ rotation.T, atol=1e-8
    )
    assert np.allclose(flipped["soc_vectors"].numpy(), reference["soc_vectors"].numpy(), atol=1e-8)
    assert np.allclose(flipped["tdm"].numpy(), -reference["tdm"].numpy(), atol=1e-8)
    assert np.allclose(turned["energy_t"].numpy(), reference["energy_t"].numpy(), atol=1e-8)


def test_soc_rotation_conserves_frobenius_norm(soc_model, backbone):
    """U_S (x) U_T is orthogonal, so the total coupling is a diabatic invariant."""
    output = run(soc_model, backbone, [make_atoms(4)])
    adiabatic = output["soc_vectors"][0]
    # undo the rotation: diabatic block from the adiabatic one
    block = torch.eye(NUM_STATES + 1, dtype=torch.float64)
    block[1:, 1:] = output["rotation"][0]
    diabatic = torch.einsum("km,mna,ln->kla", block, adiabatic, output["rotation_t"][0])
    assert torch.allclose(diabatic.square().sum(), adiabatic.square().sum())
    # S0 is not rotated: its row keeps its total intensity per adiabatic triplet
    assert torch.allclose(
        adiabatic[0].square().sum(), diabatic[0].square().sum()
    )


def test_direct_soc_mode_is_invariant_and_positive(backbone):
    torch.manual_seed(3)
    model = TransitionDipoleMACE(
        backbone=backbone, num_states=NUM_STATES, num_triplets=NUM_TRIPLETS, soc_mode="direct"
    ).double()
    model.set_target_statistics(_labelled_targets())
    with torch.no_grad():
        model.head.soc[-1].weight.normal_(std=0.5)
    model.eval()
    atoms = make_atoms()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ Rotation.random(random_state=9).as_matrix().T
    reference, turned = run(model, backbone, [atoms]), run(model, backbone, [rotated])
    assert "soc_vectors" not in reference
    assert np.allclose(turned["soc"].numpy(), reference["soc"].numpy(), atol=1e-10)
    assert bool((reference["soc"] > 0).all())


def test_statistics_fill_unlabelled_pairs(soc_model):
    """The S0 row had no labels: it takes the statistics of the labelled pairs."""
    head = soc_model.head
    log_reference = head.soc.log_reference.view(NUM_STATES + 1, NUM_TRIPLETS)
    assert torch.isfinite(log_reference).all()
    assert torch.allclose(log_reference[0], log_reference[1:].mean().expand(NUM_TRIPLETS))
    assert bool((head.triplet.energy_scale > 0).all())
    assert head.singlet.energy_reference.shape == (NUM_STATES,)


def test_loss_masks_missing_labels(soc_model, backbone):
    from mace.modules.transition_dipole import TransitionDipoleLoss, target_statistics

    targets = _labelled_targets()
    statistics = target_statistics(targets)
    assert torch.isfinite(statistics["soc_log_std"]).all()
    loss_fn = TransitionDipoleLoss(statistics)
    output = run(soc_model, backbone, [make_atoms(1), make_atoms(2)])
    prediction = {key: value.clone() for key, value in output.items()}

    # a perfect prediction has zero loss whatever is missing
    target = {
        "energy_ex": output["energy_ex"], "tdm": output["tdm"],
        "osce": oscillator_strength(output["tdm"], output["energy_ex"]),
        "energy_t": output["energy_t"].clone(), "soc": output["soc"].clone(),
    }
    target["soc"][:, 0, :] = float("nan")
    target["energy_t"][1] = float("nan")
    assert float(loss_fn(prediction, target)) < 1e-18

    # an error in an unlabelled entry costs nothing; in a labelled one it does
    prediction["soc"] = output["soc"].clone()
    prediction["soc"][:, 0, :] *= 10.0
    assert float(loss_fn(prediction, target)) < 1e-18
    prediction["soc"][:, 1, 0] *= 10.0
    assert float(loss_fn(prediction, target)) > 0.0

    # all-NaN triplet labels: the terms vanish rather than producing NaN
    target["energy_t"][:] = float("nan")
    target["soc"][:] = float("nan")
    prediction["energy_t"] = output["energy_t"] + 1.0
    assert torch.isfinite(loss_fn(prediction, target))
    assert float(loss_fn(prediction, target)) < 1e-18


def test_mixing_matrix_is_standardised(backbone):
    """Energies sit on the training reference at initialisation: no fixed eV scale."""
    torch.manual_seed(0)
    model = TransitionDipoleMACE(backbone=backbone, num_states=NUM_STATES).double()
    targets = _labelled_targets(num_configs=40)
    targets["energy_ex"] = targets["energy_ex"] + 2.0  # a molecule with higher energies
    model.set_target_statistics(targets)
    model.eval()
    output = run(model, backbone, [make_atoms()])
    reference = targets["energy_ex"].mean(0)
    scale = targets["energy_ex"].std(0).mean()
    assert np.allclose(output["energy_ex"].numpy()[0], reference.numpy(), atol=3 * float(scale))


def test_foundation_model_names():
    from mace.oscillator_strength import FOUNDATION_MODELS, load_foundation_model

    for name in ("medium_off", "mh-1", "omol"):
        assert name in FOUNDATION_MODELS
    with pytest.raises(ValueError):
        load_foundation_model("no-such-model", "cpu", torch.float64)


# ------------------------------------------------- auxiliary branch, vectors
def _aux_targets(num_configs: int = 6, seed: int = 11):
    targets = _labelled_targets(num_configs, seed)
    generator = torch.Generator().manual_seed(seed + 1)
    targets["soc_vec"] = 1.0e-4 * torch.randn(num_configs, NUM_STATES + 1, NUM_TRIPLETS, 3, generator=generator, dtype=torch.float64)
    targets["osce_t"] = 1.0e-8 * torch.exp(torch.randn(num_configs, NUM_TRIPLETS, generator=generator, dtype=torch.float64))
    targets["d_s"] = 0.05 + 0.01 * torch.randn(num_configs, NUM_STATES, generator=generator, dtype=torch.float64)
    targets["d_t"] = 0.01 + 0.003 * torch.randn(num_configs, NUM_TRIPLETS, generator=generator, dtype=torch.float64)
    targets["gp"] = 0.2 + 0.02 * torch.randn(num_configs, generator=generator, dtype=torch.float64)
    return targets


@pytest.fixture(scope="module")
def aux_model(backbone):
    torch.manual_seed(4)
    built = TransitionDipoleMACE(
        backbone=backbone, num_states=NUM_STATES, num_triplets=NUM_TRIPLETS,
        solvation=True, phosphorescence=True,
    ).double()
    built.set_target_statistics(_aux_targets())
    with torch.no_grad():
        for mlp in (built.head.d_s_mlp, built.head.d_t_mlp, built.head.gp_mlp, built.head.osce_t_mlp):
            mlp[-1].weight.normal_(std=0.5)
        built.head.soc.weights.weight.normal_(std=0.5)
    built.eval()
    return built


def test_auxiliary_outputs_are_invariant_and_sane(aux_model, backbone):
    atoms = make_atoms()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ Rotation.random(random_state=12).as_matrix().T
    reference, turned = run(aux_model, backbone, [atoms]), run(aux_model, backbone, [rotated])
    for key in ("susceptibility_s", "susceptibility_t", "ground_polarization", "osce_t"):
        assert np.allclose(turned[key].numpy(), reference[key].numpy(), atol=1e-8), key
    assert reference["susceptibility_s"].shape == (1, NUM_STATES)
    assert reference["susceptibility_t"].shape == (1, NUM_TRIPLETS)
    assert bool((reference["osce_t"] > 0).all())


def test_auxiliary_losses_do_not_reach_the_main_parameters(aux_model, backbone):
    """Solvation and phosphorescence gradients stop at the detached rotations."""
    from mace.modules.transition_dipole import TransitionDipoleLoss, target_statistics

    targets = _aux_targets()
    loss_fn = TransitionDipoleLoss(
        target_statistics(targets), dipole_weight=0.0, tensor_weight=0.0, energy_weight=0.0,
        strength_weight=0.0, triplet_energy_weight=0.0, soc_weight=0.0, soc_vector_weight=0.0,
        phosphorescence_weight=1.0, solvation_weight=1.0,
    )
    aux_model.zero_grad(set_to_none=True)
    batch = to_batch([make_atoms(1), make_atoms(2)], backbone).to_dict()
    output = aux_model(batch)
    target = {key: value[:2] for key, value in targets.items()}
    loss = loss_fn(output, target)
    assert float(loss) > 0.0
    loss.backward()
    aux_names = ("aux_mlp", "d_s_mlp", "d_t_mlp", "gp_mlp", "osce_t_mlp")
    for name, parameter in aux_model.head.named_parameters():
        grad = parameter.grad
        if name.startswith(aux_names):
            continue
        assert grad is None or float(grad.abs().max()) == 0.0, f"{name} received a gradient"
    assert any(p.grad is not None and float(p.grad.abs().max()) > 0 for n, p in aux_model.head.named_parameters() if n.startswith("d_s_mlp"))
    aux_model.zero_grad(set_to_none=True)


def test_soc_vector_loss_is_phase_free_and_zero_at_target(aux_model, backbone):
    from mace.modules.transition_dipole import TransitionDipoleLoss, target_statistics

    targets = _aux_targets()
    loss_fn = TransitionDipoleLoss(
        target_statistics(targets), dipole_weight=0.0, tensor_weight=0.0, energy_weight=0.0,
        strength_weight=0.0, triplet_energy_weight=0.0, soc_weight=0.0, soc_vector_weight=1.0,
    )
    output = run(aux_model, backbone, [make_atoms(3)])
    target = {key: value[:1] for key, value in targets.items()}
    target["soc_vec"] = output["soc_vectors"].clone()
    assert float(loss_fn(output, target)) < 1e-18
    target["soc_vec"][:, 1:] *= -1.0  # arbitrary sign per pair
    assert float(loss_fn(output, target)) < 1e-18
    target["soc_vec"][:, 1, 0] *= 2.0
    assert float(loss_fn(output, target)) > 0.0


def test_calibration_matches_training_means(aux_model, backbone):
    head = aux_model.head
    before_soc, before_t = head.soc_calibration.clone(), head.osce_t_calibration.clone()
    try:
        output = run(aux_model, backbone, [make_atoms(5), make_atoms(6), make_atoms(7)])
        target = {"soc": 2.0 * output["soc"], "osce_t": 0.5 * output["osce_t"]}
        target["soc"][:, 0, :] = float("nan")  # unlabelled pairs keep a factor of one
        head.set_calibration(output, target)
        assert torch.allclose(head.soc_calibration[1:], torch.full_like(head.soc_calibration[1:], 2.0))
        assert torch.allclose(head.soc_calibration[0], torch.ones(NUM_TRIPLETS, dtype=torch.float64))
        assert torch.allclose(head.osce_t_calibration, torch.full_like(head.osce_t_calibration, 0.5))
        calibrated = run(aux_model, backbone, [make_atoms(5)])
        assert torch.allclose(calibrated["soc"][0, 1:], 2.0 * output["soc"][0, 1:])
        assert torch.allclose(calibrated["soc_vectors"].norm(dim=-1), calibrated["soc"])
    finally:
        head.soc_calibration.copy_(before_soc)
        head.osce_t_calibration.copy_(before_t)


def test_fine_tune_layers(backbone):
    model = TransitionDipoleMACE(backbone=backbone, num_states=NUM_STATES).double()
    model.set_backbone_trainable("last", layers=1)
    trainable = {n for n, p in model.backbone.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith(("interactions.1", "products.1")) for n in trainable)
    model.set_backbone_trainable("last", layers=99)
    assert all(p.requires_grad for p in model.backbone.interactions.parameters())
    model.set_backbone_trainable("head")
    assert not any(p.requires_grad for p in model.backbone.parameters())


def test_susceptibilities_are_never_negative(aux_model, backbone):
    """D = B B^T: every rotated susceptibility is >= 0 even with large network outputs."""
    head = aux_model.head
    params = list(head.d_s_mlp.parameters()) + list(head.d_t_mlp.parameters())
    saved = [p.detach().clone() for p in params]
    try:
        with torch.no_grad():
            for mlp in (head.d_s_mlp, head.d_t_mlp):
                mlp[-1].weight.normal_(std=5.0)
                mlp[-1].bias.normal_(std=5.0)
        output = run(aux_model, backbone, [make_atoms(k) for k in range(8)])
        assert bool((output["susceptibility_s"] >= 0).all())
        assert bool((output["susceptibility_t"] >= 0).all())
    finally:
        with torch.no_grad():
            for p, value in zip(params, saved):
                p.copy_(value)

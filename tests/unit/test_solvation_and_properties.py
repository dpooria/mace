"""Tests for the solvation outputs of the TDM head and the state-property model."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mace.modules.state_properties import (
    StatePropertyLoss,
    StatePropertyMACE,
    parse_target_specs,
)
from mace.modules.transition_dipole import TransitionDipoleLoss, TransitionDipoleMACE
from tests.unit.test_transition_dipole import backbone, make_atoms, run  # noqa: F401

NUM_STATES = 3


@pytest.fixture(scope="module")
def solvation_model(backbone):  # noqa: F811
    torch.manual_seed(1)
    built = TransitionDipoleMACE(
        backbone=backbone,
        num_states=NUM_STATES,
        energy_reference=torch.tensor([3.6, 3.7, 3.9], dtype=torch.float64),
        solvation=True,
    ).double()
    built.set_solvation_statistics(
        torch.tensor([[0.05, 0.02, 0.01], [0.07, 0.03, 0.02]], dtype=torch.float64),
        torch.tensor([0.21, 0.25], dtype=torch.float64),
    )
    # move the outputs away from their initial near-constant values
    with torch.no_grad():
        for mlp in (built.head.susceptibility_mlp, built.head.polarization_mlp):
            mlp[-1].weight.normal_(std=0.5)
    built.eval()
    return built


@pytest.fixture(scope="module")
def property_model(backbone):  # noqa: F811
    torch.manual_seed(2)
    specs = parse_target_specs(["e_t1", "osce_t1:log"])
    built = StatePropertyMACE(backbone, specs, hidden_dim=16).double()
    built.set_target_statistics(
        torch.tensor([[2.7, 1.0e-5], [2.8, 3.0e-5], [2.6, 2.0e-6]], dtype=torch.float64)
    )
    with torch.no_grad():
        built.head.output_mlp[-1].weight.normal_(std=0.5)
    built.eval()
    return built


def test_solvation_outputs_are_rotation_invariant(solvation_model, backbone):  # noqa: F811
    atoms = make_atoms()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ Rotation.random(random_state=4).as_matrix().T
    reference = run(solvation_model, backbone, [atoms])
    turned = run(solvation_model, backbone, [rotated])
    for key in ("susceptibility", "ground_polarization", "susceptibility_diabatic"):
        assert np.allclose(turned[key].numpy(), reference[key].numpy(), atol=1e-8)


def test_susceptibility_follows_the_mixing_rotation(solvation_model, backbone):  # noqa: F811
    output = run(solvation_model, backbone, [make_atoms(5)])
    rotation, matrix = output["rotation"][0], output["susceptibility_diabatic"][0]
    expected = torch.diagonal(rotation.T @ matrix @ rotation)
    assert torch.allclose(output["susceptibility"][0], expected, atol=1e-10)
    # a unitary rotation preserves the trace: no susceptibility is created
    assert torch.allclose(output["susceptibility"][0].sum(), torch.trace(matrix))


def test_solvation_loss_is_zero_at_the_target(solvation_model, backbone):  # noqa: F811
    output = run(solvation_model, backbone, [make_atoms(6)])
    target = {key: value.clone() for key, value in output.items()}
    target["osce"] = torch.zeros(1, NUM_STATES, dtype=torch.float64)
    ones = torch.ones(NUM_STATES, dtype=torch.float64)
    loss = TransitionDipoleLoss(
        dipole_std=ones, tensor_std=ones, energy_std=ones, strength_std=ones,
        tensor_weight=0.0, strength_weight=0.0,
        susceptibility_std=ones, polarization_std=torch.tensor(1.0, dtype=torch.float64),
        susceptibility_weight=1.0, polarization_weight=1.0,
    )
    assert float(loss(output, target)) < 1e-20
    target["ground_polarization"] = target["ground_polarization"] + 0.5
    assert float(loss(output, target)) == pytest.approx(0.25)


def test_property_outputs_are_invariant_and_positive(property_model, backbone):  # noqa: F811
    atoms = make_atoms()
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ Rotation.random(random_state=7).as_matrix().T
    reference = run(property_model, backbone, [atoms, make_atoms(8)])
    turned = run(property_model, backbone, [rotated, make_atoms(8)])
    assert np.allclose(turned["props"].numpy(), reference["props"].numpy(), atol=1e-10)
    assert bool((reference["props"][:, 1] > 0).all())


def test_log_transform_round_trips(property_model):
    head = property_model.head
    values = torch.tensor([[2.7, 4.2e-6], [3.1, 8.0e-5]], dtype=torch.float64)
    assert torch.allclose(head.untransform(head.transform(values)), values)


def test_calibration_matches_ensemble_means_of_log_targets(property_model):
    head = property_model.head
    prediction = torch.tensor([[2.7, 1.0e-5], [2.8, 1.0e-5]], dtype=torch.float64)
    target = torch.tensor([[2.6, 1.0e-5], [2.9, 3.0e-5]], dtype=torch.float64)
    before = head.calibration.clone()
    try:
        head.set_calibration(prediction, target)
        # linear targets untouched, log targets rescaled to the mean ratio
        assert head.calibration.tolist() == pytest.approx([1.0, 2.0])
    finally:
        head.calibration.copy_(before)


def test_property_loss_skips_missing_targets(property_model):
    head = property_model.head
    loss = StatePropertyLoss(head, [1.0, 1.0])
    target = torch.tensor([[2.7, np.nan]], dtype=torch.float64)
    standardised = head.standardise(torch.tensor([[2.7, 1.0e-5]], dtype=torch.float64))
    prediction = {"standardised": standardised}
    assert float(loss(prediction, {"props": target})) < 1e-20


def test_target_specs_reject_bad_input():
    with pytest.raises(ValueError):
        parse_target_specs(["osce_t1:sqrt"])
    with pytest.raises(ValueError):
        parse_target_specs(["e_t1", "e_t1:log"])

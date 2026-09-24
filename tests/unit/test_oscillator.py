import numpy as np
import torch
from ase import Atoms
from e3nn import o3

from mace import data, tools
from mace.modules.oscillator import (
    InvariantFeatures,
    OscillatorStrengthLoss,
    latent_to_target,
    target_to_latent,
    target_statistics,
)


def test_invariant_features_are_rotation_invariant():
    irreps = [o3.Irreps("2x0e + 2x1o + 1x2e")]
    features = torch.randn(4, irreps[0].dim)
    rotation = o3.rand_matrix()
    rotated = features @ irreps[0].D_from_matrix(rotation).T
    extractor = InvariantFeatures(irreps)
    assert torch.allclose(
        extractor(features),
        extractor(rotated),
        atol=1.0e-6,
    )


def test_loss_is_finite_and_has_gradients():
    target = torch.tensor([[0.2, 0.01, 0.001], [0.3, 0.03, 0.004]])
    statistics = target_statistics(target)
    normalized = torch.zeros_like(target, requires_grad=True)
    latent = (
        normalized * statistics["latent_std"] + statistics["latent_mean"]
    )
    prediction = {
        "normalized_latent": normalized,
        "oscillator_strength": latent_to_target(latent),
    }
    loss = OscillatorStrengthLoss(**statistics)(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert normalized.grad is not None


def test_exchange_transform_round_trip():
    target = torch.tensor([[0.2, 0.01, 0.001], [0.01, 0.3, 0.004]])
    recovered = latent_to_target(target_to_latent(target))
    assert torch.allclose(target, recovered)


def test_atomic_graph_is_translation_invariant():
    atoms = Atoms("OH2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    table = tools.AtomicNumberTable([1, 8])

    def graph(positions):
        config = data.Configuration(
            atomic_numbers=atoms.numbers,
            positions=positions,
            properties={},
            property_weights={},
            cell=atoms.cell.array,
            pbc=tuple(atoms.pbc),
        )
        return data.AtomicData.from_config(config, table, 3.0)

    original = graph(atoms.positions)
    translated = graph(atoms.positions + np.array([2.0, -1.0, 4.0]))
    assert torch.equal(original.edge_index, translated.edge_index)
    assert torch.allclose(original.shifts, translated.shifts)

    original.sample_id = torch.tensor([0])
    translated.sample_id = torch.tensor([1])
    loader = tools.torch_geometric.dataloader.DataLoader(
        [original, translated], batch_size=2
    )
    batch = next(iter(loader))
    assert torch.equal(batch.sample_id, torch.tensor([0, 1]))

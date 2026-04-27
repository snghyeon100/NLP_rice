import torch

from mcsu_utils import compute_mcsu_projection_loss


def test_mcsu_loss_high_when_z_lies_in_subspace():
    U = torch.tensor([[1.0], [0.0], [0.0]])
    z = torch.tensor([[3.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    loss = compute_mcsu_projection_loss(z, U)

    assert loss.item() > 0.99


def test_mcsu_loss_near_zero_when_z_is_orthogonal():
    U = torch.tensor([[1.0], [0.0], [0.0]])
    z = torch.tensor([[0.0, 2.0, 0.0], [0.0, -1.0, 4.0]])

    loss = compute_mcsu_projection_loss(z, U)

    assert loss.item() < 1e-6


def test_removing_projection_decreases_mcsu_loss():
    U = torch.tensor([[1.0], [0.0], [0.0]])
    z = torch.tensor([[1.0, 1.0, 0.0], [2.0, -1.0, 0.5]])
    z_without_projection = z - (z @ U) @ U.T

    original_loss = compute_mcsu_projection_loss(z, U)
    removed_loss = compute_mcsu_projection_loss(z_without_projection, U)

    assert removed_loss.item() < original_loss.item()
    assert removed_loss.item() < 1e-6

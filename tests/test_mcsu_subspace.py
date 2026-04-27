import torch

from mcsu_utils import compute_language_subspace, compute_shared_subspace


def _orthogonal_unit_vector(reference, generator):
    vec = torch.randn(reference.shape[0], generator=generator)
    vec = vec - (vec @ reference) * reference
    return vec / vec.norm()


def test_shared_projector_subspace_recovers_synthetic_shared_direction():
    generator = torch.Generator().manual_seed(7)
    hidden_dim = 32
    num_examples = 96
    num_languages = 4

    shared = torch.randn(hidden_dim, generator=generator)
    shared = shared / shared.norm()

    basis_by_language = {}
    for lang_idx in range(num_languages):
        language_direction = _orthogonal_unit_vector(shared, generator)
        z = (
            shared.unsqueeze(0).repeat(num_examples, 1)
            + 0.2 * language_direction.unsqueeze(0).repeat(num_examples, 1)
            + 0.05 * torch.randn(num_examples, hidden_dim, generator=generator)
        )
        basis, _ = compute_language_subspace(
            z,
            k=2,
            row_normalize_z=True,
            mean_center_z=False,
        )
        basis_by_language[f"lang{lang_idx}"] = basis

    U_shared, _ = compute_shared_subspace(basis_by_language, k_shared=1)

    cosine = torch.abs(U_shared[:, 0] @ shared).item()
    assert cosine > 0.85

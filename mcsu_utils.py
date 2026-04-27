import torch


def last_nonpad_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pool the hidden state at the last non-padding token.

    hidden: [batch, seq_len, hidden_dim]
    attention_mask: [batch, seq_len]
    return: [batch, hidden_dim]
    """
    if hidden.dim() != 3:
        raise ValueError(f"hidden must be [batch, seq_len, hidden_dim], got shape {tuple(hidden.shape)}")
    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must be [batch, seq_len], got shape {tuple(attention_mask.shape)}")
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError(
            "hidden and attention_mask batch/sequence dimensions must match: "
            f"{tuple(hidden.shape[:2])} vs {tuple(attention_mask.shape)}"
        )
    if not torch.all(attention_mask.sum(dim=1) > 0):
        raise ValueError("Every sample must contain at least one non-padding token.")

    idx = attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


def default_layer_ids(model_config) -> list:
    n_layers = getattr(model_config, "num_hidden_layers", None)
    if n_layers is None:
        n_layers = getattr(model_config, "n_layer", None)
    if n_layers is None:
        n_layers = getattr(model_config, "num_layers", None)
    if n_layers is None:
        raise ValueError("Could not infer number of transformer layers from model.config.")

    candidates = [n_layers // 3, n_layers // 2, 2 * n_layers // 3, n_layers - 2]
    return sorted({layer for layer in candidates if 0 <= layer < n_layers})


def resolve_layer_ids(layer_ids, model_config) -> list:
    if layer_ids is None:
        return default_layer_ids(model_config)
    if isinstance(layer_ids, str):
        if layer_ids.lower() in {"null", "none", ""}:
            return default_layer_ids(model_config)
        layer_ids = [int(x.strip()) for x in layer_ids.split(",") if x.strip()]
    resolved = [int(layer) for layer in layer_ids]
    if not resolved:
        raise ValueError("layer_ids resolved to an empty list.")
    return sorted(set(resolved))


def normalize_rows(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return z / (z.norm(dim=-1, keepdim=True) + eps)


def compute_language_subspace(
    z: torch.Tensor,
    k: int,
    row_normalize_z: bool = True,
    mean_center_z: bool = False,
    eps: float = 1e-8,
):
    """
    Compute a per-language concept basis with right singular vectors.

    z: [num_examples, hidden_dim]
    returns: B [hidden_dim, min(k, rank)], singular values
    """
    if z.dim() != 2:
        raise ValueError(f"z must be [num_examples, hidden_dim], got shape {tuple(z.shape)}")
    if z.shape[0] == 0:
        raise ValueError("Cannot compute a subspace from zero examples.")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    z = z.float()
    if row_normalize_z:
        z = normalize_rows(z, eps=eps)
    if mean_center_z:
        z = z - z.mean(dim=0, keepdim=True)

    _, singular_values, vh = torch.linalg.svd(z, full_matrices=False)
    rank = min(int(k), vh.shape[0])
    basis = vh[:rank].T.contiguous()
    return basis, singular_values


def compute_shared_subspace(basis_by_language: dict, k_shared: int):
    """
    Average language projectors and return top eigenvectors.

    basis_by_language maps language -> B where B is [hidden_dim, k_lang].
    """
    if not basis_by_language:
        raise ValueError("basis_by_language is empty.")
    if k_shared <= 0:
        raise ValueError(f"k_shared must be positive, got {k_shared}")

    first_basis = next(iter(basis_by_language.values()))
    hidden_dim = first_basis.shape[0]
    accumulator = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32, device=first_basis.device)
    weight = 1.0 / len(basis_by_language)

    for language, basis in basis_by_language.items():
        if basis.shape[0] != hidden_dim:
            raise ValueError(
                f"Basis for language {language} has hidden dim {basis.shape[0]}, expected {hidden_dim}."
            )
        basis = basis.float()
        accumulator = accumulator + weight * (basis @ basis.T)

    eigvals, eigvecs = torch.linalg.eigh(accumulator)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvecs[:, : min(k_shared, eigvecs.shape[1])].contiguous(), eigvals.contiguous()


def pairwise_subspace_overlap(basis_a: torch.Tensor, basis_b: torch.Tensor, normalizer: int = None) -> float:
    if basis_a.shape[0] != basis_b.shape[0]:
        raise ValueError(f"Basis hidden dimensions differ: {basis_a.shape[0]} vs {basis_b.shape[0]}")
    if normalizer is None:
        normalizer = max(1, min(basis_a.shape[1], basis_b.shape[1]))
    overlap = torch.linalg.matrix_norm(basis_a.float().T @ basis_b.float(), ord="fro").pow(2)
    return (overlap / max(1, int(normalizer))).item()


def compute_mcsu_projection_loss(z: torch.Tensor, basis: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute ||U^T z||^2 / (||z||^2 + eps), averaged over a batch.

    z: [batch, hidden_dim]
    basis: [hidden_dim, k]
    """
    if z.dim() != 2:
        raise ValueError(f"z must be [batch, hidden_dim], got shape {tuple(z.shape)}")
    if basis.dim() != 2:
        raise ValueError(f"basis must be [hidden_dim, k], got shape {tuple(basis.shape)}")
    if z.shape[1] != basis.shape[0]:
        raise ValueError(f"z hidden dim {z.shape[1]} does not match basis dim {basis.shape[0]}")

    proj = z @ basis
    numerator = (proj ** 2).sum(dim=-1)
    denominator = (z ** 2).sum(dim=-1) + eps
    return (numerator / denominator).mean()


def parse_torch_dtype(dtype_name):
    if dtype_name is None:
        return None
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    normalized = str(dtype_name).lower()
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported torch dtype '{dtype_name}'.")
    return aliases[normalized]

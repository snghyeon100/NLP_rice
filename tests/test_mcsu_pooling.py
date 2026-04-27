import torch

from mcsu_utils import last_nonpad_pool


def test_last_nonpad_pool_right_padding():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ]
    )

    pooled = last_nonpad_pool(hidden, attention_mask)

    assert torch.equal(pooled[0], hidden[0, 2])
    assert torch.equal(pooled[1], hidden[1, 1])


def test_last_nonpad_pool_left_padding():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ]
    )

    pooled = last_nonpad_pool(hidden, attention_mask)

    assert torch.equal(pooled[0], hidden[0, 4])
    assert torch.equal(pooled[1], hidden[1, 4])


def test_last_nonpad_pool_all_valid_tokens():
    hidden = torch.arange(2 * 4 * 2, dtype=torch.float32).view(2, 4, 2)
    attention_mask = torch.ones(2, 4, dtype=torch.long)

    pooled = last_nonpad_pool(hidden, attention_mask)

    assert torch.equal(pooled[0], hidden[0, 3])
    assert torch.equal(pooled[1], hidden[1, 3])

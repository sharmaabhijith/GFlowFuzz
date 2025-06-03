import torch

dense_first = torch.nn.Linear(1000, 1000)
sparse_first = torch.nn.Linear(1000, 1000)

dense_first.weight.grad = torch.sparse_coo_tensor(
    torch.stack(
        [
            torch.arange(1000),
            torch.zeros(1000, dtype=torch.long),
        ]
    ),
    torch.ones(1000),
    (1000, 1000),
).to_dense()

sparse_first.weight.grad = torch.sparse_coo_tensor(
    torch.stack(
        [
            torch.arange(1000),
            torch.zeros(1000, dtype=torch.long),
        ]
    ),
    torch.ones(1000),
    (1000, 1000),
)

dense_first.weight.grad.shape == dense_first.weight.shape  # True
sparse_first.weight.grad.shape == sparse_first.weight.shape  # True
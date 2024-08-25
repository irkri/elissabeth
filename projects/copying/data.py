import numpy as np
import torch


def copying(
    n_samples: int,
    length: int,
    n_categories: int = 10,
    to_copy: int = 10,
    max_dilute: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if length <= 2 * to_copy:
        raise ValueError("Length has to be at least 2*to_copy + 1")
    if n_categories <= 2:
        raise ValueError("There have to be at least 3 categories")
    x = torch.empty((n_samples, length), dtype=torch.int64)
    x = torch.fill(x, n_categories-2)
    y = torch.empty((n_samples, length), dtype=torch.int64)
    y = torch.fill(y, -1)
    for i in range(n_samples):
        ind = np.sort(np.random.choice(
            to_copy+max_dilute*to_copy,
            to_copy,
            replace=False,
        ))
        x[i, ind] = torch.randint(
            n_categories-2,
            size=(to_copy, ),
        )
        y[i, -to_copy:] = x[i, ind]
    x[:, -to_copy-1] = n_categories-1
    return x, y

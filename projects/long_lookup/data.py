import numpy as np
import torch


def lookup(
    n_samples: int,
    length: int,
    characters: int = 3,
    multiple_keys: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a dataset of characters from an alphabet of given size.
    The task is to find the first occurence of the last character in the
    sequence. The answer then is the character right before this first
    occurence.
    Examples:
        "ABADABBCCDAACADDBBDBCAAABD" -> "A"
        "AABDBABD" -> "B"
        "BACA" -> "B"
    """
    x = torch.randint(
        characters,
        size=(n_samples, length),
        dtype=torch.int64,
    )
    y = x.detach().clone().roll(-1, 1)
    if multiple_keys:
        for i in range(n_samples):
            mark = np.random.randint(characters)
            indices = torch.where(x[i] == mark)[0]
            if len(indices) > 0 and indices[0] == 0:
                x[i, 0] = (mark + 1) % characters
                indices = indices[1:]
            if ((len(indices) == 1 and indices[-1] == length-1)
                    or len(indices) == 0):
                indices = torch.randint(1, length-1, (1, ), dtype=torch.int64)
            x[i, -1] = mark
            y[i, -2] = mark
            x[i, indices[0]] = mark
            y[i, -1] = x[i, indices[0]-1]
    else:
        for i in range(n_samples):
            index = np.random.randint(1, length-1)
            mark = np.random.randint(characters)
            mask = x[i] == mark
            x[i, mask] = torch.Tensor(
                np.random.choice(
                    [i for i in range(characters) if i != mark],
                    size=int(mask.sum()),
                ),
            ).long()
            x[i, index] = mark
            x[i, -1] = mark
            y[i, :-1] = x[i, 1:]
            y[i, -1] = x[i, index-1]
    return x, y

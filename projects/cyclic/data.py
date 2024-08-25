import numpy as np
import torch


def cyclic(
    n_samples: int,
    length: int,
    characters: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a dataset of characters from an alphabet of given size.
    The task is to check if the subsequence ``123...n`` is in the
    sequence. However, we also allow cyclic permutations like
    ``345...n12``. Here, ``n`` is the number of characters.
    """
    x = torch.zeros(
        size=(n_samples, length),
        dtype=torch.int64,
    )
    y = torch.zeros(n_samples, dtype=torch.int64)
    for i in range(n_samples):
        positions = np.sort(np.random.choice(
            length-1,
            size=characters-2,
            replace=False,
        ))
        if np.random.randint(2) == 1:
            marks = torch.roll(
                torch.arange(1, characters),
                shifts=np.random.randint(characters-1),
            )
            y[i] = 1
        else:
            marks = torch.arange(1, characters)
            bp = torch.where(marks == characters-1)[0] + 1
            while (all(torch.diff(marks[:bp]) == torch.ones(
                        max(len(marks[:bp])-1, 0)))
                    and all(
                        torch.diff(marks[bp:]) == torch.ones(
                            max(len(marks[bp:])-1, 0))
            )):
                marks = marks[torch.randperm(characters-1)]
                bp = torch.where(marks == characters-1)[0] + 1
        x[i, positions] = marks[:-1]
        x[i, -1] = marks[-1]
    return x, y

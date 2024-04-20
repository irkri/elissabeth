__all__ = [
    "modular_arithmetic",
    "gridworld",
    "imitate_mha",
    "streaks",
    "numberville",
    "long_lookup",
    "copying",
]

import itertools
import os
from typing import Optional

import numpy as np
import torch
from torch import nn


def modular_arithmetic(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Creates a complete dataset of integers ``(a,b,c)`` where ``a,b``
    are integers in ``{0,...,p-1}`` and ``c=(a+b)%p``.

    Args:
        p (int): Parameter for the modular addition.

    Returns:
        tuple of torch.Tensor: context and target
    """
    x = torch.zeros((p**2, 3), dtype=torch.long)
    x[:, 0:2] = torch.Tensor(list(itertools.product(range(p), repeat=2)))
    x[:, 2] = (x[:, 0] + x[:, 1]) % p
    y = x.detach().clone()
    # third token in input sequence is '=' (special), not a number
    # we encode this by the number p, which is not included in the input
    x[:, 2] = p

    return x, y


def _gridworld_automaton(
    start_at: int = 0,
    n_steps: int = 10,
    S: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a random example run of a automaton having two possible
    actions and a given number of states.

    Args:
        start_at (int, optional): The state to start in. Defaults to 0.
        n_steps (int, optional): The number of steps to generate.
            Defaults to 10.
        S (int, optional): The number of states of the automaton. States
            are enumerated starting at 0 to S-1. Defaults to 3.

    Returns:
        np.ndarray: An array with the actions used in the example.
        list[int]: The list of sequential states reached with the
            randomly generated actions.
    """
    commands = torch.randint(0, 3, size=(n_steps, )) - 1
    states = torch.zeros((n_steps+1, ))
    states[0] = start_at
    for i, command in enumerate(commands):
        states[i+1] = torch.clip(states[i]+command, 0, S-1)
    commands += 1
    return commands, states[1:]


def gridworld(
    n_samples: int = 1_000,
    n_steps: int = 10,
    S: int = 4,
    cache_path: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a datasset of example runs for an automaton with a
    given set of states and two possible actions.
    Reference::
        Liu et al - 2022 - Transformers Learn Shortcuts to Automata
        https://www.youtube.com/watch?v=g8zdumOAWzw
    """
    if cache_path is not None:
        file_commands = os.path.join(cache_path, "commands.npy")
        file_states = os.path.join(cache_path, "states.npy")
        if os.path.isfile(file_commands) and os.path.isfile(file_states):
            COMMANDS = torch.Tensor(np.load(file_commands)).type(torch.long)
            STATES = torch.Tensor(np.load(file_states)).type(torch.long)
            return COMMANDS, STATES

    COMMANDS = torch.zeros((n_samples, n_steps), dtype=torch.long)
    STATES = torch.zeros((n_samples, n_steps), dtype=torch.long)
    for i in range(n_samples):
        commands, states = _gridworld_automaton(n_steps=n_steps, S=S)
        COMMANDS[i] = commands
        STATES[i] = states

    if cache_path is not None:
        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        np.save(os.path.join(cache_path, "commands.npy"), COMMANDS.numpy())
        np.save(os.path.join(cache_path, "states.npy"), STATES.numpy())

    return COMMANDS, STATES


def imitate_mha(
    n_samples: int,
    length: int,
    embed_dim: int,
    n_heads: int = 1,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=n_heads,
        bias=False,
        batch_first=True,
    )
    X = torch.randn(n_samples, length, embed_dim)
    Y, _ = mha(X, X, X)
    Y = Y.detach()
    return X, Y


def streaks(
    n_samples: int,
    length: int,
    signal_length: int = 10,
    signal_start: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A signal of consecutive ones with given length is hidden in a
    stream of zeros and ones. The placement of this signal is
    controllable. Predict if the signal is present in the current input.
    """
    x = torch.randint(2, size=(n_samples, length+1), dtype=torch.int64)
    z = torch.randint(2, size=(n_samples, ), dtype=torch.int64)
    x[z == 1, signal_start:signal_start+signal_length] = 1
    y = x.detach().clone()
    x[:, -1] = 0
    y[:, -1] = z
    return x, y


def numberville(
    n_samples: int,
    length: int,
    start: tuple[int, int],
    spacing: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Why is 6 afraid of 7?

    A dataset of sequences of digits 0-9. Classify if the subsequence
    ``789`` exists. A given spacing fills the spaces in between these
    numbers with random digits.
    """
    x = torch.randint(10, size=(n_samples, length), dtype=torch.int64)
    z = torch.randint(2, size=(n_samples, ), dtype=torch.int64)
    for i in range(n_samples):
        if z[i] == 1:
            s = np.random.randint(start[0], start[1])
            x[i, s] = 7
            x[i, s+spacing] = 8
            x[i, s+2*spacing] = 9
    y = x.detach().clone().roll(-1, 1)
    y[:, -1] = z
    return x, y


def long_lookup(
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


def copying(
    n_samples: int,
    length: int,
    n_categories: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    if length <= 20:
        raise ValueError("Length has to be at least 21")
    if n_categories <= 2:
        raise ValueError("There have to be at least 3 categories")
    x = torch.empty((n_samples, length), dtype=torch.int64)
    x = torch.fill(x, n_categories-2)
    x[:, :10] = torch.randint(n_categories-2, size=(n_samples, 10))
    x[:, -11] = n_categories-1
    y = torch.empty((n_samples, length), dtype=torch.int64)
    y = torch.fill(y, -1)
    y[:, -10:] = x[:, :10]
    return x, y

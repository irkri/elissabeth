__all__ = ["modular_arithmetic", "gridworld", "imitate_mha"]

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
) -> tuple[np.ndarray, np.ndarray]:
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
    commands = np.random.choice([-1, 0, 1], size=n_steps)
    states = np.zeros((n_steps+1, ))
    states[0] = start_at
    for i, command in enumerate(commands):
        states[i+1] = np.clip(states[i]+command, 0, S-1)
    return commands, states


def gridworld(
    n_samples: int = 1_000,
    n_steps: int = 10,
    S: int = 4,
    cache_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
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
            COMMANDS = np.load(file_commands)
            STATES = np.load(file_states)
            return COMMANDS, STATES

    COMMANDS = np.zeros((n_samples, n_steps))
    STATES = np.zeros((n_samples, n_steps+1))
    for i in range(n_samples):
        commands, states = _gridworld_automaton(n_steps=n_steps, S=S)
        COMMANDS[i] = commands
        STATES[i] = states

    if cache_path is not None:
        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        np.save(os.path.join(cache_path, "commands.npy"), COMMANDS)
        np.save(os.path.join(cache_path, "states.npy"), STATES)

    return COMMANDS, STATES


def imitate_mha(
    n_samples: int,
    length: int,
    embed_dim: int,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=1,
        bias=False,
        batch_first=True,
    )
    X = torch.randn(n_samples, length, embed_dim)
    Y, _ = mha(X, X, X)
    Y = Y.detach()
    return X, Y

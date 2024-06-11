from collections.abc import Sequence
from typing import Literal, Optional

import torch

from ..elissabeth import Elissabeth


def reduce_append_dims(
    tensor: torch.Tensor,
    expect: Optional[int] = None,
    reduce_dims: dict[int, int] | bool = False,
    append_dims: Sequence[int] | bool = True,
) -> torch.Tensor:
    if isinstance(reduce_dims, dict):
        reduce_dims = dict(sorted(reduce_dims.items()))
        for d, i in reduce_dims.items():
            tensor = torch.index_select(tensor, dim=d, index=torch.tensor(i))
            tensor = tensor.squeeze(d)
    elif reduce_dims:
        tensor = tensor.squeeze()
        while tensor.ndim > 4:
            tensor = tensor[0]
    if isinstance(append_dims, Sequence):
        for d in append_dims:
            tensor = tensor.unsqueeze(d)
    elif append_dims:
        while tensor.ndim < 4:
            tensor = tensor.unsqueeze(0)
    if expect is not None and tensor.ndim != expect:
        raise IndexError(
            f"Expected {expect} dimensions of tensor, "
            f"but got {tensor.ndim}: {tensor.shape}."
        )
    return tensor


def get_attention_matrices(
    model: Elissabeth,
    x: torch.Tensor,
    layer: int = 0,
    length: int = 0,
    only_kernels: Optional[tuple[int, ...]] = None,
    total: bool = False,
    project_heads: tuple[int, ...] | bool = False,
) -> torch.Tensor:
    """Returns the attention matrices in an Elissabeth model generated
    by the input ``x``.

    Args:
        model (Elissabeth): Trained Elissabeth model.
        x (torch.Tensor): Example to generate the attention matrix for.
            Has to be of shape ``(T, )`` for token input or ``(T, d)``
            for vector input.
        layer (int, optional): The index of the LISS layer to extract
            the attention matrices from. Defaults to 0.
        length (int, optional): The index of the ISS level to extract
            the attention matrices from. Defaults to 0.
        only_kernels (tuple of int, optional): Restricts the attention
            matrix to kernels with the given indices. Defaults to None.
        total (bool, optional): If set to True, also calculates the
            attention matrix for the whole iterated sum ``Att_{t_1,t}``
            by calculating the iterated sum of all weightings. Defaults
            to False.
        project_heads (tuple of int | bool, optional): Whether the
            ``n_is`` iterated sums of the model should be linearly
            projected using the ``W_H`` matrix in the model. The result
            then is a tensor with first dimension of size 1. If a tuple
            of indices is given, only these iterated sums are
            considered. Defaults to False.

    Returns:
        torch.Tensor: Attention matrix of shape
            ``(n_is, length_is, T, T)`` or ``(1, length_is, T, T)`` if
            ``project_heads`` is set to true.
    """
    for weighting in model.layers[layer].levels[length].weightings:
        weighting.hooks.get("Att").attach()

    model(x.to(next(model.parameters()).device).unsqueeze(0))

    for weighting in model.layers[layer].levels[length].weightings:
        weighting.hooks.get("Att").release()

    if model.layers[layer].config("lengths") is not None:
        iss_length = model.layers[layer].config("lengths")[length]
    else:
        iss_length = length + 1

    N = model.layers[0].config("n_is")
    att_mat = torch.ones((N, iss_length, x.size(0), x.size(0)))

    if only_kernels is None:
        for weighting in model.layers[layer].levels[length].weightings:
            att_mat[:, :, :, :] *= weighting.hooks.get("Att").fwd[0]
    else:
        for j in only_kernels:
            weighting = model.layers[layer].levels[length].weightings[j]
            att_mat[:, :, :, :] *= weighting.hooks.get("Att").fwd[0]

    if isinstance(project_heads, tuple):
        att_mat = torch.tensordot(
            model.layers[layer].W_H[length, project_heads],
            att_mat[project_heads, ...],
            dims=([0], [0]),  # type: ignore
        ).detach().unsqueeze(0)
    elif project_heads:
        att_mat = torch.tensordot(
            model.layers[layer].W_H[length, :],
            att_mat,
            dims=([0], [0]),  # type: ignore
        ).detach().unsqueeze(0)

    if total:
        total_att = torch.zeros((N, x.size(0), x.size(0)))
        ind = torch.triu_indices(x.size(0), x.size(0), offset=0)
        total_att[:, :] = att_mat[:, :, 0]
        total_att[:, :, *ind] = 0
        for p in range(1, iss_length):
            if p == iss_length - 1:
                ind = torch.triu_indices(x.size(0), x.size(0), offset=1)
            mat = torch.clone(att_mat[:, :, p, :, :])
            mat[:, :, *ind] = 0
            total_att[:, :, :, :] = mat @ total_att
        if isinstance(project_heads, tuple):
            total_att = torch.tensordot(
                model.layers[layer].W_H[length, project_heads],
                total_att[project_heads, ...],
                dims=([0], [0]),  # type: ignore
            ).detach().unsqueeze(0)
        elif project_heads:
            total_att = torch.tensordot(
                model.layers[layer].W_H[length, :],
                total_att,
                dims=([0], [0]),  # type: ignore
            ).detach().unsqueeze(0)
        att_mat = torch.cat((att_mat, total_att.unsqueeze(-3)), dim=-3)

    return att_mat


def probe_qkv_transform(
    model: Elissabeth,
    which: Literal["q", "k", "v"] = "v",
    layer: int = 0,
    length: int = 0,
    weighting: int = 0,
) -> torch.Tensor:
    match which:
        case "v":
            P = model.layers[layer].levels[length].P_V
        case "q":
            P = model.layers[layer].levels[length].weightings[weighting].P_Q
        case "k":
            P = model.layers[layer].levels[length].weightings[weighting].P_K
    D = model.config("d_hidden")
    T = model.config("context_length")
    if which == "v":
        include_time = P.config("v_include_time")
    else:
        include_time = P.config("qk_include_time")
    y = torch.empty((D, T, D+int(include_time)))
    x = torch.eye(D, D).unsqueeze(1).repeat(1, T, 1)
    y[:, :, :D] = x
    if include_time:
        y[:, :, D] = torch.linspace(1/T, 1, T)
    return P.transform(y).reshape(*y.shape[:-1], *P._shape).detach()


def get_iss(
    model: Elissabeth,
    x: torch.Tensor,
    layer: int = 0,
    length: int = 0,
    project_heads: tuple[int, ...] | bool = False,
    project_values: bool = False,
) -> torch.Tensor:
    """Extracts the hook 'iss' from an Elissabeth model. This only works
    for one dimensional values.

    Args:
        model (Elissabeth): Model to extract the ISS from.
        x (torch.Tensor): Example tensor for which the model calculates
            an output.
        layer (int, optional): The index of the layer to use. Defaults
            to 0.
        length (int, optional): The index of the iss length to use.
            Defaults to 0.
        project_heads (tuple of int | bool, optional): Whether the
            ``n_is`` iterated sums of the model should be linearly
            projected using the ``W_H`` matrix in the model. The result
            then is a tensor with first dimension of size 1. If a tuple
            of indices is given, only these iterated sums are
            considered. Defaults to False.
        project_values (bool, optional): Whether the values should be
            projected to the hidden dimension using the parameters
            ``W_O`` in the model. The result then is a tensor with
            ``d_v`` replaced to ``d_hidden``. Defaults to False.

    Returns:
        torch.Tensor: Tensor of shape ``(n_is, d_v, T)``.
    """
    model.layers[layer].levels[length].hooks.get("iss").attach()
    model(x.to(next(model.parameters()).device).unsqueeze(0))
    model.layers[layer].levels[length].hooks.get("iss").release()
    iss = model.layers[layer].levels[length].hooks.get("iss").fwd[0, ...]
    iss = torch.swapaxes(iss, 0, 1)

    if isinstance(project_heads, tuple):
        iss = torch.tensordot(
            model.layers[layer].W_H[length, project_heads],
            iss[project_heads, ...],
            dims=([0], [0]),  # type: ignore
        ).detach().unsqueeze(0)
    elif project_heads:
        iss = torch.tensordot(
            model.layers[layer].W_H[length, :],
            iss,
            dims=([0], [0]),  # type: ignore
        ).detach().unsqueeze(0)
    if project_values:
        iss = torch.einsum("vwd,ntvw->ntd", model.layers[layer].W_O, iss)
        iss = iss.detach()
    else:
        iss = iss[..., 0]
    return torch.swapaxes(iss, -1, -2)


def get_query_key(
    model: Elissabeth,
    x: torch.Tensor,
    layer: int = 0,
    length: int = 0,
    weighting: int = 0,
    project_heads: tuple[int, ...] | bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts the query and key hooks from an Elissabeth model.

    Args:
        model (Elissabeth): Model to extract the hooks from.
        x (torch.Tensor): Example tensor for which the model calculates
        which ("Q" or "K"): Extract either queries "Q" or keys "K".
        layer (int, optional): The index of the layer to use. Defaults
            to 0.
        length (int, optional): The index of the iss length to use.
            Defaults to 0.
        weighting (int, optional): The index of the kernel to extract
            query and key from. Defaults to 0.
        project_heads (tuple of int | bool, optional): Whether the
            ``n_is`` iterated sums of the model should be linearly
            projected using the ``W_H`` matrix in the model. The result
            then is a tensor with first dimension of size 1. If a tuple
            of indices is given, only these iterated sums are
            considered. Defaults to False.

    Returns:
        tuple of torch.Tensor: Two tensors of shape
            ``(n_is, p, d_qk, T)``.
    """
    kernel = model.layers[layer].levels[length].weightings[weighting]
    kernel.hooks.get("Q").attach()
    kernel.hooks.get("K").attach()
    model(x.to(next(model.parameters()).device).unsqueeze(0))
    kernel.hooks.get("Q").release()
    kernel.hooks.get("K").release()
    q = kernel.hooks.get("Q").fwd[0, ...]
    k = kernel.hooks.get("K").fwd[0, ...]
    q = torch.swapaxes(torch.swapaxes(q, 0, 1), 1, 2)
    k = torch.swapaxes(torch.swapaxes(k, 0, 1), 1, 2)

    if isinstance(project_heads, tuple):
        q = torch.tensordot(
            model.layers[layer].W_H[length, project_heads],
            q[project_heads, ...],
            dims=([0], [0]),  # type: ignore
        ).detach().unsqueeze(0)
        k = torch.tensordot(
            model.layers[layer].W_H[length, project_heads],
            k[project_heads, ...],
            dims=([0], [0]),  # type: ignore
        ).detach().unsqueeze(0)
    elif project_heads:
        q = torch.tensordot(
            model.layers[layer].W_H[length, :],
            q,
            dims=([0], [0]),  # type: ignore
        ).detach().unsqueeze(0)
        k = torch.tensordot(
            model.layers[layer].W_H[length, :],
            k,
            dims=([0], [0]),  # type: ignore
        ).detach().unsqueeze(0)
    return (torch.swapaxes(q, -1, -2), torch.swapaxes(k, -1, -2))


def get_values(
    model: Elissabeth,
    x: torch.Tensor,
    layer: int = 0,
    length: int = 0,
    project_heads: tuple[int, ...] | bool = False,
    project_values: bool = False,
) -> torch.Tensor:
    """Extracts the values hook 'V' from an Elissabeth model. This only
    works for one dimensional values.

    Args:
        model (Elissabeth): Model to extract the values from.
        x (torch.Tensor): Example tensor for which the model calculates
            an output.
        layer (int, optional): The index of the layer to use. Defaults
            to 0.
        length (int, optional): The index of the iss length to use.
            Defaults to 0.
        project_heads (tuple of int | bool, optional): Whether the
            ``n_is`` iterated sums of the model should be linearly
            projected using the ``W_H`` matrix in the model. The result
            then is a tensor with first dimension of size 1. If a tuple
            of indices is given, only these iterated sums are
            considered. Defaults to False.
        project_values (bool, optional): Whether the values should be
            projected to the hidden dimension using the parameters
            ``W_O`` in the model. The result then is a tensor with
            ``d_v`` replaced to ``d_hidden``. Defaults to False.

    Returns:
        torch.Tensor: Tensor of shape ``(n_is, p, d_v, T)``.
    """
    model.layers[layer].levels[length].hooks.get("V").attach()
    model(x.to(next(model.parameters()).device).unsqueeze(0))
    model.layers[layer].levels[length].hooks.get("V").release()
    v = model.layers[layer].levels[length].hooks.get("V").fwd[0, ...]
    v = torch.swapaxes(torch.swapaxes(v, 0, 1), 1, 2)

    if isinstance(project_heads, tuple):
        v = torch.tensordot(
            model.layers[layer].W_H[length, project_heads],
            v[project_heads, ...],
            dims=([0], [0]),  # type: ignore
        ).detach().unsqueeze(0)
    elif project_heads:
        v = torch.tensordot(
            model.layers[layer].W_H[length, :],
            v,
            dims=([0], [0]),  # type: ignore
        ).detach().unsqueeze(0)
    if project_values:
        v = torch.einsum("vwd,nptvw->nptd", model.layers[layer].W_O, v)
        v = v.detach()
    else:
        v = v[..., 0]
    return torch.swapaxes(v, -1, -2)


def get_alphabet_projection(
    model: Elissabeth,
    layer: int = 0,
    length: int = 0,
    weighting: int = 0,
    n: int = 0,
    p: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.arange(model.config("input_vocab_size"))
    x = x.unsqueeze(1).repeat(1, model.config("context_length"))
    kernel = model.layers[layer].levels[length].weightings[weighting]
    model.layers[layer].levels[length].hooks.get("V").attach()
    kernel.hooks.get("Q").attach()
    kernel.hooks.get("K").attach()
    model(x.to(next(model.parameters()).device))
    model.layers[layer].levels[length].hooks.get("V").release()
    kernel.hooks.get("Q").release()
    kernel.hooks.get("K").release()
    q = kernel.hooks.get("Q").fwd[..., n, p, :]
    k = kernel.hooks.get("K").fwd[..., n, p, :]
    v = model.layers[layer].levels[length].hooks.get("V").fwd[..., n, p, :, 0]
    q = torch.swapaxes(q, 1, 2)
    k = torch.swapaxes(k, 1, 2)
    v = torch.swapaxes(v, 1, 2)
    return q, k, v

from typing import Literal, Optional

import torch

from ..elissabeth import Elissabeth


def get_attention_matrices(
    model: Elissabeth,
    x: torch.Tensor,
    layer: int = 0,
    length: int = 0,
    total: bool = False,
    project_heads: bool = False,
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
        total (bool, optional): If set to True, also calculates the
            attention matrix for the whole iterated sum ``Att_{t_1,t}``
            by calculating the iterated sum of all weightings. Defaults
            to False.
        project_heads (bool, optional): Whether the ``n_is`` iterated
            sums of the model should be linearly projected using the
            ``W_H`` matrix in the model. The result then is a tensor
            with first dimension of size 1. Defaults to False.

    Returns:
        torch.Tensor: Attention matrix of shape
            ``(n_is, length_is, T, T)`` or ``(1, length_is, T, T)`` if
            ``project_heads`` is set to true.
    """
    for liss_layer in model.layers:
        for weighting in liss_layer.levels[length].weightings:
            weighting.hooks.get("Att").attach()

    model(x.to(next(model.parameters()).device).unsqueeze(0))

    for liss_layer in model.layers:
        for weighting in liss_layer.levels[length].weightings:
            weighting.hooks.get("Att").release()

    if model.layers[layer].config("lengths") is not None:
        iss_length = model.layers[layer].config("lengths")[length]
    else:
        iss_length = length + 1

    N = model.layers[0].config("n_is")
    att_mat = torch.ones((N, iss_length, x.size(0), x.size(0)))

    for weighting in model.layers[layer].levels[length].weightings:
        att_mat[:, :, :, :] *= weighting.hooks.get("Att").fwd[0]
    if project_heads:
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
        if project_heads:
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

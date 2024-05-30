import torch

from ..elissabeth import Elissabeth


def get_attention_matrices(
    model: Elissabeth,
    x: torch.Tensor,
    total: bool = False,
    length_index: int = 0,
) -> torch.Tensor:
    """Returns the attention matrices in an Elissabeth model generated
    by the input ``x``.

    Args:
        model (Elissabeth): Trained Elissabeth model.
        x (torch.Tensor): Example to generate the attention matrix for.
            Has to be of shape ``(T, )`` for token input or ``(T, d)``
            for vector input.
        total (bool, optional): If set to True, also calculates the
            attention matrix for the whole iterated sum ``Att_{t_1,t}``
            by calculating the iterated sum of all weightings. Defaults
            to False.
        level_index (int, optional): The index of the ISS level to
            extract the attention matrices from. Defaults to 0.

    Returns:
        torch.Tensor: Attention matrix of shape
            ``(n_is, n_layers, length_is, T, T)``.
    """
    for layer in model.layers:
        for weighting in layer.levels[length_index].weightings:
            weighting.hooks.get("Att").attach()

    model(x.to(next(model.parameters()).device).unsqueeze(0))

    for layer in model.layers:
        for weighting in layer.levels[length_index].weightings:
            weighting.hooks.get("Att").release()

    n_layers = model.config("n_layers")
    if model.layers[0].config("lengths") is not None:
        iss_length = model.layers[0].config("lengths")[length_index]
    else:
        iss_length = length_index + 1
    N = model.layers[0].config("n_is")
    att_mat = torch.ones((N, n_layers, iss_length, x.size(0), x.size(0)))

    for l in range(n_layers):
        for weighting in model.layers[l].levels[length_index].weightings:
            att_mat[:, l, :, :, :] *= weighting.hooks.get("Att").fwd[0]

    if total:
        total_att = torch.zeros((N, n_layers, x.size(0), x.size(0)))
        ind = torch.triu_indices(x.size(0), x.size(0), offset=0)
        total_att[:, :] = att_mat[:, :, 0]
        total_att[:, :, *ind] = 0
        for p in range(1, iss_length):
            if p == iss_length - 1:
                ind = torch.triu_indices(x.size(0), x.size(0), offset=1)
            mat = torch.clone(att_mat[:, :, p, :, :])
            mat[:, :, *ind] = 0
            total_att[:, :, :, :] = mat @ total_att
        att_mat = torch.cat((att_mat, total_att.unsqueeze(2)), dim=2)

    return att_mat

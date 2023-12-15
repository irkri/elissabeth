import torch

from sainomore.elissabeth.liss import LISS
from sainomore.elissabeth import ElissabethConfig


def test_output() -> None:
    config = ElissabethConfig(
        context_length=13,
        input_vocab_size=4,
        n_is=1,
        n_layers=5,
        d_hidden=2,
        d_values=6,
    )

    liss = LISS(config)

    x = torch.zeros((1, 13, 2))
    for i in range(x.size(0)):
        x[i, :, 0] = 1

    torch.nn.init.zeros_(liss.W_Q)
    torch.nn.init.zeros_(liss.W_K)
    print(f"{liss.W_Q=}")
    print(f"{liss.W_K=}")
    print(f"{liss.W_V=}")
    liss.attach_all_hooks()

    y = liss(x)

    for key in liss.hooks.names:
        print(key, liss.hooks.get(key).fwd)
    print(y.detach().shape)
    # assert y.detach().shape == ()


test_output()

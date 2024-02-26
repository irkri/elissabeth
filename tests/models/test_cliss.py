import numpy as np
import torch

from sainomore.elissabeth import CLISSConfig, Elissabeth


def test_cliss_weights() -> None:
    cliss = Elissabeth(
        CLISSConfig(
            context_length=10,
            input_vocab_size=1,
            d_hidden=1,
            d_values=1,
            length_is=3,
            n_layers=1,
            layer_norm=False,
            d_query_key=1,
            exponent=3,
            input_type="vector",
            distance_weighting=True,
            alpha_multiplier=2,
            values_2D=False,
            bias_query_key=False,
            bias_value=False,
            pe_key=False,
            pe_value=False,
            share_queries=False,
            share_keys=False,
            share_values=False,
            sum_normalization=None,
            bias=False,
        )
    )
    torch.nn.init.ones_(cliss.get_parameter("layers.0.W_Q"))
    torch.nn.init.ones_(cliss.get_parameter("layers.0.W_K"))
    torch.nn.init.ones_(cliss.get_parameter("layers.0.W_V"))
    torch.nn.init.ones_(cliss.get_parameter("layers.0.W_O"))
    cliss.set_eye("unembedding.weight", requires_grad=False, dims=(0, 1))
    states = cliss.state_dict()
    alpha = torch.Tensor([1, 3, 2])
    states["layers.0.alpha"] = (
        alpha.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    )
    alpha = 1 - 1 / (alpha**2 + 1)
    cliss.load_state_dict(states)

    X = torch.randn((1, 10))
    the_result = torch.zeros_like(X)
    for t in range(X.shape[1]):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    the_result[0, t] += (X[0, t_1]
                        * X[0, t_2]
                        * X[0, t_3]
                        * torch.cos(X[0, t_2] - X[0, t_1])**3
                        * torch.cos(X[0, t_3] - X[0, t_2])**3
                        * torch.cos(X[0,   t] - X[0, t_3])**3
                        * np.exp(-2*alpha[2] * (  t - t_3 - 1)/10)
                        * np.exp(-2*alpha[1] * (t_3 - t_2 - 1)/10)
                        * np.exp(-2*alpha[0] * (t_2 - t_1 - 1)/10)
                    )

    cliss.attach_all_hooks()
    result = cliss(X.unsqueeze(-1))
    cliss.release_all_hooks()
    torch.testing.assert_close(result[0, :, :], the_result+X)


if __name__ == "__main__":
    test_cliss_weights()

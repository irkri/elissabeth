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
    cliss.load_state_dict(states)

    X = torch.randn((1, 10))
    the_result = torch.zeros_like(X)
    for k in range(X.shape[1]):
        for i_3 in range(k+1):
            for i_2 in range(i_3):
                for i_1 in range(i_2):
                    the_result[0, k] += (X[0, i_1]
                        * X[0, i_2]
                        * X[0, i_3]
                        * torch.cos(X[0, i_2] - X[0, i_1])**3
                        * torch.cos(X[0, i_3] - X[0, i_2])**3
                        * torch.cos(X[0, k] - X[0, i_3])**3
                        * np.exp(-2*torch.sigmoid(alpha[2])*(k - i_3)/10)
                        * np.exp(-2*torch.sigmoid(alpha[1])*(i_3 - i_2)/10)
                        * np.exp(-2*torch.sigmoid(alpha[0])*(i_2 - i_1)/10)
                    )

    cliss.attach_all_hooks()
    result = cliss(X.unsqueeze(-1))
    cliss.release_all_hooks()
    torch.testing.assert_close(result[0, :, :], the_result+X)


if __name__ == "__main__":
    test_cliss_weights()

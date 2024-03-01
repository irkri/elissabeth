import numpy as np
import torch

from sainomore.elissabeth import Elissabeth, Weighting


def test_liss_weights() -> None:

    config = {
        "context_length" : "10",
        "input_vocab_size" : "1",
        "d_hidden" : "1",
        "n_is" : "1",
        "d_values" : "1",
        "length_is" : "3",
        "n_layers" : "1",
        "layer_norm" : "False",
        "input_type" : "vector",
        "restrict_query_key" : "False",
        "distance_weighting" : "True",
        "alpha_multiplier" : "2",
        "values_2D" : "False",
        "bias_query_key" : "False",
        "bias_value" : "False",
        "pe_key" : "False",
        "pe_value" : "False",
        "share_queries" : "False",
        "share_keys" : "False",
        "share_values" : "False",
        "sum_normalization" : None,
        "bias" : "False",
    }

    model = Elissabeth.build(
        config,
        Weighting.RELATIVE_DISTANCE | Weighting.EXPONENTIAL,

    )
    torch.nn.init.ones_(
        model.get_parameter("layers.0.weightings.1.W_Q")
    )
    torch.nn.init.ones_(
        model.get_parameter("layers.0.weightings.1.W_K")
    )
    torch.nn.init.ones_(model.get_parameter("layers.0.W_V"))
    torch.nn.init.ones_(model.get_parameter("layers.0.W_O"))
    model.set_eye(
        "unembedding.weight",
        requires_grad=False,
        dims=(0, 1),
    )
    states = model.state_dict()
    alpha = torch.Tensor([1, 3, 2])
    states["layers.0.weightings.0.alpha"] = (
        alpha.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    )
    alpha = 1 - 1 / (alpha**2 + 1)
    model.load_state_dict(states)

    X = torch.randn((1, 10))
    the_result = torch.zeros_like(X)
    for t in range(X.shape[1]):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    the_result[0, t] += (
                          X[0, t_1]
                        * X[0, t_2]
                        * X[0, t_3]
                        * torch.exp(X[0, t_2] - X[0, t_1])
                        * torch.exp(X[0, t_3] - X[0, t_2])
                        * torch.exp(X[0,   t] - X[0, t_3])
                        * np.exp(-2*alpha[2] * (t - t_3)/10)
                        * np.exp(-2*alpha[1] * (t_3 - t_2 - 1)/10)
                        * np.exp(-2*alpha[0] * (t_2 - t_1 - 1)/10)
                    )

    result = model(X.unsqueeze(-1))
    torch.testing.assert_close(result[0, :, :], the_result+X)


if __name__ == "__main__":
    test_liss_weights()

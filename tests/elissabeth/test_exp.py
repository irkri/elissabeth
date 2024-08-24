import numpy as np
import torch
from torch import randn

from sainomore.elissabeth import Elissabeth


def test_exp_weighting() -> None:

    config = {
        "context_length" : 10,
        "input_vocab_size" : 5,
        "d_hidden" : 5,
        "n_layers" : 1,
        "layer_norm" : False,
        "residual_stream": False,
        "sum_normalization" : False,

        "n_is" : 1,
        "lengths" : [3],
        "d_values" : 5,
        "values_2D" : False,
        "pe_value" : False,
        "v_norm": False,

        "restrict_query_key" : False,

        "exp_alpha_0": .5,

        "weighting": ["ExponentialDecay", "Exponential"]
    }
    model = Elissabeth.build(config)

    q1 = randn(5)
    k1 = randn(5)
    q2 = randn(5)
    k2 = randn(5)
    q3 = randn(5)
    k3 = randn(5)

    v1 = randn(5, 5)
    v2 = randn(5, 5)
    v3 = randn(5, 5)

    state_dict = model.state_dict()

    state_dict["embedding.weight"] = torch.eye(5)

    alpha = torch.Tensor([1, 3, 2])
    state_dict["layers.0.levels.0.weightings.0.alpha"] = (
        alpha.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    )
    alpha = torch.tanh(alpha)

    state_dict["layers.0.levels.0.weightings.1.P_Q.transform.weight"] = (
        torch.stack((q1, q2, q3))
    )
    state_dict["layers.0.levels.0.weightings.1.P_K.transform.weight"] = (
        torch.stack((k1, k2, k3))
    )

    state_dict["layers.0.levels.0.P_V.transform.weight"] = torch.cat(
        (v1.T, v2.T, v3.T), dim=0,
    )

    state_dict["layers.0.W_H"] = torch.tensor(((1,),))
    state_dict["layers.0.W_O"] = torch.eye(5).unsqueeze(1)

    state_dict["unembedding.weight"] = torch.eye(5)

    model.load_state_dict(state_dict)

    X = torch.randint(0, 5, size=(10, ))
    the_result = torch.zeros(10, 5)
    for t in range(X.shape[0]):
        for t_3 in range(t+1):
            for t_2 in range(t_3):
                for t_1 in range(t_2):
                    the_result[t] += (
                          v1[X[t_1]]
                        * v2[X[t_2]]
                        * v3[X[t_3]]

                        * torch.exp(q1[X[t_2]] - k1[X[t_1]])
                        * torch.exp(q2[X[t_3]] - k2[X[t_2]])
                        * torch.exp(q3[X[  t]] - k3[X[t_3]])

                        * np.exp(-.5*alpha[2] * ( t - t_3)/10)
                        * np.exp(-.5*alpha[1] * (t_3 - t_2 - 1)/10)
                        * np.exp(-.5*alpha[0] * (t_2 - t_1 - 1)/10)
                    )

    result = model(X.unsqueeze(0))
    torch.testing.assert_close(
        result[0, :, :],
        the_result,
        rtol=1e-5,
        atol=1e-4,
    )


if __name__ == "__main__":
    test_exp_weighting()

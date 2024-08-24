import numpy as np
import torch
from torch import randn

from sainomore.elissabeth import Elissabeth


def test_cosine_weighting() -> None:

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
        "exponent": 2,
        "d_query_key": 3,

        "exp_alpha_0" : 2,

        "weighting": ["ExponentialDecay", "Cosine"]
    }
    model = Elissabeth.build(config)

    q11, q12, q13 = randn(5), randn(5), randn(5)
    k11, k12, k13 = randn(5), randn(5), randn(5)

    q21, q22, q23 = randn(5), randn(5), randn(5)
    k21, k22, k23 = randn(5), randn(5), randn(5)

    q31, q32, q33 = randn(5), randn(5), randn(5)
    k31, k32, k33 = randn(5), randn(5), randn(5)

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
        torch.stack((q11, q12, q13, q21, q22, q23, q31, q32, q33))
    )
    state_dict["layers.0.levels.0.weightings.1.P_K.transform.weight"] = (
        torch.stack((k11, k12, k13, k21, k22, k23, k31, k32, k33))
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

                        * torch.cos(q11[X[t_2]] - k11[X[t_1]])**2
                        * torch.cos(q12[X[t_2]] - k12[X[t_1]])**2
                        * torch.cos(q13[X[t_2]] - k13[X[t_1]])**2

                        * torch.cos(q21[X[t_3]] - k21[X[t_2]])**2
                        * torch.cos(q22[X[t_3]] - k22[X[t_2]])**2
                        * torch.cos(q23[X[t_3]] - k23[X[t_2]])**2

                        * torch.cos(q31[X[  t]] - k31[X[t_3]])**2
                        * torch.cos(q32[X[  t]] - k32[X[t_3]])**2
                        * torch.cos(q33[X[  t]] - k33[X[t_3]])**2

                        * np.exp(-2*alpha[2] * (  t - t_3)/10)
                        * np.exp(-2*alpha[1] * (t_3 - t_2 - 1)/10)
                        * np.exp(-2*alpha[0] * (t_2 - t_1 - 1)/10)
                    )

    result = model(X.unsqueeze(0))
    torch.testing.assert_close(result[0, :, :], the_result)


def test_cosine_decay_weighting() -> None:

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
        "exponent": 1,
        "d_query_key": 3,

        "cos_alpha_0" : 2,
        "d_alpha": 1,
        "decay_exponent": 2,

        "weighting": ["CosineDecay", "Cosine"]
    }
    model = Elissabeth.build(config)

    q11, q12, q13 = randn(5), randn(5), randn(5)
    k11, k12, k13 = randn(5), randn(5), randn(5)

    q21, q22, q23 = randn(5), randn(5), randn(5)
    k21, k22, k23 = randn(5), randn(5), randn(5)

    q31, q32, q33 = randn(5), randn(5), randn(5)
    k31, k32, k33 = randn(5), randn(5), randn(5)

    v1 = randn(5, 5)
    v2 = randn(5, 5)
    v3 = randn(5, 5)

    state_dict = model.state_dict()

    state_dict["embedding.weight"] = torch.eye(5)

    alpha = torch.Tensor([1, 3, 2])
    state_dict["layers.0.levels.0.weightings.0.alpha"] = (
        alpha.unsqueeze(-1).unsqueeze(-1)
             .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    )
    alpha = torch.tanh(alpha)

    state_dict["layers.0.levels.0.weightings.1.P_Q.transform.weight"] = (
        torch.stack((q11, q12, q13, q21, q22, q23, q31, q32, q33))
    )
    state_dict["layers.0.levels.0.weightings.1.P_K.transform.weight"] = (
        torch.stack((k11, k12, k13, k21, k22, k23, k31, k32, k33))
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

                        * torch.cos(q11[X[t_2]] - k11[X[t_1]])
                        * torch.cos(q12[X[t_2]] - k12[X[t_1]])
                        * torch.cos(q13[X[t_2]] - k13[X[t_1]])

                        * torch.cos(q21[X[t_3]] - k21[X[t_2]])
                        * torch.cos(q22[X[t_3]] - k22[X[t_2]])
                        * torch.cos(q23[X[t_3]] - k23[X[t_2]])

                        * torch.cos(q31[X[  t]] - k31[X[t_3]])
                        * torch.cos(q32[X[  t]] - k32[X[t_3]])
                        * torch.cos(q33[X[  t]] - k33[X[t_3]])

                        * np.cos(-2*alpha[2] * (  t - t_3)/10)**2
                        * np.cos(-2*alpha[1] * (t_3 - t_2 - 1)/10)**2
                        * np.cos(-2*alpha[0] * (t_2 - t_1 - 1)/10)**2
                    )

    result = model(X.unsqueeze(0))
    torch.testing.assert_close(result[0, :, :], the_result)


if __name__ == "__main__":
    test_cosine_weighting()
    test_cosine_decay_weighting()

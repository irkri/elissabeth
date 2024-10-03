import torch
from torch import randn

from sainomore.elissabeth import Elissabeth


def test_iterated_sum() -> None:

    config = {
        "context_length" : 10,
        "input_vocab_size" : 5,
        "d_hidden" : 5,
        "n_layers" : 1,
        "layer_norm" : False,
        "residual_stream": False,
        "sum_normalization" : False,
        "semiring" : "arctic",

        "n_is" : 1,
        "lengths" : [3],
        "d_values" : 5,
        "values_2D" : False,
        "pe_value" : False,
        "v_norm": False,

        "weighting": []
    }
    model = Elissabeth.build(config)

    v1 = randn(5, 5)
    v2 = randn(5, 5)
    v3 = randn(5, 5)

    state_dict = model.state_dict()

    state_dict["embedding.weight"] = torch.eye(5)

    state_dict["layers.0.levels.0.P_V.transform.weight"] = torch.cat(
        (v1.T, v2.T, v3.T), dim=0,
    )

    state_dict["layers.0.W_H"] = torch.tensor(((1,),))
    state_dict["layers.0.W_O"] = torch.eye(5).unsqueeze(1)

    state_dict["unembedding.weight"] = torch.eye(5)

    model.load_state_dict(state_dict)

    X = torch.randint(0, 5, size=(100, ))
    the_result = torch.fill(torch.empty(5), -torch.inf)
    for t_3 in range(X.shape[0]):
        for t_2 in range(t_3):
            for t_1 in range(t_2):
                result_ = (
                      v1[X[t_1]]
                    + v2[X[t_2]]
                    + v3[X[t_3]]
                )
                for i in range(5):
                    if result_[i] > the_result[i]:
                        the_result[i] = result_[i]

    result = model(X.unsqueeze(0))
    torch.testing.assert_close(result[0, -1, :], the_result)


if __name__ == "__main__":
    test_iterated_sum()

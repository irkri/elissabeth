import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from sainomore.models import Elissabeth, ElissabethConfig

config = {
    "n_samples": 10,
    "d_hidden": 5,

    "n_layers": 1,
    "length_is": 4,
}


def build_model(
    context_length: int,
    layer_norm: bool = True,
    normalize_is: bool = False,
) -> Elissabeth:
    model_config = ElissabethConfig(
        context_length=context_length,
        input_vocab_size=config["d_hidden"],
        n_layers=config["n_layers"],
        length_is=config["length_is"],
        d_hidden=config["d_hidden"],  # n_is = d_hidden
        layer_norm=layer_norm,
        normalize_is=normalize_is,
        denominator_is=False,
        single_query_key=False,
        positional_bias=False,
        distance_weighting=True,
        weighting=None,
        positional_encoding=None,
        input_type="vector",
    )
    model = Elissabeth(model_config)
    model.set_eye("unembedding.weight")

    for l in range(config["n_layers"]):
        # model.set_eye(f"layers.{l}.W_Q", requires_grad=True)
        # model.set_eye(f"layers.{l}.W_K", requires_grad=True)
        model.set_eye(f"layers.{l}.W_V", requires_grad=True)
        model.set_eye(f"layers.{l}.W_O", requires_grad=True)

    return model


def make_plot(norms, initializers, context_lengths):
    fig, ax = plt.subplots(
        len(initializers), len(context_lengths),
        figsize=(16, 9),
    )
    if len(initializers) == 1:
        ax = ax[np.newaxis, :]
    if len(context_lengths) == 1:
        ax = ax[:, np.newaxis]

    for i in range(len(initializers)):
        for j in range(len(context_lengths)):
            ax[i, j].plot(
                norms[i, j, 0].flatten(), "-x", label="No Normalization",
            )
            ax[i, j].plot(
                norms[i, j, 1].flatten(), "-x", label="ISNorm",
            )
            ax[i, j].plot(
                norms[i, j, 2].flatten(), "-x", label="LayerNorm",
            )
            ax[i, j].plot(
                norms[i, j, 3].flatten(), "-x", label="LayerNorm + ISNorm",
            )

            ax[i, j].tick_params(axis='x', which='major', labelsize=12)
            ax[i, j].tick_params(axis='x', which='minor', labelsize=0)

            ax[i, j].set_xticks(
                [l*config["length_is"]
                 for l in range(0, config["n_layers"]+1)]
            )
            ax[i, j].set_xticks(
                list(range(config["length_is"]*config["n_layers"])),
                minor=True,
            )
            ax[i, j].set_xticklabels(
                [f"Layer {l}" for l in range(config["n_layers"]+1)]
            )
            ax[i, j].vlines(
                [l*config["length_is"] for l in range(config["n_layers"]+1)],
                np.min(norms), np.max(norms),
                colors="black",  # type: ignore
                linestyles="dashed",
            )
            ax[i, j].set_yscale("log")
            ax[i, j].set_title(
                f"{initializers[i].__name__}, T={context_lengths[j]}"
            )
    ax[0, 0].legend(loc="best")

    return fig, ax


def calculate_norm():
    initializers = [
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
        # torch.nn.init.xavier_normal_,
    ]
    context_lengths = [10, 1000]
    norms = np.zeros(
        (len(initializers), len(context_lengths), 4,
         config["n_layers"], config["length_is"])
    )
    norms_grad = np.zeros(
        (len(initializers), len(context_lengths), 4,
         config["n_layers"], config["length_is"])
    )

    for il, context_length in enumerate(context_lengths):
        data = torch.empty(
            (config["n_samples"], context_length, config["d_hidden"])
        )
        for k, initializer in enumerate(initializers):
            initializer(data)
            c = 0
            for norm_layers in [False, True]:
                for norm_iss in [False, True]:
                    model = build_model(
                        context_length,
                        layer_norm=norm_layers,
                        normalize_is=norm_iss,
                    )

                    model.attach_all_hooks()

                    output = model(data)
                    loss = torch.nn.L1Loss()(
                        output,
                        torch.randn((config["n_samples"], config["d_hidden"],
                                     context_length)),
                    )
                    loss.backward()

                    for l in range(model.config.n_layers):
                        for i in range(model.config.length_is):
                            norms[k, il, c, l, i] = np.mean(np.linalg.norm(
                                model.get_hook(f"layers.{l}", f"iss.{i}").fwd,
                                axis=2,
                            ), axis=0)[-1]
                    for l in range(model.config.n_layers):
                        norms_grad[k, il, c, l, :] = np.mean(np.linalg.norm(
                            model.layers[l].W_V.grad.detach(),  # type: ignore
                            axis=2,
                        ), axis=1)
                    c += 1

                    model.release_all_hooks()

    fig1, _ = make_plot(norms, initializers, context_lengths)
    # fig2, _ = make_plot(norms_grad, initializers, context_lengths)

    fig1.suptitle(
        f"Norm of propagated input in Elissabeth\n"
        f"Samples: {config['n_samples']} | d_hidden: {config['d_hidden']}"
    )
    fig1.tight_layout()
    fig1.savefig(
        os.path.join(os.path.dirname(__file__), "norms.pdf"),
        bbox_inches="tight",
        facecolor=(0, 0, 0, 0),
    )

    # fig2.suptitle(
    #     f"Norm of gradient of W_Q in Elissabeth\n"
    #     f"Samples: {config['n_samples']} | d_hidden: {config['d_hidden']}"
    # )
    # fig2.tight_layout()
    # fig2.savefig(
    #     os.path.join(os.path.dirname(__file__), "norms_grad.pdf"),
    #     bbox_inches="tight",
    #     facecolor=(0, 0, 0, 0),
    # )

    plt.show()


def testcase():
    data = torch.ones((1, 1000, config["d_hidden"]))

    # torch.nn.init.ones_(data)
    model = build_model(
        context_length=1000,
        layer_norm=False,
        normalize_is=True,
    )

    model.attach_all_hooks(backward=False)

    output = model(data)
    loss = torch.nn.L1Loss()(
        output, torch.randint(0, 1, (1, config["d_hidden"], 1000))
    )
    loss.backward()

    for l in range(model.config.n_layers):
        # print(f"Layer {l}, Q:", np.linalg.norm(
        #     model.get_hook("layers.0", "Q").fwd[0, :, :],
        #     axis=2,
        # ))
        # print(f"Layer {l}, K:", np.linalg.norm(
        #     model.get_hook("layers.0", "Q").fwd[0, :, :],
        #     axis=2,
        # ))
        # print(f"Layer {l}, V:", np.linalg.norm(
        #     model.get_hook("layers.0", "Q").fwd[0, :, :],
        #     axis=2,
        # ))
        for i in range(model.config.length_is):
            print(f"Layer {l}, ISS {i}:", np.mean(np.linalg.norm(
                model.get_hook(f"layers.{l}", f"iss.{i}").fwd,
                axis=2,
            ), axis=0)[-1])
    # for l in range(model.config.n_layers):
    #     print("THIS", id(model.get_hook(f"layers.{l}", f"Q")))
    #     # print("Q", l, np.linalg.norm(model.get_hook(f"layers.{l}", "Q").bwd))
    #     print("V", l, np.linalg.norm(model.get_hook(f"layers.{l}", "Q").bwd))

    # plt.matshow(model.get_hook("layers.0", "weighting.0").fwd[0, :, :, 0])
    # plt.show()

    model.release_all_hooks()


if __name__ == "__main__":
    # calculate_norm()
    testcase()

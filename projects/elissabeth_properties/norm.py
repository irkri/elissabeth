import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from sainomore.models import Elissabeth, ElissabethConfig, SAINoMoreModule

config = {
    "n_samples": 100,
    "d_hidden": 2,

    "n_layers": 5,
    "iss_length": 10,
}


def build_model(
    initializer,
    context_length: int,
    normalize_layers: bool = True,
    normalize_iss: bool = False,
) -> tuple[SAINoMoreModule, ElissabethConfig]:
    model_config = ElissabethConfig(
        context_length=context_length,
        input_vocab_size=1,
        n_layers=config["n_layers"],
        iss_length=config["iss_length"],
        d_hidden=config["d_hidden"],
        single_query_key=False,
        normalize_iss=normalize_iss,
        normalize_layers=normalize_layers,
    )
    model = Elissabeth(model_config)

    initializer(model.embedding.weight)
    initializer(model.unembedding.weight)

    for l in range(config["n_layers"]):
        for i in range(config["iss_length"]):
            torch.nn.init.eye_(model.layers[l].W_Q[i])
            torch.nn.init.eye_(model.layers[l].W_K[i])
            torch.nn.init.eye_(model.layers[l].W_V[i])
        torch.nn.init.eye_(model.layers[l].W_O)

    return model, model_config


def calculate_norm():
    norms = np.zeros((2, 2, 4, config["n_layers"], config["iss_length"]))
    initializers = [torch.nn.init.uniform_, torch.nn.init.normal_]
    context_lengths = [10, 1000]

    for il, context_length in enumerate(context_lengths):
        data = torch.randint(0, 1, (config["n_samples"], context_length))
        for k, initializer in enumerate(initializers):
            c = 0
            for norm_layers in [False, True]:
                for norm_iss in [False, True]:
                    model, model_config = build_model(
                        initializer,
                        context_length,
                        normalize_layers=norm_layers,
                        normalize_iss=norm_iss,
                    )

                    model.attach_all_hooks()

                    output = model(data)

                    for l in range(model_config.n_layers):
                        for i in range(model_config.iss_length):
                            norms[k, il, c, l, i] = np.mean(np.linalg.norm(
                                model.get_hook(f"layers.{l}", f"iss.{i}").fwd,
                                axis=2,
                            ), axis=0)[-1]
                    c += 1

                    model.release_all_hooks()

    fig, ax = plt.subplots(
        len(initializers), len(context_lengths),
        figsize=(16, 9),
    )

    for i in range(len(initializers)):
        for j in range(len(context_lengths)):
            ax[i, j].plot(
                norms[i, j, 0].flatten(), "-x", label="no normalization",
            )
            ax[i, j].plot(
                norms[i, j, 1].flatten(), "-x", label="iss norm",
            )
            ax[i, j].plot(
                norms[i, j, 2].flatten(), "-x", label="layer norm",
            )
            ax[i, j].plot(
                norms[i, j, 3].flatten(), "-x", label="layer + iss norm",
            )

            ax[i, j].tick_params(axis='x', which='major', labelsize=12)
            ax[i, j].tick_params(axis='x', which='minor', labelsize=0)

            ax[i, j].set_xticks(
                [l*config["iss_length"]
                 for l in range(0, config["n_layers"]+1)]
            )
            ax[i, j].set_xticks(
                list(range(config["iss_length"]*config["n_layers"])),
                minor=True,
            )
            ax[i, j].set_xticklabels(
                [f"Layer {l}" for l in range(config["n_layers"]+1)]
            )
            ax[i, j].vlines(
                [l*config["iss_length"] for l in range(config["n_layers"]+1)],
                np.min(norms), np.max(norms),
                colors="black",  # type: ignore
                linestyles="dashed",
            )
            ax[i, j].set_yscale("log")
            ax[i, j].set_title(
                f"{initializers[i].__name__}, T={context_lengths[j]}"
            )
    ax[1, 1].legend(loc="best")

    fig.suptitle(
        f"Norm of propagated input in Elissabeth\n"
        f"Samples: {config['n_samples']} | d_hidden: {config['d_hidden']}"
    )
    fig.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "norms.pdf"),
        bbox_inches="tight",
        facecolor=(0, 0, 0, 0),
    )
    plt.show()


def testcase():
    data = torch.randint(0, 1, (1, 10))

    model, model_config = build_model(
        torch.nn.init.normal_,
        100,
        normalize_layers=False,
        normalize_iss=True,
    )

    model.attach_all_hooks()

    output = model(data)

    for l in range(model_config.n_layers):
        for i in range(model_config.iss_length):
            print(f"Layer {l}, ISS {i}:", np.mean(np.linalg.norm(
                model.get_hook(f"layers.{l}", f"iss.{i}").fwd,
                axis=2,
            ), axis=0)[-1])

    model.release_all_hooks()

if __name__ == "__main__":
    calculate_norm()
    # testcase()

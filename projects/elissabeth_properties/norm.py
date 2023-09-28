import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from sainomore.models import Elissabeth, ElissabethConfig, SAINoMoreModule

config = {
    "n_samples": 1_000,
    "context_length": 10,
    "d_hidden": 4,

    "n_layers": 10,
    "iss_length": 10,
}


def build_model(
    normalize_layers: bool = True,
    normalize_iss: bool = False,
) -> tuple[SAINoMoreModule, ElissabethConfig]:
    model_config = ElissabethConfig(
        context_length=config["context_length"],
        input_vocab_size=1,
        n_layers=config["n_layers"],
        iss_length=config["iss_length"],
        d_hidden=config["d_hidden"],
        single_query_key=False,
        normalize_iss=normalize_iss,
        normalize_layers=normalize_layers,
    )
    model = Elissabeth(model_config)

    torch.nn.init.normal_(model.embedding.weight)
    torch.nn.init.normal_(model.unembedding.weight)

    for l in range(config["n_layers"]):
        for i in range(config["iss_length"]):
            torch.nn.init.eye_(model.layers[l].W_Q[i])
            torch.nn.init.eye_(model.layers[l].W_K[i])
            torch.nn.init.eye_(model.layers[l].W_V[i])
        torch.nn.init.eye_(model.layers[l].W_O)

    return model, model_config


def calculate_norm():
    data = torch.randint(0, 1,
        (config["n_samples"], config["context_length"]),
    )

    norms = np.zeros((4, config["n_layers"], config["iss_length"]))

    c = 0
    for norm_layers in [False, True]:
        for norm_iss in [False, True]:
            model, model_config = build_model(
                normalize_layers=norm_layers,
                normalize_iss=norm_iss,
            )

            model.attach_all_hooks()

            output = model(data)

            for l in range(model_config.n_layers):
                for i in range(model_config.iss_length):
                    norms[c, l, i] = np.mean(np.linalg.norm(
                        model.get_hook(f"layers.{l}", f"iss.{i}").fwd, axis=2,
                    ), axis=0)[-1]
            c += 1

            model.release_all_hooks()

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    ax.plot(norms[0].flatten(), "-x", label="no normalization")
    ax.plot(norms[1].flatten(), "-x", label="iss norm")
    ax.plot(norms[2].flatten(), "-x", label="layer norm")
    ax.plot(norms[3].flatten(), "-x", label="layer + iss norm")

    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='x', which='minor', labelsize=0)

    ax.set_xticks(
        [i*config["iss_length"] for i in range(0, config["n_layers"]+1)]
    )
    ax.set_xticks(
        list(range(config["iss_length"]*config["n_layers"])),
        minor=True,
    )
    ax.set_xticklabels(
        [f"Layer {i}" for i in range(config["n_layers"]+1)]
    )
    ax.vlines(
        [i*config["iss_length"] for i in range(config["n_layers"]+1)],
        np.min(norms), np.max(norms),
        colors="black",  # type: ignore
        linestyles="dashed",
    )
    ax.set_title(
        "Norm of propagated normal distributed input in Elissabeth\n"
        f"Samples: {config['n_samples']} | "
        f"Context length: {config['context_length']} | "
        f"d_hidden: {config['d_hidden']}"
    )
    ax.legend(loc="best")
    ax.set_yscale("log")

    fig.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "norms.pdf"),
        bbox_inches="tight",
        facecolor=(0, 0, 0, 0),
    )
    plt.show()


if __name__ == "__main__":
    calculate_norm()

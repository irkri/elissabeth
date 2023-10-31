import argparse
import os
from typing import Optional

import numpy as np
import lightning.pytorch as L
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt
from torchmetrics.classification import MulticlassAccuracy

from data import LetterAssembler
from sainomore.callbacks import (ElissabethWeighting, GeneralConfigCallback,
                                 WeightHistory, ElissabethISTracker)
from sainomore.data import long_lookup
from sainomore.data.lightning import GivenDataModule
from sainomore.lightning import TokenPredictionModule
from sainomore.models import (DecoderOnlyTransformer,
                              DecoderOnlyTransformerConfig, Elissabeth,
                              ElissabethConfig)
from sainomore.tools import (get_liss_attention_matrix,
                             plot_liss_attention_matrix)

torch.set_float32_matmul_precision('high')

LOAD_PATH: Optional[str] = None
SAVE_PATH: Optional[str] = None

assembler = (
    LetterAssembler(os.path.join(os.path.dirname(__file__), "quotes.txt"))
)

config = {
    "context_length": assembler.context_length,
    "characters": assembler.vocab_size,

    "lr": 5e-3,
    "weight_decay": 1e-4,
    "epochs": 101,

    "batch_size": 32,
    "val_size": 0.05,
}


class PredictionCallback(Callback):

    def __init__(self, each_n_epochs: int = 1) -> None:
        super().__init__()
        self._each_n_epochs = each_n_epochs
        self._epoch = -1

    def on_train_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self._epoch = -1

    def on_training_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        a,b,c,
    ) -> None:
        self._epoch += 1
        if self._epoch % self._each_n_epochs != 0:
            return

        words = generate(pl_module)  # type: ignore
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.log_text(
                    key="samples",
                    columns=["Example"],
                    data=[[a] for a in words],
                )
                break
        else:
            print("\n".join(words))


def build_model() -> TokenPredictionModule:
    # model_config = DecoderOnlyTransformerConfig(
    #     context_length=config["context_length"],
    #     input_vocab_size=config["characters"],
    #     positional_encoding="sinusoidal",
    #     n_layers=1,
    #     n_heads=4,
    #     d_hidden=32,
    #     d_head=4,
    #     ffn_units=32,
    #     bias=True,
    # )
    # model = DecoderOnlyTransformer(model_config)

    model_config = ElissabethConfig(
        context_length=config["context_length"],
        input_vocab_size=config["characters"],
        n_layers=1,
        length_is=5,
        n_is=64,
        d_hidden=64,
        weighting="exp",
        positional_encoding=None,
        denominator_is=False,
        distance_weighting=True,
        positional_bias=True,
        positional_bias_values=False,
        single_query_key=False,
        share_queries=False,
        share_keys=False,
        share_values=False,
        normalize_is=True,
    )
    model = Elissabeth(model_config)
    # model.set_eye("embedding.weight")

    lightning_module = TokenPredictionModule(
        model,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        loss=torch.nn.CrossEntropyLoss(ignore_index=-1),
        accuracy=MulticlassAccuracy(config["characters"], ignore_index=-1),
        only_last=False,
    )
    return lightning_module


def train(
    lightning_module: TokenPredictionModule,
    use_wandb: bool = False,
    progress_bar: bool = False,
    only_test: bool = False,
) -> None:
    data_module = GivenDataModule(
        assembler.get_dataset(),
        val_size=config["val_size"],
        batch_size=config["batch_size"],
        num_workers=3,
        # somehow this option is important, atleast on CPU
        # (no more waiting between epochs)
        persistent_workers=True,
    )

    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            project="Elissabeth 3",
            checkpoint_name=LOAD_PATH,
            tags=["Elissabeth", "makemore_quotes"],
            id=LOAD_PATH.split("/")[1] if LOAD_PATH is not None else None,
            resume="must" if LOAD_PATH is not None else False,
        )

        wandb_logger.experiment.config.update(
            lightning_module.model.config.to_dict()
        )
        wandb_logger.experiment.config.update(config, allow_val_change=True)

        wandb_logger.watch(lightning_module, log="all")

    # example = long_lookup(
    #     n_samples=1,
    #     length=config["context_length"],
    #     characters=config["characters"],
    # )[0]
    # np.save(
    #     os.path.join(
    #         os.path.dirname(__file__), "data", "elissabeth_example.npy",
    #     ),
    #     example.numpy(),
    # )
    example = assembler.sample(2418)[0]
    callbacks = [
        GeneralConfigCallback(max_depth=10),
        PredictionCallback(each_n_epochs=10),
        # Progressbar(),
        WeightHistory((
                "model.layers.0.W_Q",
                "model.layers.0.W_K",
                "model.layers.0.E_K",
                "model.layers.0.W_V",
                # "model.layers.0.E_V",
                "model.layers.0.W_O",
                "model.layers.0.alpha",
                "model.embedding.weight",
                "model.unembedding.weight",
            ),
            reduce_axis=[0, 0, 2, 0, None, None, None, None],
            each_n_epochs=10,
        ),
        ElissabethISTracker(
            example,
            reduce="norm",
            each_n_epochs=5,
            use_wandb=True,
        )
        # ElissabethWeighting(
        #     example,
        #     each_n_epochs=100,
        #     save_path=os.path.join(os.path.dirname(__file__), "data"),
        # ),
    ]

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        callbacks=callbacks if not only_test else None,
        logger=wandb_logger if not only_test else None,
        default_root_dir=SAVE_PATH,
        enable_progress_bar=progress_bar,
    )

    if only_test:
        trainer.validate(lightning_module, data_module)
    else:
        trainer.fit(lightning_module, data_module, ckpt_path=LOAD_PATH)

    if use_wandb:
        wandb.finish()


def generate(lightning_module: TokenPredictionModule) -> list[str]:
    lightning_module.to("cuda")
    n_samples = 1

    start = torch.zeros((n_samples, 1)).long().to(lightning_module.device)

    # lightning_module.model.get_hook("layers.0", "iss.0").attach()
    for i in range(1, 100):#assembler.context_length):
        out = lightning_module.model(start)
        probabilities = torch.softmax(out[:, :, -1], dim=-1)
        # if torch.sum(torch.isnan(probabilities)) > 0:
        #     print(lightning_module.model.layers[0].W_Q)
        #     print(lightning_module.model.get_hook("layers.0", "iss.0").fwd)
        start = torch.cat(
            (start, torch.multinomial(probabilities, num_samples=1)),
            dim=1,
        )

    result = [assembler.translate(y) for y in start]
    return result


def plot(lightning_module: TokenPredictionModule) -> None:
    lightning_module.to("cuda")

    # print(lightning_module.model.layers[0].beta)
    # print(
    #     torch.min(lightning_module.model.layers[0].alpha),
    #     torch.max(lightning_module.model.layers[0].alpha),
    # )
    model = lightning_module.model

    x, y = assembler.sample(2418)
    x, y = x.to(lightning_module.device), y.to(lightning_module.device)
    # print(y[0, -100:])

    model.attach_all_hooks()
    metrics = lightning_module.training_step((x, y), 0)

    metrics["loss"].backward()

    gradient = lightning_module.get_parameter(
        "model.layers.0.W_V"
    ).grad.cpu().detach().numpy()
    print(np.max(gradient))
    # for l in range(model.config.n_layers):
    #     for i in range(model.config.length_is):
    #         print(f"Layer {l}, ISS {i}:", np.mean(np.linalg.norm(
    #             model.get_hook(f"layers.{l}", f"iss.{i}").fwd,
    #             axis=2,
    #         ), axis=0)[-1])

    # mat = get_liss_attention_matrix(lightning_module.model, x)
    # y_hat = lightning_module.predict_step(x, 0)
    # print(f"Input     : {x}\nTarget    : {y}\nPrediction: {y_hat}")
    # plot_liss_attention_matrix(
    #     mat[:, :, 0, :, :],
    #     example=x.cpu().numpy()[0],
    #     figsize=(50, 10),
    #     log_colormap=False,
    #     causal_mask=False,
    # )
    # plt.savefig(
    #     os.path.join(os.path.dirname(__file__), "plot.pdf"),
    #     bbox_inches="tight",
    #     facecolor=(0, 0, 0, 0),
    # )
    # plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=["train", "test", "plot", "generate"])
    parser.add_argument("--online", action="store_true")

    args = parser.parse_args()

    lightning_module = build_model()

    if LOAD_PATH is not None:
        saved_ = torch.load(LOAD_PATH)
        lightning_module.load_state_dict(saved_["state_dict"])

    if args.mode == "train":
        train(
            lightning_module,
            use_wandb=args.online,
            progress_bar=not args.online,
        )
    elif args.mode == "test":
        train(lightning_module, only_test=True)
    elif args.mode == "plot":
        plot(lightning_module)
    elif args.mode == "generate":
        print("\n".join(generate(lightning_module)))


if __name__ == '__main__':
    main()

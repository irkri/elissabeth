import argparse
import os
from typing import Optional
import json
from pathlib import Path

import lightning.pytorch as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt
from torchmetrics.classification import MulticlassAccuracy

from data import LetterAssembler
from sainomore.elissabeth import Weighting
from sainomore.callbacks import (ElissabethISTracker, ElissabethWeighting,
                                 GeneralConfigCallback, WeightHistory)
from sainomore.data.lightning import GivenDataModule
from sainomore.lightning import TokenPredictionModule
from sainomore import Elissabeth, LISSConfig
# from sainomore.tools import (get_liss_attention_matrix,
#                              plot_liss_attention_matrix)

torch.set_float32_matmul_precision('high')

SAVE_PATH: Optional[str] = None

assembler = (
    LetterAssembler(Path(__file__).parent / "quotes.txt")
)

config = {
    "context_length": assembler.context_length,
    "characters": assembler.vocab_size,

    "lr": 3e-5,
    "weight_decay": 1e-4,
    "epochs": 501,

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

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: TokenPredictionModule,
    ) -> None:
        self._epoch += 1
        if self._epoch % self._each_n_epochs != 0:
            return

        words = generate(pl_module, n_samples=5, max_length=None)
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
    with open("config.json", "r") as f:
        model_config = json.load(f)

    model_config["context_length"] = config["context_length"]
    model_config["input_vocab_size"] = config["characters"]

    model = Elissabeth.build(
        model_config,
        Weighting.ComplexExponential,
        # Weighting.ExponentialDecay,
    )

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
    load_path: Optional[str] = None,
    use_wandb: bool = False,
    progress_bar: bool = False,
    only_test: bool = False,
) -> None:
    data_module = GivenDataModule(
        assembler.get_dataset(),
        val_size=config["val_size"],
        batch_size=config["batch_size"],
        num_workers=5,
        # somehow this option is important, atleast on CPU
        # (no more waiting between epochs)
        persistent_workers=True,
    )

    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            project="Elissabeth Makemore",
            checkpoint_name=load_path,
            tags=[],
            id=load_path.split("/")[-2] if load_path is not None else None,
            resume="must" if load_path is not None else False,
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
        PredictionCallback(each_n_epochs=25),
        ModelCheckpoint(monitor="validation/loss", mode="min"),
        # WeightHistory((
        #         ("model.layers.0.W_Q", (2, 0)),
        #         ("model.layers.0.b_Q", (0, )),
        #         ("model.layers.0.W_K", (2, 0)),
        #         ("model.layers.0.b_K", (0, )),
        #         # "model.layers.0.W_V",
        #         # "model.layers.0.b_V",
        #         # "model.layers.0.W_O",
        #         # ("model.layers.0.alpha", (1, 2)),
        #         "model.embedding.weight",
        #         "model.unembedding.weight",
        #     ),
        #     each_n_epochs=25,
        # ),
        # ElissabethISTracker(
        #     example,
        #     reduce="norm",
        #     each_n_epochs=25,
        #     use_wandb=True,
        # ),
    ]

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        callbacks=callbacks if not only_test else None,
        logger=wandb_logger if not only_test else None,
        default_root_dir=SAVE_PATH,
        enable_progress_bar=progress_bar,
        detect_anomaly=True
    )

    if only_test:
        trainer.validate(lightning_module, data_module)
    else:
        trainer.fit(lightning_module, data_module, ckpt_path=load_path)

    if use_wandb:
        wandb.finish()


def generate(
    lightning_module: TokenPredictionModule,
    n_samples: int = 5,
    max_length: Optional[int] = None,
) -> list[str]:
    lightning_module.to("cuda")

    start = torch.zeros((n_samples, 1)).long().to(lightning_module.device)
    T = assembler.context_length if max_length is None else max_length
    for _ in range(1, T):
        out = lightning_module.model(start)
        probabilities = torch.softmax(out[:, -1, :], dim=-1)
        start = torch.cat(
            (start, torch.multinomial(probabilities, num_samples=1)),
            dim=1,
        )

    result = [assembler.translate(y) for y in start]
    return result


def plot(lightning_module: TokenPredictionModule) -> None:
    lightning_module.to("cuda")

    print(lightning_module.model.layers[0].alpha)
    print(
        torch.min(lightning_module.model.layers[0].alpha),
        torch.max(lightning_module.model.layers[0].alpha),
    )
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
    parser.add_argument("--load", default=None)

    args = parser.parse_args()

    lightning_module = build_model()

    load_path = None
    if args.load is not None:
        directory = Path.cwd()
        for folder in directory.iterdir():
            if not (folder.is_dir() and (folder / args.load).is_dir()):
                continue
            if (load_path := (folder / args.load / "checkpoints")).exists():
                load_path = load_path / next(load_path.iterdir())
                saved_ = torch.load(
                    load_path,
                    # map_location=torch.device("cpu"),
                )
                lightning_module.load_state_dict(saved_["state_dict"])
    if args.load is not None and load_path is None:
        raise FileExistsError(
            "Load specification does not point to a saved model"
        )

    if args.mode == "train":
        train(
            lightning_module,
            load_path=load_path,
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

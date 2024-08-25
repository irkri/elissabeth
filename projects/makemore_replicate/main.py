import argparse
import json
from pathlib import Path
from typing import Optional

import lightning.pytorch as L
import torch
import wandb
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics.classification import MulticlassAccuracy

from sainomore import Elissabeth
from sainomore.callbacks import GeneralConfigCallback
from sainomore.data import GivenDataModule
from sainomore.lightning import TokenPredictionModule

from data import LetterAssembler

torch.set_float32_matmul_precision('high')

assembler = (
    LetterAssembler(Path(__file__).parent / "quotes.txt")
)

config = {
    "context_length": assembler.context_length,
    "characters": assembler.vocab_size,

    "lr": 5e-3,
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

    model = Elissabeth.build(model_config)
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

    callbacks = [
        GeneralConfigCallback(max_depth=10),
        PredictionCallback(each_n_epochs=25),
        ModelCheckpoint(monitor="validation/loss", mode="min"),
    ]

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        callbacks=callbacks if not only_test else None,
        logger=wandb_logger if not only_test else None,
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
    n_samples: int = 2,
    max_length: Optional[int] = None,
) -> list[str]:
    # lightning_module.to("cuda")

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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=["train", "test", "generate"])
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
    elif args.mode == "generate":
        print("\n".join(generate(lightning_module)))


if __name__ == '__main__':
    main()

import argparse
import json
from pathlib import Path
from typing import Optional

import lightning.pytorch as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics.classification import MulticlassAccuracy

from data import lookup
from sainomore.callbacks import GeneralConfigCallback, ModelCheckpoint
from sainomore.data import GivenDataModule
from sainomore.elissabeth import Elissabeth
from sainomore.lightning import TokenPredictionModule

torch.set_float32_matmul_precision('high')

config = {
    "n_samples": 1000,
    "context_length": 25,
    "characters": 5,
    "multiple_keys": False,

    "lr": 5e-3,
    "weight_decay": 1e-4,
    "epochs": 10001,

    "batch_size": 64,
    "val_size": 0.2,
}


class LookUpElissabeth(Elissabeth):
    """Elissabeth, but the output is subtracted by a shifted version of
    the input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = super().forward(x)
        logits = (logits
            - torch.nn.functional.pad(
                self.embedding(x)[:, :-1], (0, 0, 1, 0)
            )
        )
        return logits


def build_model(l: int | None = None) -> TokenPredictionModule:
    with open("config.json", "r") as f:
        model_config = json.load(f)

    model_config["context_length"] = (
        config["context_length"] if l is None else l
    )
    model_config["input_vocab_size"] = config["characters"]

    model = LookUpElissabeth.build(model_config)
    model = implant_exact_solution(model, fix_parameters=True)

    lightning_module = TokenPredictionModule(
        model,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        loss=torch.nn.CrossEntropyLoss(),
        accuracy=MulticlassAccuracy(config["characters"]),
        only_last=True,
    )
    return lightning_module


def implant_exact_solution(
    model: Elissabeth,
    fix_parameters: bool = False,
) -> Elissabeth:
    state_dict = model.state_dict()

    state_dict["embedding.weight"] = torch.eye(5)

    state_dict["layers.0.levels.0.P_V.transform.weight"] = torch.Tensor([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],

        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ])

    state_dict["layers.0.W_O"] = torch.Tensor([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]).unsqueeze(1)
    state_dict["unembedding.weight"] = torch.eye(5)

    state_dict["layers.0.levels.0.weightings.1.alpha"] = torch.Tensor([[
        [100, 0]
    ]]).unsqueeze(-1).unsqueeze(-1)

    d = torch.pi / 2
    state_dict["layers.0.levels.0.weightings.0.P_Q.transform.weight"] = (
        torch.Tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],

            [0, d, 0, 0, d],
            [0, 0, d, 0, d],
            [0, 0, 0, d, 0],
        ])
    )
    state_dict["layers.0.levels.0.weightings.0.P_K.transform.weight"] = (
        torch.Tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],

            [0, d, 0, 0, d],
            [0, 0, d, 0, d],
            [0, 0, 0, d, 0],
        ])
    )
    model.load_state_dict(state_dict)

    if fix_parameters:
        for name in [
            "layers.0.levels.0.weightings.0.P_Q.transform.weight",
            "layers.0.levels.0.weightings.0.P_Q.transform.bias",
            "layers.0.levels.0.weightings.0.P_K.transform.weight",
            "layers.0.levels.0.weightings.0.P_K.transform.bias",
            "layers.0.levels.0.P_V.transform.weight",
            "layers.0.levels.0.P_V.transform.bias",
            "layers.0.W_O",
            "layers.0.W_H",
            "embedding.weight",
            "unembedding.weight",
            "layers.0.levels.0.weightings.1.alpha",
        ]:
            model.get_parameter(name).requires_grad = False
    return model


def train(
    lightning_module: TokenPredictionModule,
    use_wandb: bool = False,
    load_path: Optional[Path] = None,
    progress_bar: bool = False,
    only_test: bool = False,
) -> None:
    data_module = GivenDataModule(
        lookup(
            n_samples=config["n_samples"],
            length=config["context_length"],
            characters=config["characters"],
            multiple_keys=config["multiple_keys"],
        ),
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
            project="Elissabeth Long Lookup 2",
            checkpoint_name=load_path,
            tags=["one key"],
            id=load_path.parts[-3] if load_path is not None else None,
            resume="must" if load_path is not None else False,
        )

        wandb_logger.experiment.config.update(
            lightning_module.model.get_config()
        )
        wandb_logger.experiment.config.update(config, allow_val_change=True)

        wandb_logger.watch(lightning_module, log="all")

    callbacks: list[Callback] = [
        GeneralConfigCallback(),
        ModelCheckpoint(monitor="validation/loss", mode="min"),
    ]

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        callbacks=callbacks if not only_test else None,
        logger=wandb_logger if not only_test else None,
        enable_progress_bar=progress_bar,
    )

    if only_test:
        trainer.validate(lightning_module, data_module)
    else:
        trainer.fit(lightning_module, data_module, ckpt_path=load_path)

    if use_wandb:
        wandb.finish()


def battery(lightning_module: TokenPredictionModule) -> None:
    lengths = [10, 25, 100, 150, 200, 250, 500, 750, 1000, 10_000]
    samples = 5
    trainit = True
    multiple_keys = False

    accuracy = np.zeros((len(lengths), samples, 2))
    trainer = L.Trainer(max_epochs=500)
    for l in range(len(lengths)):
        print(f"Starting Length {lengths[l]}", end="", flush=True)
        lightning_module = build_model(lengths[l])
        if trainit:
            data_module = GivenDataModule(
                lookup(
                    n_samples=1000,
                    length=lengths[l],
                    characters=config["characters"],
                    multiple_keys=multiple_keys,
                ),
                val_size=0.0,
                batch_size=config["batch_size"],
                num_workers=5,
                # somehow this option is important, atleast on CPU
                # (no more waiting between epochs)
                persistent_workers=True,
            )
            trainer.fit(
                lightning_module,
                train_dataloaders=data_module,
            )
        for i in range(samples):
            print(".", end="", flush=True)
            data_module = GivenDataModule(
                lookup(
                    n_samples=100,
                    length=lengths[l],
                    characters=config["characters"],
                    multiple_keys=multiple_keys,
                ),
                val_size=1.0,
                batch_size=config["batch_size"],
                num_workers=5,
                # somehow this option is important, atleast on CPU
                # (no more waiting between epochs)
                persistent_workers=True,
            )
            out = trainer.validate(
                lightning_module,
                data_module,
                verbose=False,
            )[0]
            accuracy[l, i, 0] = out["validation/loss"]
            accuracy[l, i, 1] = out["validation/accuracy"]
        pd.DataFrame(
            accuracy[:(l+1), :, 0].T,
            columns=[f"Length {k}" for k in lengths[:(l+1)]],
        ).to_csv("battery_loss.csv")
        pd.DataFrame(
            accuracy[:(l+1), :, 1].T,
            columns=[f"Length {k}" for k in lengths[:(l+1)]],
        ).to_csv("battery_accuracy.csv")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=["train", "test", "battery"])
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
    elif args.mode == "battery":
        battery(lightning_module)


if __name__ == '__main__':
    main()

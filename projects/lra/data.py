"""Inspired by the equivalent file in the S4 github repository:

    https://github.com/state-spaces/s4/blob/main/src/dataloaders/lra.py
"""
from pathlib import Path
from typing import Callable

import lightning.pytorch as L
import torch
import torchvision
from einops.layers.torch import Rearrange, Reduce
from PIL import Image  # Only used for Pathfinder
from torch.utils.data import DataLoader


class PathFinderDataset(torch.utils.data.Dataset):
    """Path Finder dataset."""

    # There's an empty file in the dataset
    blacklist = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}

    def __init__(
        self,
        data_dir: Path,
        transform: Callable | None = None,
    ) -> None:
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be
                applied on a sample.
        """
        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir(), (
            f"data_dir {str(self.data_dir)} does not exist"
        )
        self.transform = transform
        samples = []
        for diff_level in ["curv_contour_length_14"]:
            path_list = sorted(
                list((self.data_dir / diff_level / "metadata").glob("*.npy")),
                key=lambda path: int(path.stem),
            )
            assert path_list, "No metadata found"
            for metadata_file in path_list:
                with open(metadata_file, "r") as f:
                    for metadata in f.read().splitlines():
                        metadata = metadata.split()
                        image_path = (
                            Path(diff_level) / metadata[0] / metadata[1]
                        )
                        if (
                            str(Path(self.data_dir.stem) / image_path)
                            not in self.blacklist
                        ):
                            label = int(metadata[3])
                            samples.append((image_path, label))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Image.Image, torch.Tensor]:
        path, target = self.samples[idx]
        with open(self.data_dir / path, "rb") as f:
            sample = Image.open(f).convert("L")  # Open in grayscale
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class PathFinder(L.LightningDataModule):


    def __init__(
        self,
        data_dir: Path,
        val_size: float = 0.2,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.val_size = val_size
        self.batch_size = batch_size
        self._kwargs = kwargs
        self.resolution = 32
        self.sequential = True
        self.tokenize = False
        self.center = False
        self.pool = 1
        self.seed = 42

    def default_transforms(self) -> Callable:
        transform_list = [torchvision.transforms.ToTensor()]
        if self.pool > 1:
            transform_list.append(
                Reduce(
                    "1 (h h2) (w w2) -> 1 h w",
                    "mean",
                    h2=self.pool,
                    w2=self.pool,
                )
            )
        if self.tokenize:
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: (x * 255).long())
            )
        else:
            if self.center:
                transform_list.append(
                    torchvision.transforms.Normalize(mean=0.5, std=0.5)
                )
        if self.sequential:
            transform_list.append(
                Rearrange("1 h w -> (h w)")
                if self.tokenize
                else Rearrange("1 h w -> (h w) 1")
            )
        else:
            transform_list.append(Rearrange("1 h w -> h w 1"))
        return torchvision.transforms.Compose(transform_list)

    def prepare_data(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"""
            Directory {str(self.data_dir)} not found.
            To get the dataset, download lra_release.gz from
            https://github.com/google-research/long-range-arena,
            then unzip it with tar -xvf lra_release.gz.
            Then point data_dir to the pathfinderX directory, where X is
            either 32, 64, 128, or 256.
            """
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            **self._kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            **self._kwargs,
        )

    def setup(self, stage: str | None = None) -> None:
        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")
        dataset = PathFinderDataset(
            self.data_dir,
            transform=self.default_transforms(),
        )
        len_dataset = len(dataset)
        val_len = int(self.val_size * len_dataset)
        train_len = len_dataset - val_len
        (
            self.dataset_train,
            self.dataset_val,
        ) = torch.utils.data.random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(self.seed),
        )

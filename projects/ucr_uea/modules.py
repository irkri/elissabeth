import torch


class Preparateur(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = torch.nn.functional.pad(x[:, 1:]-x[:, :-1], (0, 0, 1, 0))
        x = torch.cat(
            (
                x,
                dx,
                x ** 2,
                dx ** 2,
                x ** 3,
                dx ** 3,
                x ** 4,
                dx ** 4,
                x ** 5,
                dx ** 5,
            ),
            dim=-1,
        )
        return x

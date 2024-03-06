import torch

from sainomore.positional import RoPE


def test_output() -> None:
    rope = RoPE(context_length=4, d_hidden=3)

    x = torch.arange(12).reshape(1, 4, 3)

    y = rope(x)
    torch.testing.assert_close(
        torch.Tensor([[
            [ 0.0000,  1.0000,  2.0000],
            [-4.0000, 3.0000,  5.0000],
            [-6.0000, -7.0000,  8.0000],
            [10.0000, -9.0000, 11.0000],
        ]]),
        y,
    )


if __name__ == "__main__":
    test_output()

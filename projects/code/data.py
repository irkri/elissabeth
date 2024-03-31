from typing import Callable

from scipy.integrate import solve_ivp
import torch
import numpy as np


class CODEDataset:

    def __init__(
        self,
        fs: Callable[[np.ndarray], np.ndarray],
        T: int = 100,
        d_u: int = 2,
        d_x: int = 3,
        std: float = 1,
    ) -> None:
        self.fs = fs
        self.T = T
        self.d_u = d_u
        self.d_x = d_x
        self._std = std

    def _get_control(self) -> np.ndarray:
        u = np.random.normal(0, self._std, size=(self.T, self.d_u))
        u = np.cumsum(u, axis=0)
        return u

    def _solve_ode(self, u: np.ndarray) -> np.ndarray:
        du = np.zeros((u.shape[0], u.shape[1]+1))
        du[1:, :2] = u[1:] - u[:-1]
        du[:, 2] = np.arange(self.T)
        def fun(t, x):
            t = int(np.ceil(t))
            return np.sum(self.fs(x) * du[t], axis=-1)

        ex = solve_ivp(
            fun,
            t_span=(0, self.T-1),
            y0=np.array([0, 0, 0], dtype=np.double),
            t_eval=np.arange(0, self.T),
            method="RK45",
        )
        return np.swapaxes(ex.y, 0, 1)

    def get_dataset(
        self,
        N: int,
        verbose: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.empty((N, self.T, self.d_u))
        y = np.empty((N, self.T, self.d_x))

        for i in range(N):
            if verbose is not None and i % verbose == 0:
                print(f"Generated {(i/N*100):.2f}%", flush=True)
            x[i] = self._get_control()
            y[i] = self._solve_ode(x[i])

        return torch.Tensor(x), torch.Tensor(y)


if __name__ == "__main__":
    def f(x):
        return np.stack((
            np.stack((
                np.cos(x[..., 0]),
                0.05*x[..., 2],
                0.1*x[..., 1] - 0.25*x[..., 0],
            ), axis=-1),
            np.stack((
                np.sin(x[..., 0]) - np.cos(x[..., 1]),
                0.1*x[..., 0] - np.sin(x[..., 2]),
                -0.05*x[..., 1],
            ), axis=-1),
            np.stack((
                -0.005*x[..., 0],
                -0.005*x[..., 1],
                -0.005*x[..., 2],
            ), axis=-1),
        ), axis=-1)

    np.random.seed(62)
    code_ds = CODEDataset(f, T=100, d_u=2, d_x=3, std=1)
    dataset = code_ds.get_dataset(1000, verbose=50)

    torch.save(dataset[0], "code_u.pt")
    torch.save(dataset[1], "code_x.pt")

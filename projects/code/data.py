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
        u = np.pad(u, ((1, 0), (0, 0)))
        du = u[1:] - u[:-1]
        def fun(t, x):
            t = int(np.ceil(t))
            return np.sum(self.fs(x) * du[t])

        ex = solve_ivp(
            fun,
            t_span=(0, self.T-1),
            y0=np.array([0, 0, 0], dtype=np.double),
            t_eval=np.arange(0, self.T),
        )
        return np.swapaxes(ex.y, 0, 1)

    def get_dataset(self, N: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.empty((N, self.T, self.d_u))
        y = torch.empty((N, self.T, self.d_x))

        for i in range(N):
            u = self._get_control()
            y[i] = torch.from_numpy(self._solve_ode(u))
            x[i] = torch.from_numpy(u)

        return x, y


if __name__ == "__main__":
    def f(x):
        return np.stack((
            np.stack((
                np.cos(x[..., 0])*x[..., 1]**2 + 5*x[..., 2],
                x[..., 2],
                np.sqrt(1+x[..., 1]**2)
            ), axis=-1),
            np.stack((
                np.sin(x[..., 0])*x[..., 2] + np.cos(x[..., 1]),
                x[..., 0] + np.sin(x[..., 2]),
                2*x[..., 1]
            ), axis=-1),
        ), axis=-1)

    ds = CODEDataset(f, T=10, d_u=2, d_x=3, std=1)
    print(ds.get_dataset(10))

import torch
import plotly.graph_objects as go
import numpy as np



def main(context_length: int, d: int) -> None:
    position = torch.arange(context_length).unsqueeze(-1)
    div_term = torch.exp(
        torch.arange(0, d, 2) * (-np.log(1000) / d)
    )
    pe = torch.zeros(context_length, d)
    pe[:, 0::2] = torch.sin(position * div_term)
    if d % 2 != 0:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)

    fig = go.Figure(
        [go.Heatmap(z=pe.T, colorscale="Viridis")],
    )
    fig.show()


if __name__ == '__main__':
    main(1000, 128)

"""The network: a small multilayer perceptron that predicts bikes rented."""

import torch
from torch import nn


class BikeNet(nn.Module):
    """Fully connected layers with ReLU, ending in one value with no activation.

    The target is a count, not a probability, so the last layer is a plain
    linear output. forward() returns shape (N,) so that it matches the targets.
    """

    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...] = (64, 64)):
        super().__init__()
        if len(hidden_sizes) < 2:
            raise ValueError("BikeNet needs at least two hidden layers")
        layers: list[nn.Module] = []
        width = n_features
        for size in hidden_sizes:
            layers.append(nn.Linear(width, size))
            layers.append(nn.ReLU())
            width = size
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)

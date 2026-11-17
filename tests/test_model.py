"""The network: its shape, its layers and one optimizer step."""

import pytest
import torch
from torch import nn

from model import BikeNet
from train import run_epoch


@pytest.mark.parametrize("hidden_sizes", [(8, 8), (64, 64), (128, 64, 32)])
def test_network_returns_one_value_per_row(hidden_sizes):
    model = BikeNet(n_features=8, hidden_sizes=hidden_sizes)
    output = model(torch.randn(10, 8))
    assert output.shape == (10,)
    assert output.dtype == torch.float32


def test_network_has_relu_hidden_layers_and_a_plain_output():
    model = BikeNet(n_features=8, hidden_sizes=(16, 16))
    modules = list(model.layers)
    assert sum(isinstance(m, nn.ReLU) for m in modules) == 2
    assert isinstance(modules[-1], nn.Linear)
    assert modules[-1].out_features == 1

    with pytest.raises(ValueError):
        BikeNet(n_features=8, hidden_sizes=(16,))


def test_one_optimizer_step_changes_every_layer(small_batch):
    torch.manual_seed(0)
    features, targets = small_batch
    model = BikeNet(features.shape[1], (32, 32))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    run_epoch(model, [(features, targets)], nn.MSELoss(), optimizer)

    for name, parameter in model.named_parameters():
        assert not torch.equal(before[name], parameter.detach()), name

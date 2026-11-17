"""The loop: it learns, it clears gradients, it measures without training."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import BikeNet
from prepare import RentalsDataset
from train import evaluate_baselines, fit, run_epoch


def test_loop_memorizes_one_small_batch(small_batch):
    torch.manual_seed(0)
    features, targets = small_batch
    batch = [(features, targets)]
    model = BikeNet(features.shape[1], (64, 64))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    start_loss, _ = run_epoch(model, batch, loss_fn)
    for _ in range(300):
        run_epoch(model, batch, loss_fn, optimizer)
    end_loss, _ = run_epoch(model, batch, loss_fn)

    assert end_loss < 0.05 * start_loss


def test_gradients_are_cleared_before_every_step(small_batch):
    # With a learning rate of zero the weights never move, so after two
    # identical batches the stored gradient must equal the gradient of one
    # batch. If zero_grad() were missing it would be twice as large.
    torch.manual_seed(0)
    features, targets = small_batch
    model = BikeNet(features.shape[1], (16, 16))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    run_epoch(model, [(features, targets)] * 2, loss_fn, optimizer)
    after_loop = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    loss_fn(model(features), targets).backward()
    one_batch = [p.grad for p in model.parameters()]

    for looped, single in zip(after_loop, one_batch, strict=True):
        assert torch.allclose(looped, single, rtol=1e-5, atol=1e-6)


def test_validation_epoch_leaves_weights_unchanged(prepared):
    torch.manual_seed(0)
    model = BikeNet(prepared.n_features, (32, 32))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loader = DataLoader(RentalsDataset(prepared.x_val, prepared.y_val), batch_size=64)

    run_epoch(model, loader, nn.MSELoss())

    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value), key
    assert all(p.grad is None for p in model.parameters())


def test_prediction_of_shape_n_by_1_is_rejected(small_batch):
    class WrongShape(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def forward(self, x):
            return self.linear(x)  # (N, 1) instead of (N,)

    features, targets = small_batch
    with pytest.raises(ValueError, match="shape"):
        run_epoch(WrongShape(features.shape[1]), [(features, targets)], nn.MSELoss())


def test_fit_logs_every_epoch_and_restores_best_weights(prepared):
    torch.manual_seed(0)
    train_loader = DataLoader(
        RentalsDataset(prepared.x_train[:512], prepared.y_train[:512]),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    val_loader = DataLoader(
        RentalsDataset(prepared.x_val[:256], prepared.y_val[:256]), batch_size=64
    )
    model = BikeNet(prepared.n_features, (32, 32))

    history, best_epoch = fit(model, train_loader, val_loader, epochs=6, lr=5e-2)

    assert len(history) == 6
    assert [row["epoch"] for row in history] == list(range(1, 7))
    for row in history:
        assert {"epoch", "train_loss", "val_loss", "val_mae"} <= row.keys()

    val_losses = [row["val_loss"] for row in history]
    assert best_epoch == val_losses.index(min(val_losses)) + 1
    restored_loss, _ = run_epoch(model, val_loader, nn.MSELoss())
    assert restored_loss == pytest.approx(min(val_losses), rel=1e-5)


def test_mean_baseline_predicts_the_training_mean(prepared):
    results = evaluate_baselines(prepared)
    assert results["train_mean"]["value"] == pytest.approx(prepared.y_train.mean())
    assert results["linear"]["val"]["mae"] < results["train_mean"]["val"]["mae"]

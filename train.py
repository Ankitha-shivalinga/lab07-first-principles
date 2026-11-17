"""Train the bike-demand network with a hand-written loop and write reports/.

Run `python train.py` from the repository root. It recreates every file in
reports/: the data plot, the baselines, the main run, the experiments, the
learning curves and the single test result.
"""

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch import nn

from bikes_data import TARGET, make_rentals
from model import BikeNet
from prepare import SEED, Prepared, make_loaders, prepare, split_frame

REPORTS = Path(__file__).resolve().parent / "reports"


@dataclass(frozen=True)
class RunConfig:
    """One training run. Each experiment changes exactly one field from MAIN."""

    name: str
    changed: str
    lr: float = 1e-3
    hidden_sizes: tuple[int, ...] = (64, 64)
    batch_size: int = 64
    epochs: int = 150


MAIN = RunConfig(name="main", changed="nothing (main run)")
EXPERIMENTS = [
    RunConfig(name="lr_too_small", changed="learning rate 1e-3 -> 1e-5", lr=1e-5),
    RunConfig(name="lr_too_large", changed="learning rate 1e-3 -> 1e-1", lr=1e-1),
    RunConfig(
        name="too_wide",
        changed="hidden width 64 -> 1024",
        hidden_sizes=(1024, 1024),
    ),
]


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch so every run is repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one pass over the loader.

    With an optimizer the model trains; without one it only measures, with
    gradients switched off. Returns the mean loss and the mean absolute error
    in rentals, both averaged over rows.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_abs_error = 0.0
    n_rows = 0
    with torch.set_grad_enabled(training):
        for features, targets in loader:
            predictions = model(features)
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"prediction shape {tuple(predictions.shape)} does not match "
                    f"target shape {tuple(targets.shape)}"
                )
            loss = loss_fn(predictions, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_rows = len(targets)
            total_loss += loss.item() * batch_rows
            total_abs_error += (predictions - targets).abs().sum().item()
            n_rows += batch_rows
    return total_loss / n_rows, total_abs_error / n_rows


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    verbose: bool = False,
) -> tuple[list[dict], int]:
    """Train for many epochs and put back the weights with the lowest val loss.

    Returns one history row per epoch and the best epoch (1-based).
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose and (epoch == 1 or epoch % 25 == 0):
            print(
                f"  epoch {epoch:4d}  train {train_loss:9.2f}  "
                f"val {val_loss:9.2f}  val MAE {val_mae:6.2f}"
            )

    model.load_state_dict(best_state)
    return history, best_epoch


def predict(model: nn.Module, loader) -> np.ndarray:
    """Predictions for every row of a loader, in loader order."""
    model.eval()
    with torch.no_grad():
        return torch.cat([model(features) for features, _ in loader]).numpy()


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def evaluate_baselines(prepared: Prepared) -> dict[str, dict]:
    """The training mean and a linear regression, scored on the same rows."""
    train_mean = float(prepared.y_train.mean())
    linear = LinearRegression().fit(prepared.x_train, prepared.y_train)
    results: dict[str, dict] = {
        "train_mean": {"value": train_mean},
        "linear": {},
    }
    for split, x, y in [
        ("val", prepared.x_val, prepared.y_val),
        ("test", prepared.x_test, prepared.y_test),
    ]:
        results["train_mean"][split] = score(y, np.full_like(y, train_mean))
        results["linear"][split] = score(y, linear.predict(x))
    return results


def train_run(
    config: RunConfig, prepared: Prepared
) -> tuple[nn.Module, list[dict], int]:
    """Train one configuration from the same seed."""
    set_seed(SEED)
    loaders = make_loaders(prepared, batch_size=config.batch_size, seed=SEED)
    model = BikeNet(prepared.n_features, config.hidden_sizes)
    history, best_epoch = fit(
        model,
        loaders["train"],
        loaders["val"],
        epochs=config.epochs,
        lr=config.lr,
        verbose=True,
    )
    return model, history, best_epoch


def plot_rentals_by_hour(train_rows: pd.DataFrame, path: Path) -> None:
    """Rentals against the hour of the day, on the training rows only."""
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    jitter = rng.uniform(-0.25, 0.25, len(train_rows))
    ax.scatter(
        train_rows["hour"] + jitter,
        train_rows[TARGET],
        s=4,
        alpha=0.25,
        color="grey",
        label="one station-hour",
    )
    for weekend, label, colour in [
        (0, "weekday mean", "C0"),
        (1, "weekend mean", "C1"),
    ]:
        rows = train_rows[train_rows["is_weekend"] == weekend]
        means = rows.groupby("hour")[TARGET].mean()
        ax.plot(means.index, means.values, color=colour, linewidth=2.5, label=label)
    ax.set_xlabel("hour of the day")
    ax.set_ylabel("bikes rented")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Rentals by hour (training rows): two commute peaks, not a line")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(history: list[dict], best_epoch: int, title: str, path: Path) -> None:
    """Training and validation loss per epoch, with the best epoch marked."""
    frame = pd.DataFrame(history)
    fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_loss.plot(frame["epoch"], frame["train_loss"], label="training loss")
    ax_loss.plot(frame["epoch"], frame["val_loss"], label="validation loss")
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("MSE (rentals squared, log scale)")
    ax_mae.plot(frame["epoch"], frame["train_mae"], label="training MAE")
    ax_mae.plot(frame["epoch"], frame["val_mae"], label="validation MAE")
    ax_mae.set_ylabel("MAE (rentals)")
    for ax in (ax_loss, ax_mae):
        if best_epoch > 0:
            ax.axvline(best_epoch, color="k", linestyle="--", linewidth=1)
            ax.annotate(
                f"best epoch {best_epoch}",
                xy=(best_epoch, 0.95),
                xycoords=("data", "axes fraction"),
                ha="right",
                fontsize=8,
            )
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_experiments(runs: dict[str, list[dict]], path: Path) -> None:
    """Validation loss of every run on one set of axes."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, history in runs.items():
        frame = pd.DataFrame(history)
        ax.plot(frame["epoch"], frame["val_loss"], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MSE (log scale)")
    ax.set_title("Validation loss for every run")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    plt.switch_backend("Agg")
    REPORTS.mkdir(exist_ok=True)

    frame = make_rentals()
    splits = split_frame(frame, seed=SEED)
    prepared = prepare(splits)
    print(
        f"rows: train {len(splits.train)}, val {len(splits.val)}, "
        f"test {len(splits.test)}; features {prepared.n_features}"
    )
    plot_rentals_by_hour(splits.train, REPORTS / "rentals_by_hour.png")

    baselines = evaluate_baselines(prepared)
    print(
        f"baselines (val MAE): mean {baselines['train_mean']['val']['mae']:.2f}, "
        f"linear {baselines['linear']['val']['mae']:.2f}"
    )

    # Train every run. Choose the final network on validation rows only.
    models: dict[str, nn.Module] = {}
    histories: dict[str, list[dict]] = {}
    rows = []
    for config in [MAIN, *EXPERIMENTS]:
        print(f"run {config.name}: {config.changed}")
        model, history, best_epoch = train_run(config, prepared)
        models[config.name] = model
        histories[config.name] = history
        best = history[best_epoch - 1] if best_epoch > 0 else history[0]
        rows.append(
            {
                "run": config.name,
                "changed": config.changed,
                "lr": config.lr,
                "hidden_sizes": "x".join(str(s) for s in config.hidden_sizes),
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "best_epoch": best_epoch,
                "best_val_mae": round(best["val_mae"], 3),
                "best_val_loss": round(best["val_loss"], 3),
                "final_train_loss": round(history[-1]["train_loss"], 3),
                "final_val_loss": round(history[-1]["val_loss"], 3),
            }
        )
        plot_curves(
            history,
            best_epoch,
            f"{config.name}: {config.changed}",
            REPORTS / f"curves_{config.name}.png",
        )

    experiments = pd.DataFrame(rows)
    experiments.to_csv(REPORTS / "experiments.csv", index=False)
    pd.DataFrame(histories[MAIN.name]).round(4).to_csv(
        REPORTS / "history.csv", index=False
    )
    main_best = int(
        experiments.loc[experiments["run"] == MAIN.name, "best_epoch"].iloc[0]
    )
    plot_curves(
        histories[MAIN.name],
        main_best,
        "Main run: training and validation loss",
        REPORTS / "learning_curves.png",
    )
    plot_experiments(histories, REPORTS / "experiments_val_loss.png")

    # The final network is the run with the lowest validation MAE.
    selected = experiments.sort_values("best_val_mae").iloc[0]["run"]
    final_model = models[selected]
    loaders = make_loaders(prepared, seed=SEED)
    network_val = score(prepared.y_val, predict(final_model, loaders["val"]))
    # The test rows are used once, here, after every choice has been made.
    network_test = score(prepared.y_test, predict(final_model, loaders["test"]))

    metrics = {
        "seed": SEED,
        "rows": {
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
        "baselines": baselines,
        "network": {
            "selected_run": selected,
            "config": asdict(
                next(c for c in [MAIN, *EXPERIMENTS] if c.name == selected)
            ),
            "best_epoch": int(
                experiments.loc[experiments["run"] == selected, "best_epoch"].iloc[0]
            ),
            "val": network_val,
            "test": network_test,
        },
    }
    with open(REPORTS / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    print("\nResults (MAE in rentals)")
    print("| Model | Validation MAE | Test MAE |")
    print("|---|---|---|")
    for label, result in [
        ("Training mean", baselines["train_mean"]),
        ("Linear regression", baselines["linear"]),
        (f"Network ({selected})", {"val": network_val, "test": network_test}),
    ]:
        print(f"| {label} | {result['val']['mae']:.2f} | {result['test']['mae']:.2f} |")
    print("\nExperiments")
    print(experiments.drop(columns=["changed"]).to_string(index=False))


if __name__ == "__main__":
    main()

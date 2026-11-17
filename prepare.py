"""Split, prepare and load the Lakeshore Bike Share table.

The table is split into training, validation and test rows first. Only then is
anything fitted: the scaler and the one-hot encoder learn from the training
rows alone and are applied unchanged to the validation and test rows.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import STATION_TYPES, TARGET

SEED = 2026
NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL = ["station_type"]
FEATURES = NUMERIC + CATEGORICAL


@dataclass
class Splits:
    """The raw rows of each split, before any preparation."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class Prepared:
    """Feature matrices and targets for each split, plus the fitted preprocessor."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]


def split_frame(
    frame: pd.DataFrame,
    seed: int = SEED,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> Splits:
    """Split the rows 60/20/20 with a fixed seed, before any preparation."""
    rest, test = train_test_split(frame, test_size=test_size, random_state=seed)
    val_share = val_size / (1.0 - test_size)
    train, val = train_test_split(rest, test_size=val_share, random_state=seed)
    return Splits(train=train, val=val, test=test)


def make_preprocessor() -> ColumnTransformer:
    """Scale the numeric columns and one-hot encode the station type.

    The one-hot columns are scaled too, so every training feature has mean zero
    and standard deviation one.
    """
    station = Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(
                    categories=[STATION_TYPES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("station", station, CATEGORICAL),
        ]
    )


def prepare(splits: Splits) -> Prepared:
    """Fit the preprocessor on the training rows only and transform every split."""
    preprocessor = make_preprocessor()
    x_train = preprocessor.fit_transform(splits.train[FEATURES])
    x_val = preprocessor.transform(splits.val[FEATURES])
    x_test = preprocessor.transform(splits.test[FEATURES])
    return Prepared(
        x_train=x_train.astype(np.float32),
        y_train=splits.train[TARGET].to_numpy(dtype=np.float32),
        x_val=x_val.astype(np.float32),
        y_val=splits.val[TARGET].to_numpy(dtype=np.float32),
        x_test=x_test.astype(np.float32),
        y_test=splits.test[TARGET].to_numpy(dtype=np.float32),
        preprocessor=preprocessor,
    )


class RentalsDataset(Dataset):
    """One station-hour at a time: a feature vector and the number rented."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same number of rows")
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # features: shape (n_features,); target: a scalar, so a batch is (N,)
        return self.features[index], self.targets[index]


def make_loaders(
    prepared: Prepared, batch_size: int = 64, seed: int = SEED
) -> dict[str, DataLoader]:
    """Build the three loaders. Only the training loader shuffles, with a seed."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(
            RentalsDataset(prepared.x_train, prepared.y_train),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        "val": DataLoader(
            RentalsDataset(prepared.x_val, prepared.y_val),
            batch_size=batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            RentalsDataset(prepared.x_test, prepared.y_test),
            batch_size=batch_size,
            shuffle=False,
        ),
    }

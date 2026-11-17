"""Shared fixtures: the table, its split and the prepared arrays."""

import pytest
import torch

from bikes_data import make_rentals
from prepare import prepare as prepare_splits
from prepare import split_frame


@pytest.fixture(scope="session")
def frame():
    return make_rentals()


@pytest.fixture(scope="session")
def splits(frame):
    return split_frame(frame, seed=2026)


@pytest.fixture(scope="session")
def prepared(splits):
    return prepare_splits(splits)


@pytest.fixture
def small_batch(prepared):
    """Thirty-two training rows as one batch of float32 tensors."""
    features = torch.as_tensor(prepared.x_train[:32])
    targets = torch.as_tensor(prepared.y_train[:32])
    return features, targets

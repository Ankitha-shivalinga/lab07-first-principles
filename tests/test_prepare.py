"""The split, the preparation, the Dataset and the loaders."""

import numpy as np
import pytest
import torch

from prepare import NUMERIC, RentalsDataset, make_loaders


def test_splits_have_expected_sizes_add_up_and_do_not_overlap(frame, splits):
    assert len(splits.train) == 3600
    assert len(splits.val) == 1200
    assert len(splits.test) == 1200
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(frame)

    train, val, test = (set(s.index) for s in (splits.train, splits.val, splits.test))
    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == set(frame.index)


def test_scaler_is_fitted_on_training_rows_only(splits, prepared):
    # Every training feature averages zero; the validation features do not.
    assert np.allclose(prepared.x_train.mean(axis=0), 0.0, atol=1e-4)
    assert np.abs(prepared.x_val.mean(axis=0)).max() > 1e-3

    scaler = prepared.preprocessor.named_transformers_["numeric"]
    assert np.allclose(scaler.mean_, splits.train[NUMERIC].mean().to_numpy())


def test_dataset_length_dtypes_and_shapes(prepared):
    dataset = RentalsDataset(prepared.x_train, prepared.y_train)
    assert len(dataset) == len(prepared.y_train)

    features, target = dataset[0]
    assert features.dtype == torch.float32
    assert target.dtype == torch.float32
    assert features.shape == (prepared.n_features,)
    assert target.shape == ()


def test_dataset_rejects_mismatched_lengths(prepared):
    with pytest.raises(ValueError):
        RentalsDataset(prepared.x_train, prepared.y_train[:-1])


def test_train_loader_shuffles_but_is_repeatable(prepared):
    first = next(iter(make_loaders(prepared, batch_size=64, seed=7)["train"]))
    again = next(iter(make_loaders(prepared, batch_size=64, seed=7)["train"]))
    other = next(iter(make_loaders(prepared, batch_size=64, seed=8)["train"]))

    in_order = torch.as_tensor(prepared.y_train[:64])
    assert torch.equal(first[1], again[1])
    assert not torch.equal(first[1], in_order)
    assert not torch.equal(first[1], other[1])

    # The validation loader keeps the rows in order.
    val_targets = next(iter(make_loaders(prepared, batch_size=64)["val"]))[1]
    assert torch.equal(val_targets, torch.as_tensor(prepared.y_val[:64]))

    features, targets = first
    assert features.shape == (64, prepared.n_features)
    assert targets.shape == (64,)

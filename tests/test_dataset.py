from __future__ import annotations

import torch

from ml_template.config_schemas import DataConfig
from ml_template.data import SyntheticDataset, build_dataloaders


def test_synthetic_dataset_shapes() -> None:
    ds = SyntheticDataset(n_samples=64, n_features=8, n_classes=3, seed=0)
    assert len(ds) == 64
    x, y = ds[0]
    assert x.shape == (8,)
    assert y.shape == ()
    assert y.dtype == torch.int64
    assert 0 <= int(y.item()) < 3


def test_dataset_is_deterministic_given_seed() -> None:
    a = SyntheticDataset(n_samples=32, n_features=4, n_classes=2, seed=42)
    b = SyntheticDataset(n_samples=32, n_features=4, n_classes=2, seed=42)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.y, b.y)


def test_build_dataloaders_split_sizes() -> None:
    cfg = DataConfig(n_samples=100, n_features=4, n_classes=2, batch_size=10, val_fraction=0.2)
    train, val = build_dataloaders(cfg, seed=0)
    train_n = sum(x.shape[0] for x, _ in train)
    val_n = sum(x.shape[0] for x, _ in val)
    assert train_n == 80
    assert val_n == 20

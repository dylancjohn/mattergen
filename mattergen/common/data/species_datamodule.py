"""LightningDataModule for training on the species vocabulary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from neutral_layer.data.vocab import build_species_vocab
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mattergen.common.data.collate import collate
from mattergen.common.data.datamodule import worker_init_fn
from mattergen.common.data.dataset import DatasetTransform
from mattergen.common.data.species_dataset import SpeciesCrystalDataset
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.common.data.types import PropertySourceId


class SpeciesDataModule(pl.LightningDataModule):
    """DataModule serving ``SpeciesCrystalDataset`` splits.

    Builds the default species vocabulary itself, so it is Hydra-instantiable
    without an external vocab object. ``data_dir`` must contain ``train/`` and
    ``val/``; without ``test/``, the val split is reused for testing.
    ``num_workers`` and ``batch_size`` map ``train``/``val``/``test`` to values.
    ``properties`` are loaded from ``{name}.json`` in each split, and
    ``dataset_transforms`` (e.g. ``filter_sparse_properties``) are applied to
    each loaded split. ``average_density`` is unused; it exists so the
    corruption config can interpolate ``${data_module.average_density}``. Extra
    Hydra keys are absorbed by ``**_``.

    The ``set_chemical_system_string`` transform is deliberately omitted: it
    reads ``atomic_numbers`` as atomic numbers, but here they are species
    indices.
    """

    def __init__(
        self,
        data_dir: str | Path,
        num_workers: DictConfig,
        batch_size: DictConfig,
        average_density: float | None = None,
        properties: list[PropertySourceId] | None = None,
        dataset_transforms: list[DatasetTransform] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size

        vocab = build_species_vocab()
        transforms = [symmetrize_lattice]
        dataset_transforms = dataset_transforms or []
        data_dir = Path(data_dir)

        def _load_split(split: str) -> SpeciesCrystalDataset:
            dataset = SpeciesCrystalDataset.from_cache_path(
                data_dir / split,
                vocab=vocab,
                transforms=transforms,
                properties=properties,
            )
            for t in dataset_transforms:
                dataset = t(dataset)
            return dataset

        self.train_dataset = _load_split("train")
        self.val_dataset = _load_split("val")
        # Fine-tuning datasets may ship only train/val.
        test_dir = data_dir / "test"
        self.test_dataset = (
            _load_split("test") if test_dir.exists() else self.val_dataset
        )

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size.test,
            num_workers=self.num_workers.test,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )

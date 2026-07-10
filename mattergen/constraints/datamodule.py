"""datamodule.py

LightningDataModule for species-vocabulary training.

``SpeciesDataModule`` is fully Hydra-instantiable: it accepts only primitive
constructor arguments (data directory path, num_workers, batch_size) and
builds the species vocab and datasets internally. This removes the need for a
custom training script to wire the vocab through to both the embedding and the
datasets.

Modules
-------
SpeciesDataModule
    DataModule that serves SpeciesCrystalDataset splits for train/val/test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mattergen.common.data.collate import collate
from mattergen.common.data.datamodule import worker_init_fn
from mattergen.common.data.dataset import DatasetTransform
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.common.data.types import PropertySourceId
from mattergen.constraints.dataset import SpeciesCrystalDataset
from neutral_layer.vocab import build_species_vocab


class SpeciesDataModule(pl.LightningDataModule):
    """DataModule serving SpeciesCrystalDataset splits for species-vocab training.

    Builds the species vocab internally so the class is fully Hydra-instantiable
    without an external vocab object. The ``set_chemical_system_string`` transform
    is intentionally excluded: it maps ``atomic_numbers`` to element symbols via
    atomic number lookup, but in species mode ``atomic_numbers`` holds species
    indices, not raw Z values.

    Parameters
    ----------
    data_dir : str or Path
        Root directory of the dataset. Must contain ``train/`` and ``val/``
        subdirectories. If ``test/`` is absent the val split is reused for
        test evaluation.
    num_workers : DictConfig
        Mapping with ``train``, ``val``, ``test`` worker counts.
    batch_size : DictConfig
        Mapping with ``train``, ``val``, ``test`` batch sizes.
    average_density : float or None
        Ignored at runtime; present so the corruption config can reference
        ``${data_module.average_density}`` via Hydra interpolation.
    properties : list of PropertySourceId or None
        Property names to load from ``{name}.json`` files in each split
        directory (e.g. ``["dft_band_gap"]``). Passed through to
        ``SpeciesCrystalDataset.from_cache_path``.
    dataset_transforms : list of DatasetTransform or None
        Whole-dataset transforms applied after loading each split, e.g.
        ``filter_sparse_properties`` to drop structures missing a property.
    **_ : Any
        Absorbs any extra config-only keys forwarded by Hydra.
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
                data_dir / split, vocab=vocab, transforms=transforms, properties=properties
            )
            for t in dataset_transforms:
                dataset = t(dataset)
            return dataset

        self.train_dataset = _load_split("train")
        self.val_dataset = _load_split("val")
        # Fall back to val when a held-out test split is not present (e.g. fine-tuning
        # datasets that only ship train/val).
        test_dir = data_dir / "test"
        self.test_dataset = _load_split("test") if test_dir.exists() else self.val_dataset

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

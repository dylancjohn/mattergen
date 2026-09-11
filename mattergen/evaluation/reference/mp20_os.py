"""Reference dataset built from an mp-20-os split.

Wraps the flat per-atom arrays each mp-20-os split is stored as (see
`mattergen/datasets/mp-20-os/`) into a `ReferenceDataset`, so it can be passed as `evaluate()`'s
`reference` argument wherever a comparison against mp-20-os itself -- rather than the default
Alex-MP/MP2020 reference -- is wanted (e.g. `OxidationStateDistance`, or novelty/uniqueness
against the training distribution specifically).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

from mattergen.evaluation.reference.reference_dataset import ReferenceDataset


def load_mp20_os_reference_dataset(mp20_os_dir: str | Path, split: str) -> ReferenceDataset:
    """Build a `ReferenceDataset` from one mp-20-os split (`"train"`, `"val"` or `"test"`).

    Every entry is given a dummy `energy=0.0`: mp-20-os carries no DFT energies on disk. The
    resulting dataset is therefore only valid for structure/composition-based metrics (novelty,
    uniqueness, precision, recall, `OxidationStateDistance`) -- never for anything energy-based
    (stability, energy above hull).
    """
    split_dir = Path(mp20_os_dir) / split
    atomic_numbers = np.load(split_dir / "atomic_numbers.npy")
    cell = np.load(split_dir / "cell.npy")
    pos = np.load(split_dir / "pos.npy")
    num_atoms = np.load(split_dir / "num_atoms.npy")
    offsets = np.concatenate([[0], np.cumsum(num_atoms)])

    entries = []
    for i in range(len(num_atoms)):
        start, end = int(offsets[i]), int(offsets[i + 1])
        # `pos.npy` holds *fractional* coordinates -- that is what `dataset.py` writes
        # (`structure_infos["pos"].append(struct.frac_coords)`) and what every other reader in
        # the codebase assumes (`CrystalDataset` applies `% 1.0`; `eval_utils.get_crystals_list`
        # builds with `coords_are_cartesian=False`). Reading them as Cartesian collapses every
        # reference structure into a blob near the origin, silently destroying the geometry that
        # fingerprint- and StructureMatcher-based metrics depend on.
        structure = Structure(
            Lattice(cell[i]), atomic_numbers[start:end], pos[start:end], coords_are_cartesian=False
        )
        entries.append(ComputedStructureEntry(structure=structure, energy=0.0))

    return ReferenceDataset.from_entries(f"mp20_os_{split}", entries)

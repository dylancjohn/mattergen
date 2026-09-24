"""Reference dataset built from an mp-20-os split.

Wraps a split's flat per-atom ``.npy`` arrays in a `ReferenceDataset`: the reference for
`mattergen.evaluation.evaluate_os`, and usable as `evaluate()`'s `reference` for novelty and
uniqueness against the training distribution rather than the default Alex-MP/MP2020 reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

from mattergen.evaluation.reference.reference_dataset import ReferenceDataset


def load_mp20_os_reference_dataset(mp20_os_dir: str | Path, split: str) -> ReferenceDataset:
    """Build a `ReferenceDataset` from one mp-20-os split (`"train"`, `"val"` or `"test"`).

    mp-20-os stores no DFT energies, so every entry gets a dummy `energy=0.0`. Use it only for
    structure- and composition-based metrics (e.g. novelty, uniqueness, coverage, oxidation-state
    distance), never energy-based ones (stability, energy above hull).
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
        # `pos.npy` holds fractional coordinates, as for `CrystalDataset`. Reading them as
        # Cartesian would collapse every structure near the origin and silently break the
        # fingerprint- and StructureMatcher-based metrics.
        structure = Structure(
            Lattice(cell[i]), atomic_numbers[start:end], pos[start:end], coords_are_cartesian=False
        )
        entries.append(ComputedStructureEntry(structure=structure, energy=0.0))

    return ReferenceDataset.from_entries(f"mp20_os_{split}", entries)

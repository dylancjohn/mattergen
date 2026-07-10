"""dataset.py

SpeciesCrystalDataset for training MatterGen on a species (element + OS) vocabulary.

Mirrors CrystalDataset but stores 1-based species indices in ChemGraph.atomic_numbers
rather than raw atomic numbers. Preserving the 1-based convention means AtomEmbedding
(which applies Z - 1) and D3PMCorruption (offset=1) are unchanged; only the vocab
size in model config needs to differ.

Six filters are applied at load time, mirroring the pipeline used by ICSDStructureDataset
in neurosymbolic-bertos with one deliberate difference for alloys:
  1. max atoms — drop oversized structures
  2. alloy check — keep metallic-only structures only if every atom has OS=0
  3. single-species — drop elemental compounds
  4. unknown (z, os) pairs — drop structures with species absent from the vocabulary
  5. charge neutrality — always drop non-neutral structures (log_z = -inf in SPL)
  6. MV element registry — drop structures where a non-MV element carries >1 distinct OS

Modules:
    SpeciesCrystalDataset — dataset that maps (atomic_number, os) pairs to species indices.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing
import torch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.dataset import CORE_STRUCTURE_FILE_NAMES, BaseDataset
from mattergen.common.data.transform import Transform
from mattergen.common.data.types import PropertySourceId, PropertyValues
from mattergen.common.utils.globals import PROPERTY_SOURCE_IDS
from neutral_layer.vocab import MIXED_VALENCE_ELEMENTS, SpeciesVocab

OXIDATION_STATES_FILE = "oxidation_states.npy"


# Pymatgen metals by atomic number — used by the alloy filter.
# Import deferred to avoid a module-level pymatgen parse on every import.
def _build_metal_atomic_numbers() -> frozenset[int]:
    from pymatgen.core import Element as _PmgElement

    return frozenset(e.Z for e in _PmgElement if e.is_metal)


_METAL_ATOMIC_NUMBERS: frozenset[int] = _build_metal_atomic_numbers()

# Bounds matching OS_VALUES in vocab.py: -5 to +8.
_OS_MIN: int = -5
_OS_MAX: int = 8
_OS_OFFSET: int = -_OS_MIN  # shift so _OS_MIN maps to column 0
_OS_COLS: int = _OS_MAX - _OS_MIN + 1  # 14 columns

# Upper bound on atomic numbers; matches MAX_ATOMIC_NUM in globals.
_Z_MAX: int = 100


def _build_species_indices(
    atomic_numbers: numpy.typing.NDArray,
    oxidation_states: numpy.typing.NDArray,
    vocab: SpeciesVocab,
) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]:
    """Map parallel flat arrays of atomic numbers and OS to 1-based species indices.

    Uses a 2D lookup table for O(N) vectorised mapping. A value of 0 in the
    returned indices array marks atoms whose (z, os) pair is absent from the
    vocabulary; callers decide how to handle these (raise or filter).

    Parameters
    ----------
    atomic_numbers
        Flat int64 array of 1-based atomic numbers, shape [N_atoms].
    oxidation_states
        Flat int64 array of oxidation states, shape [N_atoms].
    vocab
        Species vocabulary defining valid (z, os) pairs and their indices.

    Returns
    -------
    species_indices
        Flat int64 array of 1-based species indices, shape [N_atoms].
        Entries are 0 for atoms with unknown (z, os) pairs.
    invalid_mask
        Boolean array, shape [N_atoms]. True where the (z, os) pair is not in vocab.
    """
    lookup = np.zeros((_Z_MAX + 1, _OS_COLS), dtype=np.int64)
    for (z, oxs), idx in vocab.species_to_idx.items():
        if 0 <= z <= _Z_MAX and _OS_MIN <= oxs <= _OS_MAX:
            lookup[z, oxs + _OS_OFFSET] = idx

    os_in_range = (oxidation_states >= _OS_MIN) & (oxidation_states <= _OS_MAX)
    z_in_range = (atomic_numbers >= 0) & (atomic_numbers <= _Z_MAX)

    safe_z = np.clip(atomic_numbers, 0, _Z_MAX)
    safe_os_col = np.clip(oxidation_states + _OS_OFFSET, 0, _OS_COLS - 1)
    indices = lookup[safe_z, safe_os_col]

    invalid_mask = (indices == 0) | ~os_in_range | ~z_in_range
    return indices, invalid_mask


@dataclass(frozen=True, kw_only=True)
class SpeciesCrystalDataset(BaseDataset):
    """Dataset for crystal structures using a species (element + OS) vocabulary.

    Mirrors CrystalDataset but stores 1-based species indices in ChemGraph.atomic_numbers.
    Use from_cache_path to load from a directory containing the standard MatterGen
    .npy files plus oxidation_states.npy.

    Attributes
    ----------
    pos
        Fractional coordinates, shape [N_atoms_total, 3].
    cell
        Lattice matrices, shape [N_structures, 3, 3].
    species_indices
        1-based species indices, shape [N_atoms_total]. Precomputed at load time.
    num_atoms
        Number of atoms per structure, shape [N_structures].
    structure_id
        Structure identifiers, shape [N_structures].
    vocab
        Species vocabulary used to interpret species_indices.
    properties
        Per-structure property arrays, keyed by PropertySourceId.
    transforms
        Per-sample transforms applied sequentially in __getitem__.
    """

    pos: numpy.typing.NDArray
    cell: numpy.typing.NDArray
    species_indices: numpy.typing.NDArray
    num_atoms: numpy.typing.NDArray
    structure_id: numpy.typing.NDArray
    vocab: SpeciesVocab
    properties: dict[PropertySourceId, numpy.typing.NDArray] = field(
        default_factory=dict
    )
    transforms: list[Transform] | None = None

    def __post_init__(self) -> None:
        property_names = list(self.properties.keys())
        assert all(s in PROPERTY_SOURCE_IDS for s in property_names), (
            f"Property names {property_names} are not valid. "
            f"Valid property source names: {PROPERTY_SOURCE_IDS}"
        )

    @cached_property
    def index_offset(self) -> numpy.typing.NDArray:
        """Cumulative atom offset for indexing into pos and species_indices."""
        return np.concatenate([np.array([0]), np.cumsum(self.num_atoms[:-1])])

    def __getitem__(self, index: int) -> ChemGraph:
        pos_offset = self.index_offset[index]
        num_atoms = torch.tensor(self.num_atoms[index])

        props_dict = self.get_properties_dict(index)
        data = ChemGraph(
            pos=torch.from_numpy(self.pos[pos_offset : pos_offset + num_atoms]).float()
            % 1.0,
            cell=torch.from_numpy(self.cell[index]).float().unsqueeze(0),
            atomic_numbers=torch.from_numpy(
                self.species_indices[pos_offset : pos_offset + num_atoms]
            ),
            num_atoms=num_atoms,
            num_nodes=num_atoms,
            **props_dict,  # type: ignore[arg-type]
        )

        if self.transforms is not None:
            for t in self.transforms:
                data = t(data)
        return data

    def __len__(self) -> int:
        return len(self.num_atoms)

    def subset(self, indices: Sequence[int]) -> "SpeciesCrystalDataset":
        """Return a new dataset containing only the structures at the given indices."""
        batch_indices: list[int] = []
        for index in indices:
            pos_offset = self.index_offset[index]
            batch_indices.extend(range(pos_offset, pos_offset + self.num_atoms[index]))

        idx_list = list(indices)
        return SpeciesCrystalDataset(
            pos=self.pos[batch_indices],
            cell=self.cell[idx_list],
            species_indices=self.species_indices[batch_indices],
            num_atoms=self.num_atoms[idx_list],
            structure_id=self.structure_id[idx_list],
            vocab=self.vocab,
            properties={k: v[idx_list] for k, v in self.properties.items()},
            transforms=self.transforms,
        )

    @classmethod
    def from_cache_path(  # type: ignore[override]
        cls,
        cache_path: str | Path,
        vocab: SpeciesVocab,
        transforms: list[Transform] | None = None,
        properties: list[PropertySourceId] | None = None,
        filter_unknown_species: bool = True,
        max_atoms: int = 200,
        mv_elements: frozenset[str] | None = MIXED_VALENCE_ELEMENTS,
    ) -> "SpeciesCrystalDataset":
        """Load a SpeciesCrystalDataset from a directory of .npy files.

        Reads the standard MatterGen files (pos.npy, cell.npy, atomic_numbers.npy,
        num_atoms.npy, structure_id.npy) plus oxidation_states.npy, then maps each
        (atomic_number, os) pair to a 1-based species index via vocab. Six filters
        are applied (see module docstring for ordering and rationale).

        Parameters
        ----------
        cache_path
            Directory containing the .npy files.
        vocab
            Species vocabulary for mapping (atomic_number, os) to indices.
        transforms
            Per-sample transforms applied in __getitem__.
        properties
            Property names to load from {prop}.json files in cache_path.
        filter_unknown_species
            If True (default), silently drop structures that contain any
            (atomic_number, oxidation_state) pair absent from vocab and log the
            count. If False, raise ValueError listing the unknown pairs instead.
        max_atoms
            Structures with more atoms than this threshold are dropped.
        mv_elements
            Elements permitted to carry more than one distinct OS per compound.
            Structures where any other element appears with multiple OS values are
            dropped. Pass ``None`` to disable this filter.

        Returns
        -------
        dataset
            Loaded SpeciesCrystalDataset.

        Raises
        ------
        FileNotFoundError
            If any required .npy file or property .json file is missing.
        ValueError
            If ``filter_unknown_species=False`` and any (z, os) pair is absent
            from vocab.
        """
        import logging

        log = logging.getLogger(__name__)

        cache_path = str(cache_path)

        def _load(filename: str) -> numpy.typing.NDArray:
            path = os.path.join(cache_path, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
            # allow_pickle=True is needed for structure_id arrays which are stored
            # as numpy object arrays (string IDs like "mp-1234").
            return np.load(path, allow_pickle=True)

        pos = _load(CORE_STRUCTURE_FILE_NAMES["pos"])
        cell = _load(CORE_STRUCTURE_FILE_NAMES["cell"])
        raw_atomic_numbers = _load(CORE_STRUCTURE_FILE_NAMES["atomic_numbers"])
        num_atoms = _load(CORE_STRUCTURE_FILE_NAMES["num_atoms"])
        structure_id = _load(CORE_STRUCTURE_FILE_NAMES["structure_id"])
        oxidation_states = _load(OXIDATION_STATES_FILE)

        species_indices, invalid_atom_mask = _build_species_indices(
            raw_atomic_numbers, oxidation_states, vocab
        )

        # Build per-atom → per-structure mapping and cumulative atom offsets.
        atom_struct_idx = np.repeat(np.arange(len(num_atoms)), num_atoms)
        offsets = np.concatenate([[0], np.cumsum(num_atoms)])

        # ── Filter 1: max atoms ───────────────────────────────────────────────
        too_large_mask = num_atoms > max_atoms
        if too_large_mask.any():
            log.warning(
                "%s: dropped %d/%d structures with >%d atoms",
                cache_path,
                int(too_large_mask.sum()),
                len(num_atoms),
                max_atoms,
            )

        # ── Filter 2: alloy check ─────────────────────────────────────────────
        # Metallic-only structures are kept only when every atom carries OS=0.
        # Non-zero OS on a metal-only compound is an ICSD artefact.
        is_metal_atom = np.array(
            [z in _METAL_ATOMIC_NUMBERS for z in raw_atomic_numbers.tolist()]
        )
        alloy_bad = np.zeros(len(num_atoms), dtype=bool)
        for i in range(len(num_atoms)):
            s, e = int(offsets[i]), int(offsets[i + 1])
            if is_metal_atom[s:e].all():
                alloy_bad[i] = not (oxidation_states[s:e] == 0).all()
        if alloy_bad.any():
            log.warning(
                "%s: dropped %d/%d metallic-only structures with non-zero OS",
                cache_path,
                int(alloy_bad.sum()),
                len(num_atoms),
            )

        # ── Filter 3: single-species ──────────────────────────────────────────
        single_species_bad = np.array(
            [
                len(
                    set(
                        raw_atomic_numbers[
                            int(offsets[i]) : int(offsets[i + 1])
                        ].tolist()
                    )
                )
                == 1
                for i in range(len(num_atoms))
            ],
            dtype=bool,
        )
        if single_species_bad.any():
            log.warning(
                "%s: dropped %d/%d single-species structures",
                cache_path,
                int(single_species_bad.sum()),
                len(num_atoms),
            )

        # ── Filter 4: unknown (z, os) pairs ──────────────────────────────────
        if invalid_atom_mask.any():
            unknown_pairs: set[tuple[int, int]] = set(
                zip(
                    raw_atomic_numbers[invalid_atom_mask].tolist(),
                    oxidation_states[invalid_atom_mask].tolist(),
                )
            )
            if not filter_unknown_species:
                raise ValueError(
                    f"The following (atomic_number, oxidation_state) pairs are not in the "
                    f"species vocabulary: {sorted(unknown_pairs)}"
                )
            invalid_count_per_struct = np.zeros(len(num_atoms), dtype=np.int64)
            np.add.at(
                invalid_count_per_struct,
                atom_struct_idx,
                invalid_atom_mask.astype(np.int64),
            )
            bad_from_unknown = invalid_count_per_struct > 0
            log.warning(
                "%s: dropped %d/%d structures containing unknown (z, os) pairs: %s",
                cache_path,
                int(bad_from_unknown.sum()),
                len(num_atoms),
                sorted(unknown_pairs),
            )
        else:
            bad_from_unknown = np.zeros(len(num_atoms), dtype=bool)

        # ── Filter 5: non-neutral structures ─────────────────────────────────
        # Non-neutral structures cause log_z = -inf for all timesteps once
        # their atoms are committed, triggering NaN in the SPL backward pass.
        charge_per_struct = np.zeros(len(num_atoms), dtype=np.int64)
        np.add.at(charge_per_struct, atom_struct_idx, oxidation_states)
        non_neutral_mask = charge_per_struct != 0
        if non_neutral_mask.any():
            log.warning(
                "%s: dropped %d/%d non-neutral structures (total OS ≠ 0)",
                cache_path,
                int(non_neutral_mask.sum()),
                len(num_atoms),
            )

        # ── Filter 6: MV element registry ────────────────────────────────────
        # Non-MV elements must carry exactly one distinct OS per structure.
        mv_bad = np.zeros(len(num_atoms), dtype=bool)
        if mv_elements is not None:
            from pymatgen.core import Element as _PmgEl

            z_to_sym: dict[int, str] = {e.Z: e.symbol for e in _PmgEl}
            for i in range(len(num_atoms)):
                s, e = int(offsets[i]), int(offsets[i + 1])
                el_os: dict[str, set[int]] = {}
                for z, oxs in zip(
                    raw_atomic_numbers[s:e].tolist(), oxidation_states[s:e].tolist()
                ):
                    sym = z_to_sym.get(z, "")
                    el_os.setdefault(sym, set()).add(oxs)
                for sym, charges in el_os.items():
                    if len(charges) > 1 and sym not in mv_elements:
                        mv_bad[i] = True
                        break
            if mv_bad.any():
                log.warning(
                    "%s: dropped %d/%d structures where a non-MV element carries >1 distinct OS",
                    cache_path,
                    int(mv_bad.sum()),
                    len(num_atoms),
                )

        # Load properties before filtering so the keep_struct mask can be applied.
        property_names = properties or []
        props: dict[PropertySourceId, numpy.typing.NDArray] = {}
        for prop_name in property_names:
            prop_path = os.path.join(cache_path, f"{prop_name}.json")
            if not os.path.exists(prop_path):
                raise FileNotFoundError(
                    f"{prop_name}.json does not exist in {cache_path}."
                )
            props[prop_name] = PropertyValues.from_json(prop_path).values

        # ── Combined filtering pass ───────────────────────────────────────────
        bad_struct_mask = (
            too_large_mask
            | alloy_bad
            | single_species_bad
            | bad_from_unknown
            | non_neutral_mask
            | mv_bad
        )
        if bad_struct_mask.any():
            keep_struct = ~bad_struct_mask
            cell = cell[keep_struct]
            num_atoms = num_atoms[keep_struct]
            structure_id = structure_id[keep_struct]
            for prop_name in props:
                props[prop_name] = props[prop_name][keep_struct]

            keep_atom = ~np.isin(atom_struct_idx, np.where(bad_struct_mask)[0])
            pos = pos[keep_atom]
            species_indices = species_indices[keep_atom]

        return cls(
            pos=pos,
            cell=cell,
            species_indices=species_indices,
            num_atoms=num_atoms,
            structure_id=structure_id,
            vocab=vocab,
            properties=props,
            transforms=transforms,
        )
